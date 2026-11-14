"""Executor agent: enforce policy, request dry quotes, and audit decisions.

The Executor never signs, submits, or broadcasts a transaction. Because
NEAR Intents dry quotes omit the deposit address required by generate-intent,
it can build only a clearly labeled local preview draft from the dry quote.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, localcontext
from pathlib import Path
from typing import Any, Literal, TypedDict

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "policy.json"
AUDIT_LOG_PATH = PROJECT_ROOT / "logs" / "audit.jsonl"
ONECLICK_TOKENS_URL = "https://1click.chaindefuser.com/v0/tokens"
ONECLICK_QUOTE_URL = "https://1click.chaindefuser.com/v0/quote"
REQUEST_TIMEOUT_SECONDS = 15
GENESIS_HASH = "0" * 64
AUDIT_LOCK = threading.Lock()

# The NEAR alias comes from the NEAR Intents token catalogue's documented
# wNEAR asset. The actual asset ID is still checked against the live catalogue
# before it is used in a quote request.
TOKEN_SYMBOL_ALIASES: dict[str, set[str]] = {"near": {"near", "wnear"}}


class SpendingPolicy(TypedDict):
    allowed_tokens: list[str]
    max_amount_usd: float
    min_confidence: float
    cooldown_seconds: int


class ExecutorMessage(TypedDict):
    agent: Literal["Executor"]
    status: Literal["done", "blocked"]
    timestamp: str
    decision: Literal["approved", "blocked"]
    rule: str
    policy: dict[str, Any]
    quote: dict[str, Any] | None
    draft_intent: dict[str, Any] | None
    draft_intent_label: str
    audit: dict[str, Any]
    message: str


class PolicyError(ValueError):
    """Raised when the human-authored spending policy is invalid."""


class AuditLogError(RuntimeError):
    """Raised when an audit record cannot safely be appended or verified."""


class QuoteError(RuntimeError):
    """Raised when the 1Click API cannot return usable data."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    moment = value or _utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _finite_number(value: Any, field: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{field} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise PolicyError(f"{field} is outside the supported numeric range") from exc
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        limit = f" and at most {maximum:g}" if maximum is not None else ""
        raise PolicyError(f"{field} must be finite and at least {minimum:g}{limit}")
    return result


def load_policy(path: Path | None = None) -> SpendingPolicy:
    """Load policy.json, applying the documented amount and confidence defaults."""
    policy_path = Path(path) if path is not None else POLICY_PATH
    try:
        with policy_path.open("r", encoding="utf-8") as policy_file:
            raw = json.load(policy_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"could not load policy at {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyError("policy.json must contain a JSON object")

    allowed_tokens = raw.get("allowed_tokens")
    if not isinstance(allowed_tokens, list) or not allowed_tokens:
        raise PolicyError("allowed_tokens must be a non-empty list of 1Click asset IDs")
    if any(not isinstance(token, str) or not token.strip() for token in allowed_tokens):
        raise PolicyError("every allowed_tokens entry must be a non-empty string")

    max_amount_usd = _finite_number(raw.get("max_amount_usd", 5), "max_amount_usd", 0)
    min_confidence = _finite_number(raw.get("min_confidence", 0.7), "min_confidence", 0, 1)
    cooldown_value = raw.get("cooldown_seconds", 60)
    if isinstance(cooldown_value, bool) or not isinstance(cooldown_value, int) or cooldown_value < 0:
        raise PolicyError("cooldown_seconds must be a non-negative integer")

    return {
        "allowed_tokens": [token.strip() for token in allowed_tokens],
        "max_amount_usd": max_amount_usd,
        "min_confidence": min_confidence,
        "cooldown_seconds": cooldown_value,
    }


def _parse_time(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def check_policy(
    proposal: Mapping[str, Any],
    policy: SpendingPolicy,
    *,
    last_run_at: datetime | str | None = None,
    resolved_asset_ids: tuple[str, str] | None = None,
    now: datetime | None = None,
) -> tuple[Literal["approved", "blocked"], str]:
    """Apply the spending policy using ordinary Python checks only."""
    action = proposal.get("action")
    if action != "swap":
        return "blocked", f"No swap requested (Analyst action is {action!r}); Executor will not request a quote."

    amount = proposal.get("amount_usd")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return "blocked", "Blocked: Analyst amount_usd is not a number."
    try:
        amount_value = float(amount)
    except OverflowError:
        return "blocked", "Blocked: Analyst amount_usd is outside the supported numeric range."
    if not math.isfinite(amount_value):
        return "blocked", "Blocked: Analyst amount_usd is not a finite number."
    if amount_value <= 0:
        return "blocked", "Blocked: amount_usd must be greater than 0 for a swap."
    if amount_value > policy["max_amount_usd"]:
        return "blocked", (
            f"Blocked: amount_usd ${amount_value:.2f} exceeds policy max_amount_usd "
            f"${policy['max_amount_usd']:.2f}."
        )

    confidence = proposal.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return "blocked", "Blocked: Analyst confidence is not a number."
    try:
        confidence_value = float(confidence)
    except OverflowError:
        return "blocked", "Blocked: Analyst confidence is outside the supported numeric range."
    if not math.isfinite(confidence_value):
        return "blocked", "Blocked: Analyst confidence is not a finite number."
    if confidence_value < policy["min_confidence"]:
        return "blocked", (
            f"Blocked: confidence {confidence_value:.2f} is below policy min_confidence "
            f"{policy['min_confidence']:.2f}."
        )

    from_token = proposal.get("from_token")
    to_token = proposal.get("to_token")
    if not isinstance(from_token, str) or not from_token.strip() or not isinstance(to_token, str) or not to_token.strip():
        return "blocked", "Blocked: a swap requires non-empty from_token and to_token values."
    if from_token.casefold() == to_token.casefold():
        return "blocked", "Blocked: from_token and to_token are the same token."

    if last_run_at is not None and policy["cooldown_seconds"] > 0:
        try:
            elapsed = ((_parse_time(now or _utc_now())) - _parse_time(last_run_at)).total_seconds()
        except (TypeError, ValueError) as exc:
            return "blocked", f"Blocked: last run timestamp is invalid ({exc})."
        if elapsed < policy["cooldown_seconds"]:
            remaining = max(0, math.ceil(policy["cooldown_seconds"] - elapsed))
            return "blocked", f"Blocked: policy cooldown is active for {remaining} more second(s)."

    if resolved_asset_ids is not None:
        allowed = set(policy["allowed_tokens"])
        if resolved_asset_ids[0] == resolved_asset_ids[1]:
            return "blocked", "Blocked: from_token and to_token resolve to the same NEAR Intents asset."
        for label, asset_id in zip(("from_token", "to_token"), resolved_asset_ids):
            if asset_id not in allowed:
                return "blocked", f"Blocked: {label} resolved to {asset_id!r}, which is not in policy allowed_tokens."

    return "approved", "Approved: amount, confidence, token, action, and cooldown checks passed policy."


def _fetch_supported_tokens(session: requests.Session) -> list[dict[str, Any]]:
    response = session.get(ONECLICK_TOKENS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 429:
        raise QuoteError("NEAR Intents token catalog rate-limited the request (HTTP 429).")
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, list) or not all(isinstance(token, dict) for token in body):
        raise QuoteError("NEAR Intents returned an invalid supported-token catalog.")
    return body


def _resolve_allowed_asset(
    proposed_token: str,
    catalog: list[dict[str, Any]],
    policy: SpendingPolicy,
) -> dict[str, Any]:
    allowed_ids = set(policy["allowed_tokens"])
    wanted = proposed_token.strip()
    exact_id_matches = [
        item for item in catalog if item.get("assetId") == wanted and item.get("assetId") in allowed_ids
    ]
    if exact_id_matches:
        candidates = exact_id_matches
    else:
        symbol_key = wanted.casefold()
        accepted_symbols = TOKEN_SYMBOL_ALIASES.get(symbol_key, {symbol_key})
        candidates = [
            item
            for item in catalog
            if isinstance(item.get("symbol"), str)
            and item["symbol"].casefold() in accepted_symbols
            and item.get("assetId") in allowed_ids
        ]

    # Deduplicate any repeated catalogue rows by their exact API asset ID.
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        asset_id = item.get("assetId")
        if isinstance(asset_id, str):
            unique[asset_id] = item
    if not unique:
        raise QuoteError(f"{proposed_token!r} did not resolve to an asset in policy allowed_tokens.")
    if len(unique) > 1:
        raise QuoteError(
            f"{proposed_token!r} maps to multiple allowed NEAR Intents assets; use an exact asset ID in the proposal."
        )
    token = next(iter(unique.values()))
    decimals = token.get("decimals")
    if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 36:
        raise QuoteError(f"NEAR Intents returned invalid decimals for {token.get('assetId')!r}.")
    return token


def _amount_in_smallest_unit(
    amount_usd: float,
    token: Mapping[str, Any],
    scout_metrics: Mapping[str, Any],
    proposal_token: str,
) -> str:
    prices = scout_metrics.get("prices")
    if not isinstance(prices, Mapping):
        raise QuoteError("Scout metrics did not contain a prices map; cannot convert USD to token base units.")
    price_row = prices.get(proposal_token)
    if not isinstance(price_row, Mapping):
        # Resolve case-insensitively because the UI and typed proposal may case differently.
        price_row = next(
            (row for symbol, row in prices.items() if isinstance(symbol, str) and symbol.casefold() == proposal_token.casefold() and isinstance(row, Mapping)),
            None,
        )
    if not isinstance(price_row, Mapping):
        raise QuoteError(f"Scout metrics have no USD price for origin token {proposal_token!r}.")
    try:
        price_usd = Decimal(str(price_row.get("usd")))
        requested_usd = Decimal(str(amount_usd))
        decimals = int(token["decimals"])
    except (InvalidOperation, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise QuoteError(f"Could not convert the USD proposal into token base units: {exc}") from exc
    if not price_usd.is_finite() or price_usd <= 0 or not requested_usd.is_finite() or requested_usd <= 0:
        raise QuoteError("Origin token USD price or converted base-unit amount is invalid.")
    try:
        with localcontext() as context:
            context.prec = max(50, decimals + 20)
            base_units = (requested_usd / price_usd * (Decimal(10) ** decimals)).to_integral_value(rounding=ROUND_DOWN)
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise QuoteError(f"Could not convert the USD proposal into token base units: {exc}") from exc
    if not base_units.is_finite() or base_units <= 0:
        raise QuoteError("Origin token USD price or converted base-unit amount is invalid.")
    return format(base_units, "f").split(".")[0]


def _oneclick_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _request_dry_quote(
    session: requests.Session,
    *,
    request_body: dict[str, Any],
    api_key: str | None,
) -> dict[str, Any]:
    response = session.post(
        ONECLICK_QUOTE_URL,
        headers=_oneclick_headers(api_key),
        json=request_body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 429:
        raise QuoteError("NEAR Intents dry quote was rate-limited (HTTP 429).")
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("quote"), dict):
        message = body.get("message") if isinstance(body, dict) else None
        detail = f": {message}" if isinstance(message, str) else ""
        raise QuoteError(f"NEAR Intents returned no quote object{detail}.")
    return body


def _safe_quote_view(body: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only documented, display-safe dry quote fields; omit addresses."""
    raw_request = body.get("quoteRequest")
    raw_quote = body.get("quote")
    request_fields = (
        "dry",
        "swapType",
        "slippageTolerance",
        "originAsset",
        "depositType",
        "destinationAsset",
        "amount",
        "recipientType",
        "refundType",
        "deadline",
    )
    quote_fields = (
        "amountIn",
        "amountInFormatted",
        "amountInUsd",
        "minAmountIn",
        "amountOut",
        "amountOutFormatted",
        "amountOutUsd",
        "minAmountOut",
        "timeEstimate",
        "refundFee",
        "withdrawFee",
        "deadline",
    )
    safe_request = (
        {field: raw_request[field] for field in request_fields if field in raw_request}
        if isinstance(raw_request, Mapping)
        else {}
    )
    safe_quote = (
        {field: raw_quote[field] for field in quote_fields if field in raw_quote}
        if isinstance(raw_quote, Mapping)
        else {}
    )
    return {
        "correlationId": body.get("correlationId"),
        "request": safe_request,
        "quote": safe_quote,
        "quote_note": "The route shown is the requested asset pair; the documented response has no solver route field.",
    }


def _quote_summary(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if body is None:
        return None
    return dict(body)


def _build_local_draft(
    *,
    proposal: Mapping[str, Any],
    policy_rule: str,
    requested_amount_usd: float,
    origin_asset: str,
    destination_asset: str,
    amount_units: str,
    quote_request: Mapping[str, Any],
    quote_view: Mapping[str, Any],
) -> dict[str, Any]:
    request = quote_view.get("request")
    quote = quote_view.get("quote")
    request = request if isinstance(request, Mapping) else {}
    quote = quote if isinstance(quote, Mapping) else {}
    fees = {key: quote[key] for key in ("refundFee", "withdrawFee") if key in quote}
    amounts = {
        "requested_amount_usd": requested_amount_usd,
        "requested_origin_amount_smallest_unit": amount_units,
    }
    for field in (
        "amountIn",
        "amountInFormatted",
        "amountInUsd",
        "minAmountIn",
        "amountOut",
        "amountOutFormatted",
        "amountOutUsd",
        "minAmountOut",
    ):
        if field in quote:
            amounts[field] = quote[field]
    return {
        "label": "Draft intent (not generated by generate-intent, not signed, not submitted)",
        "draft_type": "local_dry_quote_preview",
        "policy_decision": "approved",
        "policy_rule": policy_rule,
        "analyst_proposal": dict(proposal),
        "requested_route": {
            "originAsset": origin_asset,
            "destinationAsset": destination_asset,
            "note": "Requested asset pair only; NEAR Intents did not return a solver route field.",
        },
        "amounts": amounts,
        "fees_raw": fees,
        "deadline": {
            "requested": request.get("deadline", quote_request.get("deadline")),
            "returned_by_dry_quote": quote.get("deadline"),
            "note": "A dry quote does not provide an executable quote deadline; requested input is shown separately.",
        },
        "quote_correlation_id": quote_view.get("correlationId"),
        "deposit_address_note": "No depositAddress is included or inferred.",
    }


def verify_audit_log(path: Path | None = None) -> dict[str, Any]:
    """Recompute every record hash and previous-hash link in the JSONL log."""
    audit_path = Path(path) if path is not None else AUDIT_LOG_PATH
    if not audit_path.exists():
        return {"valid": True, "records": 0, "last_hash": GENESIS_HASH, "message": "Audit log is empty; no records to verify."}

    previous_hash = GENESIS_HASH
    records = 0
    try:
        with audit_path.open("r", encoding="utf-8") as audit_file:
            for line_number, line in enumerate(audit_file, start=1):
                if not line.strip():
                    return {
                        "valid": False,
                        "records": records,
                        "last_hash": previous_hash,
                        "message": f"Audit verification failed at line {line_number}: blank lines are not allowed.",
                    }
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("record_hash"), str):
                    return {
                        "valid": False,
                        "records": records,
                        "last_hash": previous_hash,
                        "message": f"Audit verification failed at line {line_number}: malformed record.",
                    }
                if record.get("previous_hash") != previous_hash:
                    return {
                        "valid": False,
                        "records": records,
                        "last_hash": previous_hash,
                        "message": f"Audit verification failed at line {line_number}: previous_hash does not match.",
                    }
                record_hash = record["record_hash"]
                unhashed_record = {key: value for key, value in record.items() if key != "record_hash"}
                expected_hash = hashlib.sha256(_canonical_json(unhashed_record).encode("utf-8")).hexdigest()
                if not hmac.compare_digest(record_hash, expected_hash):
                    return {
                        "valid": False,
                        "records": records,
                        "last_hash": previous_hash,
                        "message": f"Audit verification failed at line {line_number}: record hash does not match.",
                    }
                previous_hash = record_hash
                records += 1
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "valid": False,
            "records": records,
            "last_hash": previous_hash,
            "message": f"Audit verification failed: {type(exc).__name__}: {exc}",
        }
    return {"valid": True, "records": records, "last_hash": previous_hash, "message": f"Verified {records} audit record(s)."}


def append_audit_record(event: Mapping[str, Any], path: Path | None = None) -> str:
    """Append one tamper-evident JSONL record and return its SHA-256 hash."""
    audit_path = Path(path) if path is not None else AUDIT_LOG_PATH
    with AUDIT_LOCK:
        verification = verify_audit_log(audit_path)
        if not verification["valid"]:
            raise AuditLogError(str(verification["message"]))
        previous_hash = str(verification["last_hash"])
        record: dict[str, Any] = {**dict(event), "previous_hash": previous_hash}
        record_hash = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
        record["record_hash"] = record_hash
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8", newline="\n") as audit_file:
                audit_file.write(_canonical_json(record) + "\n")
        except OSError as exc:
            raise AuditLogError(f"could not append {audit_path}: {exc}") from exc
        return record_hash


def _finish(
    *,
    decision: Literal["approved", "blocked"],
    rule: str,
    policy: Mapping[str, Any],
    proposal: Mapping[str, Any],
    message: str,
    status: Literal["done", "blocked"],
    quote: dict[str, Any] | None = None,
    draft_intent: dict[str, Any] | None = None,
    draft_intent_label: str = "No local draft intent is available.",
    quote_status: str = "not_requested",
    audit_path: Path | None = None,
    now: datetime | None = None,
) -> ExecutorMessage:
    created_at = _timestamp(now)
    event = {
        "timestamp": created_at,
        "decision": decision,
        "rule": rule,
        "proposal": dict(proposal),
        "quote_status": quote_status,
        "quote_summary": _quote_summary(quote),
        "message": message,
    }
    try:
        record_hash = append_audit_record(event, audit_path)
        audit = {"written": True, "record_hash": record_hash, "path": str(audit_path or AUDIT_LOG_PATH)}
    except (AuditLogError, OSError, TypeError, ValueError) as exc:
        audit = {"written": False, "message": str(exc)}
        message = f"{message} Audit log write failed: {exc}"
        status = "blocked"
    return {
        "agent": "Executor",
        "status": status,
        "timestamp": created_at,
        "decision": decision,
        "rule": rule,
        "policy": dict(policy),
        "quote": quote,
        "draft_intent": draft_intent,
        "draft_intent_label": draft_intent_label,
        "audit": audit,
        "message": message,
    }


def run_executor(
    analyst_message: Mapping[str, Any],
    scout_metrics: Mapping[str, Any],
    *,
    policy_path: Path | None = None,
    audit_path: Path | None = None,
    last_run_at: datetime | str | None = None,
    account_id: str | None = None,
    api_key: str | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> ExecutorMessage:
    """Check policy, then request a 1Click dry quote for an approved swap."""
    raw_proposal = analyst_message.get("proposal")
    proposal: Mapping[str, Any] = raw_proposal if isinstance(raw_proposal, Mapping) else {}
    policy_display: dict[str, Any] = {}
    try:
        policy = load_policy(policy_path)
        policy_display = dict(policy)
    except PolicyError as exc:
        return _finish(
            decision="blocked",
            rule=f"Blocked: invalid human spending policy. {exc}",
            policy=policy_display,
            proposal=proposal,
            message="Executor could not apply the spending policy; no quote was requested.",
            status="blocked",
            audit_path=audit_path,
            now=now,
        )

    decision, rule = check_policy(proposal, policy, last_run_at=last_run_at, now=now)
    if decision == "blocked":
        return _finish(
            decision=decision,
            rule=rule,
            policy=policy_display,
            proposal=proposal,
            message=f"Executor blocked the proposal. {rule}",
            status="blocked",
            audit_path=audit_path,
            now=now,
        )

    resolved_account_id = account_id or os.getenv("NEAR_INTENTS_ACCOUNT_ID")
    if not resolved_account_id:
        note = (
            "No quote was requested because NEAR Intents requires recipient and refund fields. "
            "Set NEAR_INTENTS_ACCOUNT_ID to a public account ID to request a dry quote."
        )
        return _finish(
            decision="blocked",
            rule=note,
            policy=policy_display,
            proposal=proposal,
            message=note,
            status="blocked",
            audit_path=audit_path,
            now=now,
        )

    active_session = session or requests.Session()
    resolved_api_key = api_key or os.getenv("NEAR_INTENTS_API_KEY")
    quote_body: dict[str, Any] | None = None
    quote_status = "not_requested"
    draft_intent: dict[str, Any] | None = None
    try:
        catalog = _fetch_supported_tokens(active_session)
    except (requests.RequestException, QuoteError, ValueError, TypeError, AttributeError) as exc:
        quote_status = "token_catalog_failed"
        error_message = f"NEAR Intents token catalog request failed: {type(exc).__name__}: {exc}"
        if session is None:
            active_session.close()
        return _finish(
            decision="approved",
            rule=f"{rule} Quote could not proceed because the supported-token catalog request failed.",
            policy=policy_display,
            proposal=proposal,
            message=error_message,
            status="blocked",
            quote_status=quote_status,
            audit_path=audit_path,
            now=now,
        )

    try:
        from_symbol = str(proposal.get("from_token", ""))
        to_symbol = str(proposal.get("to_token", ""))
        from_asset = _resolve_allowed_asset(from_symbol, catalog, policy)
        to_asset = _resolve_allowed_asset(to_symbol, catalog, policy)
        from_asset_id = str(from_asset["assetId"])
        to_asset_id = str(to_asset["assetId"])
    except QuoteError as exc:
        decision = "blocked"
        rule = f"Blocked: token resolution did not satisfy policy allowed_tokens. {exc}"
        if session is None:
            active_session.close()
        return _finish(
            decision=decision,
            rule=rule,
            policy=policy_display,
            proposal=proposal,
            message=f"Executor blocked the proposal. {rule}",
            status="blocked",
            audit_path=audit_path,
            now=now,
        )

    try:
        decision, rule = check_policy(
            proposal,
            policy,
            last_run_at=last_run_at,
            resolved_asset_ids=(from_asset_id, to_asset_id),
            now=now,
        )
        if decision == "blocked":
            return _finish(
                decision=decision,
                rule=rule,
                policy=policy_display,
                proposal=proposal,
                message=f"Executor blocked the proposal. {rule}",
                status="blocked",
                audit_path=audit_path,
                now=now,
            )

        amount = float(proposal["amount_usd"])
        origin_symbol = from_symbol.strip()
        amount_units = _amount_in_smallest_unit(amount, from_asset, scout_metrics, origin_symbol)
        request_time = now or _utc_now()
        if request_time.tzinfo is None:
            request_time = request_time.replace(tzinfo=timezone.utc)
        deadline = _timestamp(request_time.astimezone(timezone.utc) + timedelta(hours=48))
        quote_request: dict[str, Any] = {
            "dry": True,
            "swapType": "EXACT_INPUT",
            "slippageTolerance": 100,
            "originAsset": from_asset_id,
            "depositType": "INTENTS",
            "destinationAsset": to_asset_id,
            "amount": amount_units,
            "recipient": resolved_account_id,
            "recipientType": "INTENTS",
            "refundTo": resolved_account_id,
            "refundType": "INTENTS",
            "deadline": deadline,
        }
        raw_quote_body = _request_dry_quote(active_session, request_body=quote_request, api_key=resolved_api_key)
        quote_status = "dry_quote_received"
        quote_body = _safe_quote_view(raw_quote_body)
        draft_intent = _build_local_draft(
            proposal=proposal,
            policy_rule=rule,
            requested_amount_usd=amount,
            origin_asset=from_asset_id,
            destination_asset=to_asset_id,
            amount_units=amount_units,
            quote_request=quote_request,
            quote_view=quote_body,
        )
        draft_intent_label = str(draft_intent["label"])
        message = (
            "NEAR Intents dry quote succeeded. A local preview draft was built from the policy decision and quote. "
            "It is not a generate-intent payload, is not signed, and is not submitted."
        )

        return _finish(
            decision="approved",
            rule=rule,
            policy=policy_display,
            proposal=proposal,
            message=message,
            status="done",
            quote=quote_body,
            draft_intent=draft_intent,
            draft_intent_label=draft_intent_label,
            quote_status=quote_status,
            audit_path=audit_path,
            now=now,
        )
    except (requests.RequestException, QuoteError, ValueError, TypeError, KeyError, AttributeError) as exc:
        quote_status = "failed"
        error_message = f"NEAR Intents request failed: {type(exc).__name__}: {exc}"
        return _finish(
            decision="approved",
            rule=f"{rule} Quote request did not complete: {exc}",
            policy=policy_display,
            proposal=proposal,
            message=error_message,
            status="blocked",
            quote=quote_body,
            draft_intent=draft_intent,
            draft_intent_label=str(draft_intent.get("label")) if draft_intent else "No local draft intent is available.",
            quote_status=quote_status,
            audit_path=audit_path,
            now=now,
        )
    finally:
        if session is None:
            active_session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify the NEAR SwarmGuard Executor.")
    parser.add_argument("--verify", action="store_true", help="Recompute and verify the audit hash chain.")
    parser.add_argument("--demo", action="store_true", help="Use saved Scout and Analyst fixtures; no API calls.")
    args = parser.parse_args()

    if args.verify:
        print(json.dumps(verify_audit_log(), indent=2, ensure_ascii=False))
        return

    from agents.analyst import run_analyst
    from agents.scout import load_sample_data

    scout_message = load_sample_data()
    if scout_message is None:
        scout_message = {
            "agent": "Scout",
            "status": "blocked",
            "data_status": "stale",
            "timestamp": _timestamp(),
            "metrics": {"network": {}, "prices": {}, "source_timestamps": {}},
            "message": "No saved Scout fixture is available.",
        }
    analyst_message = run_analyst(
        scout_message["metrics"],
        data_status=scout_message["data_status"],
        demo_mode=args.demo,
    )
    result = run_executor(analyst_message, scout_message["metrics"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
