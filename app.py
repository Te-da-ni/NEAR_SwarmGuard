"""Streamlit dashboard for the NEAR SwarmGuard sequential agent demo."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.analyst import run_analyst
from agents.executor import ExecutorMessage, SpendingPolicy, load_policy, run_executor, verify_audit_log
from agents.scout import ScoutMessage, load_sample_data, run_scout


MAX_RUNS_PER_SESSION = 5
BUTTON_COOLDOWN_SECONDS = 60
LOG_LIMIT = 100

st.set_page_config(page_title="NEAR SwarmGuard", page_icon="🛡️", layout="wide")


def _secret(name: str) -> str | None:
    """Read secrets only from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None
    if secret_value is None:
        return None
    secret_text = str(secret_value).strip()
    return secret_text or None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _init_session_state() -> None:
    st.session_state.setdefault("coordination_log", [])
    st.session_state.setdefault("run_count", 0)
    st.session_state.setdefault("last_run_epoch", None)
    st.session_state.setdefault("last_run_at", None)
    st.session_state.setdefault("last_scout", None)
    st.session_state.setdefault("last_analyst", None)
    st.session_state.setdefault("last_executor", None)
    st.session_state.setdefault(
        "agent_statuses",
        {
            "Scout": {"status": "idle", "detail": "Waiting to run."},
            "Analyst": {"status": "idle", "detail": "Waiting for Scout."},
            "Executor": {"status": "idle", "detail": "Waiting for Analyst."},
        },
    )
    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = not bool(_secret("NEAR_AI_API_KEY"))
    if "intents_account_id" not in st.session_state:
        st.session_state["intents_account_id"] = _secret("NEAR_INTENTS_ACCOUNT_ID") or ""


def _append_log(text: str, placeholder: Any | None = None) -> None:
    line = f"{_iso_now()}  {text}"
    entries = list(st.session_state.get("coordination_log", []))
    entries.append(line)
    st.session_state["coordination_log"] = entries[-LOG_LIMIT:]
    if placeholder is not None:
        placeholder.code("\n".join(st.session_state["coordination_log"]), language="text")


def _render_status_cards(slots: list[Any]) -> None:
    statuses = st.session_state["agent_statuses"]
    for slot, agent in zip(slots, ("Scout", "Analyst", "Executor")):
        status_info = statuses[agent]
        with slot.container(border=True):
            st.subheader(agent)
            st.metric("Status", str(status_info["status"]).upper())
            st.caption(str(status_info["detail"]))


def _set_agent_status(
    slots: list[Any],
    agent: str,
    status: str,
    detail: str,
) -> None:
    statuses = dict(st.session_state["agent_statuses"])
    statuses[agent] = {"status": status, "detail": detail}
    st.session_state["agent_statuses"] = statuses
    _render_status_cards(slots)


def _empty_scout(message: str) -> ScoutMessage:
    return {
        "agent": "Scout",
        "status": "blocked",
        "data_status": "stale",
        "timestamp": _iso_now(),
        "metrics": {"network": {}, "prices": {}, "source_timestamps": {}},
        "message": message,
    }


def _unexpected_analyst_failure(message: str) -> dict[str, Any]:
    return {
        "agent": "Analyst",
        "status": "blocked",
        "timestamp": _iso_now(),
        "proposal": {
            "risk_score": 100.0,
            "action": "none",
            "from_token": None,
            "to_token": None,
            "amount_usd": 0.0,
            "confidence": 0.0,
            "reasoning": "Analyst failed; no action proposed.",
        },
        "source": "fallback",
        "attempts": 0,
        "message": message,
    }


def _unexpected_executor_failure(message: str, policy: dict[str, Any]) -> ExecutorMessage:
    return {
        "agent": "Executor",
        "status": "blocked",
        "timestamp": _iso_now(),
        "decision": "blocked",
        "rule": f"Executor stopped safely after an unexpected error: {message}",
        "policy": policy,
        "quote": None,
        "draft_intent": None,
        "draft_intent_label": "No local draft intent is available.",
        "audit": {"written": False, "message": "Executor failed before it could confirm an audit record."},
        "message": message,
    }


def _run_swarm(
    *,
    demo_mode: bool,
    account_id: str | None,
    previous_run_at: str | None,
    status_slots: list[Any],
    log_placeholder: Any,
) -> None:
    run_number = int(st.session_state["run_count"]) + 1
    st.session_state["run_count"] = run_number
    st.session_state["last_run_epoch"] = time.time()
    st.session_state["last_run_at"] = _iso_now()
    _append_log(f"Run {run_number}/{MAX_RUNS_PER_SESSION} started ({'demo' if demo_mode else 'live'} mode).", log_placeholder)

    _set_agent_status(status_slots, "Scout", "running", "Fetching or replaying network and price metrics.")
    _append_log("Scout started.", log_placeholder)
    try:
        if demo_mode:
            scout_message = load_sample_data()
            if scout_message is None:
                scout_message = _empty_scout("Demo mode could not load data/sample.json.")
            else:
                scout_message = {
                    **scout_message,
                    "message": f"Demo mode replayed the saved Scout snapshot from {scout_message.get('timestamp', 'unknown time')}.",
                }
        else:
            scout_message = run_scout()
    except Exception as exc:
        scout_message = _empty_scout(f"Scout stopped safely: {type(exc).__name__}: {exc}")

    st.session_state["last_scout"] = scout_message
    scout_status = "done" if scout_message.get("status") == "done" else "blocked"
    scout_detail = str(scout_message.get("message", "Scout completed."))
    if scout_message.get("data_status") == "stale":
        scout_detail = f"STALE DATA — {scout_detail}"
    elif demo_mode:
        scout_detail = f"DEMO DATA — {scout_detail}"
    _set_agent_status(status_slots, "Scout", scout_status, scout_detail)
    _append_log(
        f"Scout → Analyst handoff at {scout_message.get('timestamp', _iso_now())}: "
        f"data_status={scout_message.get('data_status', 'unknown')}; {scout_message.get('message', '')}",
        log_placeholder,
    )

    _set_agent_status(status_slots, "Analyst", "running", "Evaluating the Scout handoff.")
    _append_log("Analyst started.", log_placeholder)
    try:
        analyst_message = run_analyst(
            scout_message["metrics"],
            data_status=scout_message.get("data_status", "stale"),
            demo_mode=demo_mode,
            api_key=None if demo_mode else _secret("NEAR_AI_API_KEY"),
        )
    except Exception as exc:
        analyst_message = _unexpected_analyst_failure(f"Analyst stopped safely: {type(exc).__name__}: {exc}")
    st.session_state["last_analyst"] = analyst_message
    analyst_status = "done" if analyst_message.get("status") == "done" else "blocked"
    proposal = analyst_message.get("proposal", {})
    analyst_detail = str(analyst_message.get("message", "Analyst completed."))
    _set_agent_status(status_slots, "Analyst", analyst_status, analyst_detail)
    _append_log(
        f"Analyst → Executor handoff at {analyst_message.get('timestamp', _iso_now())}: "
        f"action={proposal.get('action', 'none') if isinstance(proposal, dict) else 'none'}; {analyst_detail}",
        log_placeholder,
    )

    _set_agent_status(status_slots, "Executor", "running", "Checking policy before any quote request.")
    _append_log("Executor started; policy checks run locally in Python.", log_placeholder)
    try:
        try:
            policy_for_display = dict(load_policy())
        except Exception as exc:
            policy_for_display = {"error": str(exc)}
        executor_message = run_executor(
            analyst_message,
            scout_message["metrics"],
            last_run_at=previous_run_at,
            account_id=account_id,
            api_key=_secret("NEAR_INTENTS_API_KEY"),
        )
    except Exception as exc:
        executor_message = _unexpected_executor_failure(
            f"{type(exc).__name__}: {exc}",
            policy_for_display,
        )
    st.session_state["last_executor"] = executor_message
    executor_status = "done" if executor_message.get("status") == "done" else "blocked"
    _set_agent_status(status_slots, "Executor", executor_status, str(executor_message.get("message", "Executor completed.")))
    audit = executor_message.get("audit", {})
    audit_note = "audit record appended" if isinstance(audit, dict) and audit.get("written") else "audit append unavailable"
    _append_log(
        f"Executor result: decision={executor_message.get('decision')}; "
        f"rule={executor_message.get('rule')}; {audit_note}.",
        log_placeholder,
    )


def _plot_scout_metrics(scout_message: dict[str, Any] | None, analyst_message: dict[str, Any] | None) -> None:
    st.subheader("Scout metrics and Analyst risk")
    if not scout_message:
        st.info("Run the swarm to chart the Scout snapshot.")
        return

    metrics = scout_message.get("metrics", {})
    prices = metrics.get("prices", {}) if isinstance(metrics, dict) else {}
    rows: list[dict[str, Any]] = []
    if isinstance(prices, dict):
        for symbol, value in prices.items():
            if isinstance(value, dict) and isinstance(value.get("usd"), (int, float)):
                rows.append(
                    {
                        "Token": symbol,
                        "USD price": float(value["usd"]),
                        "24h change (%)": value.get("change_24h_pct"),
                    }
                )

    chart_col, gauge_col = st.columns([1.5, 1])
    with chart_col:
        if rows:
            price_frame = pd.DataFrame(rows)
            figure = px.bar(
                price_frame,
                x="Token",
                y="USD price",
                color="24h change (%)",
                color_continuous_scale="RdYlGn",
                hover_data={"USD price": ":.6f", "24h change (%)": ":.2f"},
                title="Saved or live token prices (color = 24-hour change)",
            )
            figure.update_layout(height=370, margin={"l": 12, "r": 12, "t": 56, "b": 12})
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No usable price metrics are available in the last Scout message.")

    with gauge_col:
        proposal = analyst_message.get("proposal", {}) if isinstance(analyst_message, dict) else {}
        risk_value = proposal.get("risk_score") if isinstance(proposal, dict) else None
        if isinstance(risk_value, (int, float)):
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(risk_value),
                    number={"suffix": "/100"},
                    title={"text": "Analyst risk score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#3867d6"},
                        "steps": [
                            {"range": [0, 40], "color": "#d8f3dc"},
                            {"range": [40, 70], "color": "#fff3bf"},
                            {"range": [70, 100], "color": "#ffd6d6"},
                        ],
                    },
                )
            )
            gauge.update_layout(height=370, margin={"l": 24, "r": 24, "t": 56, "b": 12})
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info("No Analyst risk score is available yet.")

    network = metrics.get("network", {}) if isinstance(metrics, dict) else {}
    if network:
        with st.expander("NEAR network snapshot", expanded=False):
            st.json(network)
    if scout_message.get("data_status") == "stale":
        st.warning("Scout is using a saved snapshot marked stale. Review its timestamp before interpreting the chart.")
    elif scout_message.get("message", "").startswith("Demo mode"):
        st.info("The chart is replaying data/sample.json in demo mode.")


def _render_dry_quote(executor_message: dict[str, Any] | None) -> None:
    st.subheader("NEAR Intents dry quote")
    quote_view = executor_message.get("quote") if isinstance(executor_message, dict) else None
    if not isinstance(quote_view, dict):
        st.info("No dry quote was requested. A quote is only requested after policy approval and when an account ID is configured.")
        return

    request = quote_view.get("request", {})
    quote = quote_view.get("quote", {})
    request = request if isinstance(request, dict) else {}
    quote = quote if isinstance(quote, dict) else {}

    origin = request.get("originAsset", "not returned")
    destination = request.get("destinationAsset", "not returned")
    st.markdown(f"**Requested route:** `{origin}` → `{destination}`")
    st.caption(str(quote_view.get("quote_note", "Route is based on the requested asset pair.")))

    amount_in = quote.get("amountInFormatted", quote.get("amountIn", "Not provided"))
    amount_out = quote.get("amountOutFormatted", quote.get("amountOut", "Not provided"))
    amount_in_usd = quote.get("amountInUsd", "Not provided")
    amount_out_usd = quote.get("amountOutUsd", "Not provided")
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Quoted input", str(amount_in))
    metrics_cols[1].metric("Quoted input USD", str(amount_in_usd))
    metrics_cols[2].metric("Quoted output", str(amount_out))
    metrics_cols[3].metric("Quoted output USD", str(amount_out_usd))

    fees = {field: quote[field] for field in ("refundFee", "withdrawFee") if field in quote}
    if fees:
        st.markdown("**Fee fields returned by the API (raw values):**")
        st.json(fees)
        st.caption("These fee values are shown as returned; the API response does not provide a unit conversion here.")
    else:
        st.caption("The dry quote did not return refundFee or withdrawFee fields.")

    returned_deadline = quote.get("deadline")
    requested_deadline = request.get("deadline")
    st.markdown(f"**Requested deadline input:** `{requested_deadline or 'not returned'}`")
    if returned_deadline:
        st.markdown(f"**Deadline returned by quote:** `{returned_deadline}`")
    else:
        st.caption("The requested deadline is an input value. NEAR Intents dry quotes omit an executable quote deadline.")

    with st.expander("Dry quote response fields", expanded=False):
        st.json(quote_view)


def _render_decision_and_policy(
    policy: SpendingPolicy | dict[str, Any] | None,
    executor_message: dict[str, Any] | None,
) -> None:
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Spending policy")
        if policy is not None:
            st.json(policy)
        else:
            st.error("policy.json could not be loaded.")

    with right_col:
        st.subheader("Executor decision")
        if not executor_message:
            st.info("No Executor decision yet.")
        else:
            decision = executor_message.get("decision", "blocked")
            if decision == "approved":
                st.success("Approved by policy")
            else:
                st.error("Blocked")
            st.write(str(executor_message.get("rule", "No policy rule recorded.")))
            st.caption(str(executor_message.get("message", "")))

    st.subheader("Draft intent JSON")
    if executor_message and isinstance(executor_message.get("draft_intent"), dict):
        st.warning(str(executor_message.get("draft_intent_label", "Local preview only.")))
        st.json(executor_message["draft_intent"])
        st.caption(
            "This local JSON is derived from the policy decision and dry quote. It is not a NEAR Intents "
            "generate-intent response, has no depositAddress, is not signed, and is not submitted."
        )
    elif executor_message:
        st.info(str(executor_message.get("draft_intent_label", "No local draft was built for this decision.")))
    else:
        st.info("A local draft will appear after an approved proposal receives a dry quote.")


def _render_audit(executor_message: dict[str, Any] | None) -> None:
    st.subheader("Audit log")
    verify_clicked = st.button("Verify audit log", key="verify_audit_log")
    if verify_clicked:
        result = verify_audit_log()
        if result.get("valid"):
            st.success(str(result.get("message", "Audit chain is valid.")))
        else:
            st.error(str(result.get("message", "Audit chain verification failed.")))
        st.caption(f"Records checked: {result.get('records', 0)}")
    if executor_message:
        audit = executor_message.get("audit", {})
        if isinstance(audit, dict) and audit.get("written"):
            st.caption(f"Last decision record hash: {audit.get('record_hash')}")
        elif isinstance(audit, dict) and audit.get("message"):
            st.warning(f"Last audit append: {audit.get('message')}")
    st.caption("The log is local at logs/audit.jsonl. Verify recomputes each record hash and previous_hash link.")


_init_session_state()

st.title("NEAR SwarmGuard")
st.caption("Scout collects public metrics, Analyst returns a proposal, and Executor applies policy before any dry quote.")
st.caption("The app does not connect to a wallet, sign, submit, or broadcast transactions.")

with st.sidebar:
    st.header("Run settings")
    demo_mode = st.toggle(
        "Demo mode",
        key="demo_mode",
        help="Replay data/sample.json and the saved Analyst response without API keys or network calls.",
    )
    account_id = st.text_input(
        "NEAR Intents account ID",
        key="intents_account_id",
        help="Public recipient/refund account used only for a dry quote. No private key is requested.",
    ).strip() or None
    if demo_mode:
        st.info("Demo mode uses the saved Scout and Analyst fixtures; no API requests are made.")
    else:
        st.caption(f"NEAR AI API key: {'available' if _secret('NEAR_AI_API_KEY') else 'not set'}")
        st.caption(f"NEAR Intents partner key: {'available' if _secret('NEAR_INTENTS_API_KEY') else 'not set (optional)'}")
    st.metric("Runs this session", f"{st.session_state['run_count']} / {MAX_RUNS_PER_SESSION}")

try:
    current_policy: SpendingPolicy | dict[str, Any] | None = load_policy()
except Exception as exc:
    current_policy = None
    st.error(f"Could not load policy.json: {exc}")

status_slots = list(st.columns(3))
_render_status_cards(status_slots)

st.subheader("Run swarm")
last_epoch = st.session_state.get("last_run_epoch")
cooldown_seconds = BUTTON_COOLDOWN_SECONDS
if isinstance(current_policy, dict) and isinstance(current_policy.get("cooldown_seconds"), int):
    cooldown_seconds = max(BUTTON_COOLDOWN_SECONDS, int(current_policy["cooldown_seconds"]))
cooldown_remaining = 0
if isinstance(last_epoch, (int, float)):
    cooldown_remaining = max(0, int(cooldown_seconds - (time.time() - last_epoch) + 0.999))

run_disabled = int(st.session_state["run_count"]) >= MAX_RUNS_PER_SESSION
run_clicked = st.button("Run swarm", type="primary", disabled=run_disabled)
st.caption(
    f"This session allows up to {MAX_RUNS_PER_SESSION} runs. Cooldown: {cooldown_seconds} seconds between run starts."
)
if run_disabled:
    st.warning("Session run limit reached. Start a new Streamlit session to reset the limit.")
elif cooldown_remaining > 0:
    st.info(f"Cooldown active: wait {cooldown_remaining} more second(s). Click Run swarm after the timer expires.")

log_placeholder = st.empty()
if st.session_state["coordination_log"]:
    log_placeholder.code("\n".join(st.session_state["coordination_log"]), language="text")
else:
    log_placeholder.caption("Handoffs and network errors will appear here with UTC timestamps.")

if run_clicked:
    now_epoch = time.time()
    last_epoch = st.session_state.get("last_run_epoch")
    if isinstance(last_epoch, (int, float)) and now_epoch - last_epoch < cooldown_seconds:
        remaining = max(1, int(cooldown_seconds - (now_epoch - last_epoch) + 0.999))
        st.warning(f"Cooldown active: wait {remaining} more second(s). This click did not start a run.")
        _append_log(f"Run request blocked by the session cooldown; {remaining} second(s) remain.", log_placeholder)
    elif int(st.session_state["run_count"]) >= MAX_RUNS_PER_SESSION:
        st.error("Session run limit reached. No agents were started.")
        _append_log("Run request blocked because the five-run session limit was reached.", log_placeholder)
    else:
        previous_run_at = st.session_state.get("last_run_at")
        _run_swarm(
            demo_mode=demo_mode,
            account_id=account_id,
            previous_run_at=previous_run_at,
            status_slots=status_slots,
            log_placeholder=log_placeholder,
        )

st.divider()
_plot_scout_metrics(st.session_state.get("last_scout"), st.session_state.get("last_analyst"))

last_scout_message = st.session_state.get("last_scout")
if last_scout_message and last_scout_message.get("data_status") == "stale":
    st.warning(str(last_scout_message.get("message", "Scout data is stale.")))

_render_dry_quote(st.session_state.get("last_executor"))

try:
    policy_for_panel: SpendingPolicy | dict[str, Any] | None = load_policy()
except Exception:
    policy_for_panel = None
_render_decision_and_policy(policy_for_panel, st.session_state.get("last_executor"))
_render_audit(st.session_state.get("last_executor"))
