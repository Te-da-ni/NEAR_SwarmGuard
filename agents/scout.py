"""Scout agent: collect public NEAR network and token market metrics.

This module only makes read-only requests. It does not connect to a wallet or
create, sign, or submit transactions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample.json"

# Public endpoints documented by NEAR and CoinPaprika. The CoinPaprika coin IDs
# are explicit IDs rather than ticker-symbol lookups to avoid symbol ambiguity.
NEAR_RPC_URL = "https://free.rpc.fastnear.com"
COINPAPRIKA_TICKER_URL = "https://api.coinpaprika.com/v1/tickers/{coin_id}"
COINS: dict[str, str] = {
    "NEAR": "near-near-protocol",
    "ETH": "eth-ethereum",
    "BTC": "btc-bitcoin",
    "SOL": "sol-solana",
}
REQUEST_TIMEOUT_SECONDS = 10


class TokenPrice(TypedDict):
    usd: float
    change_24h_pct: float | None
    market_cap_usd: float | None
    source_id: str


class ScoutMetrics(TypedDict):
    network: dict[str, Any]
    prices: dict[str, TokenPrice]
    source_timestamps: dict[str, str | None]


class ScoutMessage(TypedDict):
    agent: Literal["Scout"]
    status: Literal["done", "blocked"]
    data_status: Literal["fresh", "stale"]
    timestamp: str
    metrics: ScoutMetrics
    message: str


class ScoutFetchError(RuntimeError):
    """Raised when a live public data source cannot provide usable data."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_saved_sample(sample_path: Path = SAMPLE_PATH) -> ScoutMessage | None:
    try:
        with sample_path.open("r", encoding="utf-8") as sample_file:
            sample = json.load(sample_file)
        if not isinstance(sample, dict) or not isinstance(sample.get("metrics"), dict):
            return None
        return sample  # type: ignore[return-value]
    except (OSError, json.JSONDecodeError):
        return None


def load_sample_data() -> ScoutMessage | None:
    """Load the committed last-good snapshot for demo or fallback use."""
    return _load_saved_sample()


def _save_last_good(message: ScoutMessage, sample_path: Path = SAMPLE_PATH) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = sample_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as sample_file:
        json.dump(message, sample_file, indent=2, ensure_ascii=False)
        sample_file.write("\n")
    temporary_path.replace(sample_path)


def _fetch_network(session: requests.Session) -> tuple[dict[str, Any], str]:
    response = session.post(
        NEAR_RPC_URL,
        json={"jsonrpc": "2.0", "id": "swarmguard-scout", "method": "status", "params": []},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 429:
        raise ScoutFetchError("NEAR RPC rate-limited the status request (HTTP 429).")
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or "error" in body or not isinstance(body.get("result"), dict):
        raise ScoutFetchError("NEAR RPC returned no usable status result.")

    result = body["result"]
    sync_info = result.get("sync_info") or {}
    version = result.get("version") or {}
    if not isinstance(sync_info, dict) or not isinstance(version, dict):
        raise ScoutFetchError("NEAR RPC returned malformed status details.")
    latest_block_time = sync_info.get("latest_block_time")
    if not latest_block_time or sync_info.get("latest_block_height") is None:
        raise ScoutFetchError("NEAR RPC status was missing its latest block data.")

    network = {
        "chain_id": result.get("chain_id"),
        "latest_protocol_version": result.get("latest_protocol_version"),
        "latest_block_height": sync_info.get("latest_block_height"),
        "latest_block_time": latest_block_time,
        "syncing": bool(sync_info.get("syncing", False)),
        "uptime_seconds": result.get("uptime_sec"),
        "node_version": version.get("version"),
    }
    return network, latest_block_time


def _fetch_prices(session: requests.Session) -> tuple[dict[str, TokenPrice], dict[str, str | None]]:
    prices: dict[str, TokenPrice] = {}
    source_timestamps: dict[str, str | None] = {}

    for symbol, coin_id in COINS.items():
        url = COINPAPRIKA_TICKER_URL.format(coin_id=coin_id)
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 429:
            raise ScoutFetchError(f"CoinPaprika rate-limited the {symbol} ticker request (HTTP 429).")
        response.raise_for_status()
        ticker = response.json()
        if not isinstance(ticker, dict):
            raise ScoutFetchError(f"CoinPaprika returned malformed ticker data for {symbol}.")
        usd_quote = (ticker.get("quotes") or {}).get("USD") or {}
        price = usd_quote.get("price")
        if not isinstance(price, (int, float)) or not ticker.get("last_updated"):
            raise ScoutFetchError(f"CoinPaprika returned incomplete price data for {symbol}.")

        prices[symbol] = {
            "usd": float(price),
            "change_24h_pct": _optional_number(usd_quote.get("percent_change_24h")),
            "market_cap_usd": _optional_number(usd_quote.get("market_cap")),
            "source_id": coin_id,
        }
        source_timestamps[symbol] = str(ticker["last_updated"])

    return prices, source_timestamps


def _optional_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fallback_message(reason: str, sample_path: Path = SAMPLE_PATH) -> ScoutMessage:
    saved = _load_saved_sample(sample_path)
    if saved is None:
        return {
            "agent": "Scout",
            "status": "blocked",
            "data_status": "stale",
            "timestamp": _utc_now(),
            "metrics": {"network": {}, "prices": {}, "source_timestamps": {}},
            "message": f"Live data unavailable ({reason}); no saved sample.json snapshot is available.",
        }

    return {
        **saved,
        "status": "done",
        "data_status": "stale",
        "message": f"Live data unavailable ({reason}); using the saved snapshot from {saved.get('timestamp', 'an unknown time')}.",
    }


def run_scout(
    sample_path: Path | None = None,
    session: requests.Session | None = None,
) -> ScoutMessage:
    """Fetch a fresh snapshot or return the last saved snapshot as stale data."""
    active_sample_path = Path(sample_path) if sample_path is not None else SAMPLE_PATH
    active_session = session or requests.Session()
    try:
        network, network_timestamp = _fetch_network(active_session)
        prices, price_timestamps = _fetch_prices(active_session)
        timestamp = _utc_now()
        metrics: ScoutMetrics = {
            "network": network,
            "prices": prices,
            "source_timestamps": {"network": network_timestamp, **price_timestamps},
        }
        message: ScoutMessage = {
            "agent": "Scout",
            "status": "done",
            "data_status": "fresh",
            "timestamp": timestamp,
            "metrics": metrics,
            "message": "Scout fetched live NEAR network status and CoinPaprika USD prices.",
        }
        try:
            _save_last_good(message, active_sample_path)
        except OSError as exc:
            message["message"] += f" The live snapshot could not be saved to {active_sample_path}: {exc}."
        return message
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError, ScoutFetchError) as exc:
        return _fallback_message(f"{type(exc).__name__}: {exc}", active_sample_path)
    finally:
        if session is None:
            active_session.close()


if __name__ == "__main__":
    print(json.dumps(run_scout(), indent=2, ensure_ascii=False))
