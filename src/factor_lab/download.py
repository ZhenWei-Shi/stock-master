"""Bounded Yahoo acquisition with resumable, request-bound staging files."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .publication import replace_json as atomic_json


def fetch_panel(config: dict, staging: Path) -> tuple[pd.DataFrame, dict]:
    """Download each declared symbol, never silently replace the requested universe."""
    import yfinance as yf
    from .cache import validate_panel

    symbols = [*config["symbols"], config["benchmark"]]
    if not symbols or len(symbols) != len(set(symbols)) or any(
        not isinstance(x, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", x) for x in symbols
    ):
        raise ValueError("Unsafe or duplicate requested symbol")
    request = {"symbols": symbols, "start": config["download_start"],
               "end": config["download_end_exclusive"]}
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    staging.mkdir(parents=True, exist_ok=True)
    binding = staging / "request.json"
    if binding.exists():
        if json.loads(binding.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("Staging belongs to a different data request")
    else:
        atomic_json(binding, {"request": request, "fingerprint": fingerprint})
    yf.set_tz_cache_location(str(staging / "yfinance-cache"))
    panels, provenance, failures = [], {}, []
    columns = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
               "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividend",
               "Stock Splits": "split"}
    for number, symbol in enumerate(symbols, 1):
        csv_path, meta_path = staging / f"{symbol}.csv", staging / f"{symbol}.json"
        try:
            if csv_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if hashlib.sha256(csv_path.read_bytes()).hexdigest() != meta["sha256"]:
                    raise ValueError(f"Staging checksum mismatch: {symbol}")
                panel = validate_panel(pd.read_csv(csv_path, parse_dates=["date"]))
                if set(panel.symbol) != {symbol}:
                    raise ValueError("Staging symbol mismatch")
                print(f"[{number}/{len(symbols)}] {symbol}: reused {len(panel)} rows", flush=True)
            else:
                history = None
                for attempt in range(2):
                    try:
                        history = yf.Ticker(symbol).history(
                            start=request["start"], end=request["end"], auto_adjust=False,
                            actions=True, repair=False, interval="1d", timeout=20, raise_errors=True)
                        break
                    except Exception as exc:
                        # Read-only request: one bounded retry; never treat failure as no data.
                        permanent = type(exc).__name__ in {"YFInvalidPeriodError", "YFNotImplementedError",
                            "YFTickerMissingError", "YFPricesMissingError", "YFTzMissingError", "ValueError", "TypeError"}
                        if attempt or permanent:
                            raise RuntimeError(f"Yahoo retrieval failed for {symbol}") from exc
                        time.sleep(1)
                if history is None or history.empty:
                    raise ValueError(f"Empty history: {symbol}")
                missing = set(columns) - set(history.columns)
                if missing:
                    raise ValueError(f"Missing Yahoo fields for {symbol}: {sorted(missing)}")
                panel = history[list(columns)].rename(columns=columns).copy()
                panel.insert(0, "date", history.index.tz_localize(None).normalize())
                panel.insert(1, "symbol", symbol)
                panel = validate_panel(panel.reset_index(drop=True))
                temporary = csv_path.with_suffix(".csv.tmp")
                panel.to_csv(temporary, index=False, float_format="%.17g")
                os.replace(temporary, csv_path)
                meta = {"symbol": symbol, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(), "rows": len(panel)}
                atomic_json(meta_path, meta)
                print(f"[{number}/{len(symbols)}] {symbol}: downloaded {len(panel)} rows", flush=True)
            dates = panel.date
            if dates.min() < pd.Timestamp(request["start"]) or dates.max() >= pd.Timestamp(request["end"]):
                raise ValueError(f"History outside requested boundaries: {symbol}")
            panels.append(panel)
            provenance[symbol] = meta
        except Exception as exc:
            failures.append({"symbol": symbol, "type": type(exc).__name__, "message": str(exc)[:200]})
            print(f"[{number}/{len(symbols)}] {symbol}: FAILED ({type(exc).__name__})", flush=True)
    if failures:
        atomic_json(staging / "failures.json", {"failures": failures})
        raise RuntimeError(f"{len(failures)} requested symbols failed; no complete cache published")
    panel = validate_panel(pd.concat(panels, ignore_index=True))
    return panel, {"source": f"Yahoo Finance via yfinance {yf.__version__}",
                   "requested_start": request["start"], "requested_end": request["end"],
                   "request": request, "acquisition": provenance,
                   "universe_status": config["status"], "limitations": config["limitations"]}
