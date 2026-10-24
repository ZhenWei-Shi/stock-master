"""Explicit offline-only CLI. No schedule, broker, automatic promotion, or test unlock."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .publication import publish_json
from .data_contract import assert_cache_compatible, data_request, request_sha256
from .provenance import verify_legacy_proof, verify_diagnostic_source


def cache_compatibility(cfg, manifest, cache, legacy_registration):
    legacy = "data_request" not in manifest["metadata"]
    if legacy:
        proof = tomllib.loads(legacy_registration.read_text(encoding="utf-8"))
        verify_legacy_proof(manifest, (cache / "manifest.json").read_bytes(), proof)
    result = assert_cache_compatible(cfg, manifest, legacy_verified=legacy)
    if legacy:
        result["compatibility_mode"] = "inspected_immutable_legacy_v1"
        result["acquisition_code_commit"] = proof["acquisition_code_commit"]
    return result


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    cfg = tomllib.loads(raw.decode("utf-8"))
    required = {"schema_version", "experiment_id", "symbols", "benchmark", "factors",
                "development_start", "development_end", "validation_start", "validation_end",
                "reserved_start", "horizon_sessions", "roundtrip_cost_bps", "random_seed", "quantiles"}
    if required - cfg.keys() or cfg["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete registered configuration")
    if cfg.get("run_reserved") is not False:
        raise ValueError("This CLI cannot unlock the reserved period")
    import pandas as pd
    ds, de, vs, ve, rs = [pd.Timestamp(cfg[k]) for k in
                          ["development_start", "development_end", "validation_start", "validation_end", "reserved_start"]]
    if not ds < de < vs < ve < rs:
        raise ValueError("Time partitions must be ordered and non-overlapping")
    if len(cfg["symbols"]) != len(set(cfg["symbols"])) or cfg["benchmark"] in cfg["symbols"]:
        raise ValueError("Duplicate universe symbols or benchmark inside research universe")
    if not cfg["symbols"] or any(not isinstance(x, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", x)
                                 for x in [*cfg["symbols"], cfg["benchmark"]]):
        raise ValueError("Invalid universe symbol")
    if cfg["horizon_sessions"] != 5 or cfg["quantiles"] != 5 or cfg["roundtrip_cost_bps"] != 10.0:
        raise ValueError("Version 1 CLI supports registered 5-session/5-quantile/10bp contract only")
    return cfg, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "evaluate", "h4", "diagnose"])
    parser.add_argument("--config", type=Path, default=Path("research/factor_lab_v1.toml"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h4-input", type=Path)
    parser.add_argument("--results-input", type=Path)
    parser.add_argument("--diagnostic-config", type=Path, default=Path("research/factor_diagnostics_v1.toml"))
    parser.add_argument("--legacy-cache-registration", type=Path, default=Path("research/factor_cache_legacy_v1.toml"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an experiment output")
    cfg, cfg_hash = load_config(args.config)
    common = {"experiment_id": cfg["experiment_id"], "config_sha256": cfg_hash,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "limitations": cfg["limitations"], "reserved_evaluated": False}
    try:
        common["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        common["code_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        common["code_commit"] = None
    if args.action == "fetch":
        if args.cache is None:
            parser.error("fetch requires --cache")
        from .cache import quality_report, write_cache, read_cache
        from .download import fetch_panel
        if (args.cache / "manifest.json").exists():
            panel, manifest = read_cache(args.cache)
            compatibility = cache_compatibility(cfg, manifest, args.cache, args.legacy_cache_registration)
            reused = True
        else:
            panel, meta = fetch_panel(cfg, args.cache.with_name(args.cache.name + "-staging"))
            meta["config_sha256"] = cfg_hash
            meta["data_request"] = data_request(cfg)
            meta["data_request_sha256"] = request_sha256(meta["data_request"])
            manifest = write_cache(panel, args.cache, meta)
            compatibility = cache_compatibility(cfg, manifest, args.cache, args.legacy_cache_registration)
            reused = False
        result = {**common, "reused_cache": reused, "data_compatibility": compatibility, "cache_manifest_sha256": hashlib.sha256(
            (args.cache / "manifest.json").read_bytes()).hexdigest(), "quality": quality_report(panel, cfg["benchmark"])}
    elif args.action == "evaluate":
        if args.cache is None:
            parser.error("evaluate requires --cache")
        from .cache import read_cache
        from .factors import compute_factors
        from .evaluation import evaluate_panel
        import pandas as pd
        panel, manifest = read_cache(args.cache)
        compatibility = cache_compatibility(cfg, manifest, args.cache, args.legacy_cache_registration)
        panel = panel[panel.date <= pd.Timestamp(cfg["validation_end"])].copy()
        values = compute_factors(panel, symbols=cfg["symbols"])
        if set(values.factor) != set(cfg["factors"]):
            raise ValueError("Factor definitions differ from preregistration")
        result = {**common, "data_compatibility": compatibility, "cache_manifest_sha256": hashlib.sha256(
            (args.cache / "manifest.json").read_bytes()).hexdigest(), "periods": {}}
        for period in ["development", "validation"]:
            result["periods"][period] = evaluate_panel(
                panel, values, start=cfg[period + "_start"], end=cfg[period + "_end"],
                horizon=cfg["horizon_sessions"], cost_bps=cfg["roundtrip_cost_bps"],
                seed=cfg["random_seed"], n_quantiles=cfg["quantiles"],
                factor_names=cfg["factors"], stress_cost_bps=cfg["stress_cost_bps"], hac_lag=cfg["hac_lag"])
    elif args.action == "diagnose":
        if args.results_input is None or args.cache is not None or args.h4_input is not None:
            parser.error("diagnose requires --results-input and does not accept price cache or H4 input")
        from .diagnostics import diagnose_result
        source_bytes = args.results_input.read_bytes()
        source = json.loads(source_bytes)
        if source.get("config_sha256") != cfg_hash:
            raise ValueError("Diagnostic source configuration differs from registered experiment")
        registration_bytes = args.diagnostic_config.read_bytes()
        registration = tomllib.loads(registration_bytes.decode("utf-8"))
        verify_diagnostic_source(source, source_bytes, cfg, cfg_hash, registration)
        result = {**common, "source_result_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "source_code_commit": source.get("code_commit"),
                  "diagnostic_config_sha256": hashlib.sha256(registration_bytes).hexdigest(),
                  "diagnostics": diagnose_result(source, cfg, registration)}
    else:
        if args.h4_input is None:
            parser.error("h4 requires --h4-input")
        from .h4_validation import GATE_VERSION, evaluate_h4
        source = args.h4_input.read_bytes()
        obj = json.loads(source)
        # Accept a read-only audit snapshot, never import the original strategy.
        trades = obj["files"]["overnight_lab.json"]["data"]["trades"]
        result = {**common, "evaluation_version": GATE_VERSION,
                  "source_sha256": hashlib.sha256(source).hexdigest(), "source_captured_at": obj["captured_at_utc"],
                  "supplement": evaluate_h4(trades)}
    publish_json(args.output, result)
    print(json.dumps({"output": str(args.output), "action": args.action, "reserved_evaluated": False}))


if __name__ == "__main__":
    main()
