"""Integration safeguards independent of module author synthetic examples."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.factor_lab.__main__ import main, load_config
from src.factor_lab.provenance import verify_legacy_proof


def test_unknown_legacy_cannot_claim_policy_from_source_name():
    manifest = {"metadata": {"source": "Yahoo Finance via yfinance 1.7.0", "config_sha256": "x"}}
    raw = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match="not the inspected"):
        verify_legacy_proof(manifest, raw, {"schema_version": 1})


def test_diagnostic_cli_refuses_price_cache(tmp_path):
    with pytest.raises(SystemExit):
        main(["diagnose", "--cache", str(tmp_path), "--results-input", "unused.json",
              "--output", str(tmp_path / "out.json")])


def test_diagnostic_cli_binds_exact_source_and_never_reads_prices(tmp_path):
    source = {"config_sha256": "wrong"}
    path = tmp_path / "old-result.json"
    path.write_text(json.dumps(source))
    with patch("src.factor_lab.cache.read_cache", side_effect=AssertionError("prices read")), \
            patch("src.factor_lab.download.fetch_panel", side_effect=AssertionError("network called")), \
            patch("src.factor_lab.factors.compute_factors", side_effect=AssertionError("features computed")):
        with pytest.raises(ValueError, match="configuration differs"):
            main(["diagnose", "--results-input", str(path), "--output", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()


def test_registered_diagnostic_succeeds_without_price_or_network_access(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg, cfg_hash = load_config(root / "research/factor_lab_v1.toml")
    source = {"experiment_id": cfg["experiment_id"], "config_sha256": cfg_hash,
              "cache_manifest_sha256": "a" * 64, "reserved_evaluated": False, "periods": {}}
    for period, date in [("development", "2013-01-02"), ("validation", "2021-01-04")]:
        row = {"date": date, "n": 20, "top_net": 0.02, "benchmark_net": 0.01,
               "random_net": 0.015, "excess_random": 0.005}
        factor = {"dates": 1, "per_date": [row], "mean_net": 0.02,
                  "mean_benchmark_net": 0.01, "mean_random_net": 0.015, "mean_excess_random": 0.005}
        source["periods"][period] = {"period": {"start": cfg[period + "_start"], "end": cfg[period + "_end"],
                                              "panel_cropped_to": cfg[period + "_end"]},
                                     "cost": {"roundtrip_bps": 10.0},
                                     "factors": {name: factor for name in cfg["factors"]}}
    source_bytes = json.dumps(source).encode()
    path = tmp_path / "source.json"
    path.write_bytes(source_bytes)
    registration = (root / "research/factor_diagnostics_v1.toml").read_text(encoding="utf-8")
    import tomllib
    parsed = tomllib.loads(registration)
    registration = registration.replace(parsed["expected_source_sha256"], hashlib.sha256(source_bytes).hexdigest())
    registration = registration.replace(parsed["expected_cache_manifest_sha256"], "a" * 64)
    regpath = tmp_path / "registration.toml"
    regpath.write_text(registration, encoding="utf-8")
    output = tmp_path / "diagnostic.json"
    args = ["diagnose", "--results-input", str(path), "--diagnostic-config", str(regpath), "--output", str(output)]
    with patch("src.factor_lab.cache.read_cache", side_effect=AssertionError("prices read")), \
            patch("src.factor_lab.download.fetch_panel", side_effect=AssertionError("network called")), \
            patch("src.factor_lab.factors.compute_factors", side_effect=AssertionError("features computed")):
        main(args)
    result = json.loads(output.read_bytes())
    assert result["source_result_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert result["diagnostics"]["candidate_selection"] is False
    assert set(result["diagnostics"]["periods"]["validation"]) == set(cfg["factors"])
    # Same parsed object with changed raw bytes cannot masquerade as the frozen source.
    path.write_bytes(source_bytes + b"\n")
    args[-1] = str(tmp_path / "changed-source.json")
    with pytest.raises(ValueError, match="immutable preregistered"):
        main(args)
    assert not (tmp_path / "changed-source.json").exists()
