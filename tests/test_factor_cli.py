"""Integration boundaries: reserved prices, immutable results, acquisition failure."""
import copy
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.factor_lab.__main__ import load_config, main
from src.factor_lab.download import fetch_panel


CONFIG = Path(__file__).resolve().parents[1] / "research/factor_lab_v1.toml"


def test_reserved_cannot_be_enabled(tmp_path):
    invalid = tmp_path / "changed.toml"
    invalid.write_text(CONFIG.read_text().replace("run_reserved = false", "run_reserved = true"))
    with pytest.raises(ValueError, match="cannot unlock"):
        load_config(invalid)


def test_time_partitions_cannot_overlap(tmp_path):
    invalid = tmp_path / "changed.toml"
    invalid.write_text(CONFIG.read_text().replace('validation_start = "2021-01-01"',
                                               'validation_start = "2020-01-01"'))
    with pytest.raises(ValueError, match="non-overlapping"):
        load_config(invalid)


def test_symbol_cannot_escape_cache_directory(tmp_path):
    invalid = tmp_path / "changed.toml"
    invalid.write_text(CONFIG.read_text().replace('"AAPL"', '"../../ESCAPE"'))
    with pytest.raises(ValueError, match="Invalid universe symbol"):
        load_config(invalid)


def test_existing_result_is_never_overwritten(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("original")
    with pytest.raises(FileExistsError):
        main(["evaluate", "--output", str(output)])
    assert output.read_text() == "original"


def test_cli_never_supplies_reserved_prices_to_factor_or_evaluation(tmp_path):
    cfg, cfg_hash = load_config(CONFIG)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    panel = pd.DataFrame({"date": pd.to_datetime(["2023-12-29", "2024-01-02"]), "symbol": ["AAPL"] * 2})
    seen = []

    def factors(frame, symbols):
        assert frame.date.max() < pd.Timestamp("2024-01-01")
        return pd.DataFrame({"factor": cfg["factors"]})

    def evaluation(frame, values, **kw):
        assert frame.date.max() < pd.Timestamp("2024-01-01")
        seen.append(kw["end"])
        return {"boundary_test": True}

    with patch("src.factor_lab.cache.read_cache", return_value=(panel, {"metadata": {"config_sha256": cfg_hash}})), \
            patch("src.factor_lab.factors.compute_factors", side_effect=factors), \
            patch("src.factor_lab.evaluation.evaluate_panel", side_effect=evaluation):
        main(["evaluate", "--config", str(CONFIG), "--cache", str(cache), "--output", str(tmp_path / "out.json")])
    assert seen == ["2020-12-31", "2023-12-31"]
    assert json.loads((tmp_path / "out.json").read_text())["reserved_evaluated"] is False


def test_cli_preserves_result_finalized_after_initial_check(tmp_path):
    cfg, cfg_hash = load_config(CONFIG)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    output = tmp_path / "shared-result.json"
    panel = pd.DataFrame({"date": pd.to_datetime(["2023-12-29"]), "symbol": ["AAPL"]})

    def another_run_finishes(*args, **kwargs):
        if not output.exists():
            output.write_text('{"completed_by_other_run":true}')
        return {"current_run": True}

    with patch("src.factor_lab.cache.read_cache", return_value=(panel, {"metadata": {"config_sha256": cfg_hash}})), \
            patch("src.factor_lab.factors.compute_factors", return_value=pd.DataFrame({"factor": cfg["factors"]})), \
            patch("src.factor_lab.evaluation.evaluate_panel", side_effect=another_run_finishes):
        with pytest.raises(FileExistsError):
            main(["evaluate", "--config", str(CONFIG), "--cache", str(cache), "--output", str(output)])
    assert json.loads(output.read_text()) == {"completed_by_other_run": True}


def test_failed_source_cannot_be_published_as_empty_universe(tmp_path):
    cfg, _ = load_config(CONFIG)
    cfg = copy.deepcopy(cfg)
    cfg["symbols"] = ["AAPL"]

    class FailingTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            raise TimeoutError("simulated source timeout")

    fake = types.SimpleNamespace(Ticker=FailingTicker, set_tz_cache_location=lambda _: None, __version__="test")
    with patch.dict(sys.modules, {"yfinance": fake}), patch("src.factor_lab.download.time.sleep"):
        with pytest.raises(RuntimeError, match="2 requested symbols failed"):
            fetch_panel(cfg, tmp_path / "stage")
    failure = json.loads((tmp_path / "stage/failures.json").read_text())
    assert len(failure["failures"]) == 2
    assert not (tmp_path / "stage/manifest.json").exists()
