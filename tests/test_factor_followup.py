"""Synthetic behavior tests for reusable data identity and descriptive diagnostics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factor_lab.data_contract import (assert_cache_compatible, data_request,
                                      request_sha256)
from factor_lab.diagnostics import diagnose_result


FACTORS = ["momentum_12_1", "reversal_5", "signed_rvol_20", "low_vol_20"]


def experiment():
    return {
        "experiment_id": "synthetic-four-factors-v1", "factors": FACTORS.copy(),
        "source": "Yahoo Finance via yfinance, synthetic policy fixture",
        "symbols": ["BBB", "AAA"], "benchmark": "SPY",
        "download_start": "2012-01-01", "download_end_exclusive": "2024-01-01",
        "development_start": "2013-01-01", "development_end": "2020-12-31",
        "validation_start": "2021-01-01", "validation_end": "2023-12-31",
        "reserved_start": "2024-01-01", "roundtrip_cost_bps": 10.0,
    }


def manifest(*, explicit=True):
    cfg = experiment()
    req = data_request(cfg)
    meta = {
        "source": "Yahoo Finance via yfinance 1.7.0",
        "request": {"symbols": ["BBB", "AAA", "SPY"],
                    "start": "2012-01-01", "end": "2024-01-01"},
        "requested_start": "2012-01-01", "requested_end": "2024-01-01",
        "config_sha256": "old-experiment-hash-kept-for-provenance",
    }
    if explicit:
        meta.update(data_request=req, data_request_sha256=request_sha256(req))
    return {"metadata": meta, "files": [
        {"symbol": name, "first_date": "2013-01-02", "last_date": "2023-12-29"}
        for name in ["AAA", "BBB", "SPY"]
    ]}


def registration():
    return {
        "schema_version": 1, "source_experiment_id": "synthetic-four-factors-v1",
        "allowed_periods": ["development", "validation"],
        "allowed_years": list(range(2013, 2024)), "reserved_start": "2024-01-01",
        "cost_scenarios_bps": [0.0, 10.0, 20.0, 40.0],
        "concentration_top_dates": 5, "run_reserved": False,
        "new_hypotheses": False, "select_candidates": False,
        "limitations": ["synthetic fixture; descriptive analysis only"],
    }


def row(date, excess):
    return {"date": date, "top_net": excess - 0.001,
            "benchmark_net": 0.004, "random_net": -0.001,
            "excess_random": excess}


def source_result():
    cfg = experiment()
    development = [row(date, excess) for date, excess in zip(
        ["2013-01-02", "2013-01-03", "2013-01-04", "2013-01-07",
         "2014-01-02", "2014-01-03", "2014-01-06"],
        [0.01, 0.02, -0.01, 0.03, 0.04, 0.05, 0.06])]
    validation = [row("2021-01-04", -0.01), row("2023-12-20", 0.0)]
    periods = {}
    for name, records in [("development", development), ("validation", validation)]:
        periods[name] = {
            "period": {"start": cfg[name + "_start"], "end": cfg[name + "_end"]},
            "cost": {"roundtrip_bps": 10.0},
            "factors": {factor: {"per_date": deepcopy(records)} for factor in FACTORS},
        }
    return {"experiment_id": cfg["experiment_id"], "reserved_evaluated": False,
            "periods": periods}


def run_diagnostics(source=None, cfg=None, reg=None):
    return diagnose_result(source if source is not None else source_result(),
                           cfg if cfg is not None else experiment(),
                           reg if reg is not None else registration())


def test_data_identity_is_canonical_and_excludes_experiment_metadata():
    cfg = experiment()
    before = deepcopy(cfg)
    req = data_request(cfg)
    changed = deepcopy(cfg)
    changed.update(experiment_id="new-hypothesis", factors=["another-factor"],
                   roundtrip_cost_bps=20.0, limitations=["changed description"])
    changed["symbols"].reverse()
    assert data_request(changed) == req
    assert req == {
        "provider": "yahoo_finance_yfinance", "interval": "1d",
        "auto_adjust": False, "actions": True, "repair": False,
        "field_policy": "raw_ohlcv_adjclose_actions_v1",
        "symbols": ["AAA", "BBB", "SPY"],
        "start": "2012-01-01", "end": "2024-01-01",
    }
    canonical = json.dumps(req, sort_keys=True, separators=(",", ":")).encode()
    assert request_sha256(req) == hashlib.sha256(canonical).hexdigest()
    assert cfg == before


@pytest.mark.parametrize("key,value", [
    ("source", "unverified alternative provider"), ("symbols", ["AAA", "AAA"]),
    ("symbols", "AAA"), ("symbols", ["../AAA"]), ("benchmark", "AAA"),
    ("benchmark", "bad/ticker"), ("download_start", "2012-1-1"),
    ("download_start", "2024-01-01"), ("download_end_exclusive", "2011-12-31"),
])
def test_invalid_data_request_fails(key, value):
    cfg = experiment()
    cfg[key] = value
    with pytest.raises((ValueError, TypeError)):
        data_request(cfg)


def test_explicit_identity_reuses_with_new_experiment_and_preserves_inputs():
    cfg, cached = experiment(), manifest()
    cfg["experiment_id"] = "new-experiment"
    before = deepcopy((cfg, cached))
    out = assert_cache_compatible(cfg, cached)
    assert out["compatibility_mode"] == "explicit_data_identity_v1"
    assert out["data_request_sha256"] == request_sha256(data_request(cfg))
    assert (cfg, cached) == before


def test_legacy_compatibility_requires_explicit_verified_flag_and_is_read_only():
    cached = manifest(explicit=False)
    before = deepcopy(cached)
    out = assert_cache_compatible(experiment(), cached, legacy_verified=True)
    assert out["compatibility_mode"] == "legacy_verified_request_v1"
    assert out["data_request_sha256"] == request_sha256(data_request(experiment()))
    assert cached == before


@pytest.mark.parametrize("flag", [False, None, 1, "true"])
def test_unverified_or_truthy_legacy_flag_cannot_authorize_reuse(flag):
    with pytest.raises((ValueError, TypeError)):
        assert_cache_compatible(experiment(), manifest(explicit=False), legacy_verified=flag)


@pytest.mark.parametrize("missing", ["data_request", "data_request_sha256"])
def test_half_present_identity_pair_is_rejected_even_with_legacy_flag(missing):
    cached = manifest()
    del cached["metadata"][missing]
    with pytest.raises((ValueError, TypeError)):
        assert_cache_compatible(experiment(), cached, legacy_verified=True)


@pytest.mark.parametrize("case", ["source", "request_symbols", "request_start",
    "requested_end", "duplicate_file", "missing_file", "first_before_start",
    "last_at_end", "reversed_span", "identity_hash", "identity_policy"])
def test_cache_contract_incompatibilities_fail_without_mutation(case):
    cached = manifest()
    meta = cached["metadata"]
    if case == "source":
        meta["source"] = "different provider"
    elif case == "request_symbols":
        meta["request"]["symbols"] = ["AAA", "SPY"]
    elif case == "request_start":
        meta["request"]["start"] = "2011-01-01"
    elif case == "requested_end":
        meta["requested_end"] = "2023-12-31"
    elif case == "duplicate_file":
        cached["files"].append(deepcopy(cached["files"][0]))
    elif case == "missing_file":
        cached["files"].pop()
    elif case == "first_before_start":
        cached["files"][0]["first_date"] = "2011-12-31"
    elif case == "last_at_end":
        cached["files"][0]["last_date"] = "2024-01-01"
    elif case == "reversed_span":
        cached["files"][0].update(first_date="2023-12-29", last_date="2013-01-02")
    elif case == "identity_hash":
        meta["data_request_sha256"] = "0" * 64
    elif case == "identity_policy":
        meta["data_request"]["auto_adjust"] = True
        meta["data_request_sha256"] = request_sha256(meta["data_request"])
    before = deepcopy(cached)
    with pytest.raises((ValueError, TypeError)):
        assert_cache_compatible(experiment(), cached, legacy_verified=True)
    assert cached == before


def test_observed_span_need_not_equal_requested_interval():
    cached = manifest()
    cached["files"][0].update(first_date="2020-01-02", last_date="2022-12-30")
    assert assert_cache_compatible(experiment(), cached)["compatibility_mode"] == "explicit_data_identity_v1"


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("source_experiment_id", "foreign-experiment"),
    ("allowed_periods", ["development", "validation", "reserved"]),
    ("allowed_years", list(range(2013, 2025))), ("reserved_start", "2025-01-01"),
    ("run_reserved", True), ("new_hypotheses", True), ("select_candidates", True),
    ("run_reserved", 0), ("cost_scenarios_bps", [0.0, 10.0, 20.0, -40.0]),
    ("concentration_top_dates", 10),
])
def test_diagnostic_registration_cannot_expand_scope(key, value):
    reg = registration()
    reg[key] = value
    with pytest.raises((ValueError, TypeError)):
        run_diagnostics(reg=reg)


@pytest.mark.parametrize("case", ["identity", "reserved_true", "reserved_integer",
    "missing_period", "extra_period", "wrong_period_boundary", "missing_factor",
    "extra_factor", "empty_factor", "cost_bool", "cost_mismatch"])
def test_source_contract_failures_are_not_silently_excluded(case):
    source = source_result()
    dev = source["periods"]["development"]
    if case == "identity":
        source["experiment_id"] = "foreign-experiment"
    elif case == "reserved_true":
        source["reserved_evaluated"] = True
    elif case == "reserved_integer":
        source["reserved_evaluated"] = 0
    elif case == "missing_period":
        source["periods"].pop("validation")
    elif case == "extra_period":
        source["periods"]["reserved"] = deepcopy(dev)
    elif case == "wrong_period_boundary":
        dev["period"]["end"] = "2021-01-01"
    elif case == "missing_factor":
        dev["factors"].pop(FACTORS[-1])
    elif case == "extra_factor":
        dev["factors"]["unregistered"] = deepcopy(dev["factors"][FACTORS[0]])
    elif case == "empty_factor":
        dev["factors"][FACTORS[0]]["per_date"] = []
    elif case == "cost_bool":
        dev["cost"]["roundtrip_bps"] = True
    elif case == "cost_mismatch":
        dev["cost"]["roundtrip_bps"] = 20.0
    with pytest.raises((ValueError, TypeError)):
        run_diagnostics(source=source)


@pytest.mark.parametrize("case", ["duplicate", "unsorted", "malformed_date",
    "outside_period", "reserved_date", "nan", "infinity", "bool", "string",
    "missing_return", "inconsistent_excess"])
def test_bad_daily_rows_fail_instead_of_being_dropped(case):
    source = source_result()
    records = source["periods"]["development"]["factors"][FACTORS[0]]["per_date"]
    if case == "duplicate":
        records.insert(1, deepcopy(records[0]))
    elif case == "unsorted":
        records.reverse()
    elif case == "malformed_date":
        records[0]["date"] = "2013-1-2"
    elif case == "outside_period":
        records[0]["date"] = "2021-01-04"
    elif case == "reserved_date":
        records[0]["date"] = "2024-01-02"
    elif case in {"nan", "infinity", "bool", "string"}:
        records[0]["top_net"] = {"nan": float("nan"), "infinity": float("inf"),
                                  "bool": True, "string": "0.009"}[case]
    elif case == "missing_return":
        del records[0]["benchmark_net"]
    elif case == "inconsistent_excess":
        records[0]["excess_random"] = 0.5
    with pytest.raises((ValueError, TypeError)):
        run_diagnostics(source=source)


@pytest.mark.parametrize("key,value", [
    ("auto_adjust", True), ("actions", False), ("repair", True),
    ("interval", "1h"), ("auto_adjust", 0),
])
def test_explicit_unsupported_acquisition_policy_is_not_ignored(key, value):
    cfg = experiment()
    cfg[key] = value
    with pytest.raises((ValueError, TypeError)):
        data_request(cfg)


def test_all_four_candidates_annual_means_and_costs_match_hand_calculation():
    source, cfg, reg = source_result(), experiment(), registration()
    before = deepcopy((source, cfg, reg))
    out = diagnose_result(source, cfg, reg)
    assert (source, cfg, reg) == before
    assert out["reserved_evaluated"] is False
    assert out["candidate_selection"] is False
    assert {"descriptive_not_new_oos", "flat_common_cost_not_execution_model",
            "date_not_name_contribution"} <= set(out["limitations"])
    assert set(out["periods"]) == {"development", "validation"}
    for period in out["periods"].values():
        assert set(period) == set(FACTORS)
    for factor in FACTORS:
        dev = out["periods"]["development"][factor]
        annual = {item["year"]: item for item in dev["annual"]}
        assert set(annual) == {2013, 2014}
        assert annual[2013]["date_count"] == 4
        assert annual[2013]["mean_net"] == pytest.approx(0.0115)
        assert annual[2013]["mean_excess_random"] == pytest.approx(0.0125)
        assert annual[2013]["mean_excess_benchmark"] == pytest.approx(0.0075)
        assert annual[2014]["date_count"] == 3
        assert annual[2014]["mean_net"] == pytest.approx(0.049)
        scenarios = {item["cost_bps"]: item for item in dev["cost_scenarios"]}
        assert set(scenarios) == {0.0, 10.0, 20.0, 40.0}
        original_top = 0.193 / 7
        for cost, scenario in scenarios.items():
            adjustment = 0.001 - cost / 10000
            assert scenario["mean_top_net"] == pytest.approx(original_top + adjustment)
            assert scenario["mean_benchmark_net"] == pytest.approx(0.004 + adjustment)
            assert scenario["mean_random_net"] == pytest.approx(-0.001 + adjustment)
            assert scenario["mean_excess_random"] == pytest.approx(0.2 / 7)
            assert scenario["mean_excess_benchmark"] == pytest.approx(original_top - 0.004)
    json.dumps(out, allow_nan=False)


def test_positive_date_concentration_uses_positive_excess_not_net_profit():
    out = run_diagnostics()
    for factor in FACTORS:
        conc = out["periods"]["development"][factor]["positive_date_concentration"]
        assert conc["positive_sum"] == pytest.approx(0.21)
        assert conc["top5_positive_share"] == pytest.approx(0.20 / 0.21)
        assert [r["excess_random"] for r in conc["top_dates"]] == pytest.approx(
            [0.06, 0.05, 0.04, 0.03, 0.02])
        absent = out["periods"]["validation"][factor]["positive_date_concentration"]
        assert absent["positive_sum"] == 0.0
        assert absent["top5_positive_share"] is None
        assert absent["top_dates"] == []


def test_fewer_than_five_positive_dates_and_ties_have_stable_order():
    source = source_result()
    for factor in FACTORS:
        source["periods"]["development"]["factors"][factor]["per_date"] = [
            row("2013-01-02", 0.01), row("2013-01-03", 0.01), row("2013-01-04", 0.0)]
    out = run_diagnostics(source=source)
    conc = out["periods"]["development"][FACTORS[0]]["positive_date_concentration"]
    assert conc["top5_positive_share"] == pytest.approx(1.0)
    assert [item["date"] for item in conc["top_dates"]] == ["2013-01-02", "2013-01-03"]
