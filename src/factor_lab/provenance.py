"""Bind legacy migration and descriptive diagnostics to inspected local artifacts."""
from __future__ import annotations

import hashlib
import json
import math


def verify_legacy_proof(manifest: dict, raw_manifest: bytes, proof: dict) -> None:
    """A source-name match alone cannot establish a legacy acquisition policy."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (proof.get("schema_version") != 1 or
            hashlib.sha256(raw_manifest).hexdigest() != proof.get("manifest_sha256") or
            hashlib.sha256(canonical).hexdigest() != proof.get("manifest_canonical_sha256") or
            manifest["metadata"].get("config_sha256") != proof.get("config_sha256") or
            manifest["metadata"].get("source") != proof.get("source")):
        raise ValueError("Legacy cache is not the inspected immutable acquisition artifact")
    for name, length in [("acquisition_code_commit", 40), ("acquisition_source_sha256", 64)]:
        value = proof.get(name)
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Legacy acquisition implementation evidence is missing")
    if not proof.get("policy_evidence"):
        raise ValueError("Legacy acquisition policy evidence is missing")


def verify_diagnostic_source(source: dict, raw_source: bytes, cfg: dict, cfg_hash: str, registration: dict) -> None:
    """Validate exact source and internal counts/aggregates before summarizing."""
    if (hashlib.sha256(raw_source).hexdigest() != registration.get("expected_source_sha256") or
            cfg_hash != registration.get("expected_config_sha256") or
            source.get("config_sha256") != cfg_hash or
            source.get("cache_manifest_sha256") != registration.get("expected_cache_manifest_sha256") or
            registration.get("expected_factors") != cfg["factors"]):
        raise ValueError("Diagnostic source does not match immutable preregistered evidence")
    fields = {"top_net": "mean_net", "benchmark_net": "mean_benchmark_net",
              "random_net": "mean_random_net", "excess_random": "mean_excess_random"}
    for period in source.get("periods", {}).values():
        for factor in period.get("factors", {}).values():
            rows = factor.get("per_date")
            if not isinstance(rows, list) or not rows or type(factor.get("dates")) is not int or factor["dates"] != len(rows):
                raise ValueError("Source factor count differs from per-date evidence")
            for row in rows:
                if type(row.get("n")) is not int or row["n"] < 1:
                    raise ValueError("Source factor cohort count must be a positive integer")
            for per_date, aggregate in fields.items():
                values = [row.get(per_date) for row in rows]
                supplied = factor.get(aggregate)
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in [*values, supplied]):
                    raise ValueError("Source contains non-finite or non-numeric return evidence")
                if not math.isclose(math.fsum(values) / len(rows), supplied, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("Source factor aggregate differs from per-date evidence")
