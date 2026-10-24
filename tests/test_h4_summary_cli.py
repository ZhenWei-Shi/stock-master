"""Exercise the real --summary entry point with isolated synthetic ledgers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("records,valid", [
    ([{"date": "2026-10-12", "ticker": "S", "group": "signal", "pool": "v2"},
      {"date": "2026-10-12", "ticker": "C", "group": "control", "pool": "v2"}], True),
    ([{"date": "2026-10-12", "ticker": "S", "group": "signal", "pool": "v2", "ret": float("nan")}], False),
    ([{"date": "2026-10-12", "ticker": "S", "group": "signal", "pool": "v2", "ret": 0.01},
      {"date": "2026-10-12", "ticker": "C", "group": "control", "pool": "v2", "ret": 0.0}], True),
])
def test_summary_cli_is_read_only_and_refuses_invalid_data(tmp_path, records, valid):
    repo = Path(__file__).resolve().parents[1]
    for name in ("src/overnight_lab.py", "src/factor_lab/__init__.py",
                 "src/factor_lab/h4_validation.py", "src/factor_lab/metrics.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
    ledger = tmp_path / "data/overnight_lab.json"
    ledger.parent.mkdir()
    ledger.write_text(json.dumps({"trades": records}), encoding="utf8")
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "src.overnight_lab", "--summary"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf8", timeout=20,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert (result.returncode == 0) is valid
    assert "✅" not in result.stdout
    if valid:
        assert "样本不足" in result.stdout
    else:
        assert "无法判定" in result.stderr
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
