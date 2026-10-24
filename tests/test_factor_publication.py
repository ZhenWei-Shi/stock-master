"""Actual filesystem races and failures for immutable evidence publication."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

from src.factor_lab.publication import publish_json, replace_json


def test_preexisting_complete_result_is_preserved(tmp_path):
    target = tmp_path / "result.json"
    original = b'{"original":true}\n'
    target.write_bytes(original)
    with pytest.raises(FileExistsError):
        publish_json(target, {"replacement": True})
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_simultaneous_writers_publish_exactly_one_complete_result(tmp_path):
    for iteration in range(8):
        folder = tmp_path / str(iteration)
        folder.mkdir()
        target = folder / "result.json"
        barrier = Barrier(2)
        payloads = [{"writer": writer, "body": str(writer) * 100_000} for writer in range(2)]

        def write(writer):
            barrier.wait(timeout=5)
            try:
                publish_json(target, payloads[writer])
                return (writer, "published")
            except FileExistsError:
                return (writer, "exists")

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(write, [0, 1]))
        assert sorted(status for _, status in outcomes) == ["exists", "published"]
        winner = next(writer for writer, status in outcomes if status == "published")
        assert json.loads(target.read_text()) == payloads[winner]
        assert list(folder.iterdir()) == [target]


def test_failed_publication_leaves_no_partial_result_and_preserves_other_temp(tmp_path):
    target = tmp_path / "result.json"
    unrelated = tmp_path / ".result.json.other-writer.tmp"
    unrelated.write_text("belongs to another writer")
    with patch("src.factor_lab.publication.os.link", side_effect=OSError("unsupported filesystem")):
        with pytest.raises(OSError, match="unsupported filesystem"):
            publish_json(target, {"value": 1})
    assert not target.exists()
    assert unrelated.read_text() == "belongs to another writer"
    assert list(tmp_path.iterdir()) == [unrelated]
    publish_json(target, {"retried": True})
    assert json.loads(target.read_text()) == {"retried": True}


def test_serialization_failure_preserves_existing_evidence(tmp_path):
    target = tmp_path / "result.json"
    target.write_text('{"original":true}')
    with pytest.raises(ValueError):
        publish_json(target, {"value": float("nan")})
    assert json.loads(target.read_text()) == {"original": True}
    assert list(tmp_path.iterdir()) == [target]


def test_mutable_metadata_can_be_replaced(tmp_path):
    target = tmp_path / "status.json"
    replace_json(target, {"attempt": 1})
    replace_json(target, {"attempt": 2})
    assert json.loads(target.read_text()) == {"attempt": 2}
    assert list(tmp_path.iterdir()) == [target]


def test_failed_mutable_replace_keeps_previous_metadata(tmp_path):
    target = tmp_path / "status.json"
    replace_json(target, {"original": True})
    with patch("src.factor_lab.publication.os.replace", side_effect=OSError("simulated write failure")):
        with pytest.raises(OSError, match="simulated write failure"):
            replace_json(target, {"replacement": True})
    assert json.loads(target.read_text()) == {"original": True}
    assert list(tmp_path.iterdir()) == [target]
