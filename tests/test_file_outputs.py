from pathlib import Path

import pytest

from movierec.paths import replace_file


def test_atomic_output_retries_a_brief_reader_lock(tmp_path, monkeypatch):
    source, destination = tmp_path / "new", tmp_path / "output"
    source.write_text("complete new data")
    destination.write_text("old data")
    original = Path.replace
    calls = []

    def locked_replace(self, target):
        calls.append(self)
        if len(calls) < 3:
            assert destination.read_text() == "old data"
            raise PermissionError("Temporary Windows reader lock")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", locked_replace)
    replace_file(source, destination, attempts=3, delay=0)
    assert destination.read_text() == "complete new data"
    assert len(calls) == 3


def test_persistent_output_permission_error_is_not_hidden(tmp_path, monkeypatch):
    source, destination = tmp_path / "new", tmp_path / "output"
    source.write_text("new data")
    destination.write_text("old data")

    def denied(self, target):
        raise PermissionError("Persistent access denied")

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError, match="Persistent"):
        replace_file(source, destination, attempts=2, delay=0)
    assert source.read_text() == "new data"
    assert destination.read_text() == "old data"
