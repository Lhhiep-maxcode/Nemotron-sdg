"""Durable JSONL output keyed by sample ID, with separate checksum journals."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path


def _payload(row: dict) -> str:
    return json.dumps({"messages": row["messages"], "tools": row["tools"], "id": row["id"]},
                      ensure_ascii=False, default=str)


def _entry(sample_id: str, payload: str) -> str:
    return json.dumps({"sample_id": sample_id, "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()})


def _sync_write(handle, text: str) -> None:
    handle.write(text + "\n")
    handle.flush()
    os.fsync(handle.fileno())


class CheckpointedJsonl:
    """Journal each sample ID and payload checksum BEFORE writing its output.

    Recovery matches the files line by line. A pending journal entry or partial
    output at EOF is discarded, while every completed output is retained. The
    output id is the deterministic sample ID, also used to control resume.
    """

    def __init__(self, path: Path, checkpoint_path: Path | None = None):
        self.path = path
        self.checkpoint_path = checkpoint_path or path.with_suffix(".state.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.failed = False
        self._rewrite_ids = False
        self._migrate_output()
        self.done = self._recover()
        if self._rewrite_ids:
            self._migrate_output(from_checkpoint=True)

    def _migrate_output(self, *, from_checkpoint: bool = False) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as source:
            first = next((line for line in source if line.strip()), b"")
        try:
            first_row = json.loads(first)
            legacy = isinstance(first_row, dict) and "sample_id" in first_row
        except (ValueError, UnicodeError):
            legacy = False
        if not legacy and not from_checkpoint:
            return
        temporary = []
        try:
            for destination in (self.path, self.checkpoint_path):
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                                 suffix=".tmp", delete=False) as handle:
                    temporary.append(Path(handle.name))
            with ExitStack() as stack:
                source = stack.enter_context(self.path.open("rb"))
                prior_state = stack.enter_context(self.checkpoint_path.open("rb")) if from_checkpoint else None
                output = stack.enter_context(temporary[0].open("w", encoding="utf-8", newline="\n"))
                checkpoint = stack.enter_context(temporary[1].open("w", encoding="utf-8", newline="\n"))
                while line := source.readline():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeError):
                        if source.read(1):
                            raise ValueError(f"Corrupt legacy output before EOF: {self.path}")
                        break  # omit only a torn final record
                    sample_id = json.loads(prior_state.readline())["sample_id"] if prior_state else \
                        row.get("sample_id") or row.get("id")
                    if not isinstance(sample_id, str) or not sample_id:
                        raise ValueError(f"Legacy output has no sample ID: {self.path}")
                    payload = _payload({**row, "id": sample_id})
                    output.write(payload + "\n")
                    state = json.loads(_entry(sample_id, payload))
                    if from_checkpoint:
                        state["previous_sha256"] = hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()
                    checkpoint.write(json.dumps(state) + "\n")
                for handle in (output, checkpoint):
                    handle.flush()
                    os.fsync(handle.fileno())
            # Install the journal first. If interrupted before replacing output,
            # legacy metadata remains available, or previous_sha256 still validates
            # the original-ID output, so this migration can safely run again.
            temporary[1].replace(self.checkpoint_path)
            temporary[0].replace(self.path)
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()

    def _recover(self) -> set[str]:
        done = set()
        self.path.touch(exist_ok=True)
        self.checkpoint_path.touch(exist_ok=True)
        with self.path.open("r+b") as output, self.checkpoint_path.open("r+b") as checkpoint:
            while True:
                output_start, checkpoint_start = output.tell(), checkpoint.tell()
                line, entry = output.readline(), checkpoint.readline()
                if not line and not entry:
                    break
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    valid_output = False
                else:
                    if (not isinstance(row, dict) or set(row) != {"messages", "tools", "id"}
                            or not isinstance(row["messages"], list) or not isinstance(row["tools"], list)):
                        raise ValueError(f"Invalid output schema: {self.path}")
                    valid_output = True
                if not valid_output:
                    if output.read(1) or checkpoint.read(1):
                        raise ValueError(f"Corrupt output/checkpoint before EOF: {self.path}")
                    for handle, start in ((output, output_start), (checkpoint, checkpoint_start)):
                        handle.seek(start)
                        handle.truncate()
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                try:
                    state = json.loads(entry)
                    sample_id = state["sample_id"]
                    valid_state = isinstance(sample_id, str) and bool(sample_id)
                    valid_state &= hashlib.sha256(line.rstrip(b"\r\n")).hexdigest() in (
                        state["sha256"], state.get("previous_sha256"))
                except (ValueError, UnicodeError, KeyError, TypeError):
                    valid_state = False
                if not valid_state:
                    raise ValueError(f"Missing or mismatched checkpoint for {self.path}. "
                                     "Keep the matching .state directory when resuming.")
                self._rewrite_ids |= row["id"] != sample_id
                done.add(sample_id)
                for handle, data in ((output, line), (checkpoint, entry)):
                    if not data.endswith(b"\n"):
                        handle.seek(0, os.SEEK_END)
                        handle.write(b"\n")
                        handle.flush()
                        os.fsync(handle.fileno())
        return done

    def __enter__(self):
        self.output = self.path.open("a", encoding="utf-8", newline="\n")
        try:
            self.checkpoint = self.checkpoint_path.open("a", encoding="utf-8", newline="\n")
        except BaseException:
            self.output.close()
            raise
        return self

    def __exit__(self, *args):
        try:
            self.output.close()
        finally:
            self.checkpoint.close()

    def append(self, sample_id: str, row: dict) -> None:
        if self.failed:
            raise RuntimeError("A previous output write failed; resume the file to recover")
        payload = _payload({**row, "id": sample_id})
        try:
            _sync_write(self.checkpoint, _entry(sample_id, payload))
            _sync_write(self.output, payload)
        except BaseException:
            self.failed = True  # prevent other ready async workers appending after a torn write
            raise
        self.done.add(sample_id)


def completed_samples(path: Path, checkpoint_path: Path | None = None) -> set[str]:
    return CheckpointedJsonl(path, checkpoint_path).done
