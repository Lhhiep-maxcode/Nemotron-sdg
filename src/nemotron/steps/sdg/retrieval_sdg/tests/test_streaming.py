"""Offline checks for incremental output, resume and the real multiprocessing CLI."""

import asyncio
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from retrieval_sdg.conversation.verifiers import ToolCallVerifier
from retrieval_sdg.retrieval.local import LocalRetrievalClient, calculate
from retrieval_sdg.streaming import completed_samples, output_path, process_file

HERE = Path(__file__).resolve().parents[1]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result(sample_id, source):
    return {"sample_id": sample_id, "source": source,
            "messages": [{"role": "assistant", "content": "completed"}]}


def test_writes_before_slow_sample_finishes_and_bounds_concurrency(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    out = tmp_path / "result.raw.jsonl"
    release = threading.Event()
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = maximum = 0

    def generate(raw, sample_id, provenance, cfg, engine):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if provenance["line"] <= 2:
                barrier.wait(timeout=5)
            if provenance["line"] == 1:
                assert release.wait(5)
            return _result(sample_id, provenance)
        finally:
            with lock:
                active -= 1

    async def run():
        task = asyncio.create_task(process_file({"batch": {"concurrency_per_process": 2}}, source,
                                                "input.jsonl", out, synthesizer=generate))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not out.exists() or not out.stat().st_size:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)
            assert not task.done()
            assert _rows(out)[0]["source"]["line"] == 2
        finally:
            release.set()
            stats = await task
        assert stats == {"saved": 4, "resumed": 0, "failed": 0}

    asyncio.run(run())
    assert maximum == 2
    assert len(_rows(out)) == 4


def test_resume_retries_only_failures_and_repairs_partial_tail(tmp_path):
    source, out = tmp_path / "input.jsonl", tmp_path / "out.raw.jsonl"
    source.write_text('{}\n\n{}\n{}\n', encoding="utf-8")
    cfg = {"batch": {"concurrency_per_process": 3}}
    calls = []
    first_run = True

    def generate(raw, sample_id, provenance, cfg, engine):
        calls.append(provenance["line"])
        if first_run and provenance["line"] == 3:
            raise RuntimeError("temporary model error")
        return _result(sample_id, provenance)

    stats = asyncio.run(process_file(cfg, source, "x/input.jsonl", out, synthesizer=generate))
    assert stats == {"saved": 2, "resumed": 0, "failed": 1}
    assert {row["source"]["line"] for row in _rows(out)} == {1, 4}
    with out.open("ab") as handle:
        handle.write(b'{"sample_id": "interrupted", "messages": [')
    calls.clear()
    first_run = False
    stats = asyncio.run(process_file(cfg, source, "x/input.jsonl", out, synthesizer=generate))
    assert stats == {"saved": 1, "resumed": 2, "failed": 0}
    assert calls == [3]
    assert len({row["sample_id"] for row in _rows(out)}) == 3
    calls.clear()
    stats = asyncio.run(process_file(cfg, source, "x/input.jsonl", out, synthesizer=generate))
    assert stats == {"saved": 0, "resumed": 3, "failed": 0}
    assert calls == []
    assert _rows(out.with_suffix(".errors.jsonl"))[0]["source"]["line"] == 3


def test_concurrency_can_exceed_default_asyncio_executor_size(tmp_path):
    source, out = tmp_path / "input.jsonl", tmp_path / "out.raw.jsonl"
    source.write_text("{}\n" * 36, encoding="utf-8")
    barrier = threading.Barrier(36)

    def generate(raw, sample_id, provenance, cfg, engine):
        barrier.wait(timeout=5)
        return _result(sample_id, provenance)

    stats = asyncio.run(process_file({"batch": {"concurrency_per_process": 36}}, source,
                                     "input.jsonl", out, synthesizer=generate))
    assert stats == {"saved": 36, "resumed": 0, "failed": 0}


def test_resume_preserves_complete_row_without_newline_and_rejects_middle_corruption(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_bytes(b'{"sample_id":"ok"}')
    assert completed_samples(path) == {"ok"}
    assert path.read_bytes().endswith(b"\n")
    bad = b'{"sample_id":"ok"}\nBAD\n{"sample_id":"also-ok"}\n'
    path.write_bytes(bad)
    with pytest.raises(ValueError, match="before EOF"):
        completed_samples(path)
    assert path.read_bytes() == bad


def test_relative_output_paths_do_not_flatten_names(tmp_path):
    root = tmp_path / "input"
    assert output_path(root / "a/b.jsonl", root, tmp_path / "out") == tmp_path / "out/a/b.raw.jsonl"
    assert output_path(root / "a__b.jsonl", root, tmp_path / "out") == tmp_path / "out/a__b.raw.jsonl"


def test_local_search_returns_actual_vietnamese_passages_and_stable_ids():
    text = "Đá granit có RMR bằng 54.\n" * 40 + "Món phở dùng nước dùng và bánh phở.\n" * 40
    client = LocalRetrievalClient(text, "sample", "book", chunk_chars=200, overlap_chars=30)
    result = client.retrieve(unicodedata.normalize("NFD", "bánh phở"), 3, rng=random.Random(7))
    assert result and all(c.doc_id == "book" and c.text in text for c in result)
    assert any("phở" in c.text for c in result)
    assert all(c.id.startswith("h") and len(c.id) == 13 for c in result)
    assert not client.retrieve("xyzzy", 3, rng=random.Random(7))
    again = LocalRetrievalClient(text, "sample", "book", chunk_chars=200, overlap_chars=30)
    assert [c.id for c in client.chunks] == [c.id for c in again.chunks]
    different = LocalRetrievalClient(text, "other", "book", chunk_chars=200, overlap_chars=30)
    assert not {c.id for c in client.chunks} & {c.id for c in different.chunks}


@pytest.mark.parametrize("expression,result", [("(54 + 7.3) / 2", 30.65), ("250 * 1.5", 375),
                                                ("-2 ** 2", -4), ("100 / 4", 25)])
def test_calculator(expression, result):
    assert calculate(expression)["result"] == pytest.approx(result)


@pytest.mark.parametrize("expression", ["1/0", "10**1000", "__import__('os').getcwd()",
                                         "[1,2]", "True + 1", "1e999", "(-1)**0.5"])
def test_calculator_rejects_non_arithmetic_or_unbounded_values(expression):
    assert "error" in calculate(expression)


def test_cli_multiprocess_generation_and_resume(tmp_path):
    requests = []
    request_lock = threading.Lock()
    active = maximum = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal active, maximum
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with request_lock:
                requests.append(payload)
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.02)
                messages = payload["messages"]
                if not payload.get("tools"):
                    message = {"role": "assistant", "content": json.dumps({"query": "Đá granit có RMR bao nhiêu?"})}
                elif messages[-1]["role"] == "tool":
                    chunk = json.loads(messages[-1]["content"])["results"][0]
                    message = {"role": "assistant", "content": f"RMR bằng 54 [{chunk['id']}]."}
                else:
                    message = {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {
                            "name": "search", "arguments": '{"query":"đá granit RMR"}'}}]}
                body = json.dumps({"id": "test", "object": "chat.completion", "created": 0,
                                   "model": "test", "choices": [
                                       {"index": 0, "message": message, "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with request_lock:
                    active -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        inp, output = tmp_path / "input", tmp_path / "output"
        for rel in ("math/data.jsonl", "books/data.jsonl"):
            path = inp / rel
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps({"id": f"{rel}:{i}", "text": "Đá granit có RMR bằng 54."},
                                              ensure_ascii=False) + "\n" for i in range(2)), encoding="utf-8")
        settings = OmegaConf.load(HERE / "config/knowledge.yaml")
        settings.pipeline_config = str(HERE / "config/pipeline.yaml")
        settings.batch.num_process = 1  # CLI flags must override these settings
        settings.batch.concurrency_per_process = 1
        settings.engine.min_turns = settings.engine.max_turns = 1
        settings.engine.min_hops = 1
        config_path = tmp_path / "knowledge.yaml"
        OmegaConf.save(settings, config_path)
        env = dict(os.environ, ASSISTANT_ENDPOINT=f"http://127.0.0.1:{server.server_port}/v1",
                   USER_MODEL_ENDPOINT=f"http://127.0.0.1:{server.server_port}/v1", NO_PROXY="127.0.0.1,localhost",
                   PYTHONIOENCODING="utf-8")
        command = [sys.executable, str(HERE / "run_knowledge_sdg.py"), "--config", str(config_path),
                   "--input-path", str(inp), "--output-dir", str(output),
                   "--num-process", "2", "--concurrency-per-process", "2"]
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "maximum concurrent samples=4" in result.stdout
        assert 1 < maximum <= 4
        assert len(requests) == 12  # question, search call, final answer per source row
        paths = sorted(output.rglob("*.raw.jsonl"))
        assert len(paths) == 2
        for path in paths:
            rows = _rows(path)
            assert len(rows) == 2
            for row in rows:
                messages = row["messages"]
                assert messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls")
                calls = [call for message in messages for call in message.get("tool_calls", [])]
                assert calls and all(ToolCallVerifier().verify_single(call, row["tools"])[0] for call in calls)
                chunks = [chunk for message in messages if message["role"] == "tool"
                          for chunk in json.loads(message["content"])["results"]]
                cited = set(re.findall(r"h[0-9a-f]{12}", messages[-1]["content"]))
                assert len(cited) == 1 and cited <= {chunk["id"] for chunk in chunks}
                assert all(chunk["text"] == "Đá granit có RMR bằng 54." for chunk in chunks)
        before = {path: path.read_bytes() for path in paths}
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(requests) == 12  # no model calls for completed samples
        assert before == {path: path.read_bytes() for path in paths}
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
