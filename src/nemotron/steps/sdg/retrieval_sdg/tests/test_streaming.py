"""Offline checks for incremental output, resume and the real multiprocessing CLI."""

import asyncio
import hashlib
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
from retrieval_sdg.checkpoint import CheckpointedJsonl, completed_samples
from retrieval_sdg.conversation.verifiers import ToolCallVerifier
from retrieval_sdg.core.messages import format_judge_history, source_passages_from_messages
from retrieval_sdg.progress import RunProgress, count_inputs
from retrieval_sdg.retrieval.local import LocalRetrievalClient, calculate, knowledge_tool_handlers
from retrieval_sdg.streaming import (
    _engine_config,
    _load_personas,
    _personas,
    output_path,
    process_file,
    synthesize,
)

HERE = Path(__file__).resolve().parents[1]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result(sample_id, source):
    return {"id": sample_id, "tools": [],
            "messages": [{"role": "assistant", "content": str(source["line"])}]}


def _config():
    return OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(HERE / "config/pipeline.yaml"),
                                                 OmegaConf.load(HERE / "config/knowledge.yaml")), resolve=True)


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
            assert _rows(out)[0]["messages"][0]["content"] == "2"
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
    assert {row["messages"][0]["content"] for row in _rows(out)} == {"1", "4"}
    with out.open("ab") as handle:
        handle.write(b'{"sample_id": "interrupted", "messages": [')
    calls.clear()
    first_run = False
    stats = asyncio.run(process_file(cfg, source, "x/input.jsonl", out, synthesizer=generate))
    assert stats == {"saved": 1, "resumed": 2, "failed": 0}
    assert calls == [3]
    assert len({row["id"] for row in _rows(out)}) == 3
    assert all(set(row) == {"messages", "tools", "id"} for row in _rows(out))
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


def test_progress_counts_selected_nonblank_rows_per_file(tmp_path):
    paths = [tmp_path / f"{i}.jsonl" for i in range(3)]
    for path, text in zip(paths, ["{}\n\n{}\n{}\n", "\n\n", "{}\n"]):
        path.write_text(text, encoding="utf-8")
    assert count_inputs(iter(paths), None) == (3, 4)
    assert count_inputs(iter(paths), 2) == (3, 3)


def test_progress_eta_excludes_resume_skips_and_ignores_stale_reports(monkeypatch):
    from retrieval_sdg import progress as progress_module
    now = [100.0]
    monkeypatch.setattr(progress_module, "monotonic", lambda: now[0])
    with RunProgress(2, 100, enabled=False) as progress:
        progress.register("a.jsonl")
        now[0] = 110.0
        progress.report("a.jsonl", {"saved": 2, "resumed": 90, "failed": 1})
        rate, eta = progress.estimates()
        assert rate == pytest.approx(0.3) and eta == pytest.approx(7 / 0.3)
        progress.report("a.jsonl", {"saved": 1, "resumed": 80, "failed": 0})
        progress.finish("a.jsonl", {"saved": 2, "resumed": 90, "failed": 1})
        progress.report("a.jsonl", {"saved": 3, "resumed": 90, "failed": 1})
        assert progress.stats == {"saved": 2, "resumed": 90, "failed": 1}
        assert not progress.active
    with RunProgress(1, 10, enabled=False) as progress:
        progress.register("complete.jsonl")
        progress.finish("complete.jsonl", {"saved": 0, "resumed": 10, "failed": 0})
        assert progress.estimates() == (None, 0.0)


def test_progress_reports_only_durable_saves(tmp_path):
    source, out = tmp_path / "input.jsonl", tmp_path / "out.raw.jsonl"
    source.write_text("{}\n{}\n", encoding="utf-8")
    notifications = []

    def generate(raw, sample_id, provenance, cfg, engine):
        return _result(sample_id, provenance)

    def report(stats, error):
        assert len(_rows(out)) == stats["saved"]
        assert len(completed_samples(out)) == stats["saved"]
        notifications.append((stats, error))

    stats = asyncio.run(process_file({"batch": {"concurrency_per_process": 2}}, source,
                                     "input.jsonl", out, synthesizer=generate, on_progress=report))
    assert notifications[-1] == (stats, None)
    assert stats == {"saved": 2, "resumed": 0, "failed": 0}


def test_cancellation_retains_saved_samples_and_resumes_with_progress(tmp_path):
    source, out = tmp_path / "input.jsonl", tmp_path / "out.raw.jsonl"
    source.write_text("{}\n" * 6, encoding="utf-8")
    cfg = {"batch": {"concurrency_per_process": 2}}
    release = threading.Event()

    def blocked(raw, sample_id, provenance, cfg, engine):
        if provenance["line"] != 1:
            assert release.wait(5)
        return _result(sample_id, provenance)

    def fast(raw, sample_id, provenance, cfg, engine):
        return _result(sample_id, provenance)

    async def run():
        saved = asyncio.Event()

        def report(stats, error):
            if stats["saved"]:
                saved.set()

        task = asyncio.create_task(process_file(cfg, source, "input.jsonl", out,
                                                synthesizer=blocked, on_progress=report))
        try:
            await asyncio.wait_for(saved.wait(), timeout=5)
        finally:
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        previous = len(_rows(out))
        assert 0 < previous < 6
        reports = []
        stats = await process_file(cfg, source, "input.jsonl", out, synthesizer=fast,
                                   on_progress=lambda stats, error: reports.append(stats))
        assert stats == {"saved": 6 - previous, "resumed": previous, "failed": 0}
        assert reports[-1] == stats

    asyncio.run(run())
    assert len(_rows(out)) == len({row["id"] for row in _rows(out)}) == 6


def test_resume_preserves_complete_row_without_newline_and_rejects_middle_corruption(tmp_path):
    path = tmp_path / "raw.jsonl"
    with CheckpointedJsonl(path) as writer:
        writer.append("ok", {"id": "original-id", "messages": [], "tools": []})
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert completed_samples(path) == {"ok"}
    assert path.read_bytes().endswith(b"\n")
    with CheckpointedJsonl(path) as writer:
        writer.append("also-ok", {"id": "another-original-id", "messages": [], "tools": []})
    first, second = path.read_bytes().splitlines(keepends=True)
    bad = first + b"BAD\n" + second
    path.write_bytes(bad)
    with pytest.raises(ValueError, match="before EOF"):
        completed_samples(path)
    assert path.read_bytes() == bad


def test_legacy_output_is_normalized_without_losing_resume_ids(tmp_path):
    path = tmp_path / "raw.jsonl"
    legacy = {"sample_id": "old", "messages": [{"role": "assistant", "content": "answer"}],
              "tools": [], "source": {"file": "book.jsonl", "line": 1, "id": "original-source-id"},
              "query": "question", "hops_taken": 2}
    current = {"id": "new", "messages": [], "tools": []}
    path.write_text(json.dumps(legacy) + "\n" + json.dumps(current) + "\n", encoding="utf-8")
    assert completed_samples(path) == {"old", "new"}
    assert _rows(path) == [{"messages": legacy["messages"], "tools": [], "id": "old"}, current]
    assert completed_samples(path) == {"old", "new"}


@pytest.mark.parametrize("fail_after_checkpoint", [False, True])
def test_failed_legacy_conversion_leaves_original_file_intact(tmp_path, monkeypatch, fail_after_checkpoint):
    path = tmp_path / "raw.jsonl"
    original = json.dumps({"sample_id": "old", "messages": [], "tools": [], "query": "q"}) + "\n"
    path.write_text(original, encoding="utf-8")
    replace = Path.replace

    def fail_replace(self, target):
        fail_target = path if fail_after_checkpoint else path.with_suffix(".state.jsonl")
        if Path(target) == fail_target:
            raise OSError("interrupted migration")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted migration"):
        completed_samples(path)
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(Path, "replace", replace)
    assert completed_samples(path) == {"old"}
    assert set(_rows(path)[0]) == {"messages", "tools", "id"}


@pytest.mark.parametrize("stage", ["before_output", "partial_output", "after_output", "partial_journal"])
def test_checkpoint_recovers_each_write_boundary(tmp_path, monkeypatch, stage):
    from retrieval_sdg import checkpoint as checkpoint_module
    path = tmp_path / "raw.jsonl"
    row = {"id": "same-original-id", "messages": [], "tools": []}
    sync_write = checkpoint_module._sync_write

    def interrupted_write(handle, text):
        is_output = Path(handle.name) == path
        if stage == "partial_journal" and not is_output:
            handle.write(text[:15])
            handle.flush()
            raise OSError("interrupted journal")
        if is_output:
            if stage == "partial_output":
                handle.write(text[:15])
                handle.flush()
            if stage == "after_output":
                sync_write(handle, text)
            raise OSError("interrupted output")
        sync_write(handle, text)

    with CheckpointedJsonl(path) as writer:
        writer.append("sample-1", row)
        monkeypatch.setattr(checkpoint_module, "_sync_write", interrupted_write)
        with pytest.raises(OSError, match="interrupted"):
            writer.append("sample-2", row)
        with pytest.raises(RuntimeError, match="previous output write failed"):
            writer.append("sample-3", row)
    monkeypatch.setattr(checkpoint_module, "_sync_write", sync_write)
    expected = {"sample-1", "sample-2"} if stage == "after_output" else {"sample-1"}
    assert completed_samples(path) == expected
    with CheckpointedJsonl(path) as writer:
        writer.append("sample-3", row)
    assert completed_samples(path) == expected | {"sample-3"}
    assert len(_rows(path)) == len(expected) + 1
    assert {record["id"] for record in _rows(path)} == expected | {"sample-3"}
    assert all(record["messages"] == row["messages"] and record["tools"] == row["tools"] for record in _rows(path))


def test_original_id_output_requires_its_matching_checkpoint(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"id":"original", "messages":[], "tools":[]}\n', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Keep the matching .state directory"):
        completed_samples(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("interruption", [None, "checkpoint", "output"])
def test_document_id_outputs_migrate_to_sample_ids_without_regeneration(tmp_path, monkeypatch, interruption):
    path = tmp_path / "raw.jsonl"
    checkpoint = path.with_suffix(".state.jsonl")
    original_rows = [{"id": "repeated-document-id", "messages": [{"role": "assistant", "content": f"Answer {i}"}],
                      "tools": []} for i in range(2)]
    payloads = [json.dumps(row, ensure_ascii=False) for row in original_rows]
    path.write_text("\n".join(payloads) + "\n", encoding="utf-8")
    checkpoint.write_text("".join(json.dumps({"sample_id": f"sample-{i}",
                                             "sha256": hashlib.sha256(payload.encode()).hexdigest()}) + "\n"
                                   for i, payload in enumerate(payloads)), encoding="utf-8")
    original = path.read_bytes()
    replace = Path.replace
    if interruption:
        def fail_replace(self, target):
            if Path(target) == (checkpoint if interruption == "checkpoint" else path):
                raise OSError("interrupted ID migration")
            return replace(self, target)

        monkeypatch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="interrupted ID migration"):
            completed_samples(path)
        assert path.read_bytes() == original
        assert not list(tmp_path.glob("*.tmp"))
        monkeypatch.setattr(Path, "replace", replace)
    assert completed_samples(path) == {"sample-0", "sample-1"}
    assert _rows(path) == [{**row, "id": f"sample-{i}"} for i, row in enumerate(original_rows)]
    converted = path.read_bytes()
    assert completed_samples(path) == {"sample-0", "sample-1"}
    assert path.read_bytes() == converted


def test_merged_config_offers_six_tools_without_search_and_three_to_twenty_turns():
    cfg = _config()
    expected = {"calculator", "memory_read", "memory_write", "translation",
                "citation_format", "timeline_extract"}
    assert {tool["function"]["name"] for tool in cfg["tools"]} == expected
    assert len(cfg["tools"]) == 6 and all(tool["type"] == "function" for tool in cfg["tools"])
    engine = _engine_config(cfg)
    assert (engine.min_turns, engine.max_turns, engine.max_tools) == (3, 20, 6)
    assert engine.retrieval_tools == [] and engine.min_hops == 0 and not engine.force_first_tool
    assert engine.require_reasoning
    request_options = engine.model_clients["assistant_model"]["params"]["extra_body"]
    assert request_options["include_reasoning"]
    assert request_options["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 25}
    cfg["engine"]["min_turns"] = 21
    with pytest.raises(ValueError, match="min_turns"):
        _engine_config(cfg)


def test_memory_is_shared_within_a_conversation_and_isolated_between_samples():
    handlers = knowledge_tool_handlers()
    assert handlers["memory_read"]("citation_style") == {"key": "citation_style", "found": False, "value": None}
    assert handlers["memory_write"]("citation_style", "inline")["stored"]
    assert handlers["memory_read"]("citation_style")["value"] == "inline"
    assert not knowledge_tool_handlers()["memory_read"]("citation_style")["found"]


def test_auxiliary_tools_use_aux_response_model(monkeypatch):
    from retrieval_sdg.conversation import tools as tool_module
    cfg = _config()
    calls = []

    def fake_llm(models, alias, messages):
        calls.append((alias, messages))
        return {"content": '{"transformed":true}'}

    monkeypatch.setattr(tool_module, "call_llm", fake_llm)
    env = tool_module.ToolEnvironment(_engine_config(cfg), None, cfg["tools"], handlers=knowledge_tool_handlers())
    for name in ("translation", "citation_format", "timeline_extract"):
        content, retrieved = env.respond({"function": {"name": name, "arguments": "{}"}}, {}, "q", random.Random(7))
        assert json.loads(content) == {"transformed": True} and not retrieved
        assert name in calls[-1][1][-1]["content"]
    assert [alias for alias, _ in calls] == ["aux_response_model"] * 3
    env.respond({"function": {"name": "memory_write", "arguments": '{"key":"k","value":"v"}'}},
                {}, "q", random.Random(7))
    assert len(calls) == 3


def test_personas_are_loaded_once_and_selected_deterministically(tmp_path, monkeypatch):
    from retrieval_sdg import streaming
    from retrieval_sdg.core import persona_loader
    path = tmp_path / "personas.jsonl"
    path.write_text('{"persona":"A curious student."}\n{"persona":"A teacher."}\n', encoding="utf-8")
    cfg = _config()
    cfg["persona"] = {"enabled": True, "local_path": str(path), "seed": 7}
    _load_personas.cache_clear()
    calls, captured, source_contexts = [], [], []
    load = persona_loader.load_persona_pool

    def tracked_load(config):
        calls.append(config)
        return load(config)

    class FakeRunner:
        def __init__(self, config, **kwargs):
            source_contexts.append(kwargs["source_passages"])

        def generate(self, data):
            captured.append(data["persona"])
            return {"conversation_messages": '[{"role":"assistant","content":"answer",'
                                               '"reasoning_content":"Mock model rationale."}]',
                    "conversation_status": True}

    monkeypatch.setattr(persona_loader, "load_persona_pool", tracked_load)
    monkeypatch.setattr(streaming, "call_llm", lambda *args: {"content": '{"query":"question"}'})
    monkeypatch.setattr(streaming, "DirectConversationRunner", FakeRunner)
    monkeypatch.setattr(streaming, "LocalRetrievalClient",
                        lambda *args, **kwargs: pytest.fail("Search client was created"))
    engine = _engine_config(cfg)
    raw = b'{"id":"document", "text":"Source text with sufficient content."}'
    first = synthesize(raw, "sample", {"file": "book.jsonl", "line": 1}, cfg, engine)
    second = synthesize(raw, "sample", {"file": "book.jsonl", "line": 1}, cfg, engine)
    assert first == second and set(first) == {"messages", "tools", "id"} and first["id"] == "sample"
    assert captured[0] == captured[1] and captured[0] in _personas(cfg)
    assert source_contexts[0] == source_contexts[1]
    assert source_contexts[0][0]["text"] == "Source text with sufficient content."
    assert len(calls) == 1
    assert synthesize(b'{"id":0,"text":"Source text."}', "zero-sample", {}, cfg, engine)["id"] == "zero-sample"
    assert synthesize(b'{"text":"Source text."}', "missing-id", {}, cfg, engine)["id"] == "missing-id"


def test_missing_persona_file_is_reported_without_exiting_the_worker(tmp_path):
    with pytest.raises(ValueError, match="local_path not found"):
        _personas({"persona": {"enabled": True, "local_path": str(tmp_path / "missing.jsonl")}})


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


def test_provided_evidence_parser_and_judge_history_handle_quoted_markers():
    from retrieval_sdg.conversation.prompts import SOURCE_ASSISTANT_SYSTEM_PROMPT
    sources = [{"id": "habcdef012345", "text": "Source data can quote </SOURCE_PASSAGES> without ending the JSON."}]
    messages = [{"role": "system", "content": SOURCE_ASSISTANT_SYSTEM_PROMPT.format(
        tools="calculator", passages=json.dumps(sources))}, {"role": "user", "content": "Explain this."}]
    assert source_passages_from_messages(messages) == sources
    history = format_judge_history(messages)
    assert sources[0]["text"] in history and "Explain this." in history
    assert not source_passages_from_messages([{"role": "tool", "content": messages[0]["content"]}])


def test_source_grounded_answers_can_finish_without_forced_tool_calls(monkeypatch):
    from retrieval_sdg import streaming
    from retrieval_sdg.conversation import engine as engine_module
    cfg = _config()
    cfg["persona"]["enabled"] = False
    cfg["engine"].update(min_turns=1, max_turns=1)
    monkeypatch.setattr(streaming, "call_llm", lambda *args: {"content": '{"query":"What value is provided?"}'})
    monkeypatch.setattr(engine_module, "configure_direct_clients", lambda *args: None)

    def answer(models, alias, messages, tools, **kwargs):
        assert kwargs["tool_choice"] == "auto"
        assert "search" not in {tool["function"]["name"] for tool in tools}
        source = source_passages_from_messages(messages)[0]
        return {"content": f"{source['text']} [{source['id']}]",
                "reasoning_content": "The source explicitly provides the requested value."}

    monkeypatch.setattr(engine_module, "call_llm_with_majority_vote", answer)
    row = synthesize(b'{"text":"The source value is 54."}', "sample", {}, cfg, _engine_config(cfg))
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    assert "The source value is 54." in row["messages"][-1]["content"]
    assert len(row["tools"]) == 6 and set(row) == {"messages", "tools", "id"}
    assert row["messages"][-1]["reasoning_content"]


@pytest.mark.parametrize("missing", ["tool", "answer"])
def test_required_reasoning_rejects_missing_tool_or_final_rationale(monkeypatch, missing):
    from retrieval_sdg import streaming
    from retrieval_sdg.conversation import engine as engine_module
    cfg = _config()
    cfg["persona"]["enabled"] = False
    cfg["engine"].update(min_turns=1, max_turns=1)
    monkeypatch.setattr(streaming, "call_llm", lambda *args: {"content": '{"query":"Double the value?"}'})
    monkeypatch.setattr(engine_module, "configure_direct_clients", lambda *args: None)
    responses = [
        {"content": "", "reasoning_content": "" if missing == "tool" else "Use the calculator.", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "calculator", "arguments": '{"expression":"54*2"}'}}]},
        {"content": "108.", "reasoning_content": "" if missing == "answer" else "Use the returned calculation."},
    ]
    pending = iter(responses)
    monkeypatch.setattr(engine_module, "call_llm_with_majority_vote", lambda *args, **kwargs: next(pending))
    raw = b'{"text":"The source value is 54."}'
    with pytest.raises(ValueError, match="no reasoning_content"):
        synthesize(raw, "sample", {}, cfg, _engine_config(cfg))
    cfg["engine"]["require_reasoning"] = False
    pending = iter(responses)
    row = synthesize(raw, "sample", {}, cfg, _engine_config(cfg))
    assert row["id"] == "sample" and row["messages"][-1]["content"] == "108."


def test_evaluator_accepts_provided_sources_and_rejects_fabricated_citations():
    pytest.importorskip("data_designer")
    import evaluate
    from retrieval_sdg.conversation.prompts import SOURCE_ASSISTANT_SYSTEM_PROMPT
    tools = _config()["tools"]
    source_id = "habcdef012345"
    source_text = "Source evidence provides the numerical value 54."
    messages = [
        {"role": "system", "content": SOURCE_ASSISTANT_SYSTEM_PROMPT.format(
            tools="calculator", passages=json.dumps([{"id": source_id, "text": source_text}]))},
        {"role": "user", "content": "What is double that value?"},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "calculator", "arguments": '{"expression":"54*2"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result":108}'},
        {"role": "assistant", "content": f"{source_text} Double is 108 [{source_id}]."},
    ]
    row = {"messages": messages, "tools": tools}
    result = evaluate._objective(row, ToolCallVerifier())
    assert result["objective_ok"] and result["retrievals"] == 0 and result["source_passages"] == 1
    assert evaluate._evidence_overlap(row) > 0
    messages[-1]["content"] = "Invented source [h000000000000]."
    assert not evaluate._objective(row, ToolCallVerifier())["objective_ok"]
    messages[-1]["content"] = f"Double is 108 [{source_id}]."
    messages[0]["content"] = "You are a research assistant."
    assert not evaluate._objective(row, ToolCallVerifier())["objective_ok"]
    search_tool = {"type": "function", "function": {"name": "search", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    messages[2]["tool_calls"][0]["function"] = {"name": "search", "arguments": '{"query":"value"}'}
    messages[3]["content"] = json.dumps({"results": [{"id": source_id, "text": source_text}]})
    assert evaluate._objective({"messages": messages, "tools": [search_tool]}, ToolCallVerifier())["objective_ok"]


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
                    message = {"role": "assistant", "content": json.dumps({
                        "query": "Giá trị RMR 54 gấp đôi là bao nhiêu?"})}
                elif messages[-1]["role"] == "tool":
                    assert json.loads(messages[-1]["content"])["result"] == 108
                    chunk = source_passages_from_messages(messages)[0]
                    message = {"role": "assistant", "content": f"RMR gấp đôi là 108 [{chunk['id']}].",
                               "reasoning_content": "The calculator returned double the provided source value."}
                else:
                    message = {"role": "assistant", "content": "", "reasoning": "Calculate using the provided value.",
                               "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {
                            "name": "calculator", "arguments": '{"expression":"54*2"}'}}]}
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
            path.write_text("".join(json.dumps({"id": "repeated-source-id", "text": "Đá granit có RMR bằng 54."},
                                              ensure_ascii=False) + "\n" for i in range(2)), encoding="utf-8")
        settings = OmegaConf.load(HERE / "config/knowledge.yaml")
        settings.pipeline_config = str(HERE / "config/pipeline.yaml")
        settings.batch.num_process = 1  # CLI flags must override these settings
        settings.batch.concurrency_per_process = 1
        settings.persona.enabled = False
        settings.engine.min_turns = settings.engine.max_turns = 1
        settings.engine.min_hops = 0
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
        assert "Counting input" in result.stderr and "Samples:" in result.stderr and "Files:" in result.stderr
        assert "saved=" in result.stderr and "resumed=" in result.stderr and "ETA=" in result.stderr
        assert 1 < maximum <= 4
        assert len(requests) == 12  # question, calculator call, final answer per source row
        paths = sorted(output.rglob("*.raw.jsonl"))
        assert len(paths) == 2
        output_ids = set()
        for path in paths:
            rows = _rows(path)
            assert len(rows) == 2
            for row in rows:
                assert set(row) == {"messages", "tools", "id"}
                assert re.fullmatch(r"[0-9a-f]{64}", row["id"])
                assert row["id"] not in output_ids
                output_ids.add(row["id"])
                assert len(row["tools"]) == 6
                messages = row["messages"]
                assert all(message.get("reasoning_content") for message in messages if message["role"] == "assistant")
                assert all("reasoning" not in message for message in messages)
                assert messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls")
                calls = [call for message in messages for call in message.get("tool_calls", [])]
                assert calls and all(ToolCallVerifier().verify_single(call, row["tools"])[0] for call in calls)
                assert all(call["function"]["name"] != "search" for call in calls)
                chunks = source_passages_from_messages(messages)
                assert chunks
                cited = set(re.findall(r"h[0-9a-f]{12}", messages[-1]["content"]))
                assert len(cited) == 1 and cited <= {chunk["id"] for chunk in chunks}
                assert all(chunk["text"] == "Đá granit có RMR bằng 54." for chunk in chunks)
        assert all(len(request["tools"]) == 6 and request["tool_choice"] == "auto"
                   and all(tool["function"]["name"] != "search" for tool in request["tools"])
                   for request in requests if request.get("tools"))
        assert all(request["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 25}
                   and request["include_reasoning"] is True for request in requests if request.get("tools"))
        assert sorted(output.rglob("*.jsonl")) == paths  # resume state stays outside the training output tree
        states = list(output.with_name(output.name + ".state").rglob("*.state.jsonl"))
        assert len(states) == 2
        assert {state["sample_id"] for path in states for state in _rows(path)} == output_ids
        before = {path: path.read_bytes() for path in paths}
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(requests) == 12  # no model calls for completed samples
        assert before == {path: path.read_bytes() for path in paths}
        with (inp / "math/data.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"id":"bad-row-without-text"}\n')
        result = subprocess.run(command + ["--no-progress"], cwd=tmp_path, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=45)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Counting input" not in result.stderr and "Samples:" not in result.stderr
        assert len(requests) == 12
        assert sorted(output.rglob("*.jsonl")) == paths
        assert before == {path: path.read_bytes() for path in paths}
        error_file = output.with_name(output.name + ".errors") / "math/data.errors.jsonl"
        assert _rows(error_file)[0]["source"] == {"file": "math/data.jsonl", "line": 3}
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
