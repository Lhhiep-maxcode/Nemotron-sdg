"""Bounded file processes and async sample workers with durable per-row output."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import random
import signal
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from itertools import chain
from pathlib import Path
from queue import Empty

from tqdm import tqdm

from .checkpoint import CheckpointedJsonl
from .conversation.config import ConversationSimulatorConfig
from .conversation.engine import DirectConversationRunner
from .core.llm import call_llm, configure_direct_clients
from .progress import RunProgress, count_inputs
from .query_gen.generate import generate_query
from .query_gen.sampler import GenUnit
from .retrieval.local import LocalRetrievalClient, knowledge_tool_handlers, split_document

_PROGRESS_QUEUE = None


def output_path(input_file: Path, input_root: Path, output_dir: Path) -> Path:
    """a/b.jsonl -> <output_dir>/a/b.raw.jsonl (no flattened-name collisions)."""
    return output_dir / input_file.relative_to(input_root).with_suffix(".raw.jsonl")


def _append(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _engine_config(cfg: dict) -> ConversationSimulatorConfig:
    providers = {p["name"]: p for p in cfg["providers"]}
    models = {
        m["alias"]: {"model": m["model"], "base_url": providers[m["provider"]]["endpoint"],
                     "api_key_env": providers[m["provider"]].get("api_key", "NVIDIA_API_KEY"),
                     "params": dict(m.get("inference_parameters", {}))}
        for m in cfg["models"]
    }
    for alias in ("assistant_model", "user_model", "aux_response_model", "judge_model"):
        if not models.get(alias, {}).get("base_url"):
            raise ValueError(f"Configure a provider endpoint for {alias}")
    engine = cfg.get("engine", {})
    knobs = {k: v for k, v in engine.items() if k in ConversationSimulatorConfig.model_fields}
    retrieval = cfg.get("retrieval", {})
    names = {(tool.get("tool", tool)).get("function", {}).get("name") for tool in cfg["tools"]}
    retrieval_tools = [name for name in retrieval.get("tools", []) if name in names]
    knobs.update(name=engine.get("column_name", "conversation_messages"), model_clients=models,
                 retrieval_endpoint="", retrieval_tools=retrieval_tools, top_k=int(retrieval.get("top_k", 4)),
                 oversample_factor=int(retrieval.get("oversample_factor", 2)))
    config = ConversationSimulatorConfig(**knobs)
    if not 1 <= config.min_turns <= config.max_turns:
        raise ValueError("Require 1 <= engine.min_turns <= engine.max_turns")
    return config


@lru_cache(maxsize=1)
def _load_personas(config_json: str) -> list[dict]:
    from .core.persona_loader import load_persona_pool
    try:
        pool = load_persona_pool(json.loads(config_json))
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc  # report a file error instead of killing a pool worker
    if not pool:
        raise ValueError("The configured persona dataset is empty")
    return pool


def _personas(cfg: dict) -> list[dict]:
    config = cfg.get("persona", {}) or {}
    return _load_personas(json.dumps(config, sort_keys=True)) if config.get("enabled", False) else []


def synthesize(raw: bytes, sample_id: str, source: dict, cfg: dict,
               engine_config: ConversationSimulatorConfig) -> dict:
    """One complete unit: document -> question -> conversation -> output record."""
    record = json.loads(raw)
    batch = cfg["batch"]
    if not isinstance(record, dict):
        raise ValueError("Input row must be a JSON object")
    text = record.get(batch.get("text_field", "text"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input row has no nonempty text field")
    original_id = record.get(batch.get("id_field", "id"))
    if original_id is None:
        original_id = sample_id
    doc_id = str(original_id)
    chunk_options = {"chunk_chars": int(batch.get("chunk_chars", 1800)),
                     "overlap_chars": int(batch.get("overlap_chars", 200))}
    client = None
    if engine_config.retrieval_tools:
        client = LocalRetrievalClient(text, sample_id, doc_id, **chunk_options,
                                      oversample_factor=engine_config.oversample_factor)
    chunks = client.chunks if client else split_document(text, sample_id, doc_id, **chunk_options)
    rng = random.Random(f"{batch.get('seed', 7)}:{sample_id}")
    weights = batch.get("kind_weights") or {"multi_hop": 4, "comparative": 3, "factual": 2}
    kind = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
    count = min(len(chunks), int(batch.get("query_chunks", 3)))
    max_chars = int(batch.get("max_chars_per_chunk", 1600))
    if count < 1 or max_chars < 1:
        raise ValueError("query_chunks and max_chars_per_chunk must be positive")
    chosen = [chunks[i] for i in sorted(rng.sample(range(len(chunks)), count))]

    def caller(system, user):
        if client is None:
            system = system.replace("which can search a knowledge base", "which is given source material")
            system += ("\nWhen supported by the source, favor natural requests involving calculations, translation, "
                       "chronology or citation formatting. Do not invent facts or force an irrelevant tool.")
        return call_llm({}, "user_model", [{"role": "system", "content": system},
                                           {"role": "user", "content": user}]).get("content", "")

    seed = generate_query(GenUnit(kind=kind, chunks=chosen), caller,
                          max_chars_per_chunk=max_chars)
    if seed is None:
        raise ValueError("Question generation returned no valid query")
    seed.update(cluster_id=f"doc_{sample_id[:12]}", tools=cfg["tools"])
    personas = _personas(cfg)
    if personas:
        persona_rng = random.Random(f"{cfg['persona'].get('seed', 7)}:{sample_id}")
        seed[engine_config.persona_column] = dict(personas[persona_rng.randrange(len(personas))])
    runner = DirectConversationRunner(engine_config, client=client,
                                      tool_handlers=knowledge_tool_handlers(), source_passages=None if client else [
                                          {"id": chunk.id, "doc_id": chunk.doc_id,
                                           "text": " ".join(chunk.text.split())[:max_chars]} for chunk in chosen])
    generated = runner.generate(seed)
    messages = json.loads(generated["conversation_messages"])
    if not messages or not generated.get("conversation_status"):
        raise ValueError(f"No completed conversation: {generated.get('trajectory_judgment', '')}")
    return {"messages": messages, "tools": cfg["tools"], "id": sample_id}


async def process_file(cfg: dict, input_file: Path, relative: str, out: Path,
                       *, synthesizer=None, error_path: Path | None = None,
                       checkpoint_path: Path | None = None, on_progress=None) -> dict:
    batch = cfg["batch"]
    concurrency = int(batch["concurrency_per_process"])
    if concurrency < 1:
        raise ValueError("concurrency_per_process must be positive")
    writer = CheckpointedJsonl(out, checkpoint_path)
    done = writer.done
    stats = {"saved": 0, "resumed": 0, "failed": 0}
    engine_config = _engine_config(cfg) if synthesizer is None else None
    if synthesizer is None:
        _personas(cfg)  # load once per file worker; cached across its samples and files
        configure_direct_clients(engine_config.model_clients)  # initialize once, before threads start
        synthesizer = synthesize
    queue = asyncio.Queue(maxsize=2 * concurrency)

    def report(error=None):
        if on_progress is not None:
            on_progress(dict(stats), error)
        elif error:
            print(error, flush=True)

    async def produce():
        seen = 0
        with input_file.open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                if not raw.strip():
                    continue
                if batch.get("limit") is not None and seen >= int(batch["limit"]):
                    break
                seen += 1
                sample_id = hashlib.sha256(f"{relative}:{line_number}:".encode() + raw.rstrip(b"\r\n")).hexdigest()
                if sample_id in done:
                    stats["resumed"] += 1
                    if stats["resumed"] % 1000 == 0:
                        report()
                else:
                    await queue.put((raw, sample_id, {"file": relative, "line": line_number}))
                if seen % 1000 == 0:
                    await asyncio.sleep(0)
        report()
        for _ in range(concurrency):
            await queue.put(None)

    # Writes occur on the event loop, so concurrent samples never interleave lines.
    # Keep the executor explicit: asyncio's default pool can otherwise cap large
    # --concurrency-per-process values at 32 threads.
    with ThreadPoolExecutor(max_workers=concurrency) as executor, writer:
        loop = asyncio.get_running_loop()

        async def consume():
            while True:
                item = await queue.get()
                if item is None:
                    return
                raw, sample_id, source = item
                try:
                    row = await loop.run_in_executor(executor, synthesizer, raw, sample_id, source, cfg, engine_config)
                except Exception as exc:
                    error_file = error_path or out.with_suffix(".errors.jsonl")
                    error_file.parent.mkdir(parents=True, exist_ok=True)
                    with error_file.open("a", encoding="utf-8") as errors:
                        _append(errors, {"sample_id": sample_id, "source": source,
                                         "error": f"{type(exc).__name__}: {exc}"})
                    stats["failed"] += 1
                    report(f"[{relative}:{source['line']}] failed: {exc}")
                else:
                    writer.append(sample_id, row)
                    stats["saved"] += 1
                    report()
                    if on_progress is None:
                        print(f"[{relative}] saved={stats['saved']} resumed={stats['resumed']} "
                              f"failed={stats['failed']}", flush=True)

        tasks = [asyncio.create_task(produce())] + [asyncio.create_task(consume()) for _ in range(concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return stats


def _init_worker(progress_queue):
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent stops the pool on Ctrl+C


def _report_progress(relative: str, stats: dict, error: str | None):
    _PROGRESS_QUEUE.put((relative, stats, error))


def _run_file(cfg: dict, path: Path, relative: str, out: Path) -> dict:
    output_dir = Path(cfg["batch"]["output_dir"])
    error_root = output_dir.with_name(output_dir.name + ".errors")
    error_path = error_root / Path(relative).with_suffix(".errors.jsonl")
    checkpoint_root = output_dir.with_name(output_dir.name + ".state")
    checkpoint_path = checkpoint_root / Path(relative).with_suffix(".state.jsonl")
    on_progress = partial(_report_progress, relative) if _PROGRESS_QUEUE is not None else None
    return asyncio.run(process_file(cfg, path, relative, out, error_path=error_path,
                                    checkpoint_path=checkpoint_path, on_progress=on_progress))


def run_directory(cfg: dict) -> int:
    """At most num_process active files; at most concurrency_per_process samples/file."""
    batch = cfg["batch"]
    processes, concurrency = int(batch["num_process"]), int(batch["concurrency_per_process"])
    if processes < 1 or concurrency < 1:
        raise ValueError("--num-process and --concurrency-per-process must be positive")
    if batch.get("limit") is not None and int(batch["limit"]) < 1:
        raise ValueError("--limit must be positive")
    inp, output_dir = Path(batch["input_path"]).resolve(), Path(batch["output_dir"]).resolve()
    if not inp.exists() or (inp.is_file() and inp.suffix != ".jsonl"):
        raise ValueError(f"Input must be an existing JSONL file or folder: {inp}")
    root = inp.parent if inp.is_file() else inp
    if output_dir == root or root in output_dir.parents:
        raise ValueError("output_dir must be outside the input tree (prevents ingesting generated JSONL)")
    files = iter([inp]) if inp.is_file() else (path for path in inp.rglob("*.jsonl") if path.is_file())
    first = next(files, None)
    if first is None:
        raise ValueError(f"No JSONL files found under {inp}")
    _engine_config(cfg)  # validate before launching children
    show_progress = bool(batch.get("progress", True))
    file_total = sample_total = None
    if show_progress:
        file_total, sample_total = count_inputs(chain((first,), files), batch.get("limit"))
        files = iter([inp]) if inp.is_file() else (path for path in inp.rglob("*.jsonl") if path.is_file())
        first = next(files, None)
        if first is None:
            raise ValueError("Input files disappeared after counting")
    print(f"[streaming] files={processes} samples/file={concurrency} "
          f"maximum concurrent samples={processes * concurrency}", flush=True)
    file_count = 0
    file_errors = 0
    context = multiprocessing.get_context("spawn")
    progress_queue = context.Queue() if show_progress else None
    try:
        with RunProgress(file_total, sample_total, enabled=show_progress) as progress, \
                context.Pool(processes=processes, initializer=_init_worker, initargs=(progress_queue,)) as pool:
            file_count, file_errors = _process_files(cfg, root, output_dir, first, files, pool,
                                                   progress, progress_queue, processes)
    finally:
        if progress_queue is not None:
            progress_queue.close()
            progress_queue.join_thread()
    print(f"[streaming] files={file_count} {progress.stats} file_errors={file_errors}", flush=True)
    return progress.stats["failed"] + file_errors


def _process_files(cfg, root, output_dir, first, files, pool, progress, progress_queue, processes):
    """Schedule bounded file jobs and drain lightweight UI events from workers."""
    file_count = file_errors = 0
    pending = []

    def submit(path):
        relative = path.relative_to(root).as_posix()
        progress.register(relative)
        pending.append((relative, pool.apply_async(_run_file, (cfg, path, relative,
                                                              output_path(path, root, output_dir)))))

    submit(first)
    exhausted = False
    while pending:
        if progress_queue is not None:
            while True:
                try:
                    progress.report(*progress_queue.get_nowait())
                except Empty:
                    break
        progress.refresh()
        while len(pending) < processes and not exhausted:
            path = next(files, None)
            if path is None:
                exhausted = True
            else:
                submit(path)
        ready = [(name, job) for name, job in pending if job.ready()]
        if not ready:
            pending[0][1].wait(0.2)
            continue
        for name, job in ready:
            pending.remove((name, job))
            file_count += 1
            try:
                result = job.get()
            except Exception as exc:
                file_errors += 1
                progress.finish(name, None)
                tqdm.write(f"[{name}] file failed: {exc}")
            else:
                progress.finish(name, result)
                if not progress.enabled:
                    tqdm.write(f"[{name}] done: {result}")
        if not pending and not exhausted:
            path = next(files, None)
            if path is not None:
                submit(path)
    return file_count, file_errors
