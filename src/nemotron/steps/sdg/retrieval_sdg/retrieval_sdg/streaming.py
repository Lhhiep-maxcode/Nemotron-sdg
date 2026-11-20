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
from pathlib import Path

from .conversation.config import ConversationSimulatorConfig
from .conversation.engine import DirectConversationRunner
from .core.llm import call_llm, configure_direct_clients
from .query_gen.generate import generate_query
from .query_gen.sampler import GenUnit
from .retrieval.local import LocalRetrievalClient, calculate


def output_path(input_file: Path, input_root: Path, output_dir: Path) -> Path:
    """a/b.jsonl -> <output_dir>/a/b.raw.jsonl (no flattened-name collisions)."""
    return output_dir / input_file.relative_to(input_root).with_suffix(".raw.jsonl")


def _append(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def completed_samples(path: Path) -> set[str]:
    """Use raw.jsonl as the checkpoint; repair only an interrupted final record.

    A complete JSON record missing its newline is retained. Corruption before the
    final record is an error, so existing successful data cannot be silently lost.
    """
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r+b") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                if handle.read(1):
                    raise ValueError(f"Corrupt output before EOF: {path}, byte {start}")
                handle.seek(start)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                print(f"[resume] removed interrupted final record from {path}", flush=True)
                break
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
                raise ValueError(f"Output has no sample_id: {path}, byte {start}")
            done.add(row["sample_id"])
            if not line.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return done


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
    knobs.update(name=engine.get("column_name", "conversation_messages"), model_clients=models,
                 retrieval_endpoint="", retrieval_tools=["search"], top_k=int(retrieval.get("top_k", 4)),
                 oversample_factor=int(retrieval.get("oversample_factor", 2)))
    return ConversationSimulatorConfig(**knobs)


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
    doc_id = str(record.get(batch.get("id_field", "id")) or sample_id)
    client = LocalRetrievalClient(text, sample_id, doc_id,
                                  chunk_chars=int(batch.get("chunk_chars", 1800)),
                                  overlap_chars=int(batch.get("overlap_chars", 200)),
                                  oversample_factor=engine_config.oversample_factor)
    rng = random.Random(f"{batch.get('seed', 7)}:{sample_id}")
    weights = batch.get("kind_weights") or {"multi_hop": 4, "comparative": 3, "factual": 2}
    kind = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
    count = min(len(client.chunks), int(batch.get("query_chunks", 3)))
    chosen = [client.chunks[i] for i in sorted(rng.sample(range(len(client.chunks)), count))]

    def caller(system, user):
        return call_llm({}, "user_model", [{"role": "system", "content": system},
                                           {"role": "user", "content": user}]).get("content", "")

    seed = generate_query(GenUnit(kind=kind, chunks=chosen), caller,
                          max_chars_per_chunk=int(batch.get("max_chars_per_chunk", 1600)))
    if seed is None:
        raise ValueError("Question generation returned no valid query")
    seed.update(cluster_id=f"doc_{sample_id[:12]}", tools=cfg["tools"])
    runner = DirectConversationRunner(engine_config, client=client,
                                      tool_handlers={"calculator": calculate})
    generated = runner.generate(seed)
    messages = json.loads(generated["conversation_messages"])
    if not messages or not generated.get("conversation_status"):
        raise ValueError(f"No completed conversation: {generated.get('trajectory_judgment', '')}")
    row = {"sample_id": sample_id, "source": {**source, "id": doc_id},
           "query": seed["query"], "kind": kind, "messages": messages, "tools": cfg["tools"]}
    metadata = cfg.get("engine", {}).get("metadata_fields", ["cluster_id", "hops_taken", "retrieval_log"])
    for key in metadata:
        if key in generated:
            row[key] = generated[key]
    return row


async def process_file(cfg: dict, input_file: Path, relative: str, out: Path,
                       *, synthesizer=None) -> dict:
    batch = cfg["batch"]
    concurrency = int(batch["concurrency_per_process"])
    if concurrency < 1:
        raise ValueError("concurrency_per_process must be positive")
    done = completed_samples(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"saved": 0, "resumed": 0, "failed": 0}
    engine_config = _engine_config(cfg) if synthesizer is None else None
    if synthesizer is None:
        configure_direct_clients(engine_config.model_clients)  # initialize once, before threads start
        synthesizer = synthesize
    queue = asyncio.Queue(maxsize=2 * concurrency)

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
                else:
                    await queue.put((raw, sample_id, {"file": relative, "line": line_number}))
                if seen % 1000 == 0:
                    await asyncio.sleep(0)
        for _ in range(concurrency):
            await queue.put(None)

    # Writes occur on the event loop, so concurrent samples never interleave lines.
    # Keep the executor explicit: asyncio's default pool can otherwise cap large
    # --concurrency-per-process values at 32 threads.
    with ThreadPoolExecutor(max_workers=concurrency) as executor, out.open("a", encoding="utf-8") as output:
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
                    with out.with_suffix(".errors.jsonl").open("a", encoding="utf-8") as errors:
                        _append(errors, {"sample_id": sample_id, "source": source,
                                         "error": f"{type(exc).__name__}: {exc}"})
                    stats["failed"] += 1
                    print(f"[{relative}:{source['line']}] failed: {exc}", flush=True)
                else:
                    _append(output, row)  # durable output IS the completion marker
                    done.add(sample_id)
                    stats["saved"] += 1
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


def _ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent stops the pool on Ctrl+C


def _run_file(cfg: dict, path: Path, relative: str, out: Path) -> dict:
    return asyncio.run(process_file(cfg, path, relative, out))


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
    print(f"[streaming] files={processes} samples/file={concurrency} "
          f"maximum concurrent samples={processes * concurrency}", flush=True)
    total = {"saved": 0, "resumed": 0, "failed": 0}
    file_count = 0
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=processes, initializer=_ignore_interrupt) as pool:
        pending = []

        def submit(path):
            relative = path.relative_to(root).as_posix()
            pending.append((relative, pool.apply_async(_run_file, (cfg, path, relative,
                                                                  output_path(path, root, output_dir)))))

        submit(first)
        exhausted = False
        while pending:
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
                    total["failed"] += 1
                    print(f"[{name}] file failed: {exc}", flush=True)
                else:
                    for key in total:
                        total[key] += result[key]
                    print(f"[{name}] done: {result}", flush=True)
            if not pending and not exhausted:
                path = next(files, None)
                if path is not None:
                    submit(path)
    print(f"[streaming] files={file_count} {total}", flush=True)
    return total["failed"]
