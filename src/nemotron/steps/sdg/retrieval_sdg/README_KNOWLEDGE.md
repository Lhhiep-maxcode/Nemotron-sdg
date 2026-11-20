# Streaming knowledge-document SDG

This runner accepts your processed Vietnamese documents (`text`, `id`, `metadata`),
recursively discovers `*.jsonl`, and reuses the existing question prompts and
multi-turn conversation engine. Each nonblank input row independently generates
one question and one conversation. Each completed conversation is appended,
flushed and synced to disk immediately.

## Configure and run

Run from `src/nemotron/steps/sdg/retrieval_sdg/` with Python 3.10+.
The standalone runner needs:

```bash
python -m pip install openai 'pydantic>=2' omegaconf
```

1. Edit [`config/knowledge.yaml`](config/knowledge.yaml). The requested input and
   output folders are already set. Adjust `batch.num_process` and
   `batch.concurrency_per_process`, plus chunk sizes and question-kind weights.
2. Model names, providers and inference settings come from
   [`config/pipeline.yaml`](config/pipeline.yaml). You can override `models`,
   `providers` or `engine` directly in `knowledge.yaml`; lists replace the entire
   inherited list. Set your model endpoints and API keys there or through the
   existing environment variables, for example:

   ```bash
   export ASSISTANT_ENDPOINT=http://your-assistant-server:8000/v1
   export USER_MODEL_ENDPOINT=http://your-user-model-server:8000/v1
   # For authenticated services:
   export ASSISTANT_API_KEY=your-key
   export USER_MODEL_API_KEY=your-key
   ```

3. Execute the same file for both a new run and a resumed run:

   ```bash
   python run_knowledge_sdg.py
   ```

After setup, no other stage or command is required to produce raw trajectories.
For a smoke test, point at **one input file** and cap its input rows:

```bash
python run_knowledge_sdg.py --input-path /path/to/example.jsonl --limit 2 \
  --num-process 1 --concurrency-per-process 2
```

Command-line options override the YAML. For the full directory:

```bash
python run_knowledge_sdg.py --num-process 4 --concurrency-per-process 8
```

`--num-process` controls simultaneous file workers. Each worker processes one
file at a time and is reused for another file when finished. Inside each worker,
an async queue runs up to `--concurrency-per-process` independent samples using
the existing synchronous model/tool engine in a bounded thread pool. With 4 and
8, at most 32 samples run concurrently; fewer input files reduce this total.
`--limit N` selects the first N nonblank rows **per file**, including resumed rows.

## Tools and document scope

- `search`: real BM25 passage search within the current source document, with
  stable citation IDs. Supports Vietnamese text and retrieves actual source
  passages for questions about mathematics, literature, recipes, books and
  education. It needs no retrieval or embedding service.
- `calculator`: real, bounded arithmetic for ratios, percentages, recipe scaling
  and numerical exercises, using values from retrieved evidence.

Each document is a separate knowledge base. Multi-hop questions combine passages
from that document. Questions follow the source language; follow-ups and answers
follow the conversation language. Source text is treated as data, including any
instructions quoted within it. The configured tools run locally rather than being
LLM-simulated.

This mode performs no global query deduplication, embedding, clustering or
cross-document grouping. The `query_gen` and `query_prep` blocks and Data Designer
checkpoint settings apply to the original `pipeline.py` workflow.

## Output and resume

The relative input path becomes the output prefix, preserving directories:

```text
<input>/giao-duc/math/book.jsonl
  -> /workspace/storage-shared/nlp/datasets/POST-TRAINING/SFT/VAI-SFT-v3/agentic_vmlu_by_hieplh8/giao-duc/math/book.raw.jsonl
```

Output rows retain `messages`, `tools`, generation metadata, and add `sample_id`,
`query`, `kind` and `source: {file, line, id}`. Completion order can differ from
input order. IDs include the relative filename, physical line number and input
content, so repeated document IDs cannot collide.

To resume after Ctrl+C or a stopped job, rerun the same command. Existing raw rows
are the checkpoint: saved samples are skipped and unfinished/failed samples are
retried. A partial final JSONL row is removed before appending; complete rows are
retained. Failed samples are logged beside the output as `book.raw.errors.jsonl`
and are not marked complete. This log retains earlier failures even after a
successful retry. The command exits with code 1 if a sample or file failed, and
130 when interrupted by Ctrl+C.

Run one job per output directory. Keep input contents and row order stable while
resuming; use a new output directory when changing data or generation settings.
Memory holds only the active/queued input rows, active document indexes and the
current file's completed sample IDs; it never loads all generated conversations.
Only a fully saved conversation is a checkpoint; an interrupted conversation
starts again from question generation.

These are raw trajectories. Quality filtering remains a separate operation using
the existing evaluator and its Data Designer dependencies; pass an explicit file:

```bash
python evaluate.py --config config/pipeline.yaml \
  --input /path/to/book.raw.jsonl --out /path/to/book.sft.jsonl --judge
```

The focused offline tests include a local mock model server exercising the actual
CLI, multiprocessing, tool calls, citations and resume:

```bash
python -m pip install pytest
python -m pytest -q tests/test_streaming.py
```
