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
python -m pip install openai 'pydantic>=2' omegaconf tqdm
```

Your supplied config enables a local Parquet persona dataset, which also needs:

```bash
python -m pip install pandas pyarrow
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
   # For authenticated services, set providers[*].api_key to the corresponding
   # environment-variable NAME before exporting its value:
   export ASSISTANT_API_KEY=your-key
   export USER_MODEL_API_KEY=your-key
   ```

   In your supplied providers, `api_key: dummy` refers to an environment variable
   literally named `dummy`; it is not a literal API key. If unset, direct clients
   send the placeholder `EMPTY`, which is suitable for an unauthenticated server.
   `verify_ssl: false` is not consumed by the streaming direct client; your current
   endpoint URLs use HTTP, so TLS verification does not apply.

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

Your supplied config sets 128 processes and 8 samples per process: **up to 1,024
concurrent samples**, with all model roles sharing the same endpoint. These values
are preserved in the checked-in config. Choose concurrency to fit your model
server capacity and available memory. Enabled personas are loaded and cached once
per worker process, so their memory usage also scales with the process count.

## Progress and ETA

`tqdm` shows two overall bars in the parent process, regardless of the worker
count: `Samples` and `Files`. The sample bar includes saved, resumed and failed
counts, the rate of newly attempted samples, elapsed time and estimated time
remaining. Updates for saved samples happen only after durable output writes.
Resume skips advance the bar without inflating the generation rate. ETA appears
after the first new attempt finishes and remains approximate as conversation
lengths and server throughput vary.

By default the runner first counts selected nonblank input rows, respecting
`--limit` per file. This adds one streaming input scan for accurate totals without
loading the dataset into memory. Disable both the bars and this counting pass
with `batch.progress: false` in the config or:

```bash
python run_knowledge_sdg.py --no-progress
```

The sample bar counts saved rows, resume skips and failed attempts. A failed
attempt is not saved and will be retried on the next run. File errors can leave
the sample bar incomplete; see the final summary and error logs.

## Turn limits and model roles

The knowledge config explicitly overrides these values:

```yaml
engine:
  min_turns: 3
  max_turns: 20
  max_tools: 6
  min_hops: 0
  force_first_tool: false
  require_reasoning: true
  persona_voice: false
  gate_query: false
  inline_judge: false
```

The planner samples a count between 3 and 20, favoring the middle of the range.
A turn is a main user request plus the assistant's research and answer; tool calls
are internal steps. Conversations can finish early when no follow-up or answer is
produced, and clarification exchanges can add messages. `max_tools: 6` offers all
six remaining tools; the assistant calls them when useful. Search is not offered,
and tool calls are not forced to meet a minimum search count.
The inherited `max_tool_calls_per_turn: 3` limits each assistant tool-call batch;
one research turn can contain multiple batches.

| Model alias | Role |
|---|---|
| `assistant_model` | Answers from the provided source passages and selects auxiliary tools when helpful. |
| `user_model` | Generates the opening question, follow-ups, and clarification replies. |
| `aux_response_model` | Produces JSON responses for translation, citation formatting, and timeline extraction. |
| `judge_model` | Checks queries/trajectories when generation gates are enabled, or scores output through `evaluate.py --judge`. |

These are four roles, not a requirement to deploy four different models. Your
provided config assigns `DeepSeek-V4.1-Flash-1017` to all four. The judge makes no
generation calls while `gate_query` and `inline_judge` are both false.

`persona.enabled: true` now attaches one reproducibly sampled persona to each
conversation. With `persona_voice: false`, the opening query keeps its original
wording, while generated follow-ups and clarification replies use that persona.
The persona file must exist on the machine running the job. Set `persona.enabled`
to false to skip persona loading.

Your `chunk_chars: 32768`, `query_chunks: 3`, `max_chars_per_chunk: 32768`, and
assistant `max_tokens: 65536` allow large requests. The query prompt alone can
contain up to 98,304 source characters. The same bounded source passages are
provided directly to the assistant in its system context and saved in `messages`.
Ensure your served context window fits the prompt plus requested completion;
compression uses estimates and retains recent tool results, so its threshold is
not a strict model context limit. Model availability, context capacity, endpoint
access, and the cluster persona path require checking on your deployment.

## Assistant reasoning

The assistant's returned reasoning is saved inside each assistant message:

```json
{"role": "assistant", "reasoning_content": "Model-returned reasoning text.", "content": "Final answer."}
```

Assistant messages that call tools also keep `reasoning_content` beside
`tool_calls`. The outer record still has only `messages`, `tools` and `id`.
Both the current vLLM `reasoning` field and the older `reasoning_content` field
are normalized to `reasoning_content`; a complete leading `<think>...</think>`
block is also supported. Missing reasoning is never synthesized from the answer.

Thinking is now explicitly requested for `assistant_model` in
`config/pipeline.yaml`:

```yaml
inference_parameters:
  temperature: 0.7
  max_tokens: 65536
  extra_body:
    include_reasoning: true
    chat_template_kwargs:
      thinking: true
      reasoning_effort: 25
```

The thinking controls follow the official
[vLLM DeepSeek V4.1 recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml).
Ensure the matching `deepseek_v41` reasoning parser is enabled on your existing
vLLM server. Other model aliases retain their existing request parameters.
For the field-name change and response visibility, see
[vLLM reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).

`engine.require_reasoning: true` rejects a newly generated sample if any saved
assistant message has no nonempty reasoning. Rejected samples are logged as
failures and retried when resuming. Set this to false to allow messages without
reasoning while keeping whatever reasoning the server returns. Reasoning and
answer tokens share the completion budget; allow enough tokens for both and run
a small smoke test against your endpoint first. Saving reasoning does not replay
old reasoning in subsequent model requests; the existing compact input view is
retained. The evaluator keeps reasoning unless you pass `--strip-reasoning`.

Restart the runner to load these settings. Existing completed samples are skipped
on resume, so use a new output directory to regenerate previously saved samples
with reasoning.

## Tools and document scope

- `calculator`: real, bounded arithmetic for ratios, percentages, recipe scaling
  and numerical exercises, using values from the provided source passages.
- `memory_read` / `memory_write`: real state shared across turns within one
  generated conversation. It is isolated from other conversations and is not
  persisted across independent samples or sessions.
- `translation`: LLM-backed translation of the supplied text, with optional
  preservation of domain terms.
- `citation_format`: LLM-backed formatting of supplied source identifiers;
  bibliographic details that were not provided must not be invented.
- `timeline_extract`: LLM-backed chronology from the supplied evidence snippets.

Each document supplies the passages used for its opening question. Those same
passages, with stable citation IDs, are supplied directly to the assistant. It
has no search tool and cannot look up additional passages; follow-ups stay within
the available evidence. Questions follow the source language; follow-ups and
answers follow the conversation language. Source text is treated as data,
including any instructions quoted within it. Arithmetic and memory execute
locally; the three text-transformation tools use `aux_response_model`.

No search index, HTTP retrieval call or embedding service is used in this mode.
The inherited `retrieval.top_k` and `oversample_factor` do not apply without a
retrieval tool. The original `pipeline.py` retrieval workflow remains available;
the knowledge config overrides its tools and routing.

This mode performs no global query deduplication, embedding, clustering or
cross-document grouping. The `query_gen` and `query_prep` blocks and Data Designer
checkpoint settings apply to the original `pipeline.py` workflow.

## Output and resume

The relative input path becomes the output prefix, preserving directories:

```text
<input>/math/book.jsonl
  -> <output>/math/book.raw.jsonl
```

Every raw row contains **exactly three top-level fields**:

```json
{"messages": [], "tools": [], "id": "deterministic-sample-id"}
```

`messages` contains the full conversation and tool results, and `tools` contains
the six tool schemas. `id` is the original deterministic `sample_id`, renamed
to `id`: a SHA-256 hash of the relative input filename, physical line number and
input content. It is stable across resumed runs and distinct for repeated input
document IDs. `batch.id_field` still identifies the source document inside tool
evidence; it does not set the output row's ID. Completion order can differ from
input order.

To resume after Ctrl+C or a stopped job, rerun the same command. Existing raw rows
are paired with a separate checkpoint journal: saved samples are skipped and
unfinished/failed samples are retried. Internal IDs include the relative input
filename, physical line number and content, so repeated original document IDs
remain safe to resume. A journal entry is synced before its output is written;
recovery retains completed outputs and removes only an unfinished trailing pair.

Press **Ctrl+C** to stop the current command, wait for it to exit, then run
`python run_knowledge_sdg.py` again with the same config and paths. No `--resume`
flag is needed. Progress is reconstructed from the checkpoints; in-flight
conversations start again from question generation.

Checkpoint and error files live in **sibling directories outside the dataset**:

```text
processed-book/math/book.raw.jsonl          # training records: messages/tools/id only
processed-book.state/math/book.state.jsonl  # internal resume IDs and checksums
processed-book.errors/math/book.errors.jsonl # created only for failed samples
```

Keep the matching `.state` directory with the output when resuming. Legacy raw
rows containing `sample_id` are automatically converted to the three-field
schema using that value as `id`. Outputs from the previous document-ID version
are converted using their matching checkpoint's sample IDs. Messages and tools
are preserved. Conversion uses temporary files and can be retried after an
interruption without regenerating completed conversations.

Failed samples are not marked complete. Error logs retain earlier failures even
after a successful retry. The command exits with code 1 if a sample or file
failed, and 130 when interrupted by Ctrl+C.

Run one job per output directory. Keep input contents and row order stable while
resuming; use a new output directory when changing data or generation settings.
In particular, samples saved before removing search retain their original tool
schemas and calls. Use a new output directory to regenerate a wholly search-free
dataset; resuming skips those existing samples.
Memory holds only the active/queued input rows, active document indexes and the
current file's completed sample IDs and the enabled persona pool; it never loads
all generated conversations.
Only a fully saved conversation is a checkpoint; an interrupted conversation
starts again from question generation.

These are raw trajectories. Quality filtering remains a separate operation using
the existing evaluator and its Data Designer dependencies; pass an explicit file.
The evaluator accepts the provided source passages as grounding evidence and
checks citations against their IDs. It still requires at least one valid tool
call for an agentic SFT example, so raw conversations that needed no tools are
filtered out:

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
