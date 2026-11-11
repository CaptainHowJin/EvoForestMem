# EvoForestMem

Evidence-guided adaptive memory for streaming multi-party conversations.

EvoForestMem stores original messages and organizes their references into three view types: **Context View**, **Attribution View**, and **Referent View**. Each view can contain multiple trees. Hierarchical summaries guide retrieval, while original messages supply the answer evidence. Local structural updates and audited associative recall adapt memory as new messages arrive. Question feedback is applied after each batch of questions sharing an ask time.

## Repository structure

```text
EvoForestMem/
├── evoforestmem/
│   ├── cli.py               # Dataset validation, preparation, execution and scoring
│   ├── config/              # Runtime settings, method configuration and prompts
│   ├── log/                 # Original messages, registries, tags and validity
│   ├── routers/             # Routing into the three view types
│   ├── forest/              # Trees, placement and hierarchical summaries
│   ├── index/               # Sparse and dense indexes and embedding cache
│   ├── retrieve/            # Navigation and evidence assembly
│   ├── structure_update/    # Structural adaptation, feedback and recovery
│   ├── associations/        # Associative recall and evidence auditing
│   ├── stream/              # Incremental memory session
│   ├── llm/                 # Model transport and request budgeting
│   └── prompts/             # Message tagging and summary prompts
├── datasets/
│   ├── Ever-Stream/
│   ├── Group-Stream/
│   ├── manifest.json
│   └── README.md
├── .env.example
├── pyproject.toml
└── LICENSE
```

Internal view identifiers are `channel`, `actor` and `artifact`, respectively. Model weights, generated indexes and run outputs are obtained or created separately.

## Installation

Use Python 3.11 or newer. From the repository directory:

```bash
pip install -e '.[dense]'
python -m evoforestmem validate-data --data-root datasets
```

The `dense` extra supplies the local embedding and tokenizer dependencies required for execution. The installed `evoforestmem` command also provides the same CLI.

## Datasets

| Dataset | Partitions | Questions | Messages | Question IDs |
|---|---|---:|---:|---|
| Ever-Stream | 004, 005, 010, 011, 016 | 2,866 | 51,023 | `EVER-0001`–`EVER-2866` |
| Group-Stream | Finance, Technology, Manufacturing, Healthcare | 614 | 120,000 | `GROUP-0001`–`GROUP-0614` |

Release 1.0 contains 3,480 questions. Each partition includes `messages.jsonl` and `qas.jsonl`, retaining the complete conversation history. The manifest defines partition paths, question IDs and file checksums; it is required by all commands. Results identify questions by `(subset, qa_id)`.

Subset names are `ever_004`, `ever_005`, `ever_010`, `ever_011`, `ever_016`, `group_Finance`, `group_Technology`, `group_Manufacturing` and `group_Healthcare`. Use a separate output directory for each subset. See [dataset documentation](datasets/README.md) for fields and upstream sources.

## Configuration

```bash
cp evoforestmem/config/runtime.json runtime.json
```

Edit the copied configuration before running:

| Field | Purpose |
|---|---|
| `model`, `base_url` | Qwen reader and summary/tagging service; URL includes `/v1` |
| `judge_model`, `judge_base_url` | Judge model and its OpenAI-compatible service |
| `embedding_model` | SentenceTransformers-compatible local embedding model path or identifier |
| `tokenizer` | Local directory containing the matching Qwen tokenizer |
| `device` | Embedding device, such as `cpu` or `cuda:0` |
| `concurrency` | Tagging and summary request concurrency |

The supplied settings use local service URLs. Obtain the model weights and start the services separately. The reader/summary client requires a loopback URL, Qwen model identity, `enable_thinking=false` support, actual token usage in responses, and `max_model_len` in `/v1/models`. Tagging and judging use structured JSON responses. The tokenizer directory needs `tokenizer.json`, `tokenizer_config.json` and a chat template, either in the tokenizer configuration or `chat_template.jinja`.

If authentication is required, export `EVOFORESTMEM_LLM_API_KEY` and `EVOFORESTMEM_JUDGE_API_KEY` in your shell. `.env.example` lists these names; copying it alone does not configure every client. Keep credentials out of version control.

## Prepare, run and score

For one partition:

```bash
python -m evoforestmem prepare --subset ever_004 --data-root datasets --output-dir runs/ever_004 --config runtime.json
python -m evoforestmem run --subset ever_004 --data-root datasets --output-dir runs/ever_004 --config runtime.json
python -m evoforestmem judge --subset ever_004 --data-root datasets --output-dir runs/ever_004 --config runtime.json
```

Preparation enriches message tags using the current message and up to three preceding messages from the same speaker, then computes document embeddings. It makes model calls and writes reusable prepared inputs. Run ingests messages chronologically up to each question's ask time, retrieves evidence and generates answers. Questions sharing an ask time observe the same memory prefix. Reference answers and annotated evidence are reserved for evaluation.

Multiple-choice answers use exact option matching. Open-ended answers are scored by a majority of three judge votes.

Useful options:

- `--dry-run`: validate the released dataset and print the selected runtime configuration without model calls. It does not check service availability or tokenizer files.
- `run --limit-qa N`: run the first N questions while ingesting their full causal message prefix. Preparation still processes the complete partition.
- `run --resume`: resume an interrupted run with identical code, data, configuration and QA limit. Code changes require a new run directory.

Preparation refuses an already completed output directory. Each independent run needs its own prepared output directory. Re-running `judge` reuses completed request records.

## Outputs

All generated files live under `--output-dir`:

| Output | Contents |
|---|---|
| `runtime/`, `prepared.json` | Normalized inputs, tags, embeddings and preparation hashes |
| `run.json` | Code, data and configuration identity |
| `results.json` | Answers, evidence provenance and scoring fields |
| `summary.json`, `summary.per_question.json` | Retrieval summary and per-question details |
| `summary.progress.json`, `point_objects/`, `structure/` | Progress and recovery state |
| `requests/`, `judging_requests/`, `summary_cache/` | Persistent model request records and caches |
| `execution/` | Launch metadata, timestamps, exit status and stdout/stderr |

Judging records `scoring_complete` and `correct` in `results.json`. A limited or interrupted run covers only its processed questions; use its actual row count when reporting a score.

## License

Code is distributed under the [MIT license](LICENSE). Dataset sources and applicable upstream terms are documented [separately](datasets/README.md#sources).
