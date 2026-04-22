# CodeGen Agent

A multi-agent code generation pipeline built with LangGraph. Given a natural language task, the system plans, generates, tests, critiques, and formats production-ready code — with a human-in-the-loop review gate before final output.

---

## Architecture

```
User Task
    │
    ▼
┌─────────────┐
│  Sanitizer  │  Validates and normalises the input task
└──────┬──────┘
       │ valid
       ▼
┌─────────────┐
│   Planner   │  Detects language, extracts structured requirements
└──────┬──────┘
       │
       ▼
┌───────────────┐
│ Doc Retriever │  Fetches relevant docs from ChromaDB (optional)
└──────┬────────┘
       │
       ▼
┌─────────────┐ ◄────────────────────────────────────────┐
│    Coder    │  Generates implementation + test suite   │
└──────┬──────┘                                          │
       │                                                 │ NEEDS_REVISION
       ▼                                                 │ (iteration < max)
┌─────────────┐                                          │
│   Tester    │  Executes tests in a sandbox             │
└──────┬──────┘                                          │
       │                                                 │
       ▼                                                 │
┌─────────────┐ ────────────────────────────────────────►┘
│   Critic    │  Reviews code quality and test results
└──────┬──────┘
       │ APPROVED or max_iterations reached
       ▼
  ⏸  HUMAN REVIEW  (approve / edit / reject)
       │
       ▼
┌─────────────┐
│  Formatter  │  Produces structured markdown final answer
└──────┬──────┘
       │
      END
```

### Agents

| Agent | Responsibility |
|---|---|
| **Sanitizer** | Rule-based input validation — rejects prompt injections, empty tasks, malicious patterns |
| **Planner** | Detects target language, extracts numbered requirements list |
| **Doc Retriever** | Semantic search over ingested documentation via ChromaDB + OpenAI embeddings |
| **Coder** | Generates implementation and full `unittest` test suite |
| **Tester** | Executes tests in an isolated `tempfile` sandbox, captures output |
| **Critic** | Reviews code quality, test coverage, and requirement compliance — returns `APPROVED` or `NEEDS_REVISION` with feedback |
| **Formatter** | Produces a five-section markdown answer: Approach, Implementation, Tests, Usage, Requirements Satisfied |

---

## Project Structure

```
codegen_agent/
├── codegen_agent/
│   ├── __init__.py
│   ├── state.py              # LangGraph TypedDict state schema
│   ├── config.py             # LLM + checkpointer configuration
│   ├── graph.py              # StateGraph construction + conditional edges
│   ├── main.py               # CLI entry point, streaming, HITL review
│   ├── logger.py             # Structured JSON logger (structlog)
│   ├── agents/
│   │   ├── sanitizer.py
│   │   ├── planner.py
│   │   ├── coder.py
│   │   ├── tester.py
│   │   ├── critic.py
│   │   └── formatter.py
│   └── tools/
│       └── doc_retriever.py  # ChromaDB retriever tool
├── scripts/
│   └── ingest_docs.py        # One-time PDF ingestion into ChromaDB
├── tests/
│   ├── conftest.py
│   ├── test_sanitizer.py
│   ├── test_planner.py
│   ├── test_coder.py
│   ├── test_tester.py
│   ├── test_critic.py
│   └── test_graph.py
├── docs/                     # Place PDFs here for ingestion
├── pyproject.toml
└── .env
```

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone <repo-url>
cd codegen_agent
uv pip install -e .
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```env
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.2

# Checkpointer — memory | sqlite | postgres
CHECKPOINTER=memory
SQLITE_DB_PATH=checkpoints.db
POSTGRES_URL=postgresql://user:password@localhost:5432/codegen

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=codegen-agent
```

---

## Usage

### Generate code

```bash
uv run python -m codegen_agent.main "Write a Python binary search function"
```

### With a custom task

```bash
uv run python -m codegen_agent.main "Create a REST API client for the GitHub API using httpx"
```

### Resume a previous session (sqlite / postgres only)

```bash
uv run python -m codegen_agent.main --resume <thread_id>
```

### Example output

```
🚀 Starting code generation...

📋 Language detected: python
   • Accept a sorted list and a target value
   • Return the index of the target or -1 if not found
   • Use iterative binary search (no recursion)

📚 Doc retriever: no matching docs found

💻 Code generated (iteration 1): 487 chars
🧪 Tests: ✅ PASS
🔍 Critic (iteration 1): ✅ APPROVED

============================================================
⏸  HUMAN REVIEW REQUIRED
Critic verdict : ✅ APPROVED
Iterations used: 1 / 3

--- Generated Code ---
def binary_search(arr: list[int], target: int) -> int:
    ...

[A]pprove  [E]dit  [R]eject  > a

============================================================
## Approach
...
## Implementation
...
## Tests
...
## Usage
...
## Requirements Satisfied
...
============================================================
```

---

## Doc Retrieval (Optional)

Place PDF documentation files in the `docs/` directory, then run the ingestion script once:

```bash
uv run python scripts/ingest_docs.py
```

This creates a `chroma_db/` directory. The doc retriever will automatically use it on the next run. To re-ingest after adding new PDFs, run the script again.

> If `chroma_db/` does not exist, the pipeline continues without doc retrieval — no crash.

---

## Persistence

Controlled via the `CHECKPOINTER` environment variable:

| Value | Behaviour | Use case |
|---|---|---|
| `memory` | In-process only, wiped on exit | Development / testing |
| `sqlite` | Persists to `checkpoints.db` | Local dev with crash recovery |
| `postgres` | Persists to Postgres | Production |

> **Note:** `AsyncSqliteSaver` requires LangGraph ≥ 0.2. For earlier versions, use `memory`.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model name |
| `OPENAI_TEMPERATURE` | `0.2` | LLM temperature |
| `CHECKPOINTER` | `memory` | Persistence backend |
| `SQLITE_DB_PATH` | `checkpoints.db` | SQLite file path |
| `POSTGRES_URL` | — | Postgres connection string |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | `default` | LangSmith project name |

---

## Running Tests

```bash
uv run pytest tests/ -v
```

```bash
# Run a specific test file
uv run pytest tests/test_sanitizer.py -v

# Run with coverage
uv run pytest tests/ --cov=codegen_agent --cov-report=term-missing
```

---

## Human-in-the-Loop (HITL)

The graph pauses before the `formatter` node regardless of critic verdict. This gives you the chance to:

- **[A]pprove** — accept the generated code and produce the final formatted answer
- **[E]dit** — paste your own corrected code; the coder re-runs with your edit as context
- **[R]eject** — discard the run entirely

The interrupt is implemented via LangGraph's `interrupt_before=["formatter"]` at compile time, with state persisted to the configured checkpointer between Phase 1 (generation) and Phase 3 (formatting).

---

## Logging

Structured JSON logs are written to stdout via `structlog`. Each log entry includes:

- `node` — which agent emitted the log
- `iteration` — current refinement loop count
- `event` — human-readable description

To view pretty-printed logs during development:

```bash
uv run python -m codegen_agent.main "..." | jq .
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Multi-agent graph orchestration |
| `langchain-openai` | OpenAI LLM integration |
| `langchain-community` | ChromaDB vector store integration |
| `chromadb` | Local vector database for doc retrieval |
| `openai` | Embeddings for doc ingestion |
| `structlog` | Structured logging |
| `langsmith` | Tracing and observability |
| `python-dotenv` | Environment variable loading |
| `langgraph-checkpoint-sqlite` | SQLite persistence (optional) |

---



