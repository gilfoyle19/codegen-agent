from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from codegen_agent.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    """Return the configured ChatOpenAI LLM instance."""
    api_key     = os.environ.get("OPENAI_API_KEY")
    model       = os.environ.get("OPENAI_MODEL", "gpt-4o")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))

    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set",
            extra={"extra": {"module": "config"}}
        )

    logger.info(
        "LLM initialized",
        extra={"extra": {"model": model, "temperature": temperature}}
    )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        streaming=True,
    )


# ── Compile-time checkpointer (always sync MemorySaver) ──────────────────────

def get_compile_checkpointer() -> MemorySaver:
    """
    Returns a MemorySaver used ONLY for graph.compile().
    LangGraph requires a sync-compatible saver at compile time.
    The real persistence saver is injected at runtime via get_runtime_checkpointer().
    """
    return MemorySaver()


# ── Runtime checkpointer (async, persistent) ─────────────────────────────────

@asynccontextmanager
async def get_runtime_checkpointer():
    """
    Async context manager yielding the runtime checkpointer.
    This is passed to graph.astream() / graph.ainvoke() — NOT to graph.compile().

    Controlled via CHECKPOINTER env var:
        memory   — MemorySaver (default, no persistence)
        sqlite   — AsyncSqliteSaver (persists to local .db file)
        postgres — AsyncPostgresSaver (persists to Postgres)

    Usage in main.py:
        async with get_runtime_checkpointer() as saver:
            async for event in graph.astream(..., checkpointer=saver):
                ...
    """
    mode = os.environ.get("CHECKPOINTER", "memory").lower()

    if mode == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        db_path = os.environ.get("SQLITE_DB_PATH", "checkpoints.db")
        logger.info(
            "Runtime checkpointer: AsyncSqliteSaver",
            extra={"extra": {"type": "sqlite", "path": db_path}}
        )
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            yield saver

    elif mode == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        postgres_url = os.environ.get("POSTGRES_URL")
        if not postgres_url:
            logger.error(
                "POSTGRES_URL not set — falling back to MemorySaver",
                extra={"extra": {"module": "config"}}
            )
            yield MemorySaver()
            return
        logger.info(
            "Runtime checkpointer: AsyncPostgresSaver",
            extra={"extra": {"type": "postgres"}}
        )
        async with AsyncPostgresSaver.from_conn_string(postgres_url) as saver:
            yield saver

    else:
        logger.info(
            "Runtime checkpointer: MemorySaver",
            extra={"extra": {"type": "memory"}}
        )
        yield MemorySaver()


# ── LangSmith ─────────────────────────────────────────────────────────────────

def check_langsmith_config() -> None:
    """Warn at startup if LangSmith tracing is not configured."""
    if not os.environ.get("LANGCHAIN_API_KEY"):
        logger.warning(
            "LangSmith tracing disabled — set LANGCHAIN_API_KEY to enable",
            extra={"extra": {"module": "config"}}
        )
    else:
        logger.info(
            "LangSmith tracing enabled",
            extra={"extra": {
                "project": os.environ.get("LANGCHAIN_PROJECT", "default")
            }}
        )