from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from codegen_agent.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    """Return the configured ChatOpenAI LLM instance."""
    api_key    = os.environ.get("OPENAI_API_KEY")
    model      = os.environ.get("OPENAI_MODEL", "gpt-4o")
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

# ── Checkpointer ──────────────────────────────────────────────────────────────

def get_checkpointer():
    """
    Return the appropriate checkpointer based on CHECKPOINTER env var.

    Options:
        memory   — In-memory only. No persistence across restarts. (default)
        sqlite   — Persists to a local SQLite file. Survives restarts.
        postgres — Persists to Postgres. For production / multi-instance.
    """
    mode = os.environ.get("CHECKPOINTER", "memory").lower()

    if mode == "sqlite":
        return _get_sqlite_checkpointer()
    elif mode == "postgres":
        return _get_postgres_checkpointer()
    else:
        return _get_memory_checkpointer()


def _get_memory_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver
    logger.info(
        "Checkpointer: MemorySaver (in-memory, no persistence)",
        extra={"extra": {"type": "memory"}}
    )
    return MemorySaver()


def _get_sqlite_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    db_path = os.environ.get("SQLITE_DB_PATH", "checkpoints.db")
    logger.info(
        "Checkpointer: SqliteSaver",
        extra={"extra": {"type": "sqlite", "path": db_path}}
    )
    return SqliteSaver.from_conn_string(db_path)


def _get_postgres_checkpointer():
    """
    Returns a synchronous Postgres checkpointer.
    For async usage in production, swap to AsyncPostgresSaver.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    postgres_url = os.environ.get("POSTGRES_URL")
    if not postgres_url:
        logger.error(
            "POSTGRES_URL not set — falling back to MemorySaver",
            extra={"extra": {"module": "config"}}
        )
        return _get_memory_checkpointer()

    logger.info(
        "Checkpointer: PostgresSaver",
        extra={"extra": {"type": "postgres"}}
    )
    return PostgresSaver.from_conn_string(postgres_url)

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