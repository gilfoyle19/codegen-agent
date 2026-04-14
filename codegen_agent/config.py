import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from codegen_agent.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    """Return the configured ChatOpenAI LLM instance."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set",
            extra={"extra": {"module": "config"}}
        )

    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))

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


def get_checkpointer() -> MemorySaver:
    """
    Return a checkpointer for LangGraph state persistence.
    Uses MemorySaver by default (in-process, no setup needed).
    Swap for SqliteSaver or AsyncPostgresSaver in production.
    """
    logger.info(
        "Checkpointer initialized",
        extra={"extra": {"type": "MemorySaver"}}
    )
    return MemorySaver()


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