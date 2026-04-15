from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR      = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K           = 3

# Maps programming language → collection name
# Used as fallback when no library keyword is detected
LANGUAGE_COLLECTIONS: dict[str, str] = {
    "python":     "python_stdlib",
    "typescript": "typescript",
    "javascript": "javascript",
    "rust":       "rust",
    "go":         "go",
    "java":       "java",
}

# Maps library keyword → collection name
# Scanned against task + requirements text
LIBRARY_KEYWORDS: dict[str, str] = {
    "cadquery":     "cadquery",
    "cq.":          "cadquery",
    "import cq":    "cadquery",
    "trimesh":      "trimesh",
    "numpy":        "numpy",
    "np.":          "numpy",
    "pandas":       "pandas",
    "pd.":          "pandas",
    "scipy":        "scipy",
    "shapely":      "shapely",
    # Add more as you ingest more PDFs:
    # "mylib":      "mylib",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _get_vectorstore(collection_name: str) -> Chroma:
    """
    Load and cache a Chroma collection by name.
    lru_cache ensures each collection is loaded only once per process.
    """
    logger.debug(
        "Loading Chroma collection",
        extra={"extra": {"collection": collection_name}}
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=collection_name,
    )


def _detect_collections(state: CodeGenState) -> list[str]:
    """
    Resolve which Chroma collections to query by:
    1. Scanning task + requirements for library keywords (primary)
    2. Falling back to the detected programming language collection

    Returns a deduplicated ordered list of collection names.
    """
    search_text = " ".join([
        state.get("sanitized_task", ""),
        *state.get("requirements", []),
    ]).lower()

    collections: list[str] = []

    # 1 — Library keyword scan (most specific)
    for keyword, collection in LIBRARY_KEYWORDS.items():
        if keyword in search_text and collection not in collections:
            collections.append(collection)
            logger.debug(
                "Library keyword matched",
                extra={"extra": {"keyword": keyword, "collection": collection}}
            )

    # 2 — Language-level fallback
    lang = state.get("language", "").lower()
    lang_collection = LANGUAGE_COLLECTIONS.get(lang)
    if lang_collection and lang_collection not in collections:
        collections.append(lang_collection)

    return collections


def _chroma_available() -> bool:
    """Check if the Chroma DB directory exists on disk."""
    return os.path.isdir(CHROMA_DIR)


async def _query_collection(collection_name: str, query: str) -> list[str]:
    """
    Query a single Chroma collection and return formatted snippet strings.
    Returns empty list on any error (non-fatal).
    """
    try:
        db = _get_vectorstore(collection_name)
        docs = await db.asimilarity_search(query, k=TOP_K)
        return [
            f"[{collection_name} | {d.metadata.get('source', 'unknown')} | p.{d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        ]
    except Exception as e:
        logger.error(
            "Collection query failed",
            extra={"extra": {"collection": collection_name, "error": str(e)}}
        )
        return []

# ── Async retrieval tool ──────────────────────────────────────────────────────

@tool
async def retrieve_docs(query: str, language: str = "python") -> str:
    """
    Retrieve relevant documentation snippets for a given query and language.
    Returns an empty string if no collection is found or Chroma is unavailable.
    """
    collection = LANGUAGE_COLLECTIONS.get(language.lower())

    if not collection:
        logger.debug(
            "No collection mapped for language",
            extra={"extra": {"language": language}}
        )
        return ""

    if not _chroma_available():
        logger.warning(
            "Chroma DB not found — run scripts/ingest_docs.py first",
            extra={"extra": {"chroma_dir": CHROMA_DIR}}
        )
        return ""

    snippets = await _query_collection(collection, query)
    return "\n\n".join(snippets).strip()

# ── LangGraph node ────────────────────────────────────────────────────────────

@traceable(run_type="tool", name="DocRetriever")
async def doc_retriever_node(
    state: CodeGenState,
    retriever_tool=None,
) -> dict:
    logger.info(
        "DocRetriever started",
        extra={"extra": {
            "node":     "doc_retriever",
            "language": state.get("language", "unknown"),
            "retriever_active": retriever_tool is not None,
        }}
    )

    if retriever_tool is None:
        logger.info(
            "DocRetriever skipped — no retriever configured",
            extra={"extra": {"node": "doc_retriever"}}
        )
        return {"docs_context": ""}

    if not _chroma_available():
        logger.warning(
            "DocRetriever skipped — Chroma DB not found",
            extra={"extra": {"node": "doc_retriever", "chroma_dir": CHROMA_DIR}}
        )
        return {"docs_context": ""}

    # Resolve all relevant collections from task + language
    collections = _detect_collections(state)

    if not collections:
        logger.info(
            "DocRetriever — no matching collections detected",
            extra={"extra": {"node": "doc_retriever"}}
        )
        return {"docs_context": ""}

    logger.info(
        "DocRetriever querying collections",
        extra={"extra": {
            "node":        "doc_retriever",
            "collections": collections,
        }}
    )

    # Query all matched collections and merge results
    all_snippets: list[str] = []
    for collection_name in collections:
        snippets = await _query_collection(collection_name, state["sanitized_task"])
        all_snippets.extend(snippets)

    docs_context = "\n\n".join(all_snippets).strip()

    logger.info(
        "DocRetriever completed",
        extra={"extra": {
            "node":              "doc_retriever",
            "collections_hit":   collections,
            "total_snippets":    len(all_snippets),
            "docs_len":          len(docs_context),
        }}
    )
    return {"docs_context": docs_context}