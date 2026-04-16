from __future__ import annotations

from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from codegen_agent.state import CodeGenState
from codegen_agent.agents.sanitizer import sanitizer_node
from codegen_agent.agents.planner import planner_node
from codegen_agent.agents.coder import coder_node
from codegen_agent.agents.tester import tester_node
from codegen_agent.agents.critic import critic_node
from codegen_agent.agents.formatter import formatter_node
from codegen_agent.tools.doc_retriever import doc_retriever_node, retrieve_docs
from codegen_agent.logger import get_logger

logger = get_logger(__name__)

# ── Conditional edge functions ────────────────────────────────────────────────

def after_sanitizer(state: CodeGenState) -> str:
    """
    Route to END immediately if sanitization failed.
    Otherwise proceed to planner.
    """
    if state.get("sanitization_error"):
        logger.info(
            "Graph routing: sanitizer → END (input rejected)",
            extra={"extra": {"reason": state["sanitization_error"]}}
        )
        return END
    logger.info("Graph routing: sanitizer → planner")
    return "planner"


def after_critic(state: CodeGenState) -> str:
    """
    Route based on critic verdict and iteration count:
    - APPROVED                        → formatter (HITL interrupt)
    - NEEDS_REVISION + under cap      → coder (refinement loop)
    - NEEDS_REVISION + at cap         → formatter (best effort)
    """
    iteration     = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    review_passed  = state.get("review_passed", False)

    if review_passed:
        logger.info(
            "Graph routing: critic → formatter (approved)",
            extra={"extra": {"iteration": iteration}}
        )
        return "formatter"

    if iteration >= max_iterations:
        logger.info(
            "Graph routing: critic → formatter (max iterations reached)",
            extra={"extra": {"iteration": iteration, "max_iterations": max_iterations}}
        )
        return "formatter"

    logger.info(
        "Graph routing: critic → coder (needs revision)",
        extra={"extra": {"iteration": iteration, "max_iterations": max_iterations}}
    )
    return "coder"

# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    llm,
    checkpointer,
    use_doc_retriever: bool = True,
) -> CompiledStateGraph:
    """
    Build and compile the CodeGen LangGraph StateGraph.

    Args:
        llm:               Configured ChatOpenAI (or compatible) instance.
        checkpointer:      LangGraph checkpointer (MemorySaver, SqliteSaver, etc.)
        use_doc_retriever: Wire in the doc retriever tool (requires chroma_db).
                           Set False to skip doc retrieval entirely.

    Returns:
        Compiled StateGraph with HITL interrupt before formatter.
    """
    graph = StateGraph(CodeGenState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("sanitizer",     sanitizer_node)
    graph.add_node("planner",       partial(planner_node,    llm=llm))
    graph.add_node("coder",         partial(coder_node,      llm=llm))
    graph.add_node("tester",        partial(tester_node,     llm=llm))
    graph.add_node("critic",        partial(critic_node,     llm=llm))
    graph.add_node("formatter",     partial(formatter_node,  llm=llm))
    graph.add_node(
        "doc_retriever",
        partial(
            doc_retriever_node,
            retriever_tool=retrieve_docs if use_doc_retriever else None,
        )
    )

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("sanitizer")

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Sanitizer → planner (valid) or END (rejected)
    graph.add_conditional_edges(
        "sanitizer",
        after_sanitizer,
        {"planner": "planner", END: END},
    )

    # Linear pipeline: planner → doc_retriever → coder → tester → critic
    graph.add_edge("planner",       "doc_retriever")
    graph.add_edge("doc_retriever", "coder")
    graph.add_edge("coder",         "tester")
    graph.add_edge("tester",        "critic")

    # Critic → formatter (approved / max iter) or coder (needs revision)
    graph.add_conditional_edges(
        "critic",
        after_critic,
        {"formatter": "formatter", "coder": "coder"},
    )

    # Formatter → END (after HITL resume)
    graph.add_edge("formatter", END)

    # ── Compile with HITL interrupt ───────────────────────────────────────────
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["formatter"],   # mandatory human approval gate
    )

    logger.info(
        "Graph compiled",
        extra={"extra": {
            "interrupt_before":   ["formatter"],
            "use_doc_retriever":  use_doc_retriever,
        }}
    )

    return compiled