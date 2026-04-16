from __future__ import annotations

import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert {language} developer. Write clean, idiomatic, production-quality code.

Rules:
- Output ONLY the raw code — no markdown fences, no explanation, no preamble.
- Satisfy every requirement exactly — do not skip or partially implement any.
- Include type hints for all function signatures.
- Include a concise docstring for every function and class.
- Add inline comments only where the logic is non-obvious.
- If critic feedback is provided, address every point explicitly.
- If a human edit is provided, incorporate it directly and do not revert it.
- If relevant docs are provided, use the correct API calls shown in them.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences if the LLM includes them despite instructions."""
    code = re.sub(r"^```[\w]*\n", "", code.strip())
    code = re.sub(r"\n```$", "", code.strip())
    return code.strip()


def _build_user_message(state: CodeGenState, iteration: int) -> str:
    """
    Compose the full user message from all available state fields.
    Sections are only included when they contain meaningful content.
    """
    parts: list[str] = [
        f"Task: {state['sanitized_task']}",
        "",
        "Requirements:",
        *[f"  {r}" for r in state.get("requirements", [])],
    ]

    # Previous code on refinement iterations
    if state.get("generated_code") and iteration > 1:
        parts += [
            "",
            f"--- PREVIOUS CODE (iteration {iteration - 1}) ---",
            state["generated_code"],
        ]

    # Execution errors from tester sandbox
    if state.get("execution_output") and not state.get("execution_success"):
        parts += [
            "",
            "--- EXECUTION ERRORS (fix these) ---",
            state["execution_output"],
        ]

    # Critic feedback on failed review
    if state.get("review_feedback") and not state.get("review_passed"):
        parts += [
            "",
            "--- CRITIC FEEDBACK (address every point) ---",
            state["review_feedback"],
        ]

    # Human edit takes highest priority
    if state.get("human_edit"):
        parts += [
            "",
            "--- HUMAN EDIT (incorporate exactly as provided) ---",
            state["human_edit"],
        ]

    # Retrieved documentation
    if state.get("docs_context"):
        parts += [
            "",
            "--- RELEVANT DOCS (use these API calls) ---",
            state["docs_context"],
        ]

    return "\n".join(parts)

# ── LangGraph node ────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="CoderAgent")
async def coder_node(state: CodeGenState, llm) -> dict:
    iteration = state.get("iteration", 0) + 1

    logger.info(
        "Coder started",
        extra={"extra": {
            "node":      "coder",
            "iteration": iteration,
            "language":  state.get("language", "python"),
            "has_feedback":   bool(state.get("review_feedback")),
            "has_human_edit": bool(state.get("human_edit")),
            "has_docs":       bool(state.get("docs_context")),
        }}
    )

    system = SYSTEM_PROMPT.format(language=state.get("language", "python"))
    user_message = _build_user_message(state, iteration)

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_message),
    ]

    try:
        response = await llm.ainvoke(messages)
        logger.info(
            "LLM response received",
            extra={"extra": {
                "node":  "coder",
                "model": getattr(llm, "model_name", "unknown"),
            }}
        )
    except Exception as e:
        logger.error(
            "Coder failed",
            extra={"extra": {"node": "coder", "iteration": iteration, "error": str(e)}}
        )
        raise

    generated_code = _strip_code_fences(response.content)

    logger.info(
        "Coder completed",
        extra={"extra": {
            "node":      "coder",
            "iteration": iteration,
            "code_len":  len(generated_code),
        }}
    )

    return {
        "generated_code": generated_code,
        "messages":       [response],
        "iteration":      iteration,
    }