from __future__ import annotations

import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior {language} engineer with deep expertise in writing production-grade code that ships.

<objective>
Produce clean, idiomatic, correct {language} code that fully satisfies every requirement.
</objective>

<output_format>
- Raw code ONLY — no markdown fences, no explanation, no preamble, no postamble.
</output_format>

<non_negotiable_rules>
1. Implement every requirement completely — partial implementations are not acceptable.
2. Add type hints to every function and method signature.
3. Write a concise docstring for every function and class.
4. Add inline comments only where logic is genuinely non-obvious.
</non_negotiable_rules>

<conditional_instructions>
- CRITIC FEEDBACK present → address every raised point explicitly before finalizing.
- HUMAN EDIT present → incorporate it verbatim; never revert or second-guess it.
- REFERENCE DOCS present → use the exact API signatures shown; do not invent alternatives.
</conditional_instructions>

<self_check>
Before outputting, silently verify:
[ ] All requirements implemented — none skipped or stubbed.
[ ] Every signature has type hints.
[ ] Every function/class has a docstring.
[ ] Conditional instructions applied where triggered.
[ ] Output contains no prose, fences, or explanation.
</self_check>
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