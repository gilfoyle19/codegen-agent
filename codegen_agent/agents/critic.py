from __future__ import annotations

import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a staff engineer conducting a rigorous code review. Your verdict directly gates whether code proceeds to production.

<objective>
Evaluate the provided code against the stated requirements and test execution results.
Produce a binary verdict and precise, actionable feedback.
</objective>

<evaluation_criteria>
Assess strictly in this order — fail fast on any violated criterion:

CORRECTNESS (blocking):
- Every requirement is fully implemented — no stubs, no partial logic.
- All provided tests pass — verify against the execution output, do not assume.
- No logic bugs, off-by-one errors, or incorrect edge case handling.

CODE QUALITY (blocking):
- All function and method signatures have type hints.
- All functions and classes have docstrings.
- No bare excepts, no silent failures, no swallowed errors.
- No hardcoded values where parameters or constants are expected.

IDIOM & STYLE (non-blocking unless egregious):
- Code follows established language best practices and conventions.
- Tests are meaningful and cover edge cases beyond the happy path.
</evaluation_criteria>

<verdict_rules>
APPROVED   → ALL blocking criteria pass AND all tests pass.
NEEDS_REVISION → ANY blocking criterion fails OR any test fails.

When APPROVED: feedback may still list non-blocking style suggestions.
When NEEDS_REVISION: every blocking failure must appear as a feedback point.
</verdict_rules>

<feedback_rules>
- Each point must name the exact location (function name, line, or block).
- Each point must state what is wrong and what the fix should be.
- Do not repeat points addressed in previous iterations — treat prior feedback as resolved unless evidence shows otherwise.
- Non-blocking points must be prefixed with [STYLE].
</feedback_rules>

<output_format>
Respond in this EXACT format — no prose, no preamble:

VERDICT: APPROVED | NEEDS_REVISION
FEEDBACK:
- <specific actionable point>
- <specific actionable point>
...

If no feedback points exist, output:
FEEDBACK: None
</output_format>

<self_check>
Before outputting, silently verify:
[ ] Verdict is based on execution output — not assumed test results.
[ ] Every blocking failure has a corresponding feedback point.
[ ] Every feedback point names a specific location and action.
[ ] No previously addressed points are repeated.
[ ] Output matches the exact format — no extra prose.
</self_check>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_review_context(state: CodeGenState) -> str:
    """Assemble the full review context from state fields."""
    tests_passed = state.get("execution_success", False)
    iteration    = state.get("iteration", 1)

    parts: list[str] = [
        f"Language: {state.get('language', 'python')}",
        f"Iteration: {iteration} / {state.get('max_iterations', 3)}",
        "",
        "Requirements:",
        *[f"  {r}" for r in state.get("requirements", [])],
        "",
        "--- Generated Code ---",
        state.get("generated_code", ""),
        "",
        "--- Test Code ---",
        state.get("test_code", ""),
        "",
        "--- Execution Output ---",
        state.get("execution_output", "(no output)"),
        f"Tests Passed: {tests_passed}",
    ]

    # Include previous feedback so critic doesn't repeat resolved points
    if state.get("review_feedback"):
        parts += [
            "",
            "--- Previous Feedback (already sent to coder) ---",
            state["review_feedback"],
        ]

    return "\n".join(parts)


def _parse_verdict(text: str) -> tuple[bool, list[str]]:
    """
    Parse VERDICT and FEEDBACK from the critic's response.
    Returns (review_passed, feedback_lines).
    """
    review_passed   = bool(re.search(r"VERDICT\s*:\s*APPROVED", text, re.IGNORECASE))
    feedback_lines: list[str] = []
    in_feedback     = False

    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"FEEDBACK\s*:", stripped, re.IGNORECASE):
            in_feedback = True
            continue
        if in_feedback and stripped:
            # Accept bullet points starting with -, *, •, or plain text
            clean = re.sub(r"^[-*•]\s*", "", stripped)
            if clean:
                feedback_lines.append(clean)

    return review_passed, feedback_lines

# ── LangGraph node ────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="CriticAgent")
async def critic_node(state: CodeGenState, llm) -> dict:
    iteration = state.get("iteration", 1)

    logger.info(
        "Critic started",
        extra={"extra": {
            "node":              "critic",
            "iteration":         iteration,
            "execution_success": state.get("execution_success", False),
        }}
    )

    context = _build_review_context(state)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    try:
        response = await llm.ainvoke(messages)
        logger.info(
            "LLM response received",
            extra={"extra": {
                "node":  "critic",
                "model": getattr(llm, "model_name", "unknown"),
            }}
        )
    except Exception as e:
        logger.error(
            "Critic failed",
            extra={"extra": {"node": "critic", "iteration": iteration, "error": str(e)}}
        )
        raise

    review_passed, feedback_lines = _parse_verdict(response.content.strip())
    review_feedback = "\n".join(f"- {line}" for line in feedback_lines)

    logger.info(
        "Critic completed",
        extra={"extra": {
            "node":           "critic",
            "iteration":      iteration,
            "review_passed":  review_passed,
            "feedback_points": len(feedback_lines),
        }}
    )

    return {
        "review_feedback": review_feedback,
        "review_passed":   review_passed,
        "messages":        [response],
    }