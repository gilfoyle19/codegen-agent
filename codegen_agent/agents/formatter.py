from __future__ import annotations

from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a technical writer producing a clean, professional final response.
Given approved code and its test suite, produce the following sections IN ORDER:

1. ## Approach
   2-4 sentences explaining the overall design decisions and why this approach was chosen.
   Mention key libraries or APIs used.

2. ## Implementation
   The full implementation in a properly labelled markdown code block.
   Do not truncate or summarise — include the complete code.

3. ## Tests
   The full test suite in a separate markdown code block.
   Do not truncate or summarise — include all test cases.

4. ## Usage
   A short, runnable usage example showing how to call the main function(s).
   Include expected output as a comment where applicable.

5. ## Requirements Satisfied
   A bullet list mapping each original requirement to how it was satisfied.
   Format: `- ✅ <requirement> — <one sentence explanation>`

Rules:
- Use clean markdown formatting throughout.
- Code blocks must be labelled with the correct language (e.g. ```python).
- Do not add any section beyond the five listed above.
- Do not include apologies, preamble, or closing remarks.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_formatter_input(state: CodeGenState) -> str:
    """Assemble everything the formatter needs to produce the final answer."""
    parts: list[str] = [
        f"Task: {state['sanitized_task']}",
        f"Language: {state.get('language', 'python')}",
        f"Iterations taken: {state.get('iteration', 1)}",
        "",
        "Original Requirements:",
        *[f"  {r}" for r in state.get("requirements", [])],
        "",
        "--- Implementation ---",
        state.get("generated_code", ""),
        "",
        "--- Test Suite ---",
        state.get("test_code", ""),
        "",
        "--- Test Execution Output ---",
        state.get("execution_output", "(not available)"),
        f"All Tests Passed: {state.get('execution_success', False)}",
    ]

    # Include critic's final feedback for context
    if state.get("review_feedback"):
        parts += [
            "",
            "--- Final Critic Review ---",
            state["review_feedback"],
        ]

    return "\n".join(parts)

# ── LangGraph node ────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="FormatterAgent")
async def formatter_node(state: CodeGenState, llm) -> dict:
    logger.info(
        "Formatter started",
        extra={"extra": {
            "node":      "formatter",
            "iteration": state.get("iteration", 1),
            "language":  state.get("language", "python"),
        }}
    )

    formatter_input = _build_formatter_input(state)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=formatter_input),
    ]

    try:
        response = await llm.ainvoke(messages)
        logger.info(
            "LLM response received",
            extra={"extra": {
                "node":  "formatter",
                "model": getattr(llm, "model_name", "unknown"),
            }}
        )
    except Exception as e:
        logger.error(
            "Formatter failed",
            extra={"extra": {"node": "formatter", "error": str(e)}}
        )
        raise

    final_answer = response.content.strip()

    logger.info(
        "Formatter completed",
        extra={"extra": {
            "node":             "formatter",
            "final_answer_len": len(final_answer),
        }}
    )

    return {
        "final_answer": final_answer,
        "messages":     [response],
    }