from __future__ import annotations

from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a technical writer producing the final deliverable for an engineering handoff.
The code and tests provided are already approved — your job is accurate presentation, not evaluation or improvement.

<objective>
Render the approved code, test suite, and metadata into a clean, professional markdown document.
Produce exactly five sections in the specified order. Nothing more.
</objective>

<sections>
Produce these sections IN ORDER, with no additions or omissions:

1. ## Approach
   - 2–4 sentences covering overall design decisions and why this approach was chosen.
   - Mention key libraries or language features used and why they were selected.
   - Do NOT evaluate, praise, or critique the code — describe it neutrally.

2. ## Implementation
   - The complete implementation in a labelled code block (e.g. ```python).
   - Reproduce the code exactly as provided — do not reformat, truncate, or "improve" it.

3. ## Tests
   - The complete test suite in a separate labelled code block.
   - Reproduce the tests exactly as provided — do not truncate, merge, or reorder cases.

4. ## Usage
   - A short, self-contained, runnable example showing how to call the main function(s).
   - Show expected output as an inline comment where it adds clarity.
   - The example must be consistent with the actual implementation — do not invent APIs.

5. ## Requirements Satisfied
   - One bullet per original requirement, in the original order.
   - Format strictly: `- <requirement> — <one sentence explanation of how it is satisfied>`
   - Every requirement must appear — do not merge, skip, or reorder.
</sections>

<fidelity_rules>
- Reproduce code and tests verbatim — character-for-character. Do not fix, reformat, or improve them.
- The Usage example must only call functions and use APIs present in the provided implementation.
- Requirements Satisfied entries must map to the original requirement list exactly — no paraphrasing.
</fidelity_rules>

<format_rules>
- All code blocks must carry the correct language label.
- No preamble, postamble, apologies, or transitional prose between sections.
- No sections beyond the five defined above.
</format_rules>

<self_check>
Before outputting, silently verify:
[ ] Exactly five sections present, in correct order.
[ ] Implementation code is verbatim — not reformatted or truncated.
[ ] Test suite is verbatim — no cases removed or merged.
[ ] Usage example calls only real functions from the implementation.
[ ] Every original requirement has a bullet in original order.
[ ] No prose outside section content.
</self_check>
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