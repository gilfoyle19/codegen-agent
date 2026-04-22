from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState
import re

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior software architect. Your sole job is to decompose a coding task into a language declaration and a precise, testable requirement list.

<objective>
Analyse the given coding task and extract:
1. The target programming language.
2. A numbered list of atomic, testable requirements the code must satisfy.
</objective>

<requirement_rules>
- Each requirement must be a single, independently testable statement.
- State WHAT the code must do — never HOW it should do it.
- Cover all implicit requirements (error handling, edge cases, types) — not just the happy path.
- No duplicates, no overlaps between requirements.
</requirement_rules>

<output_format>
Respond in this EXACT format with no deviations:

LANGUAGE: <language>
REQUIREMENTS:
1. <requirement>
2. <requirement>
...
</output_format>

<self_check>
Before outputting, silently verify:
[ ] Language is a single lowercase word (e.g. "python", "typescript").
[ ] Every requirement is atomic — one behaviour per line.
[ ] No requirement describes implementation details or algorithms.
[ ] No prose, preamble, or closing remarks in the output.
[ ] Edge cases and error conditions are covered.
</self_check>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_planner_response(text: str) -> tuple[str, list[str]]:
    """
    Parse the LLM response into (language, requirements).
    Falls back to 'python' if no LANGUAGE line is found.
    """
    language = "python"
    requirements: list[str] = []
    in_requirements = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper().startswith("LANGUAGE:"):
            language = stripped.split(":", 1)[1].strip().lower()
            continue

        if stripped.upper().startswith("REQUIREMENTS:"):
            in_requirements = True
            continue

        if in_requirements and stripped and stripped[0].isdigit():
            # Strip leading number + dot/period (e.g. "1. ", "2) ")
            requirement = re.sub(r"^\d+[\.\)]\s*", "", stripped)
            if requirement:
                requirements.append(requirement)

    return language, requirements


# ── LangGraph node ────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="PlannerAgent")
async def planner_node(state: CodeGenState, llm) -> dict:
    logger.info(
        "Planner started",
        extra={"extra": {"node": "planner", "task_len": len(state["sanitized_task"])}}
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {state['sanitized_task']}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        logger.info(
            "LLM response received",
            extra={"extra": {"node": "planner", "model": getattr(llm, 'model_name', 'unknown')}}
        )
    except Exception as e:
        logger.error(
            "Planner failed",
            extra={"extra": {"node": "planner", "error": str(e)}}
        )
        raise

    language, requirements = _parse_planner_response(response.content.strip())

    logger.info(
        "Planner completed",
        extra={"extra": {
            "node": "planner",
            "language": language,
            "num_requirements": len(requirements),
        }}
    )

    return {
        "language": language,
        "requirements": requirements,
        "messages": [response],
    }