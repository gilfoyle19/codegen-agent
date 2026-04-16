from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a test engineer specialising in {language}.
Write comprehensive unit tests for the provided code.

Rules:
- For Python: use the `unittest` framework. Do NOT use pytest.
- Output ONLY the raw test code — no markdown fences, no explanation.
- The test file must be fully self-contained and importable.
- Import the implementation using: from solution import <name>
- Cover: happy path, edge cases, invalid inputs, boundary values.
- Each test method must have a clear descriptive name.
- Do NOT mock the implementation — test the real code.
- Do NOT include any sys.path manipulation — it is already handled externally.
- Do NOT import or modify sys in any way.
- Do NOT include if __name__ == '__main__' blocks — the runner handles execution.
- Assume 'solution.py' is always importable directly — never try to add it to the path.
"""

# ── Sandbox execution ─────────────────────────────────────────────────────────

def _run_tests_sync(generated_code: str, test_code: str) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory() as tmpdir:
        impl_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "test_solution.py")

        # Normalize to forward slashes — safe on all platforms including Windows
        safe_tmpdir = tmpdir.replace("\\", "/")

        full_test = (
            "import sys\n"
            f"sys.path.insert(0, r'{tmpdir}')\n"  # raw string handles backslashes
            "\n"
            f"{test_code}"
        )

        with open(impl_path, "w", encoding="utf-8") as f:
            f.write(generated_code)

        with open(test_path, "w", encoding="utf-8") as f:
            f.write(full_test)

        try:
            result = subprocess.run(
                ["python", "-m", "unittest", "test_solution", "-v"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir,
            )
            output = (result.stdout + result.stderr).strip()
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            output = "❌ Sandbox timeout — test execution exceeded 30 seconds"
            success = False
        except Exception as e:
            output = f"❌ Sandbox error: {str(e)}"
            success = False

    return output, success


def _truncate_output(output: str, max_chars: int = 3000) -> str:
    """Truncate long sandbox output to avoid flooding the state."""
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        output[:half]
        + f"\n\n... [truncated {len(output) - max_chars} chars] ...\n\n"
        + output[-half:]
    )


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences if the LLM includes them despite instructions."""
    import re
    code = re.sub(r"^```[\w]*\n", "", code.strip())
    code = re.sub(r"\n```$", "", code.strip())
    return code.strip()

# ── LangGraph node ────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="TesterAgent")
async def tester_node(state: CodeGenState, llm) -> dict:
    language = state.get("language", "python")

    logger.info(
        "Tester started",
        extra={"extra": {
            "node":      "tester",
            "language":  language,
            "iteration": state.get("iteration", 1),
        }}
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(language=language)),
        HumanMessage(content=(
            f"Code to test:\n{state['generated_code']}\n\n"
            f"Requirements:\n" + "\n".join(
                f"  {r}" for r in state.get("requirements", [])
            )
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
        logger.info(
            "LLM response received",
            extra={"extra": {
                "node":  "tester",
                "model": getattr(llm, "model_name", "unknown"),
            }}
        )
    except Exception as e:
        logger.error(
            "Tester LLM call failed",
            extra={"extra": {"node": "tester", "error": str(e)}}
        )
        raise

    test_code = _strip_code_fences(response.content)

    # ── Sandbox execution (Python only) ──────────────────────────────────────
    if language == "python":
        logger.info(
            "Running sandbox execution",
            extra={"extra": {"node": "tester", "iteration": state.get("iteration", 1)}}
        )
        loop = asyncio.get_event_loop()
        execution_output, execution_success = await loop.run_in_executor(
            None, _run_tests_sync, state["generated_code"], test_code
        )
        execution_output = _truncate_output(execution_output)

        logger.info(
            "Sandbox execution completed",
            extra={"extra": {
                "node":              "tester",
                "execution_success": execution_success,
                "output_len":        len(execution_output),
            }}
        )

    else:
        # Non-Python: skip sandbox, let critic evaluate statically
        execution_output = (
            f"[Sandbox: '{language}' execution not supported — "
            f"deferring to critic for static review]"
        )
        execution_success = True
        logger.info(
            "Sandbox skipped for non-Python language",
            extra={"extra": {"node": "tester", "language": language}}
        )

    logger.info(
        "Tester completed",
        extra={"extra": {
            "node":              "tester",
            "execution_success": execution_success,
        }}
    )

    return {
        "test_code":         test_code,
        "execution_output":  execution_output,
        "execution_success": execution_success,
        "messages":          [response],
    }