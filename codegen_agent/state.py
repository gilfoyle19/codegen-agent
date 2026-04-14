from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class CodeGenState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str                        # raw user input
    sanitized_task: str              # cleaned, validated task string
    sanitization_error: str          # non-empty if input was rejected
    language: str                    # detected programming language
    requirements: list[str]          # atomic requirements from planner
    generated_code: str              # latest code from coder
    test_code: str                   # unit tests from tester
    docs_context: str                # retrieved docs (optional)
    execution_output: str            # stdout/stderr from sandbox
    execution_success: bool          # True if all tests pass
    review_feedback: str             # critic's structured feedback
    review_passed: bool              # True when critic approves
    human_approved: bool             # True after human confirms at interrupt
    human_edit: str                  # optional code edit from human reviewer
    iteration: int                   # current refinement iteration
    max_iterations: int              # safety cap (default 3)
    final_answer: str                # polished output shown to user