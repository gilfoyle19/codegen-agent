from __future__ import annotations

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv

from codegen_agent.config import (
    check_langsmith_config,
    get_runtime_checkpointer,
    get_llm,
)
from codegen_agent.graph import build_graph
from codegen_agent.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Stream event printer ──────────────────────────────────────────────────────

_printed: set[str] = set()


def _print_event(event: dict) -> None:
    if event.get("sanitization_error") and "sanitization_error" not in _printed:
        _printed.add("sanitization_error")
        print(f"\n❌ Input rejected: {event['sanitization_error']}")

    if event.get("language") and "planner" not in _printed:
        _printed.add("planner")
        print(f"\n📋 Language detected: {event['language']}")
        for r in event.get("requirements", []):
            print(f"   • {r}")

    if "docs_context" in event and "doc_retriever" not in _printed:
        _printed.add("doc_retriever")
        docs_len = len(event.get("docs_context", ""))
        if docs_len:
            print(f"\n📚 Docs retrieved: {docs_len} chars")
        else:
            print("\n📚 Doc retriever: no matching docs found")

    iteration = event.get("iteration")
    coder_key = f"coder_{iteration}"
    if event.get("generated_code") and iteration and coder_key not in _printed:
        _printed.add(coder_key)
        print(f"\n💻 Code generated (iteration {iteration}): "
              f"{len(event['generated_code'])} chars")

    tester_key = f"tester_{iteration}"
    if event.get("execution_output") and iteration and tester_key not in _printed:
        _printed.add(tester_key)
        status = "✅ PASS" if event.get("execution_success") else "❌ FAIL"
        print(f"\n🧪 Tests: {status}")
        if not event.get("execution_success"):
            print(f"   {event['execution_output'][:400]}")

    critic_key = f"critic_{iteration}"
    feedback = event.get("review_feedback", "")
    if feedback and iteration and critic_key not in _printed:
        _printed.add(critic_key)
        verdict = "✅ APPROVED" if event.get("review_passed") else "🔄 NEEDS REVISION"
        print(f"\n🔍 Critic (iteration {iteration}): {verdict}")
        if not event.get("review_passed"):
            for line in feedback.splitlines():
                print(f"   {line}")

    if event.get("final_answer") and "final_answer" not in _printed:
        _printed.add("final_answer")
        print("\n" + "=" * 60)
        print(event["final_answer"])
        print("=" * 60)


# ── Human review ──────────────────────────────────────────────────────────────

def _human_review(state_vals: dict) -> tuple[bool, str]:
    print("\n" + "=" * 60)
    print("⏸  HUMAN REVIEW REQUIRED")
    print("=" * 60)

    critic_status = "✅ APPROVED" if state_vals.get("review_passed") else "❌ NEEDS REVISION"
    print(f"\nCritic verdict : {critic_status}")
    print(f"Iterations used: {state_vals.get('iteration', 1)} / "
          f"{state_vals.get('max_iterations', 3)}")

    if state_vals.get("review_feedback"):
        print(f"\nFeedback:\n{state_vals['review_feedback']}")

    print("\n--- Generated Code ---")
    print(state_vals.get("generated_code", "(no code)"))

    print("\n[A]pprove  [E]dit  [R]eject  > ", end="", flush=True)
    choice = input().strip().lower()

    if choice == "r":
        logger.info("Human rejected output")
        print("\n🛑 Rejected.")
        return False, ""

    if choice == "e":
        print("\nPaste your edited code. Enter 'END' on a new line when done:\n")
        lines: list[str] = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        human_edit = "\n".join(lines)
        logger.info(
            "Human provided edit",
            extra={"extra": {"edit_len": len(human_edit)}}
        )
        return True, human_edit

    logger.info("Human approved output")
    return True, ""


# ── Main runner ───────────────────────────────────────────────────────────────

async def run(
    task: str,
    max_iterations: int = 3,
    use_doc_retriever: bool = True,
) -> str:
    check_langsmith_config()

    if use_doc_retriever and not os.path.isdir("./chroma_db"):
        print("⚠️  No chroma_db found — doc retrieval disabled.")
        print("   Run: uv run python scripts/ingest_docs.py")
        print("   Continuing without docs...\n")
        use_doc_retriever = False

    llm       = get_llm()
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "task":               task,
        "sanitized_task":     "",
        "sanitization_error": "",
        "messages":           [],
        "language":           "",
        "requirements":       [],
        "generated_code":     "",
        "test_code":          "",
        "docs_context":       "",
        "execution_output":   "",
        "execution_success":  False,
        "review_feedback":    "",
        "review_passed":      False,
        "human_approved":     False,
        "human_edit":         "",
        "iteration":          0,
        "max_iterations":     max_iterations,
        "final_answer":       "",
    }

    logger.info(
        "Graph execution started",
        extra={"extra": {"thread_id": thread_id, "task_len": len(task)}}
    )

    print("\n🚀 Starting code generation...\n")

    # ── Compile graph INSIDE context so checkpointer is live ─────────────────
    async with get_runtime_checkpointer() as checkpointer:
        graph = build_graph(llm, checkpointer, use_doc_retriever=use_doc_retriever)

        # Phase 1: stream until HITL interrupt
        async for event in graph.astream(
            initial_state, config=config, stream_mode="values"
        ):
            _print_event(event)

        # Check for sanitization rejection
        current_state = graph.get_state(config)
        if current_state.values.get("sanitization_error"):
            return ""

        if not current_state.next:
            logger.warning("Graph exited without reaching HITL interrupt")
            return ""

        # Phase 2: HITL review
        approved, human_edit = _human_review(current_state.values)
        if not approved:
            return ""

        # Phase 3: resume to formatter
        resume_state = {
            "human_approved": True,
            "human_edit":     human_edit,
            "review_passed":  not bool(human_edit),
        }

        final_event: dict = {}
        async for event in graph.astream(
            resume_state, config=config, stream_mode="values"
        ):
            _print_event(event)
            final_event = event

    logger.info(
        "Graph execution completed",
        extra={"extra": {"thread_id": thread_id}}
    )

    return final_event.get("final_answer", "")


# ── Resume a previous session ─────────────────────────────────────────────────

async def resume(thread_id: str) -> str:
    """Resume a previous session. Only works with sqlite or postgres."""
    check_langsmith_config()
    llm    = get_llm()
    config = {"configurable": {"thread_id": thread_id}}

    async with get_runtime_checkpointer() as checkpointer:
        graph         = build_graph(llm, checkpointer)
        current_state = graph.get_state(config)

        if not current_state or not current_state.values:
            print(f"❌ No checkpoint found for thread_id: {thread_id}")
            return ""

        print(f"✅ Resuming thread: {thread_id}")
        print(f"   Next node(s): {current_state.next}")

        approved, human_edit = _human_review(current_state.values)
        if not approved:
            return ""

        resume_state = {
            "human_approved": True,
            "human_edit":     human_edit,
            "review_passed":  not bool(human_edit),
        }

        final_event: dict = {}
        async for event in graph.astream(
            resume_state, config=config, stream_mode="values"
        ):
            _print_event(event)
            final_event = event

    return final_event.get("final_answer", "")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "--resume":
        asyncio.run(resume(args[1]))
        sys.exit(0)

    task = " ".join(args).strip()

    if not task:
        print("Usage:")
        print('  uv run python -m codegen_agent.main "<your coding task>"')
        print('  uv run python -m codegen_agent.main --resume <thread_id>')
        sys.exit(1)

    result = asyncio.run(run(task))
    if not result:
        sys.exit(1)