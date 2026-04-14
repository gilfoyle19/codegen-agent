import re
import unicodedata
from dataclasses import dataclass
from typing import Optional
from langsmith import traceable
from codegen_agent.logger import get_logger
from codegen_agent.state import CodeGenState

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MIN_LENGTH = 10
MAX_LENGTH = 4000

INJECTION_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"ignore (all |previous |above |prior )?instructions",
    r"you are now\b",
    r"disregard (all |your |the )?(?:above|prior|previous|instructions)",
    r"forget (all |everything|your instructions)",
    r"act as (a |an )?(?!python|javascript|typescript|rust|go|java|developer|engineer|assistant)",
    r"jailbreak",
    r"do anything now",
    r"bypass (safety|filter|restriction|guardrail)",
    r"(reveal|leak|print|output|show).{0,20}(system prompt|instructions|context)",
    r"<\|.*?\|>",              # special tokens e.g. <|endoftext|>
    r"\[INST\]|\[\/INST\]",   # instruction tokens
    r"###\s*(instruction|system|human|assistant)",  # few-shot injection headers
    r"begin\s+DAN\b",
    r"pretend (you are|to be|that)",
    r"override (previous|all|your)",
]]

DANGEROUS_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    # Malware / offensive tools
    r"(write|create|build|generate|code).{0,40}(virus|malware|ransomware|keylogger|rootkit|spyware|trojan|worm|botnet|exploit|backdoor|payload|shellcode)",
    # Auth bypass
    r"(bypass|crack|brute.?force|circumvent).{0,40}(password|auth|login|2fa|mfa|captcha|security)",
    # Data exfiltration
    r"(exfiltrate|steal|harvest|scrape|extract).{0,40}(credential|password|token|api.?key|secret|pii|personal data)",
    # Network attacks
    r"(launch|perform|execute|run).{0,30}(ddos|dos attack|sql.?injection|xss|csrf|mitm|phishing)",
    # Reverse shells / C2
    r"(reverse.?shell|bind.?shell|command.?and.?control|c2.?server|remote.?access.?tool)",
    # Privilege escalation
    r"(privilege.?escalat|root.?exploit|kernel.?exploit|local.?privilege)",
]]

# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SanitizationResult:
    cleaned: str
    error: Optional[str] = None

    @property
    def rejected(self) -> bool:
        return self.error is not None

# ── Individual sanitization checks ───────────────────────────────────────────

def _check_length(task: str) -> Optional[str]:
    stripped = task.strip()
    if len(stripped) < MIN_LENGTH:
        return f"Input too short: {len(stripped)} chars (minimum {MIN_LENGTH})"
    if len(task) > MAX_LENGTH:
        return f"Input too long: {len(task)} chars (maximum {MAX_LENGTH})"
    return None


def _check_injection(task: str) -> Optional[str]:
    for pattern in INJECTION_PATTERNS:
        if pattern.search(task):
            return f"Prompt injection detected: '{pattern.pattern}'"
    return None


def _check_dangerous(task: str) -> Optional[str]:
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(task):
            return f"Dangerous request detected: '{pattern.pattern}'"
    return None


def _normalize(task: str) -> str:
    # Strip leading/trailing whitespace
    text = task.strip()

    # Normalize unicode to NFC form (consistent representation)
    text = unicodedata.normalize("NFC", text)

    # Decode/re-encode to drop invalid UTF-8 sequences
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    # Remove non-printable characters (keep newlines and tabs)
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")

    # Collapse 3+ consecutive newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse horizontal whitespace runs (not newlines) → single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove zero-width and invisible unicode characters
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]", "", text)

    # Collapse repeated punctuation that may signal abuse (e.g. "!!!!!!!")
    text = re.sub(r"([!?.]){4,}", r"\1\1\1", text)

    return text.strip()

# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_sanitization(task: str) -> SanitizationResult:
    """Run all checks in order and return a SanitizationResult."""
    checks = [_check_length, _check_injection, _check_dangerous]
    for check in checks:
        if error := check(task):
            return SanitizationResult(cleaned="", error=error)
    return SanitizationResult(cleaned=_normalize(task))

# ── LangGraph node ────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="SanitizerNode")
async def sanitizer_node(state: CodeGenState) -> dict:
    logger.info(
        "Sanitizer started",
        extra={"extra": {"node": "sanitizer", "input_len": len(state["task"])}}
    )

    result = _run_sanitization(state["task"])

    if result.rejected:
        logger.warning(
            "Sanitizer rejected input",
            extra={"extra": {"node": "sanitizer", "reason": result.error}}
        )
        return {"sanitized_task": "", "sanitization_error": result.error}

    logger.info(
        "Sanitizer completed",
        extra={"extra": {"node": "sanitizer", "output_len": len(result.cleaned)}}
    )
    return {"sanitized_task": result.cleaned, "sanitization_error": ""}