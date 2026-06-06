"""
LLM augmentation helpers with deterministic fallback.

These wrap an LLMClient so that LLM output is *optional and verified*: if the
LLM is unconfigured, errors, or returns something invalid, the caller's
deterministic result is returned instead. This is the backbone of the
"deterministic core + LLM layer" architecture — the LLM never becomes a
single point of failure, and every LLM step stays offline-testable.

Two shapes:
  - generate_text   — free-form text (e.g. a narrative report), fall back to
                      a deterministic string.
  - generate_model  — strict JSON parsed into a Pydantic model (e.g. LLM
                      extraction), fall back to a deterministic object.
"""

import json
import re

from typing import Any, TypeVar

from pydantic import BaseModel

from app.services.llm_client import LLMClient

T = TypeVar("T", bound=BaseModel)


def generate_text(
    llm: LLMClient,
    prompt: str,
    system: str | None = None,
    fallback: str = "",
) -> str:
    """Return LLM-generated text, or *fallback* if the LLM can't be used.

    Falls back when the client is unconfigured, the call raises, or the reply
    is empty.

    Args:
        llm: The LLM client.
        prompt: User prompt.
        system: Optional system prompt.
        fallback: Deterministic text to return if the LLM is unavailable.

    Returns:
        The LLM text (stripped) or the fallback.
    """
    if not llm.is_configured():
        return fallback
    try:
        text = llm.complete(prompt, system=system)
    except Exception:
        return fallback
    text = (text or "").strip()
    return text or fallback


def extract_json(raw: str) -> Any:
    """Extract a JSON value from a model reply.

    Handles plain JSON, ```json fenced blocks, and JSON embedded in prose by
    locating the first object/array.

    Args:
        raw: The raw model reply.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If no JSON can be parsed.
    """
    text = raw.strip()

    # Strip a surrounding code fence if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))  # may raise; caller treats as failure

    raise ValueError("no JSON found in model reply")


_JSON_ONLY_SYSTEM = (
    "Respond with a single valid JSON value only. Do not include explanations, "
    "prose, apologies, or markdown code fences — output JSON and nothing else."
)


def _strict_system(system: str | None) -> str:
    """System prompt for corrective retries: enforce JSON-only output.

    Keeps the caller's system instructions (if any) and appends the strict
    JSON-only directive.
    """
    if system:
        return f"{system}\n\n{_JSON_ONLY_SYSTEM}"
    return _JSON_ONLY_SYSTEM


def _repair_prompt(
    original: str, previous: str | None, error: Exception, model_cls: type[T]
) -> str:
    """Build a corrective prompt that feeds the failure back to the model."""
    parts = [
        original,
        "",
        "Your previous response could not be used.",
    ]
    if previous is not None:
        parts += ["Previous response:", previous]
    parts += [
        f"Error: {error}",
        f"Required JSON schema: {json.dumps(model_cls.model_json_schema())}",
        "Return ONLY corrected JSON that satisfies the schema.",
    ]
    return "\n".join(parts)


def generate_model(
    llm: LLMClient,
    prompt: str,
    model_cls: type[T],
    fallback: T,
    system: str | None = None,
    retries: int = 1,
) -> T:
    """Return an LLM-produced, schema-validated model, or *fallback*.

    The LLM is asked for JSON, which is parsed and validated against
    *model_cls*. On failure, the next attempt is a **corrective retry**: the
    bad output and the specific error are fed back with the target schema,
    asking the model to fix it (a plain re-prompt would just reproduce the same
    failure at low temperature). After *retries* corrective attempts, the
    deterministic fallback is returned.

    Args:
        llm: The LLM client.
        prompt: User prompt (should ask for JSON matching the schema).
        model_cls: The Pydantic model to validate into.
        fallback: Deterministic instance returned on any failure.
        system: Optional system prompt.
        retries: Number of corrective attempts after the first (default 1, so
            up to two attempts total). Use 0 to disable retrying.

    Returns:
        A validated *model_cls* instance, or *fallback*.
    """
    if not llm.is_configured():
        return fallback

    current_prompt = prompt
    current_system = system
    for _ in range(retries + 1):
        raw: str | None = None
        try:
            raw = llm.complete(current_prompt, system=current_system)
            data = extract_json(raw)
            return model_cls.model_validate(data)
        except Exception as exc:
            # Feed the bad output + error back, and tighten the system prompt to
            # demand JSON-only, so the next attempt can correct itself.
            current_prompt = _repair_prompt(prompt, raw, exc, model_cls)
            current_system = _strict_system(system)
    return fallback
