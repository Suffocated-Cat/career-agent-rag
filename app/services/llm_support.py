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
    *model_cls*. Transient failures (call error, unparseable JSON, validation
    error) are retried up to *retries* times — LLMs often fix malformed JSON on
    a second pass — before yielding the deterministic fallback.

    Args:
        llm: The LLM client.
        prompt: User prompt (should ask for JSON matching the schema).
        model_cls: The Pydantic model to validate into.
        fallback: Deterministic instance returned on any failure.
        system: Optional system prompt.
        retries: Number of extra attempts after the first (default 1, so up to
            two attempts total). Use 0 to disable retrying.

    Returns:
        A validated *model_cls* instance, or *fallback*.
    """
    if not llm.is_configured():
        return fallback

    for _ in range(retries + 1):
        try:
            raw = llm.complete(prompt, system=system)
            data = extract_json(raw)
            return model_cls.model_validate(data)
        except Exception:
            # Transient: bad JSON / schema / call error — try again, then fall back.
            continue
    return fallback
