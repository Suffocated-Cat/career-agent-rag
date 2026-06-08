"""
LLMClient — thin wrapper over an OpenAI-compatible chat completion API.

Reads credentials from settings (``LLM_API_KEY`` / ``LLM_BASE_URL`` /
``LLM_MODEL``), so it works with any OpenAI-compatible endpoint. The SDK client
is created lazily on first use, and a pre-built client can be injected for
testing or to share one instance.
"""

from typing import Any

from app.core.config import settings
from app.services.usage import TokenUsage, UsageTracker


class LLMClient:
    """Minimal chat-completion client for an OpenAI-compatible endpoint.

    Usage::

        llm = LLMClient()
        if llm.is_configured():
            text = llm.complete("Say hi", system="You are terse.")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        usage_tracker: UsageTracker | None = None,
    ):
        """Configure the client.

        Args:
            api_key: API key (default: settings.LLM_API_KEY).
            base_url: Base URL of the OpenAI-compatible API
                (default: settings.LLM_BASE_URL).
            model: Model id to call (default: settings.LLM_MODEL).
            client: A pre-built OpenAI-compatible client to use directly;
                when given, no SDK client is created lazily.
            usage_tracker: Optional tracker to accumulate token usage across
                calls.
        """
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL
        self._client = client
        self.usage_tracker = usage_tracker
        self.last_usage: TokenUsage | None = None

    def is_configured(self) -> bool:
        """True if an API key and model are set (ready to make calls)."""
        return bool(self.api_key and self.model)

    def _get_client(self) -> Any:
        """Lazily construct (and cache) the OpenAI SDK client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Run a single-turn chat completion and return the text.

        Args:
            prompt: The user message.
            system: Optional system message.
            temperature: Sampling temperature (default 0.0 for determinism).
            **kwargs: Extra parameters passed to the completion call.

        Returns:
            The assistant message content.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self._is_deepseek():
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        self._record_usage(response)
        return response.choices[0].message.content

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ):
        """Stream a chat completion, yielding text deltas as they arrive.

        Args:
            prompt: The user message.
            system: Optional system message.
            temperature: Sampling temperature (default 0.0).
            **kwargs: Extra parameters passed to the completion call.

        Yields:
            Content fragments (strings) in order; empty deltas are skipped.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self._is_deepseek():
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body

        stream = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                self._record_usage(chunk)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            text = getattr(choices[0].delta, "content", None)
            if text:
                yield text

    def _is_deepseek(self) -> bool:
        """True when the configured endpoint/model is DeepSeek-compatible."""
        target = f"{self.base_url or ''} {self.model or ''}".lower()
        return "deepseek" in target

    def _record_usage(self, response: Any) -> None:
        """Capture token usage from a response, if present."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.last_usage = TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
        if self.usage_tracker is not None:
            self.usage_tracker.add(self.last_usage)
