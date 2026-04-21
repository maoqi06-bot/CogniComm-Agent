"""OpenAI-compatible LLM client."""

from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_client import BaseLLMClient, LLMError


class OpenAIClient(BaseLLMClient):
    """Wrapper for OpenAI and OpenAI-compatible gateways.

    Official OpenAI-style setups can use the Responses API, while many
    third-party compatible gateways still behave more reliably with the
    chat-completions API.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-5",
        base_url: str = "",
        timeout: int = 600,
    ) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is not installed. Please run: pip install openai")

        super().__init__(api_key, model=model, base_url=base_url, timeout=timeout)
        self.api_style = self._resolve_api_style()

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        **extra: Any,
    ) -> Dict[str, Any]:
        """Send a generation request to the OpenAI-compatible API."""

        try:
            if self.api_style == "chat_completions":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **extra,
                )
                return {"response": response, "api_style": self.api_style}

            input_text = self._convert_messages_to_input(messages)
            response = self.client.responses.create(
                model=self.model,
                input=input_text,
                **extra,
            )
            return {"response": response, "api_style": self.api_style}
        except Exception as e:  # noqa: BLE001 - normalize SDK errors
            raise LLMError(f"OpenAI API call failed: {e}")

    def extract_text(self, data: Dict[str, Any]) -> str:
        """Extract text from an OpenAI SDK response payload."""

        if not isinstance(data, dict):
            raise LLMError("Unexpected response payload type.")

        response = data.get("response")
        api_style = data.get("api_style", self.api_style)
        if response:
            try:
                if api_style == "chat_completions":
                    choice = response.choices[0]
                    content = choice.message.content
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        parts = [
                            part.text
                            for part in content
                            if getattr(part, "type", None) in {"text", "output_text"}
                            and getattr(part, "text", None)
                        ]
                        if parts:
                            return "".join(parts).strip()
                return response.output_text.strip()
            except Exception as e:  # noqa: BLE001
                raise LLMError(f"Unable to extract text from OpenAI response: {e}")

        raise LLMError("Unable to extract text from OpenAI response.")

    def _resolve_api_style(self) -> str:
        configured = os.getenv("OPENAI_API_STYLE", "auto").strip().lower()
        if configured in {"responses", "chat_completions"}:
            return configured

        base = (self.base_url or "").lower()
        if base and "api.openai.com" not in base:
            return "chat_completions"
        return "responses"

    def _convert_messages_to_input(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat-style messages into a single Responses API input."""

        input_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                input_parts.append(f"System: {content}")
            elif role == "user":
                input_parts.append(f"User: {content}")
            elif role == "assistant":
                input_parts.append(f"Assistant: {content}")
            else:
                input_parts.append(content)
        return "\n\n".join(input_parts)
