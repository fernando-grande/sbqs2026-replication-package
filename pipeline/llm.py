"""Access to Gemini, shared by the grounding (extract/) and augmentation stages.

Two routes to the same model (``config.GEMINI_MODEL``):

* ``google``     - the Gemini API, key in GOOGLE_API_KEY (google-genai package);
* ``openrouter`` - OpenRouter's OpenAI-compatible endpoint, key in
                   OPENROUTER_API_KEY, standard library only. OpenRouter
                   forwards the request to Google (AI Studio or Vertex).

``auto`` picks OpenRouter when its key is set. Keys may live in .env,
which is git-ignored. Every backend returns ``(text, served_by, truncated)``,
so callers can record which provider answered and detect answers cut at the
output limit.
"""
from __future__ import annotations

import json
import os

from . import config


class ApiError(RuntimeError):
    """HTTP-level failure, carrying the status code like google-genai errors do."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code} {message}")
        self.code = code


def load_env_file() -> None:
    """Read KEY=value lines from .env (git-ignored) into os.environ.

    Variables already set in the environment win over the file.
    """
    path = config.PACKAGE_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        os.environ.setdefault(name, value.strip().strip("\"'"))


def resolve_provider(provider: str = config.GROUNDING_PROVIDER) -> str:
    """``auto`` picks OpenRouter when its key is set, the Google API otherwise."""
    load_env_file()
    if provider == "auto":
        return "openrouter" if os.getenv("OPENROUTER_API_KEY") else "google"
    return provider


def has_api_key(provider: str = config.GROUNDING_PROVIDER) -> bool:
    if resolve_provider(provider) == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


class GoogleBackend:
    """Google Gemini API (google-genai), key in GOOGLE_API_KEY."""

    name = "google"

    def __init__(self, temperature: float | None, json_mode: bool, max_tokens: int) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client()
        self._types = types
        options = {
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if temperature is not None:
            options["temperature"] = temperature
        if json_mode:
            options["response_mime_type"] = "application/json"
        self._config = types.GenerateContentConfig(**options)

    def generate(self, prompt: str) -> tuple[str, str, bool]:
        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=self._config,
        )
        finish = response.candidates[0].finish_reason if response.candidates else None
        truncated = finish == self._types.FinishReason.MAX_TOKENS
        return response.text or "", "google-ai-studio", truncated


class OpenRouterBackend:
    """The same Gemini model through OpenRouter, key in OPENROUTER_API_KEY."""

    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, temperature: float | None, json_mode: bool, max_tokens: int) -> None:
        self._key = os.environ["OPENROUTER_API_KEY"]
        self._options: dict = {"max_tokens": max_tokens}
        if temperature is not None:
            self._options["temperature"] = temperature
        if json_mode:
            self._options["response_format"] = {"type": "json_object"}

    def generate(self, prompt: str) -> tuple[str, str, bool]:
        import urllib.error
        import urllib.request

        body = {
            "model": config.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            **self._options,
        }
        request = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "X-Title": "ECPCESE",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ApiError(exc.code, exc.read().decode("utf-8", "replace")[:500]) from exc

        if "error" in data:
            error = data["error"]
            code = error.get("code", 500)
            raise ApiError(code if isinstance(code, int) else 500, str(error.get("message", error))[:500])
        choice = data["choices"][0]
        truncated = choice.get("finish_reason") == "length"
        return choice["message"]["content"] or "", f"openrouter/{data.get('provider', 'unknown')}", truncated


def make_backend(
    provider: str = config.GROUNDING_PROVIDER,
    *,
    temperature: float | None,
    json_mode: bool,
    max_tokens: int,
):
    """Backend for ``provider`` ("auto", "openrouter" or "google")."""
    if resolve_provider(provider) == "openrouter":
        return OpenRouterBackend(temperature, json_mode, max_tokens)
    return GoogleBackend(temperature, json_mode, max_tokens)
