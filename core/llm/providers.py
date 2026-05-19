"""
core/llm/providers.py
---------------------
LLM provider abstraction layer with automatic retry on 503.
"""

from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES    = 4
RETRY_DELAYS   = [2, 5, 10, 20]   # seconds between retries


class BaseLLMProvider(ABC):
    @abstractmethod
    def reason(self, prompt: str) -> str: ...

    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    def name(self) -> str:
        return self.__class__.__name__


class GeminiProvider(BaseLLMProvider):
    MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None):
        from google import genai
        
        self._keys = []
        
        # 1. Check for individual numbered environment variables (GEMINI_API_KEY, GEMINI_API_KEY_2, etc.)
        for i in range(1, 6):
            suffix = f"_{i}" if i > 1 else ""
            env_val = os.environ.get(f"GEMINI_API_KEY{suffix}")
            if env_val:
                # Support comma-separated within any of these variables as well
                self._keys.extend([k.strip() for k in env_val.split(",") if k.strip()])

        # 2. Check if an explicit api_key was passed in
        if api_key:
            self._keys = [k.strip() for k in api_key.split(",") if k.strip()]

        if not self._keys:
            raise ValueError("No Gemini API keys found. Please set GEMINI_API_KEY, GEMINI_API_KEY_2, etc.")
        
        self._current_key_index = 0
        self._client = genai.Client(api_key=self._keys[0])
        print(f"  [Gemini] Initialised with {len(self._keys)} API keys for rotation.")

    def _rotate_key(self):
        from google import genai
        self._current_key_index = (self._current_key_index + 1) % len(self._keys)
        new_key = self._keys[self._current_key_index]
        print(f"  [Gemini] Switching to API key {self._current_key_index + 1}/{len(self._keys)}")
        self._client = genai.Client(api_key=new_key)

    def _call_with_retry(self, prompt: str) -> str:
        from google.genai.errors import ServerError, ClientError
        import re

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                )
                return response.text.strip()

            except ServerError as e:
                # 503 is often a temporary overload or a soft rate limit
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    if len(self._keys) > 1:
                        self._rotate_key()
                        time.sleep(1) # tiny sleep before retry with new key
                        continue # retry immediately with new key
                    
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_DELAYS[attempt]
                        print(f"  [Gemini] 503 unavailable — retrying in {wait}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            "Gemini is unavailable after multiple retries.") from e
                else:
                    raise

            except ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if len(self._keys) > 1:
                        self._rotate_key()
                        time.sleep(1)
                        continue # retry immediately with new key

                    # extract suggested retry delay from error message
                    match = re.search(r"retry in (\d+)", str(e))
                    wait  = int(match.group(1)) + 2 if match else 30
                    if attempt < MAX_RETRIES - 1:
                        print(f"  [Gemini] Rate limited — retrying in {wait}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            "Gemini rate limit exceeded. "
                            "Wait a few minutes before trying again."
                        ) from e
                else:
                    raise

        return ""

    def reason(self, prompt: str) -> str:
        return self._call_with_retry(prompt)

    def generate(self, prompt: str) -> str:
        return self._call_with_retry(prompt)

    def name(self) -> str:
        return "gemini"


class AnthropicProvider(BaseLLMProvider):
    MODEL = "claude-3-5-haiku-20241022"

    def __init__(self, api_key: str | None = None):
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found.")
        self._client = anthropic.Anthropic(api_key=key)

    def _call(self, prompt: str, max_tokens: int = 1024) -> str:
        message = self._client.messages.create(
            model=self.MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def reason(self, prompt: str) -> str:
        return self._call(prompt, max_tokens=512)

    def generate(self, prompt: str) -> str:
        return self._call(prompt, max_tokens=1024)

    def name(self) -> str:
        return "anthropic"


def get_provider(
    name: str = "gemini",
    api_key: str | None = None,
) -> BaseLLMProvider:
    name = name.lower().strip()
    if name == "gemini":
        return GeminiProvider(api_key=api_key)
    elif name in ("anthropic", "claude"):
        return AnthropicProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider '{name}'. Use 'gemini' or 'anthropic'.")