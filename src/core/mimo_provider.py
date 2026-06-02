import time
from typing import Dict, Any, Optional, Generator, List

from openai import OpenAI

from src.core.llm_provider import LLMProvider


class MimoProvider(LLMProvider):
    def __init__(
        self,
        model_name: str = "mimo-v2.5-pro",
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
    ):
        super().__init__(model_name, api_key)

        key_candidates = [key for key in (api_keys or []) if key]
        if api_key:
            key_candidates.insert(0, api_key)

        unique_keys: List[str] = []
        for key in key_candidates:
            if key not in unique_keys:
                unique_keys.append(key)

        if not unique_keys:
            raise ValueError("At least one Mimo API key must be provided")

        self.api_keys = unique_keys
        self._client_index = 0

    def _get_client(self, api_key: str) -> OpenAI:
        return OpenAI(
            api_key=api_key,
            base_url="https://token-plan-sgp.xiaomimimo.com/v1"
        )

    def _next_key_index(self) -> int:
        index = self._client_index
        self._client_index = (self._client_index + 1) % len(self.api_keys)
        return index

    def _iter_api_keys(self):
        start = self._next_key_index()
        for offset in range(len(self.api_keys)):
            yield self.api_keys[(start + offset) % len(self.api_keys)]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:

        start_time = time.time()

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        last_error: Optional[Exception] = None

        for api_key in self._iter_api_keys():
            client = self._get_client(api_key)
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=300,
                    temperature=0
                )
                break

            except Exception as exc:
                last_error = exc
                continue
        else:
            if last_error:
                raise last_error
            raise RuntimeError("MimoProvider failed to obtain a response")

        latency_ms = int(
            (time.time() - start_time) * 1000
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "latency_ms": latency_ms,
            "provider": "mimo"
        }

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        last_error: Optional[Exception] = None

        for api_key in self._iter_api_keys():
            client = self._get_client(api_key)
            try:
                stream = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True
                )

                for chunk in stream:
                    if (
                        chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
                        yield chunk.choices[0].delta.content
                return

            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            raise last_error