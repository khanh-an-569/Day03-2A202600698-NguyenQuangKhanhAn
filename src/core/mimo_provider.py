import time
from typing import Dict, Any, Optional, Generator

from openai import OpenAI, RateLimitError

from src.core.llm_provider import LLMProvider


class MimoProvider(LLMProvider):
    def __init__(
        self,
        model_name: str = "mimo-v2.5-pro",
        api_key: Optional[str] = None
    ):
        super().__init__(model_name, api_key)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://token-plan-sgp.xiaomimimo.com/v1"
        )

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

        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages
        # )
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=300,
                    temperature=0
                )
                break

            except RateLimitError:
                if attempt == 2:
                    raise

                time.sleep(5)

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

        stream = self.client.chat.completions.create(
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