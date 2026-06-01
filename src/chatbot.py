import os
import json
import time
from datetime import datetime

from src.core.local_provider import LocalProvider
from src.telemetry.logger import logger


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def save_result(data):
    filename = datetime.now().strftime(
        "logs/chatbot_%Y%m%d_%H%M%S.json"
    )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
    llm = LocalProvider(model_path=model_path)

    system_prompt = (
        "You are a helpful assistant. "
        "Be brief and factual."
    )

    tests = [
        "Find the cheapest price for 2 iPhones and compute total with 10% tax.",
        "List steps to buy two items from different stores and combine shipping.",
        "What is the meaning of capital of France?"
    ]

    for t in tests:
        logger.log_event("CHATBOT_START", {"input": t})

        start = time.time()

        response = llm.generate(
            t,
            system_prompt=system_prompt
        )

        latency = time.time() - start

        result = {
            "timestamp": datetime.now().isoformat(),
            "type": "chatbot",
            "input": t,
            "response": response["content"],
            "latency_seconds": latency,
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"],
        }

        save_result(result)

        logger.log_event(
            "CHATBOT_END",
            {
                "input": t,
                "latency_seconds": latency,
            }
        )

        print(f"INPUT: {t}")
        print(f"RESPONSE: {response}")
        print(f"LATENCY: {latency:.2f}s")
        print("-" * 50)


if __name__ == "__main__":
    main()