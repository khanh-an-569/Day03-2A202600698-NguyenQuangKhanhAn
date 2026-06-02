import os
import sys

from dotenv import load_dotenv

from src.agent.agent import ReActAgent
from src.core.provider_factory import get_provider_from_env
from src.tools.web_search import web_search


def build_agent() -> ReActAgent:
    load_dotenv(override=True)
    llm = get_provider_from_env()

    tools = [
        {
            "name": "web_search",
            "description": (
                "Search the web for similar products, prices, ratings, and review snippets. "
                "Argument: a search query string."
            ),
            "func": web_search,
        }
    ]

    # Read max_steps from env so it can be tuned without code changes
    max_steps = int(os.getenv("AGENT_MAX_STEPS", "3"))
    return ReActAgent(llm=llm, tools=tools, max_steps=max_steps)


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "iPhone 15 case"

    print(f"[Agent] Query: {query}")
    agent = build_agent()
    result = agent.run(query)

    print("\n=== FINAL OUTPUT ===")
    print(result)


if __name__ == "__main__":
    main()