import os

from dotenv import load_dotenv

from .llm_provider import LLMProvider
from .local_provider import LocalProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .mimo_provider import MimoProvider

def get_provider_from_env() -> LLMProvider:
    """Return an initialized LLM provider based on .env settings.

    Reads DEFAULT_PROVIDER, DEFAULT_MODEL, LOCAL_MODEL_PATH,
    OPENAI_API_KEY, GEMINI_API_KEY.

    Supported providers: local, openai, google/gemini
    """
    load_dotenv(override=True)  # override=True ensures last duplicate key in .env wins
    provider = os.getenv("DEFAULT_PROVIDER", "local").lower()


    if provider == "local":
        model_path = os.getenv("LOCAL_MODEL_PATH")
        if not model_path:
            raise ValueError("LOCAL_MODEL_PATH not set in .env for local provider")
        return LocalProvider(model_path=model_path)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env for openai provider")
        # OpenAIProvider(model_name, api_key) — match the actual constructor signature
        model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        return OpenAIProvider(model_name=model, api_key=api_key)

    if provider in ("google", "gemini"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env for gemini provider")
        model = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
        return GeminiProvider(model_name=model, api_key=api_key)

    if provider == "mimo":
        model = os.getenv("DEFAULT_MODEL", "mimo-v2.5-pro")

        api_keys = [
            os.getenv("MIMO_API_KEY_1"),
            os.getenv("MIMO_API_KEY_2"),
            os.getenv("MIMO_API_KEY"),
        ]
        api_keys = [key for key in api_keys if key]

        if not api_keys:
            raise ValueError(
                "No Mimo API key found. Set MIMO_API_KEY_1 and/or MIMO_API_KEY_2 (or MIMO_API_KEY) in .env"
            )

        return MimoProvider(
            model_name=model,
            api_keys=api_keys,
            api_key=api_keys[0],
        )

    raise ValueError(
        f"Unsupported DEFAULT_PROVIDER: '{provider}'. "
        "Valid options: local, openai, google, gemini, mimo"
    )