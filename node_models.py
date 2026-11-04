"""Independent RouterAI/OpenAI/OpenRouter/Ollama configuration; no model fallback."""
import os

from handbook_graph import NodeModels


class OllamaStructuredModel:
    """Ollama uses format=JSON Schema, without OpenAI's strict argument."""
    def __init__(self, model):
        self.model = model

    def with_structured_output(self, schema, **kwargs):
        kwargs.pop("strict", None)
        return self.model.with_structured_output(schema, **kwargs)


def create_node_model(role):
    if role not in {"supervisor", "research", "writer", "guardrails"}:
        raise ValueError(f"Unknown graph role: {role}")
    prefix = role.upper()
    provider = os.getenv(f"{prefix}_PROVIDER", "routerai").strip().lower()
    if provider not in {"routerai", "openai", "openrouter", "ollama"}:
        raise ValueError(f"{prefix}_PROVIDER must be routerai, openai, openrouter or ollama")
    if provider == "ollama":
        default = "qwen2.5:3b-instruct"
    elif provider in {"routerai", "openrouter"}:
        default = {"supervisor": "google/gemma-3-4b-it", "research": "qwen/qwen-2.5-7b-instruct"}.get(
            role, "openai/gpt-4o-mini")
    else:
        default = os.getenv("LLM_MODEL", "gpt-4o-mini")
    name = os.getenv(f"{prefix}_MODEL", default).strip()
    if not name:
        raise ValueError(f"Set {prefix}_MODEL")
    token_limits = {"supervisor": 300, "research": 600, "writer": 2500, "guardrails": 1000}
    max_tokens = int(os.getenv(f"{prefix}_MAX_TOKENS", str(token_limits[role])))
    if max_tokens < 1:
        raise ValueError(f"{prefix}_MAX_TOKENS must be positive")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        context = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        if context < 1024:
            raise ValueError("OLLAMA_NUM_CTX must be at least 1024")
        model = ChatOllama(model=name, base_url=os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
                           temperature=0, num_ctx=context, num_predict=max_tokens,
                           client_kwargs={"timeout": 180}, validate_model_on_init=False)
        return OllamaStructuredModel(model)
    if provider == "routerai":
        if not os.getenv("ROUTERAI_API_KEY"):
            raise ValueError(f"Set ROUTERAI_API_KEY for {role}")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name, api_key=os.environ["ROUTERAI_API_KEY"],
            base_url=os.getenv("ROUTERAI_BASE_URL") or "https://routerai.ru/api/v1",
            timeout=60, max_retries=1, max_tokens=max_tokens,
            use_responses_api=False, stream_usage=False,
        )
    if provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(f"Set OPENROUTER_API_KEY for {role}")
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=name, api_key=os.environ["OPENROUTER_API_KEY"],
            # ChatOpenRouter uses milliseconds; ChatOpenAI below uses seconds.
            temperature=0, timeout=60_000, max_retries=1, max_tokens=max_tokens,
            # Skip endpoints that would silently ignore JSON Schema parameters.
            openrouter_provider={"require_parameters": True},
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(f"Set OPENAI_API_KEY for {role}, or configure {prefix}_PROVIDER=ollama")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=name, timeout=60, max_retries=1, max_tokens=max_tokens,
                      base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")


def create_models():
    return NodeModels(**{role: create_node_model(role)
                         for role in ("supervisor", "research", "writer", "guardrails")})
