from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""

    # LLM model — all three LLM nodes (planner, evaluator, summarizer) use this model
    llm_model: str = "claude-sonnet-4-6"
    # Temperatures per scope.md: 0 for deterministic planning/evaluation, 0.3 for prose
    planner_temperature: float = 0.0
    evaluator_temperature: float = 0.0
    summarizer_temperature: float = 0.3

    # NLM Clinical Tables API — trailing slash so httpx joins paths correctly
    nlm_api_base: str = "https://clinicaltables.nlm.nih.gov/api/"
    api_timeout: float = 10.0
    api_max_retries: int = 2      # retries after initial failure (3 total attempts)
    api_backoff_base: float = 1.0  # first retry delay in seconds; doubles each attempt

    fetch_results: int = 10   # results fetched per system per executor call
    display_results: int = 5  # results kept in the final consolidated response

    confidence_threshold: float = 0.5  # consolidator score floor (reserved) — evaluator uses semantic judgment, not this threshold


settings = Settings()

MAX_ITERATIONS = 2  # cap enforced in route_after_evaluator, not in LLM prompts
NODE_PLANNER = "planner"
NODE_CONSOLIDATOR = "consolidator"
