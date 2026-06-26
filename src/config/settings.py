from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# This is the project's root folder (scholarship-rag/), computed from this
# file's location. We use this to build absolute paths to data/, logs/, etc.
# so the app works correctly no matter which directory you run it FROM.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    All configuration for the Scholarship RAG system, in one place.

    Every field below is read from environment variables (or a .env file)
    by Pydantic automatically — the field name `openai_api_key` maps to
    the environment variable `OPENAI_API_KEY` (case-insensitive by default).
    """

    # Pydantic-specific config: tells Pydantic WHERE to look for the .env
    # file, and to ignore any extra env vars it doesn't recognize (rather
    # than crashing on unrelated env vars present on the machine).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -----------------------------------------------------------------
    # LLM PROVIDER CONFIG
    # -----------------------------------------------------------------
    # WHY a "provider" field instead of hardcoding OpenAI everywhere:
    # This single field lets us swap the entire LLM backend (OpenAI vs
    # a local HuggingFace model) by changing ONE line in .env, instead
    # of editing code. See generation/llm_factory.py for how this is used.
    llm_provider: Literal["openai", "huggingface"] = Field(
        default="openai",
        description="Which LLM provider to use for answer generation.",
    )

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key. Required if llm_provider='openai'.",
    )

    llm_model_name: str = Field(
        default="gpt-4o-mini",
        description=(
            "Model name for the chosen provider. "
            "We default to gpt-4o-mini over gpt-3.5-turbo (used in the "
            "original prototype) because it is cheaper AND more capable "
            "as of 2025 — there is no longer a cost/quality tradeoff "
            "that favors gpt-3.5-turbo."
        ),
    )

    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description=(
            "Controls randomness of LLM output. We default to 0.2 (LOW), "
            "not the prototype's 0.5, because RAG answers should be "
            "FACTUAL and CONSISTENT, not creative. A low temperature "
            "means the model sticks closely to the retrieved context "
            "instead of improvising. If you raised this to e.g. 1.0, "
            "you would get more 'creative' phrasing but a higher risk "
            "of the model adding details not present in the source "
            "documents (hallucination)."
        ),
    )

    llm_max_output_tokens: int = Field(
        default=512,
        gt=0,
        description="Maximum tokens the LLM may generate in one response.",
    )

    # -----------------------------------------------------------------
    # EMBEDDING MODEL CONFIG
    # -----------------------------------------------------------------
    embedding_provider: Literal["openai", "huggingface"] = Field(
        default="openai",
        description="Which provider generates vector embeddings.",
    )

    embedding_model_name: str = Field(
        default="text-embedding-3-small",
        description=(
            "We default to text-embedding-3-small over the prototype's "
            "implicit text-embedding-ada-002 because it is both cheaper "
            "AND scores higher on retrieval benchmarks (MTEB). "
            "If budget were not a concern at all, text-embedding-3-large "
            "scores slightly higher still, at ~6x the cost."
        ),
    )

    embedding_dimension: int = Field(
        default=1536,
        description=(
            "Vector size produced by the embedding model. This MUST "
            "match the model above — text-embedding-3-small produces "
            "1536-dim vectors. Qdrant needs this number to create the "
            "collection with the correct vector size. If you change "
            "the embedding model, you MUST update this value too, or "
            "Qdrant will reject every insert with a dimension mismatch error."
        ),
    )

    # -----------------------------------------------------------------
    # CHUNKING CONFIG
    # -----------------------------------------------------------------
    chunk_size: int = Field(
        default=800,
        gt=0,
        description=(
            "Maximum characters per chunk. The prototype used 500 with "
            "no justification. We use 800 because scholarship documents "
            "often have eligibility criteria that span multiple sentences "
            "(e.g. 'Applicants must be enrolled full-time AND maintain a "
            "GPA above 3.0 AND be a resident of...') — a 500-char chunk "
            "risks splitting a single eligibility rule across two chunks, "
            "so the retriever might only find HALF of a requirement. "
            "TRADEOFF: larger chunks = more complete context per chunk, "
            "but FEWER, less precise retrieval hits (more irrelevant text "
            "comes along for the ride). 800 is a reasonable middle ground "
            "for this domain — we validate this experimentally in Phase 7 "
            "(evaluation) rather than just asserting it."
        ),
    )

    chunk_overlap: int = Field(
        default=120,
        ge=0,
        description=(
            "Characters of overlap between consecutive chunks (prototype "
            "used 50). Overlap exists so a sentence that gets cut at a "
            "chunk boundary still appears WHOLE in the next chunk. "
            "We use 120 (15% of chunk_size) because that is a common "
            "industry rule of thumb (10-20% overlap) — too little "
            "overlap risks losing context at boundaries, too much "
            "overlap wastes storage and creates near-duplicate chunks "
            "that crowd out genuinely different content in retrieval."
        ),
    )

    # -----------------------------------------------------------------
    # RETRIEVAL CONFIG
    # -----------------------------------------------------------------
    retrieval_top_k: int = Field(
        default=10,
        gt=0,
        description=(
            "How many chunks to retrieve BEFORE reranking. The prototype "
            "used k=3 directly as the FINAL answer context, with no "
            "reranking step. We retrieve MORE (10) initially, then "
            "rerank and keep only the best few — this is the standard "
            "'retrieve-then-rerank' pattern. Retrieving only 3 chunks "
            "with no reranking means if the 4th-most-similar chunk was "
            "actually the most RELEVANT one, you'd never see it."
        ),
    )

    rerank_top_k: int = Field(
        default=4,
        gt=0,
        description=(
            "How many chunks survive reranking and get sent to the LLM. "
            "Must be <= retrieval_top_k."
        ),
    )

    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description=(
            "Cross-encoder model used to rerank retrieved chunks. "
            "We chose this model because it runs locally (no API cost "
            "per rerank call) and is small enough (~22M params) to run "
            "on CPU with acceptable latency, while still meaningfully "
            "improving ranking quality over raw vector similarity. "
            "ALTERNATIVE: Cohere's hosted rerank API scores slightly "
            "higher but costs money per call and adds network latency."
        ),
    )

    hybrid_search_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for combining dense (semantic) vs sparse (BM25/"
            "keyword) search scores. 1.0 = pure dense search, 0.0 = pure "
            "keyword search. We default to 0.5 (equal weight) as a "
            "neutral starting point. WHY HYBRID AT ALL: dense embeddings "
            "are excellent at understanding MEANING ('financial aid' ~ "
            "'scholarship money') but can miss EXACT terms users actually "
            "type, like a specific scholarship name or a deadline date "
            "in a particular format. BM25 (keyword) search is the "
            "opposite: it nails exact terms but doesn't understand "
            "synonyms or paraphrasing. Combining both covers each "
            "other's blind spots."
        ),
    )

    # -----------------------------------------------------------------
    # VECTOR STORE (QDRANT) CONFIG
    # -----------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL of the Qdrant server. Points to the docker-compose service in production.",
    )

    qdrant_collection_name: str = Field(
        default="scholarship_documents",
        description="Name of the Qdrant collection storing our document chunks.",
    )

    qdrant_api_key: str | None = Field(
        default=None,
        description="API key for Qdrant Cloud. Not needed for local Docker Qdrant.",
    )

    # -----------------------------------------------------------------
    # MEMORY CONFIG
    # -----------------------------------------------------------------
    conversation_memory_window: int = Field(
        default=5,
        gt=0,
        description=(
            "Number of past conversation TURNS (question+answer pairs) "
            "kept in memory. The prototype used ConversationBufferMemory, "
            "which keeps the ENTIRE conversation forever — on a long "
            "chat session this means every single new request re-sends "
            "the whole history to the LLM, growing token cost linearly "
            "and eventually overflowing the context window. We use a "
            "WINDOWED memory that only keeps the last N turns instead."
        ),
    )

    # -----------------------------------------------------------------
    # GUARDRAILS CONFIG
    # -----------------------------------------------------------------
    max_query_length: int = Field(
        default=1000,
        gt=0,
        description="Maximum characters allowed in a user's question, to prevent abuse.",
    )

    toxicity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Toxicity score (0-1) above which content is blocked.",
    )

    rate_limit_requests_per_minute: int = Field(
        default=20,
        gt=0,
        description="Max requests a single user/session can make per minute.",
    )

    # -----------------------------------------------------------------
    # PATHS
    # -----------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")
    eval_reports_dir: Path = Field(default=PROJECT_ROOT / "eval_reports")
    golden_set_path: Path = Field(
        default=PROJECT_ROOT / "src" / "evaluation" / "golden_set.json"
    )

    # -----------------------------------------------------------------
    # APP / API CONFIG
    # -----------------------------------------------------------------
    app_env: Literal["development", "production", "test"] = Field(
        default="development",
    )

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # -----------------------------------------------------------------
    # VALIDATORS — run automatically when Settings() is instantiated
    # -----------------------------------------------------------------

    @field_validator("rerank_top_k")
    @classmethod
    def rerank_k_must_not_exceed_retrieval_k(cls, v: int, info) -> int:
        """
        Ensures rerank_top_k never exceeds retrieval_top_k.

        WHY THIS MATTERS: if you accidentally set rerank_top_k=15 but
        retrieval_top_k=10, the reranker would be asked to pick 15 best
        chunks out of only 10 candidates — a logically impossible config.
        We catch this at STARTUP with a clear message, instead of letting
        it cause a confusing silent bug during retrieval at request time.
        """
        retrieval_k = info.data.get("retrieval_top_k")
        if retrieval_k is not None and v > retrieval_k:
            raise ValueError(
                f"rerank_top_k ({v}) cannot exceed retrieval_top_k ({retrieval_k})"
            )
        return v

    @field_validator("openai_api_key")
    @classmethod
    def warn_if_openai_key_missing(cls, v: str | None, info) -> str | None:
        """
        We do NOT raise an error here directly, because this validator
        runs on the FIELD level and doesn't know the value of
        llm_provider yet (field order isn't guaranteed). Instead, the
        real "is everything actually configured correctly" check
        happens in validate_startup() below, which has access to the
        FULLY constructed Settings object and can check field
        relationships safely.
        """
        return v

    def validate_startup(self) -> None:
        """
        Performs cross-field validation that requires the FULLY loaded
        settings object (not just one field in isolation).

        WHY THIS IS A SEPARATE METHOD instead of more @field_validators:
        Checking "is openai_api_key set, GIVEN that llm_provider == openai"
        requires looking at TWO fields together. Pydantic field validators
        can technically do this via `info.data`, but it gets fragile
        because field validation order isn't guaranteed. It's clearer
        and more reliable to do these checks explicitly, once, right
        after the object is built — and call this method explicitly
        at application startup (see app/api.py's startup event).

        This is what makes the app FAIL FAST: if you forgot to set
        OPENAI_API_KEY in your .env file, you find out the INSTANT you
        start the app, with a clear message — not three hours later
        when a user's query happens to trigger an LLM call and you get
        a confusing 401 error from OpenAI's API instead.
        """
        from src.exceptions import ConfigurationError

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                missing_setting="OPENAI_API_KEY",
                reason="required because LLM_PROVIDER is set to 'openai'",
            )

        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                missing_setting="OPENAI_API_KEY",
                reason="required because EMBEDDING_PROVIDER is set to 'openai'",
            )

        # Ensure required directories exist. We create them here rather
        # than assuming they exist, so a fresh clone of the repo works
        # immediately without manual setup steps.
        for directory in (self.data_dir, self.logs_dir, self.eval_reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Returns a SINGLE cached instance of Settings, shared across the
    whole application.

    WHY @lru_cache: without this, every module that does `get_settings()`
    would create a NEW Settings object, re-reading and re-parsing the
    .env file every single time. With @lru_cache, the env file is read
    ONCE, and every subsequent call returns the SAME object instantly.
    This is the standard FastAPI pattern for settings management.
    """
    return Settings()


# A module-level instance for convenient importing as `from src.config.settings import settings`
settings = get_settings()
