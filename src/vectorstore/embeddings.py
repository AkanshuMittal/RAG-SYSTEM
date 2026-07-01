from functools import lru_cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)
from src.config import settings
from src.exceptions import EmbeddingError

_RETRYABLE_EXCEPTION_TYPES = (TimeoutError, ConnectionError, OSError)


def _retry_decorator():
    """
    Builds the retry policy for OpenAI API calls.

    WHY EXPONENTIAL BACKOFF WITH JITTER, NOT A FIXED DELAY:
    If we retried instantly (or with a FIXED delay) every time, and
    OpenAI is rate-limiting MANY of our requests at once, all our
    retries would land at the SAME moment again and get rate-limited
    again — a "thundering herd" problem. Exponential backoff (wait
    LONGER each retry: ~2s, then ~4s, then ~8s) gives the rate limit
    window time to clear. Adding RANDOM JITTER on top means even
    multiple concurrent requests from our OWN app don't all retry at
    the exact same moment as each other.

    WHY stop_after_attempt(3): three tries balances "give transient
    failures a real chance to recover" against "don't make the user
    wait 30+ seconds before we give up and report a real failure".
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10) + wait_random(0, 1),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTION_TYPES),
        reraise=True,
    )


class EmbeddingProvider:
    """
    A unified interface for generating embeddings, regardless of which
    underlying provider (OpenAI API or local HuggingFace model) is
    actually doing the work.

    WHY A CLASS WITH A SINGLE embed() METHOD, RATHER THAN TWO SEPARATE
    FUNCTIONS (embed_with_openai, embed_with_huggingface) THAT CALLERS
    CHOOSE BETWEEN:
    Callers (retriever.py, ingest_pipeline integration below) should
    NEVER need an if/else checking which provider is configured — they
    just call `provider.embed(text)` and get a vector back. The CHOICE
    of provider is made ONCE, in get_embedding_provider() below, based
    on settings — not scattered across every call site.
    """

    def __init__(self):
        self._provider_name = settings.embedding_provider
        self._model_name = settings.embedding_model_name
        self._client = None  # lazily initialized on first use — see _ensure_client()

    def _ensure_client(self) -> None:
        """
        Lazily creates the underlying embedding client on first use.

        WHY LAZY INITIALIZATION INSTEAD OF CREATING THE CLIENT IN
        __init__: importing and initializing the HuggingFace
        sentence-transformers library is SLOW (it loads a multi-hundred
        MB model into memory) and entirely unnecessary if the app is
        configured to use OpenAI instead. We only pay that cost if/when
        a HuggingFace embedding is actually requested.
        """
        if self._client is not None:
            return  # already initialized, nothing to do

        if self._provider_name == "openai":
            try:
                from openai import OpenAI
            except ImportError as error:
                raise EmbeddingError(
                    "the 'openai' package is required for EMBEDDING_PROVIDER=openai "
                    "but is not installed"
                ) from error

            if not settings.openai_api_key:
                raise EmbeddingError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
                )

            self._client = OpenAI(api_key=settings.openai_api_key)

        elif self._provider_name == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise EmbeddingError(
                    "the 'sentence-transformers' package is required for "
                    "EMBEDDING_PROVIDER=huggingface but is not installed"
                ) from error

            # WHY bge-large-en-v1.5 AS THE DEFAULT HUGGINGFACE MODEL when
            # the configured embedding_model_name doesn't look like a
            # HuggingFace model path: this is a well-regarded, free,
            # locally-runnable embedding model with strong retrieval
            # benchmark scores (MTEB), making it a sensible default for
            # development without an OpenAI key.
            model_name = settings.embedding_model_name
            if model_name == "text-embedding-3-small":
                # the OpenAI-specific default doesn't apply here — fall
                # back to a sensible HuggingFace default instead
                model_name = "BAAI/bge-large-en-v1.5"

            self._client = SentenceTransformer(model_name)

        else:
            # This should be unreachable because settings.py's Literal
            # type already restricts embedding_provider to valid values
            # — but we guard anyway, since defensive code here is cheap
            # and protects against future refactors that might loosen
            # that type constraint.
            raise EmbeddingError(f"unknown embedding provider: '{self._provider_name}'")

    def embed(self, text: str) -> list[float]:
        """
        Converts a single piece of text into a vector embedding.

        Args:
            text: The text to embed (a chunk's content, or a user's
                  question — both go through the exact same path).

        Returns:
            A list of floats representing the embedding vector. Its
            length MUST match settings.embedding_dimension, or Qdrant
            will reject it later.

        Raises:
            EmbeddingError: if the embedding call fails for any reason
                            (API error, rate limit, empty input, etc.)
        """
        if not text or not text.strip():
            raise EmbeddingError("cannot embed empty or whitespace-only text")

        self._ensure_client()

        if self._provider_name == "openai":
            return self._embed_with_openai(text)
        else:
            return self._embed_with_huggingface(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds multiple texts in one call.

        WHY THIS EXISTS SEPARATELY FROM CALLING embed() IN A LOOP:
        Both OpenAI's API and HuggingFace's sentence-transformers
        support BATCH embedding, which is significantly faster than
        embedding one text at a time — fewer network round-trips for
        OpenAI, and better GPU/CPU utilization for HuggingFace. During
        ingestion (Phase 3's integration with ingest_pipeline.py below),
        we may have hundreds of chunks to embed, so batching matters
        for real performance, not just a marginal optimization.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors, in the SAME ORDER as the
            input texts.
        """
        if not texts:
            return []

        self._ensure_client()

        if self._provider_name == "openai":
            return self._embed_batch_with_openai(texts)
        else:
            return self._embed_batch_with_huggingface(texts)

    @_retry_decorator()
    def _call_openai_embeddings_api(self, model: str, input_value):
        """
        Makes the raw OpenAI API call, WITH retry applied directly to it.

        WHY THIS IS ITS OWN METHOD, SEPARATE FROM THE try/except THAT
        WRAPS FAILURES INTO EmbeddingError:
        Retry must see the ORIGINAL exception type (ConnectionError,
        TimeoutError) to decide whether to retry. If we wrapped
        everything into EmbeddingError FIRST and retried around THAT,
        tenacity's retry_if_exception_type check would never match,
        because by the time it sees the exception, it's already been
        converted into our own custom type, which is correctly NOT in
        the retryable list (we don't want to retry e.g. a "you sent
        invalid input" error, only genuine transient network issues).
        Keeping the raw call separate from the wrapping logic lets retry
        operate on the TRUE underlying exception type.
        """
        return self._client.embeddings.create(model=model, input=input_value)

    def _embed_with_openai(self, text: str) -> list[float]:
        try:
            response = self._call_openai_embeddings_api(self._model_name, text)
        except Exception as error:
            raise EmbeddingError(
                f"OpenAI embedding call failed after retries "
                f"({type(error).__name__}: {error})"
            ) from error

        return response.data[0].embedding

    def _embed_batch_with_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._call_openai_embeddings_api(self._model_name, texts)
        except Exception as error:
            raise EmbeddingError(
                f"OpenAI batch embedding call failed after retries "
                f"({type(error).__name__}: {error})"
            ) from error

        # OpenAI's API guarantees response.data is in the SAME ORDER as
        # the input list, so a direct list comprehension is safe here.
        return [item.embedding for item in response.data]

    def _embed_with_huggingface(self, text: str) -> list[float]:
        try:
            vector = self._client.encode(text, normalize_embeddings=True)
        except Exception as error:
            raise EmbeddingError(
                f"HuggingFace embedding call failed ({type(error).__name__}: {error})"
            ) from error

        return vector.tolist()

    def _embed_batch_with_huggingface(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._client.encode(texts, normalize_embeddings=True)
        except Exception as error:
            raise EmbeddingError(
                f"HuggingFace batch embedding call failed ({type(error).__name__}: {error})"
            ) from error

        return [vector.tolist() for vector in vectors]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """
    Returns a single, shared EmbeddingProvider instance for the whole app.

    WHY @lru_cache HERE, MATCHING THE PATTERN FROM settings.py's
    get_settings(): creating an EmbeddingProvider is cheap, but the
    underlying client it lazily creates (especially a HuggingFace
    SentenceTransformer, which loads a real ML model into memory) is
    NOT cheap. We want exactly ONE of these for the app's lifetime,
    not a new one constructed every time some function needs to embed text.
    """
    return EmbeddingProvider()
