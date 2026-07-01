from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.exceptions import CollectionNotFoundError, VectorStoreConnectionError, VectorStoreError
from src.ingestion.chunker import Chunk
from src.vectorstore.embeddings import get_embedding_provider


@dataclass
class SearchResult:
    """
    A single retrieved chunk, with its similarity score and original
    metadata, returned from a vector search.

    WHY WE DEFINE OUR OWN SearchResult INSTEAD OF RETURNING QDRANT'S
    RAW ScoredPoint OBJECTS DIRECTLY: same adapter principle as Chunk
    and LoadedDocument — callers in retrieval/retriever.py (Phase 4)
    should work with OUR clean data structure, not need to know Qdrant's
    internal response shape (which payload keys exist, how scores are
    nested, etc).
    """

    chunk_id: str
    text: str
    source_file: str
    page_number: int
    score: float


class VectorStore:
    """
    Wraps Qdrant operations: collection setup, inserting chunks,
    and similarity search.
    """

    def __init__(self):
        self._client = self._create_client()
        self._collection_name = settings.qdrant_collection_name
        self._embedding_provider = get_embedding_provider()

    def _create_client(self) -> QdrantClient:
        """
        Creates the underlying Qdrant client.

        WHY WE DETECT LOCAL-FILE-MODE VS SERVER-MODE HERE:
        Qdrant's Python client supports two distinct modes: connecting
        to a REAL Qdrant SERVER over HTTP (via `url=`), or running
        Qdrant ENTIRELY LOCALLY with no server at all, persisting to a
        folder on disk (via `path=`). These are different constructor
        arguments — passing a filesystem path as `url` does not work.
        We treat any qdrant_url value that does NOT start with "http"
        as a local folder path, which lets you develop and test this
        entire system with ZERO infrastructure (no Docker, no server)
        by simply setting QDRANT_URL=./local_qdrant_data in your .env,
        then switching to a real http:// URL for staging/production
        with no code changes.

        WHY WE TRY/EXCEPT AROUND CLIENT CREATION even though connecting
        to Qdrant's REMOTE server doesn't actually happen until the
        first real operation (QdrantClient's constructor doesn't ping
        the server immediately for a URL-based client) — for the LOCAL
        file-based mode, construction CAN fail immediately if the
        storage path is invalid or locked.
        """
        try:
            if settings.qdrant_url.startswith("http"):
                return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            else:
                # Local file-based mode — no server, no network, pure disk storage.
                return QdrantClient(path=settings.qdrant_url)
        except Exception as error:
            raise VectorStoreConnectionError(
                qdrant_url=settings.qdrant_url,
                reason=f"{type(error).__name__}: {error}",
            ) from error

    def ensure_collection_exists(self) -> None:
        """
        Creates the Qdrant collection if it doesn't already exist.

        WHY THIS IS IDEMPOTENT (safe to call every time, not just once):
        Calling this at app startup every time means a fresh environment
        (first-ever run) gets the collection created automatically,
        while an existing environment with the collection already
        present is a harmless no-op. This removes an entire category of
        "did someone remember to run the setup script" deployment bugs.

        Raises:
            VectorStoreConnectionError: if Qdrant is unreachable.
            VectorStoreError: if collection creation fails for any
                               other reason (e.g. invalid vector config).
        """
        try:
            existing_collections = {
                collection.name for collection in self._client.get_collections().collections
            }
        except Exception as error:
            raise VectorStoreConnectionError(
                qdrant_url=settings.qdrant_url,
                reason=f"could not list collections ({type(error).__name__}: {error})",
            ) from error

        if self._collection_name in existing_collections:
            return  # already exists — nothing to do

        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    # WHY COSINE DISTANCE: cosine similarity measures the
                    # ANGLE between two vectors, ignoring their magnitude.
                    # This is the standard choice for text embeddings,
                    # because embedding magnitude often reflects text
                    # LENGTH rather than meaning — we want "are these two
                    # chunks talking about the same thing", not "which
                    # chunk produced a longer/larger vector".
                    distance=Distance.COSINE,
                ),
            )
        except Exception as error:
            raise VectorStoreError(
                f"failed to create collection '{self._collection_name}': "
                f"{type(error).__name__}: {error}"
            ) from error

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embeds and inserts (or updates) chunks into Qdrant.

        WHY "UPSERT" (insert-or-update) RATHER THAN A PLAIN INSERT:
        This is what makes ingestion IDEMPOTENT — re-running ingestion
        on the same files produces chunks with the SAME chunk_id (see
        chunker.py's content-hash-based ID generation). Qdrant's upsert
        operation will simply OVERWRITE the existing point with the
        same ID rather than creating a duplicate. Re-running ingestion
        ten times in a row leaves you with the same data as running it
        once — no duplicate vectors piling up.

        Args:
            chunks: Chunks produced by the ingestion pipeline (Phase 2).

        Returns:
            The number of chunks successfully upserted.

        Raises:
            VectorStoreError: if the upsert operation fails.
        """
        if not chunks:
            return 0

        self.ensure_collection_exists()

        # Batch-embed all chunk texts in ONE call rather than one at a
        # time — see embeddings.py's embed_batch() docstring for why
        # this matters for performance.
        texts = [chunk.text for chunk in chunks]
        vectors = self._embedding_provider.embed_batch(texts)

        points = [
            PointStruct(
                # WHY chunk_id NEEDS CONVERTING: Qdrant point IDs must be
                # either an unsigned integer or a valid UUID string — our
                # chunk_id is a SHA-256 hex digest, which is neither. We
                # convert it into a UUID deterministically (the SAME
                # input hash always produces the SAME UUID) so upserts
                # remain idempotent even after this conversion step.
                id=_chunk_id_to_qdrant_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        try:
            self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as error:
            raise VectorStoreError(
                f"failed to upsert {len(points)} chunks: {type(error).__name__}: {error}"
            ) from error

        return len(points)

    def search(self, query_text: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Finds the most semantically similar chunks to a query.

        This is PLAIN DENSE search only — hybrid search (combining this
        with BM25 keyword search) and reranking are built on TOP of this
        function in Phase 4's retriever.py. This function's job is just
        "talk to Qdrant correctly", not "implement the full retrieval
        strategy".

        Args:
            query_text: The user's question (or any text to search for).
            top_k: How many results to return. Defaults to
                   settings.retrieval_top_k if not specified.

        Returns:
            A list of SearchResult objects, ordered by similarity
            (most similar first).

        Raises:
            CollectionNotFoundError: if ingestion has never been run.
            VectorStoreError: for other Qdrant failures.
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        query_vector = self._embedding_provider.embed(query_text)

        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=top_k,
            ).points
        except UnexpectedResponse as error:
            # Qdrant's SERVER mode returns a 404-style UnexpectedResponse
            # when you query a collection that doesn't exist.
            if "404" in str(error) or "doesn't exist" in str(error).lower():
                raise CollectionNotFoundError(self._collection_name) from error
            raise VectorStoreError(
                f"search failed: {type(error).__name__}: {error}"
            ) from error
        except ValueError as error:
            # Qdrant's LOCAL FILE-BASED mode (used for development/testing
            # without a server, see _create_client() above) raises a plain
            # ValueError for the SAME underlying problem instead of an
            # HTTP-style exception. We check the message text to detect
            # this specific case and translate it the same way, so callers
            # get a consistent CollectionNotFoundError regardless of which
            # Qdrant mode is running underneath — they shouldn't have to
            # care about this implementation detail.
            if "not found" in str(error).lower():
                raise CollectionNotFoundError(self._collection_name) from error
            raise VectorStoreError(
                f"search failed: {type(error).__name__}: {error}"
            ) from error
        except Exception as error:
            raise VectorStoreError(
                f"search failed: {type(error).__name__}: {error}"
            ) from error

        return [
            SearchResult(
                chunk_id=point.payload["chunk_id"],
                text=point.payload["text"],
                source_file=point.payload["source_file"],
                page_number=point.payload["page_number"],
                score=point.score,
            )
            for point in results
        ]

    def search_with_filter(
        self,
        query_text: str,
        source_file: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Same as search(), but restricted to chunks from one specific
        source file.

        WHY THIS EXISTS NOW, EVEN THOUGH RBAC FILTERING IS A PHASE 6
        FEATURE: this method demonstrates that Qdrant's PAYLOAD
        FILTERING genuinely works end-to-end, using a simple, concrete
        filter (by source file) we can test today. Phase 6's RBAC
        access_control.py will use this EXACT same underlying mechanism
        — filtering by a "visibility" payload field instead of
        "source_file" — so proving this works now de-risks that future work.
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        query_vector = self._embedding_provider.embed(query_text)

        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
                ),
                limit=top_k,
            ).points
        except Exception as error:
            raise VectorStoreError(
                f"filtered search failed: {type(error).__name__}: {error}"
            ) from error

        return [
            SearchResult(
                chunk_id=point.payload["chunk_id"],
                text=point.payload["text"],
                source_file=point.payload["source_file"],
                page_number=point.payload["page_number"],
                score=point.score,
            )
            for point in results
        ]

    def count_points(self) -> int:
        """Returns how many chunks are currently stored — useful for health checks and tests."""
        try:
            result = self._client.count(collection_name=self._collection_name)
            return result.count
        except Exception:
            return 0

    def health_check(self) -> bool:
        """
        Returns True if Qdrant is reachable, False otherwise.

        WHY THIS RETURNS A BOOLEAN INSTEAD OF RAISING: this method is
        designed to be called from the FastAPI /health endpoint (Phase
        8), where we want a simple "is everything OK?" signal rather
        than needing to catch an exception just to check connectivity.
        """
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


def _chunk_id_to_qdrant_point_id(chunk_id: str) -> str:
    """
    Converts our SHA-256 hex chunk_id into a UUID Qdrant will accept.

    WHY uuid5 (NAME-BASED, DETERMINISTIC) INSTEAD OF uuid4 (RANDOM):
    uuid5 generates the SAME UUID every time for the SAME input string.
    This is essential for idempotency — if we used random UUIDs, every
    re-run of ingestion would generate a NEW random ID for the same
    chunk content, defeating the entire purpose of upsert-based
    deduplication.
    """
    import uuid

    # Using a fixed namespace UUID (Qdrant has no opinion on which one)
    # just needs to be CONSISTENT across calls, which a module-level
    # constant guarantees.
    NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(NAMESPACE, chunk_id))
