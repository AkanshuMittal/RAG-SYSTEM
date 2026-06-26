import hashlib
from dataclasses import dataclass, field

# NOTE ON THIS IMPORT — a real lesson, not just theory:
# Older LangChain tutorials (and your ORIGINAL prototype) import this as
# `from langchain.text_splitter import RecursiveCharacterTextSplitter`.
# That path is DEPRECATED in current LangChain versions — text splitters
# were moved into their own standalone package, `langchain_text_splitters`,
# so they could be released and versioned independently of the main
# langchain package. Using the old path either fails outright or emits
# a deprecation warning depending on the exact version installed.
# THE LESSON: pin your dependency versions in requirements.txt (we do,
# see Phase 8) and always import from the package the CURRENT docs
# recommend, not whatever an older tutorial shows.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from src.config import settings
from src.exceptions import ChunkingError
from src.ingestion.loader import LoadedDocument


@dataclass
class Chunk:
    """
    A single chunk of text, ready to be embedded and stored in Qdrant.

    WHY THIS IS A SEPARATE CLASS FROM LoadedDocument (defined in loader.py):
    A LoadedDocument represents ONE FULL PAGE. A Chunk represents a
    SMALLER PIECE of that page (since a page might be split into 2-3
    chunks if it's long, or multiple short pages might NOT be merged
    because we chunk per-page to keep page-number citations accurate).
    Keeping these as distinct types makes it impossible to accidentally
    pass a whole unchunked page where a chunk was expected.
    """

    text: str
    source_file: str
    page_number: int

    # Position of this chunk WITHIN its source page (0-indexed).
    # WHY WE TRACK THIS: if a page produces 3 chunks, this tells us
    # "chunk 0 of page 4" vs "chunk 1 of page 4" — useful for debugging
    # and for the parent-child retrieval pattern (see retrieval/retriever.py
    # in Phase 4), where we may want to fetch NEIGHBORING chunks for
    # more context around a matched chunk.
    chunk_index: int

    # A unique, stable ID for this exact chunk's content.
    # WHY THIS MATTERS FOR IDEMPOTENT INGESTION:
    # If you run ingestion twice on the same PDF, the SAME text content
    # will produce the SAME hash both times. ingest_pipeline.py uses this
    # to check "does Qdrant already have a chunk with this exact ID?" and
    # skips re-inserting it — this is what makes re-running ingestion safe
    # instead of creating duplicate vectors every time you run it.
    chunk_id: str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            # We hash source_file + page_number + chunk_index + text together.
            # WHY ALL FOUR, not just the text: if the SAME sentence appears
            # on two different pages (e.g. a repeated disclaimer), hashing
            # text alone would treat them as the same chunk and only store
            # ONE of them — losing the fact that it appears on both pages.
            # Including location info makes each chunk's ID unique to its
            # exact position in the source material.
            unique_string = f"{self.source_file}:{self.page_number}:{self.chunk_index}:{self.text}"
            self.chunk_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()


def _loaded_documents_to_langchain_documents(
    loaded_documents: list[LoadedDocument],
) -> list[LangChainDocument]:
    """
    Converts our LoadedDocument objects into LangChain's Document format.

    This is the ADAPTER step mentioned in the module docstring above —
    it's the ONLY place in this file where we touch LangChain's data
    structure directly.

    We put source_file and page_number into LangChain's `metadata` dict
    so that information survives the splitting process and we can read
    it back out afterward (see _langchain_documents_to_chunks below).
    """
    langchain_documents = []

    for loaded_document in loaded_documents:
        langchain_documents.append(
            LangChainDocument(
                page_content=loaded_document.text,
                metadata={
                    "source_file": loaded_document.source_file,
                    "page_number": loaded_document.page_number,
                },
            )
        )

    return langchain_documents


def _langchain_documents_to_chunks(
    split_documents: list[LangChainDocument],
) -> list[Chunk]:
    """
    Converts LangChain's split Document objects back into our own Chunk
    dataclass, reading the metadata we attached before splitting.

    WHY WE TRACK chunk_index PER PAGE (not globally across the whole file):
    We reset the counter every time the page_number changes, so the
    first chunk produced from each page is always "chunk_index=0",
    making it easy to reason about "which chunk number is this, within
    its own page" rather than an arbitrary running total across the
    entire document.
    """
    chunks: list[Chunk] = []

    # Tracks how many chunks we've seen so far for the CURRENT page,
    # so each page's chunks are numbered 0, 1, 2... independently.
    current_page_number: int | None = None
    chunk_index_within_page = 0

    for split_doc in split_documents:
        page_number = split_doc.metadata["page_number"]
        source_file = split_doc.metadata["source_file"]

        if page_number != current_page_number:
            # We've moved to a new page — reset the per-page counter.
            current_page_number = page_number
            chunk_index_within_page = 0

        chunks.append(
            Chunk(
                text=split_doc.page_content,
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index_within_page,
            )
        )

        chunk_index_within_page += 1

    return chunks


def create_chunks(loaded_documents: list[LoadedDocument]) -> list[Chunk]:
    """
    Splits loaded documents into smaller, embeddable chunks.

    Uses RecursiveCharacterTextSplitter, which tries to split on
    paragraph breaks first, then sentence breaks, then words — only
    falling back to a hard character cut if no natural boundary exists
    nearby. This is "structure-aware" in a lightweight sense: it
    respects natural language boundaries instead of cutting blindly
    every N characters regardless of where a sentence ends.

    WHY WE READ chunk_size/chunk_overlap FROM settings INSTEAD OF
    HARDCODING THEM HERE (like the original prototype did):
    See config/settings.py for the full reasoning on WHY 800/120 were
    chosen for this domain. The important architectural point is that
    this VALUE lives in exactly one place, so tuning it for an experiment
    is a one-line .env change, not a source code edit.

    Args:
        loaded_documents: Output from loader.py's load_document() calls.

    Returns:
        A list of Chunk objects, each carrying its source file, page
        number, position, and a stable content hash.

    Raises:
        ChunkingError: if the splitter produces zero chunks from
                       non-empty input (indicates a configuration bug,
                       e.g. chunk_size set to 0).
    """
    if not loaded_documents:
        # Not an error — an empty INPUT list is a valid (if unusual) case,
        # e.g. discover_files() found no files. We return an empty list
        # rather than raising, leaving the decision of whether "0 files
        # ingested" is a problem to the CALLER (ingest_pipeline.py),
        # which has more context about whether that's expected.
        return []

    langchain_documents = _loaded_documents_to_langchain_documents(loaded_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # WHY THIS SEPARATOR ORDER: RecursiveCharacterTextSplitter tries
        # each separator in order until chunks fit within chunk_size.
        # Paragraph breaks first (cleanest split), then line breaks,
        # then sentences, then words, then characters as a last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        split_documents = text_splitter.split_documents(langchain_documents)
    except Exception as error:
        raise ChunkingError(
            reason=f"text splitter failed ({type(error).__name__}: {error})"
        ) from error

    if len(split_documents) == 0:
        raise ChunkingError(
            reason=(
                f"splitter produced 0 chunks from {len(loaded_documents)} "
                f"input document(s) — check chunk_size configuration"
            )
        )

    chunks = _langchain_documents_to_chunks(split_documents)

    return chunks


def filter_low_value_chunks(chunks: list[Chunk], min_length: int = 20) -> list[Chunk]:
    """
    Removes chunks that are too short to carry meaningful information.

    WHY THIS EXISTS: page headers, footers, page numbers, or stray
    whitespace sometimes survive as their own tiny "chunk" (e.g. a
    chunk containing only "Page 3" or a copyright footer). These add
    noise to the vector store without adding retrieval value — they
    can occasionally even get retrieved by accident due to a spurious
    similarity match, displacing a genuinely useful chunk from the
    top-k results.

    Args:
        chunks: Chunks to filter.
        min_length: Minimum character length to keep a chunk. Default
                    of 20 is intentionally low — we only want to catch
                    truly trivial fragments, not aggressively discard
                    short-but-meaningful content.

    Returns:
        Chunks with length >= min_length.
    """
    return [chunk for chunk in chunks if len(chunk.text) >= min_length]
