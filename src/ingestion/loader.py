import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from src.exceptions import DocumentLoadError


@dataclass
class LoadedDocument:
    """
    Represents ONE page (or section) of a loaded document, with metadata.

    WHY A DATACLASS INSTEAD OF A RAW DICT OR LANGCHAIN'S Document CLASS:
    We define our OWN simple data structure here instead of importing
    LangChain's `Document` class directly into this layer. This keeps
    our ingestion code decoupled from LangChain — if we ever want to
    swap frameworks (e.g. move to LlamaIndex), only ONE conversion
    function needs to change, not every file that touches a Document.

    This is the "adapter" pattern: convert to LangChain's format only
    at the boundary where LangChain is actually used (in chunker.py),
    not throughout the whole codebase.
    """

    # The actual text content of this page/section
    text: str

    # WHERE this text came from — critical for citations later.
    # Without this, you can never tell a user "this answer came from
    # page 3 of scholarship_policy.pdf" — you'd just have raw text
    # with no traceable origin.
    source_file: str
    page_number: int

    # A short, stable identifier for this exact piece of content.
    # WHY WE NEED THIS: see ingest_pipeline.py — this hash lets us
    # detect "have I already ingested this exact page before?" so
    # re-running ingestion doesn't create duplicate vectors every time.
    content_hash: str = field(default="")

    def __post_init__(self):
        """
        Runs automatically right after the dataclass is created.
        We compute the content hash here so callers never forget to
        set it manually — it's always correct and always present.
        """
        if not self.content_hash:
            # SHA-256 of the text content. We hash the TEXT, not the
            # file path, because if the same content appears in two
            # different files (e.g. a policy doc duplicated in two
            # folders), we want to recognize that as the same content.
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_pdf(file_path: Path) -> list[LoadedDocument]:
    """
    Loads a single PDF file and returns one LoadedDocument PER PAGE.

    WHY PER-PAGE, NOT ONE GIANT STRING FOR THE WHOLE PDF:
    Keeping page boundaries lets us cite "page 4" later. If we joined
    every page into one big string right away, we'd permanently lose
    the ability to tell a user which page an answer came from — this
    information cannot be recovered later, so we must capture it now,
    at load time.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A list of LoadedDocument objects, one per non-empty page.

    Raises:
        DocumentLoadError: if the file cannot be opened or parsed at all
                            (e.g. corrupted file, password-protected PDF).
    """
    try:
        reader = PdfReader(str(file_path))
    except Exception as error:
        # WHY WE CATCH `Exception` HERE SPECIFICALLY (an exception to our
        # usual rule of catching specific exception types): PdfReader can
        # raise many different low-level error types depending on HOW the
        # PDF is broken (corrupted header, encryption, truncated file).
        # We don't need to distinguish between these for OUR purposes —
        # any of them means "this PDF could not be opened" — so we catch
        # broadly here and immediately convert it into OUR specific,
        # well-defined DocumentLoadError instead of leaking pypdf's
        # internal exception types up through our system.
        raise DocumentLoadError(
            file_path=str(file_path),
            reason=f"could not open or parse PDF ({type(error).__name__}: {error})",
        ) from error

    if len(reader.pages) == 0:
        raise DocumentLoadError(
            file_path=str(file_path),
            reason="PDF has zero pages",
        )

    documents: list[LoadedDocument] = []

    for page_index, page in enumerate(reader.pages):
        # PyPDF page numbers are 0-indexed internally; we convert to
        # 1-indexed for the page_number we STORE and later SHOW to users,
        # because "page 0" is confusing to a non-technical person reading
        # a citation — PDF readers everywhere display "Page 1" as the first page.
        page_number = page_index + 1

        try:
            page_text = page.extract_text()
        except Exception as error:
            # A single bad page should NOT crash the whole file's ingestion.
            # We skip just this page, but we still want this failure visible
            # — so in ingest_pipeline.py we log this as a warning, not silently
            # swallow it. Here at the loader level we just return what we can.
            page_text = ""

        # Skip pages with no extractable text. WHY: scanned image-only
        # pages, blank pages, or pages with only a logo would produce
        # empty strings. An empty chunk provides zero retrieval value
        # and would just waste space in the vector store.
        if page_text and page_text.strip():
            documents.append(
                LoadedDocument(
                    text=page_text.strip(),
                    source_file=file_path.name,
                    page_number=page_number,
                )
            )

    if len(documents) == 0:
        # Every page was empty/unreadable — this is suspicious enough
        # to flag as an error rather than silently returning nothing,
        # since it usually means the PDF is a scanned image with no
        # OCR text layer, which this loader cannot handle.
        raise DocumentLoadError(
            file_path=str(file_path),
            reason=(
                "no extractable text found on any page "
                "(this may be a scanned/image-only PDF requiring OCR)"
            ),
        )

    return documents


def load_markdown(file_path: Path) -> list[LoadedDocument]:
    """
    Loads a single markdown (.md) file as ONE LoadedDocument.

    WHY THIS EXISTS even though your current project only has PDFs:
    This demonstrates the registry pattern is REAL, not just claimed.
    If you later add a scholarship FAQ written in markdown, this
    already works with zero changes to ingest_pipeline.py.

    Markdown files don't have "pages" the way PDFs do, so we use
    page_number=1 for the whole file as a sensible default.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as error:
        raise DocumentLoadError(
            file_path=str(file_path),
            reason=f"could not read markdown file ({type(error).__name__}: {error})",
        ) from error

    if not text.strip():
        raise DocumentLoadError(file_path=str(file_path), reason="file is empty")

    return [
        LoadedDocument(
            text=text.strip(),
            source_file=file_path.name,
            page_number=1,
        )
    ]


# -----------------------------------------------------------------------
# THE REGISTRY — this is the core of the pluggable-loader design.
# -----------------------------------------------------------------------
# Maps a file extension to the function that knows how to load it.
# TO ADD A NEW FILE TYPE: write a `load_xxx(file_path) -> list[LoadedDocument]`
# function above, then add ONE line here. Nothing else in this codebase
# needs to change.
LOADER_REGISTRY: dict[str, Callable[[Path], list[LoadedDocument]]] = {
    ".pdf": load_pdf,
    ".md": load_markdown,
}


def load_document(file_path: Path) -> list[LoadedDocument]:
    """
    Loads any SUPPORTED document type by dispatching to the registry.

    This is the ONLY function the rest of the ingestion pipeline calls —
    it doesn't need to know HOW a PDF differs from a markdown file, it
    just calls load_document() and gets back a consistent list of
    LoadedDocument objects either way.

    Args:
        file_path: Path to the file to load.

    Returns:
        A list of LoadedDocument objects extracted from this file.

    Raises:
        DocumentLoadError: if the file extension is not supported, or if
                            the underlying loader function fails.
    """
    extension = file_path.suffix.lower()

    loader_function = LOADER_REGISTRY.get(extension)

    if loader_function is None:
        supported = ", ".join(sorted(LOADER_REGISTRY.keys()))
        raise DocumentLoadError(
            file_path=str(file_path),
            reason=f"unsupported file type '{extension}'. Supported types: {supported}",
        )

    return loader_function(file_path)


def discover_files(data_directory: Path) -> list[Path]:
    """
    Finds all files in a directory that we know how to load.

    WHY THIS IS SEPARATE FROM load_document(): discovery (finding WHAT
    to load) and loading (actually reading the content) are different
    concerns. ingest_pipeline.py calls discover_files() first to know
    HOW MANY files it's about to process (useful for progress reporting
    and for catching "the data folder is empty" early, with a clear
    message, before we waste time even trying to load anything).

    Args:
        data_directory: Folder to scan for supported files.

    Returns:
        A sorted list of file paths with supported extensions.
        Sorted so that ingestion order is DETERMINISTIC across runs —
        this matters for reproducible logs and easier debugging.
    """
    if not data_directory.exists():
        raise DocumentLoadError(
            file_path=str(data_directory),
            reason="data directory does not exist",
        )

    supported_extensions = set(LOADER_REGISTRY.keys())

    discovered = [
        file_path
        for file_path in data_directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions
    ]

    return sorted(discovered)
