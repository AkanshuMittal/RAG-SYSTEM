from dataclasses import dataclass, field
from pathlib import Path
from src.exceptions import DocumentLoadError, IngestionError
from src.ingestion.chunker import Chunk, create_chunks, filter_low_value_chunks
from src.ingestion.loader import discover_files, load_document
@dataclass
class FileIngestionResult:
    """Result of attempting to ingest ONE file — success or failure, with details."""

    file_path: str
    success: bool
    pages_loaded: int = 0
    chunks_produced: int = 0
    error_message: str | None = None

@dataclass
class IngestionReport:
    """
    Summary of an entire ingestion run across all files in the data directory.

    WHY WE RETURN A STRUCTURED REPORT INSTEAD OF JUST THE CHUNKS:
    If ingest_pipeline.py only returned `list[Chunk]`, you would have NO
    visibility into whether something went wrong. Returning a report
    object lets the CALLER (scripts/ingest.py, or the API's /ingest
    endpoint in Phase 8) decide what to do with failures — log them,
    alert someone, or just print a friendly summary — without the
    pipeline itself needing to know about logging or HTTP responses.
    """

    total_files_discovered: int = 0
    successful_files: list[FileIngestionResult] = field(default_factory=list)
    failed_files: list[FileIngestionResult] = field(default_factory=list)
    all_chunks: list[Chunk] = field(default_factory=list)

    @property
    def total_chunks_produced(self) -> int:
        return len(self.all_chunks)

    @property
    def success_count(self) -> int:
        return len(self.successful_files)

    @property
    def failure_count(self) -> int:
        return len(self.failed_files)

    def summary_text(self) -> str:
        """
        Human-readable summary, suitable for printing to console or
        logging. Kept as a method (not a __str__ override) so it's
        clear this is for DISPLAY purposes, not for debugging the
        object's internal state (which Python's default repr handles).
        """
        lines = [
            f"Ingestion complete: {self.total_files_discovered} file(s) discovered",
            f"  Succeeded: {self.success_count}",
            f"  Failed:    {self.failure_count}",
            f"  Total chunks produced: {self.total_chunks_produced}",
        ]

        if self.failed_files:
            lines.append("\nFailed files:")
            for failed in self.failed_files:
                lines.append(f"  - {failed.file_path}: {failed.error_message}")

        return "\n".join(lines)


def _ingest_single_file(file_path: Path) -> tuple[FileIngestionResult, list[Chunk]]:
    """
    Attempts to load and chunk ONE file, catching any failure locally
    so it does not propagate up and kill the entire batch.

    WHY THIS IS A PRIVATE HELPER (leading underscore) rather than part
    of run_ingestion_pipeline() directly: extracting the "try one file"
    logic into its own function makes run_ingestion_pipeline() read as
    a clean loop ("for each file, try to ingest it") instead of mixing
    the looping logic with the try/except details.

    Returns:
        A tuple of (result, chunks). `chunks` is an empty list if the
        file failed — the result.success flag tells you which case you're in.
    """
    try:
        loaded_pages = load_document(file_path)
    except DocumentLoadError as error:
        # We catch the SPECIFIC exception type here, not bare Exception.
        # WHY THIS MATTERS: if some OTHER unexpected error type occurred
        # (e.g. a bug in our own code causing an AttributeError), we want
        # THAT to propagate up loudly and crash visibly during development
        # and testing — not get silently swallowed and misreported as
        # "just a normal file loading failure". Only errors we EXPLICITLY
        # designed for (DocumentLoadError) get this graceful treatment.
        result = FileIngestionResult(
            file_path=str(file_path),
            success=False,
            error_message=error.message,
        )
        return result, []

    try:
        chunks = create_chunks(loaded_pages)
        chunks = filter_low_value_chunks(chunks)
    except IngestionError as error:
        result = FileIngestionResult(
            file_path=str(file_path),
            success=False,
            pages_loaded=len(loaded_pages),
            error_message=error.message,
        )
        return result, []

    result = FileIngestionResult(
        file_path=str(file_path),
        success=True,
        pages_loaded=len(loaded_pages),
        chunks_produced=len(chunks),
    )

    return result, chunks


def run_ingestion_pipeline(data_directory: Path | None = None) -> IngestionReport:
    """
    Runs the full ingestion pipeline: discover files -> load -> chunk ->
    filter, across an entire directory, with per-file fault isolation.

    This function is the public interface to "ingest everything in the
    data folder".

    Args:
        data_directory: Folder containing source documents. Defaults to
                         settings.data_dir if not provided, so callers
                         don't need to know the path — it's centrally
                         configured (see Phase 1's settings.py).

    Returns:
        An IngestionReport summarizing successes, failures, and all
        chunks produced (ready to be embedded and stored in Qdrant —
        that hand-off happens in Phase 3).

    Raises:
        IngestionError: only if NO files were discovered at all — this
                         is the one case we treat as a hard failure of
                         the whole run, rather than a per-file failure,
                         because there is nothing at all to report.
    """
    from src.config import settings

    if data_directory is None:
        data_directory = settings.data_dir

    discovered_files = discover_files(data_directory)

    if len(discovered_files) == 0:
        raise IngestionError(
            message=(
                f"No supported files found in '{data_directory}'. "
                f"Add PDF or markdown files to this folder before running ingestion."
            ),
            error_code="NO_FILES_FOUND",
        )

    report = IngestionReport(total_files_discovered=len(discovered_files))

    for file_path in discovered_files:
        result, chunks = _ingest_single_file(file_path)

        if result.success:
            report.successful_files.append(result)
            report.all_chunks.extend(chunks)
        else:
            report.failed_files.append(result)

    return report
