"""Utilities for serializing retrieved document chunks."""

from typing import List

from langchain_core.documents import Document


def serialize_chunks(docs: List[Document]) -> str:
    """Serialize a list of Document objects into a formatted CONTEXT string.

    Formats chunks with indices and page numbers as specified in the PRD:
    - Chunks are numbered (Chunk 1, Chunk 2, etc.)
    - Page numbers are included in the format "page=X"
    - Produces a clean CONTEXT section for agent consumption

    Args:
        docs: List of Document objects with metadata.

    Returns:
        Formatted string with all chunks serialized.
    """
    context_parts = []

    for idx, doc in enumerate(docs, start=1):
        # Extract page number from metadata
        page_num = doc.metadata.get("page") or doc.metadata.get(
            "page_number", "unknown"
        )

        # Format chunk with index and page number
        chunk_header = f"Chunk {idx} (page={page_num}):"
        chunk_content = doc.page_content.strip()

        context_parts.append(f"{chunk_header}\n{chunk_content}")

    return "\n\n".join(context_parts)



def serialize_chunks_with_ids(docs: list[Document]) -> tuple[str, dict]:
    """
    Returns:
    - context string with chunk IDs
    - citation map for UI / API
    """
    context_parts = []
    citation_map = {}

    for i, doc in enumerate(docs):
        chunk_id = f"C{i+1}"
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"[{chunk_id}] (Page {page})\n{doc.page_content}"
        )

        citation_map[chunk_id] = {
            "page": page,
            "source": source,
            "snippet": doc.page_content[:120] + "..."
        }

    return "\n\n".join(context_parts), citation_map
