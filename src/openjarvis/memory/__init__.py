"""Memory pillar — persistent searchable storage."""

from __future__ import annotations

# Always-available backend
import openjarvis.memory.sqlite  # noqa: F401

# Optional backends — import to trigger registration
try:
    import openjarvis.memory.bm25  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.memory.faiss_backend  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.memory.colbert_backend  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.memory.hybrid  # noqa: F401
except ImportError:
    pass

from openjarvis.memory._stubs import MemoryBackend, RetrievalResult
from openjarvis.memory.chunking import Chunk, ChunkConfig, chunk_text
from openjarvis.memory.context import ContextConfig, inject_context
from openjarvis.memory.ingest import ingest_path, read_document

__all__ = [
    "Chunk",
    "ChunkConfig",
    "ContextConfig",
    "MemoryBackend",
    "RetrievalResult",
    "chunk_text",
    "inject_context",
    "ingest_path",
    "read_document",
]
