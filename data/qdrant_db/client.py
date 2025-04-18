from qdrant_client import QdrantClient
from data.settings import settings

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=settings.QDRANT_TIMEOUT
    )