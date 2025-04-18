import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from typing import List

from data.settings import settings

logger = logging.getLogger(__name__)

def create_collection(qdrant_client: QdrantClient,
                      collection_name: str,
                      vector_dimensions: int = settings.QDRANT_VECTOR_DIMENSIONS
                ):
    """Creates new collection if it doesn't exist"""

    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Creating new Qdrant collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dimensions,
                distance=Distance.COSINE
            )
        )

def insert_to_collection(qdrant_client: QdrantClient,
                        collection_name: str,
                        points: List[PointStruct]
                ):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully inserted {len(points)} points into collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Error inserting content: {e}")
        raise

def scroll_collection(client: QdrantClient, collection_name: str, limit: int = 10000) -> List[PointStruct]:
    """
    Scroll through a Qdrant collection to retrieve all points.

    Args:
        client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the Qdrant collection.
        limit (int): Maximum number of points to retrieve in one scroll.

    Returns:
        List[PointStruct]: List of points retrieved from the collection.
    """
    response = client.scroll(collection_name=collection_name, with_payload=True, with_vectors=True, limit=limit)
    return response[0]  # response[0] contains the points


