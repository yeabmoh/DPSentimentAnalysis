import os
import numpy as np
from data.qdrant_db import get_qdrant_client, scroll_collection

def is_qdrant_data_empty(qdrant_data_path: str) -> bool:
    """
    Check if the Qdrant data directory is empty.

    Args:
        qdrant_data_path (str): Path to the Qdrant data directory.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    return not any(os.scandir(qdrant_data_path))

def load_data_from_qdrant(collection_name: str):
    """
    Load vectors and labels from Qdrant.

    Args:
        collection_name (str): Name of the Qdrant collection.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix (X) and labels (y).
    """
    # Get Qdrant client
    qdrant_client = get_qdrant_client()

    # Scroll through the collection to retrieve all points
    points = scroll_collection(qdrant_client, collection_name)
    print(f"Retrieved points: {points}")  # Debugging

    # Extract vectors and labels
    vectors = [point.vector for point in points]
    labels = [point.payload["label"] for point in points]  # Assuming "label" is stored in the payload

    if not all(len(v) == len(vectors[0]) for v in vectors):
        raise ValueError("Inconsistent vector lengths in the dataset.")
        
    return np.array(vectors), np.array(labels)