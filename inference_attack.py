import argparse
import torch
import numpy as np
from qdrant_client.http.models import PointStruct
from data.qdrant_db.client import get_qdrant_client
from sklearn.metrics.pairwise import cosine_similarity

# -----------------HIGHKEY IGNORE FOR NOW------------------

def load_sparsified_data(collection_name, qdrant_client):
    """
    Load sparsified embeddings from the Qdrant collection.
    """
    points = qdrant_client.scroll(collection_name=collection_name, limit=1000)
    embeddings = []
    labels = []
    for point in points:
        embeddings.append(point.vector)
        labels.append(point.payload['label'])
    return np.array(embeddings), np.array(labels)


def perform_inference_attack(embeddings, labels):
    """
    Perform an inference attack on the sparsified embeddings.
    """
    # Example attack: Measure similarity between embeddings and infer labels
    similarity_matrix = cosine_similarity(embeddings)
    inferred_labels = np.argmax(similarity_matrix, axis=1)

    # Evaluate the attack
    accuracy = np.mean(inferred_labels == labels)
    print(f"Inference Attack Accuracy: {accuracy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="Name of the Qdrant collection containing sparsified embeddings."
    )
    args = parser.parse_args()

    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()

    # Load sparsified data
    print("Loading sparsified data...")
    embeddings, labels = load_sparsified_data(args.collection_name, qdrant_client)

    # Perform inference attack
    print("Performing inference attack...")
    perform_inference_attack(embeddings, labels)


if __name__ == "__main__":
    main()