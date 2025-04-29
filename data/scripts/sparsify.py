import torch
from datasets import load_dataset
from qdrant_client.http.models import PointStruct
from data.qdrant_db.utils import create_collection, insert_to_collection
from data.qdrant_db.client import get_qdrant_client
from sae_lens import SAE
from transformer_lens import HookedTransformer
import numpy as np

def add_differential_privacy_noise(embeddings, epsilon=1.0):
    """
    Add differentially private noise to embeddings.
    Args:
        embeddings (torch.Tensor): The sparse embeddings.
        epsilon (float): Privacy budget parameter.
    Returns:
        torch.Tensor: Noisy embeddings.
    """
    sensitivity = 1.0  # Sensitivity of the embeddings
    scale = sensitivity / epsilon
    noise = torch.normal(mean=0, std=scale, size=embeddings.shape, device=embeddings.device)
    return embeddings + noise

def preprocess_and_store_noisy_decoded_embeddings():
    """
    Preprocess the dataset, generate sparse embeddings using pythia-70m-deduped,
    add differentially private noise, decode them back to the original basis,
    and store them in a Qdrant collection.
    """
    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()

    # Define target collection
    collection_name = "tweet_sentiment_noisy_decoded_vectors_train"

    # Load the pretrained sparse autoencoder and model
    model = HookedTransformer.from_pretrained("pythia-70m-deduped")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="pythia-70m-deduped",  # Pretrained autoencoder release
        sae_id="blocks.8.hook_resid_pre",  # Hook point for the autoencoder
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create a new collection for noisy decoded embeddings
    create_collection(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        vector_dimensions=sae.cfg.feature_size,  # Use the feature size from the SAE config
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("mteb/tweet_sentiment_extraction")

    # Preprocessing function
    print("Processing dataset...")
    def preprocess_function(examples):
        # Tokenize the texts
        tokens = model.to_tokens(examples['text'], prepend_bos=True)

        # Generate embeddings, add noise, and decode
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            dense_embeddings = cache[sae.cfg.hook_name]
            sparse_embeddings = sae.encode(dense_embeddings)
            noisy_sparse_embeddings = add_differential_privacy_noise(sparse_embeddings)
            decoded_embeddings = sae.decode(noisy_sparse_embeddings).cpu().numpy()

        return {'noisy_decoded_embeddings': decoded_embeddings}

    # Apply preprocessing to the dataset
    embedded_dataset = dataset.map(preprocess_function, batched=True, batch_size=32)

    # Insert data into Qdrant
    print("Inserting data into Qdrant...")
    for split in ['train', 'test']:
        points = []
        for i, record in enumerate(embedded_dataset[split]):
            vector = record['noisy_decoded_embeddings'].tolist()  # Convert numpy array to list
            payload = {"label": record['label']}
            points.append(PointStruct(id=i, vector=vector, payload=payload))

        # Batch insert points into Qdrant
        insert_to_collection(qdrant_client, collection_name, points)
        print(f"Data successfully stored in Qdrant collection '{collection_name}' for split '{split}'.")

if __name__ == "__main__":
    preprocess_and_store_noisy_decoded_embeddings()