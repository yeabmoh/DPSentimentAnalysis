
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datasets import load_dataset
from transformers import BertTokenizerFast, BertModel
from qdrant_client.http.models import PointStruct
from data.qdrant_db import get_qdrant_client, create_collection, insert_to_collection
from data.settings import settings


def preprocess_and_store_data():
    """
    Preprocess the dataset and store it in Qdrant.
    """
    # Load dataset
    dataset = load_dataset("mteb/tweet_sentiment_extraction")

    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    import torch

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Preprocessing function
    def preprocess_function(examples):
        inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings = embeddings[:, :128]  
        return {'embeddings': embeddings}


    embedded_dataset = dataset.map(preprocess_function, batched=True, batch_size=32)

    qdrant_client = get_qdrant_client()

    for split in ['train', 'test']:
        collection_name = f"tweet_sentiment_vectors_{split}"

        # Create Qdrant collection with correct vector size (768 for BERT base)
        create_collection(qdrant_client, collection_name, vector_dimensions=768)

        points = []
        for i, record in enumerate(embedded_dataset[split]):
            vector = record['embeddings']
            vector = [float(v) for v in vector]  # Ensure all values are float
            payload = {"label": record['label']}
            points.append(PointStruct(id=i, vector=vector, payload=payload))

        # Batch insert points into Qdrant
        BATCH_SIZE = 256
        for start_idx in range(0, len(points), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(points))
            batch = points[start_idx:end_idx]
            insert_to_collection(qdrant_client, collection_name, batch)
            print(f"Inserted points {start_idx} to {end_idx} into collection '{collection_name}'.")