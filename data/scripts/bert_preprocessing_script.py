import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Run these commands in your terminal in this order:
# 2. docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant

from datasets import load_dataset
from transformers import BertTokenizerFast
from qdrant_client.http.models import PointStruct
from data.qdrant_db import get_qdrant_client, create_collection, insert_to_collection
from data.settings import settings

def preprocess_and_store_data():
    """
    Preprocess the dataset and store it in Qdrant.
    """
    # Load dataset
    dataset = load_dataset("mteb/tweet_sentiment_extraction")

    # Initialize BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    # Apply tokenizer to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Optional: Remove the raw text if you're just training
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])

    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()

    for split in ['train', 'test']:
        # Define Qdrant collection names
        collection_name = f"tweet_sentiment_vectors_{split}"

        # Create Qdrant collection
        create_collection(qdrant_client, collection_name)

        # Prepare and insert data into Qdrant
        points = []
        for i, record in enumerate(tokenized_dataset[split]):
            vector = record['input_ids']  # Use input_ids as the vector
            # Ensure vector is a list of floats
            vector = [float(v) for v in vector]
            payload = {"label": record['label']}  # Store label as metadata
            # print(f"Processing record {i}: vector={vector}, payload={payload}")
            # Create PointStruct object
            points.append(PointStruct(id=i, vector=vector, payload=payload))

        # Batch insert points into Qdrant
        insert_to_collection(qdrant_client, collection_name, points)
        print(f"Data successfully stored in Qdrant collection '{collection_name}'.")