import torch
from datasets import load_dataset
from qdrant_client.http.models import PointStruct
from data.qdrant_db.utils import create_collection, insert_to_collection
from data.qdrant_db.client import get_qdrant_client
from sae_lens import SAE
from transformer_lens import HookedTransformer
import numpy as np
import logging
logger = logging.getLogger(__name__)



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
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the pretrained sparse autoencoder and model
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="pythia-70m-deduped-res-sm",  # Pretrained autoencoder release
        sae_id="blocks.5.hook_resid_post", 
        device="cpu",
    )
    sae = sae.to(device)
    print("did we miss it??: ",cfg_dict) #all good? gonna run it mhm bru did it not print, am i bugging it did not yea no it did not print

    def preprocess_function(examples):
        # Tokenize the texts
        decoded_embeddings = []
        print("no of examples: ", len(examples['text'])) # is that right? yh i reckon cool ummmmm oh wait, I may have fucked up again lol where? Wait so map maos preprocess to every datapoint in the dataset - like what is examples 
        
        for text in examples['text']:
            print("text: ", text) # sanity check
            tokens = model.to_tokens(text, prepend_bos=True)

            # Generate embeddings, add noise, and decode
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
                dense_embeddings = cache[sae.cfg.hook_name]
                sparse_embeddings = sae.encode(dense_embeddings)
                noisy_sparse_embeddings = add_differential_privacy_noise(sparse_embeddings)
                decoded_embeddings_text = sae.decode(noisy_sparse_embeddings).cpu().numpy()
            
            # print("tokens per text:", tokens.shape)                  
            # print("dense per text:", dense_embeddings.shape)        
            # print("decoded per text:", decoded_embeddings_text.shape)
            decoded_embeddings.append(decoded_embeddings_text[0, -1, :])  # use last vector as the embedding as this is after last token has been passed in    
            # I'm here
        # trying to fix docker issues. I think this was caused when I restarted. SIGH gotchu
        #idk why it is not printing! maybe it is piping into the txt file? but that does not make sense because the program is not done running
       


        print("total no. of embedddings: ", len(decoded_embeddings))
        return {'noisy_decoded_embeddings': decoded_embeddings}

    
    

    # # Load dataset
    # print("Loading dataset...")
    # dataset_large = load_dataset("mteb/tweet_sentiment_extraction")
    # dataset = dataset_large.select(range(0,100))

    # # Preprocessing function
    # print("Processing dataset...")
    # # Apply preprocessing to the dataset
    # embedded_dataset = dataset.map(preprocess_function, batched=True, batch_size=32)


    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()

    # Insert data into Qdrant
    print("Inserting data into Qdrant...")

    for split in ['train', 'test']:
        collection_name = f"tweet_sentiment_noisy_decoded_vectors_{split}"

        create_collection(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            vector_dimensions=sae.cfg.d_in,  # Use the feature size from the SAE config
        )

        # Load dataset
        print("Loading dataset...")
        
        
        dataset = load_dataset("mteb/tweet_sentiment_extraction", split=split)
        # I added this so I could run quickly for testing, but change back for gpu runs
        # dataset_large = load_dataset("mteb/tweet_sentiment_extraction", split=split)
        # dataset = dataset_large.select(range(0,100)) # Select a subset of the dataset for testing

        print(dataset.info)

        # Preprocessing function
        print("Processing dataset...")
        # Apply preprocessing to the dataset
        embedded_dataset = dataset.map(preprocess_function, batched=True, batch_size=32)

        points = []
        for i, record in enumerate(embedded_dataset): # embeded_dataset[split]
            print("vector: ", record['noisy_decoded_embeddings']) # sanity check
            vector = record['noisy_decoded_embeddings'] 
            payload = {"label": record['label']}
            points.append(PointStruct(id=i, vector=vector, payload=payload))
    
        # Batch insert points into Qdrant
        BATCH_SIZE = 256
        for start_idx in range(0, len(points), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(points))
            batch = points[start_idx:end_idx]
            insert_to_collection(qdrant_client, collection_name, batch)
            print(f"Inserted points {start_idx} to {end_idx} into collection '{collection_name}'.")

if __name__ == "__main__":
    preprocess_and_store_noisy_decoded_embeddings()