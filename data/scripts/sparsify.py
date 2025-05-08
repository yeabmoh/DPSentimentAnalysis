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



def calculate_sensitivity(embeddings): # Estimated this by running a few and taking a max
    """
    Calculate the global sensitivity of the dataset.

    Args:
        embeddings (torch.Tensor): A tensor of shape (n_samples, embedding_dim) containing all embeddings.

    Returns:
        float: The global sensitivity (maximum Euclidean distance between any two embeddings).
    """
    """
    Calculate the global sensitivity of the dataset using vectorized operations.
    """
    distances = torch.cdist(embeddings, embeddings, p=2)  # Compute pairwise distances
    max_distance = distances.max().item()  # Find the maximum distance
    print(f"Sensitivity: {max_distance}")
    return max_distance


def add_differential_privacy_noise(embeddings, epsilon=1.0, sensitivity=0.0, word_embeddings=None):
    """
    Add differentially private noise to embeddings using the MADLIB mechanism.

    Args:
        embeddings (torch.Tensor): The sparse embeddings to privatize.
        epsilon (float): Privacy budget parameter.
        word_embeddings (torch.Tensor): A tensor of shape (vocab_size, embedding_dim) containing valid word embeddings.

    Returns:
        torch.Tensor: Noisy embeddings mapped back to the nearest valid word embedding.
    """
    # Step 1: Calculate sensitivity (global or precomputed)
    sensitivity = sensitivity

    # Step 2: Add noise proportional to sensitivity and epsilon
    scale = sensitivity / epsilon
    noise = torch.normal(mean=0, std=scale, size=embeddings.shape, device=embeddings.device)
    noisy_embeddings = embeddings + noise

    # Step 3: Map noisy embeddings back to the nearest valid word embedding
    if word_embeddings is not None:
        # Compute distances to all word embeddings
        distances = torch.cdist(noisy_embeddings, word_embeddings, p=2)
        # Find the nearest word embedding for each noisy embedding
        nearest_indices = distances.argmin(dim=1)
        noisy_embeddings = word_embeddings[nearest_indices]

    return noisy_embeddings

def preprocess_and_store_noisy_decoded_embeddings(replace_prob=0.5, epsilon=1.0):
    """
    Preprocess the dataset, generate sparse embeddings using pythia-70m-deduped,
    add differentially private noise, decode them back to the original basis,
    and store them in a Qdrant collection.

    Args:
        replace_prob (float): Probability of replacing the original token with the noisy one.
        epsilon (float): Privacy budget parameter for differential privacy.
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
    print("SAE configuration ",cfg_dict) 

    def decode_with_hooks(tokens, hook_func, max_new_tokens=50):
            """
            Decoding with hooks and randomized response.

            Args:
                tokens (torch.Tensor): Original input tokens.
                hook_func (callable): Hook function to modify activations.
                max_new_tokens (int): Maximum number of tokens to generate.

            Returns:
                torch.Tensor: Generated tokens.
            """
            generated_tokens = tokens.clone()
            # Use run_with_hooks to apply the hook during the forward pass
            logits = model.run_with_hooks(
                generated_tokens,
                return_type="logits",
                fwd_hooks=[(sae.cfg.hook_name, hook_func)],
            )

            for i in range(max_new_tokens):
                # Select the most probable token from logits
                next_token = logits[:, i, :].argmax(dim=-1, keepdim=True)

                # Randomized response: Replace token with probability `replace_prob`
                if torch.rand(1).item() < replace_prob:
                    generated_tokens[:, i] = next_token.squeeze(-1)

                # Stop if EOS token is generated
                if next_token.item() == model.tokenizer.eos_token_id:
                    break

            return generated_tokens

    def preprocess_function(examples):
        # Tokenize the texts
        decoded_embeddings = []

        for text in examples['text']:
            print("text: ", text)  # Sanity check
            tokens = model.to_tokens(text, prepend_bos=True)
   
            def noise_hook_sae(activation_value, hook):
                # Modify activations using the decoded embeddings
                with torch.no_grad():
                    # Encode to sparse basis
                    sparse_embeddings = sae.encode(activation_value)
                    # sensitivity= calculate_sensitivity(sparse_embeddings[0])
                    sensitivity=35.0 # Precomputed sensitivity value
                    # Add noise in the sparse basis
                    noisy_sparse_embeddings = add_differential_privacy_noise(sparse_embeddings, sensitivity=sensitivity, epsilon=epsilon)
                    # Decode back to the original basis
                    decoded_embeddings_text = sae.decode(noisy_sparse_embeddings)
                    return decoded_embeddings_text

            # Generate embeddings, add noise, and decode
            with torch.no_grad():
                output_tokens = decode_with_hooks(tokens, noise_hook_sae, max_new_tokens=tokens.shape[1])


            # # Decode the output tokens to text
            # # approximate_text = model.tokenizer.decode(output.argmax(dim=-1)[0])
            approximate_text = model.tokenizer.decode(output_tokens[0])
            print("approximate text: ", approximate_text)


            # Use run_with_cache to reembed embeddings with any adjustments after mechanisms
            with torch.no_grad():
                _, cache = model.run_with_cache(output_tokens[0])

            # Extract the embedding from the desired layer (e.g., final hidden state or specific layer)
            final_embedding = cache[sae.cfg.hook_name][0, -1, :].cpu().numpy()

            # Store the final embedding
            decoded_embeddings.append(final_embedding)



        print("total no. of embeddings: ", len(decoded_embeddings))
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