import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped").to(device)
sae, _, _ = SAE.from_pretrained(
    release="pythia-70m-deduped-res-sm",
    sae_id="blocks.5.hook_resid_post",
    device="cpu",          # weights load on CPU
).to(device)               # then move to GPU/MPS if you have one

ds = load_dataset("mteb/tweet_sentiment_extraction", split="train").shuffle(seed=0)
sample_texts = ds[:8]["text"]             # just eight examples

tokens = model.to_tokens(sample_texts, prepend_bos=True)
print("tokens:", tokens.shape)            # ──> (batch, seq_len)

with torch.no_grad():
    _, cache = model.run_with_cache(tokens)
    dense = cache[sae.cfg.hook_name]
    sparse = sae.encode(dense)
    decoded = sae.decode(sparse)

print("dense :", dense.shape)            
print("sparse:", sparse.shape)            
print("decoded:", decoded.shape)          
