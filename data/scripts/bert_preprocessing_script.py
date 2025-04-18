# Run these commands in your terminal in this order:
# 1. pip install datasets transformers
# 2. python3 bert_preprocessing_script.py

from datasets import load_dataset
from transformers import BertTokenizerFast

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

# Set format for PyTorch
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
