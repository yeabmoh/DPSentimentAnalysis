import os
import argparse
from data.scripts.load_data import is_qdrant_data_empty, load_data_from_qdrant
from data.scripts.bert_preprocessing_script import preprocess_and_store_data
from models.logistic import LogisticModel
from models.dp_logistic import DPLogisticModel
from models.mlp import MLPModel
import numpy as np

# Path to Qdrant data directory
QDRANT_DATA_PATH = os.path.abspath("qdrant_data/collections")
TRAIN_COLLECTION_NAME = "tweet_sentiment_vectors_train"
TEST_COLLECTION_NAME = "tweet_sentiment_vectors_test"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run sentiment analysis with Logistic Regression or DP-Logistic Regression.")
parser.add_argument(
    "--model",
    type=str,
    choices=["logistic", "dp_logistic", "mlp"],
    default="logistic",
    help="Specify which model to use: 'logistic' or 'dp_logistic'."
)
args = parser.parse_args()
# preprocess_and_store_data()
# Check if Qdrant data is empty
if is_qdrant_data_empty(QDRANT_DATA_PATH):
    print("Qdrant data is empty. Running preprocessing...")

else:
    print("Qdrant data found. Loading data...")

# Load train and test data from Qdrant
X_train, y_train = load_data_from_qdrant(TRAIN_COLLECTION_NAME)
X_test, y_test = load_data_from_qdrant(TEST_COLLECTION_NAME)

input_dim = X_train.shape[1]
num_classes = len(set(y_train))

if args.model == "logistic":
    print("Using Logistic Regression baseline.")
    model = LogisticModel()

elif args.model == "mlp":
    print("Using non-DP MLP baseline.")
    model = MLPModel(input_dim=input_dim, num_classes=num_classes,
                     hidden=256, batch_size=64, epochs=50)

elif args.model == "dp_logistic":
    print("Using DP-MLP (DP-SGD) model.")
    model = DPLogisticModel(input_dim=input_dim, num_classes=num_classes,
                            noise_multiplier=0.6, max_grad_norm=1.0,
                            batch_size=64, epochs=50)

else:
    raise ValueError(f"Unknown model flag: {args.model}")

# Train the model
model.train(X_train, y_train)

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
print("Accuracy:", metrics["accuracy"])
print("Classification Report:", metrics["classification_report"])



# Save the model
model.save(f"{args.model}_model.joblib")

# Load the model (optional, for demonstration purposes)

# model.load(f"{args.model}_model.joblib")