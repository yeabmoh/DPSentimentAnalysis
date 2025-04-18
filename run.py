import os
import argparse
from data.scripts.load_data import is_qdrant_data_empty, load_data_from_qdrant
from data.scripts.bert_preprocessing_script import preprocess_and_store_data
from models.logistic import LogisticModel
from models.dp_logistic import DPLogisticModel

# Path to Qdrant data directory
QDRANT_DATA_PATH = os.path.abspath("qdrant_data/collections")
TRAIN_COLLECTION_NAME = "tweet_sentiment_vectors_train"
TEST_COLLECTION_NAME = "tweet_sentiment_vectors_test"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run sentiment analysis with Logistic Regression or DP-Logistic Regression.")
parser.add_argument(
    "--model",
    type=str,
    choices=["logistic", "dp_logistic"],
    default="logistic",
    help="Specify which model to use: 'logistic' or 'dp_logistic'."
)
args = parser.parse_args()

# Check if Qdrant data is empty
if is_qdrant_data_empty(QDRANT_DATA_PATH):
    print("Qdrant data is empty. Running preprocessing...")
    preprocess_and_store_data()
else:
    print("Qdrant data found. Loading data...")

# Load train and test data from Qdrant
X_train, y_train = load_data_from_qdrant(TRAIN_COLLECTION_NAME)
X_test, y_test = load_data_from_qdrant(TEST_COLLECTION_NAME)

# Initialize the model
if args.model == "logistic":
    print("Using Logistic Regression model.")
    model = LogisticModel(max_iter=1000)
elif args.model == "dp_logistic":
    print("Using DP-Logistic Regression model.")
    model = DPLogisticModel(max_iter=1000)  # Add DP-SGD-specific parameters if needed

# Train the model
model.train(X_train, y_train)

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
print("Accuracy:", metrics["accuracy"])
print("Classification Report:", metrics["classification_report"])

# Save the model
model.save(f"{args.model}_model.joblib")

# Load the model (optional, for demonstration purposes)
model.load(f"{args.model}_model.joblib")