import os
import argparse
import numpy as np
from data.scripts.load_data import load_data_from_qdrant
from data.scripts.bert_preprocessing_script import preprocess_and_store_data
from data.scripts.sparsify import preprocess_and_store_noisy_decoded_embeddings
from models.logistic import LogisticModel
from models.dp_logistic import DPLogisticModel
from models.mlp import MLPModel

# Path to Qdrant data directory
QDRANT_DATA_PATH = os.path.abspath("qdrant_data/collections")
TRAIN_COLLECTION_NAME = "tweet_sentiment_vectors_train"
TEST_COLLECTION_NAME = "tweet_sentiment_vectors_test"
SPARSE_TRAIN_COLLECTION_NAME = "tweet_sentiment_noisy_decoded_vectors_train"
SPARSE_TEST_COLLECTION_NAME = "tweet_sentiment_noisy_decoded_vectors_test"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run sentiment analysis with Logistic Regression or DP-Logistic Regression.")
parser.add_argument("--model", type=str, choices=["logistic", "dp_logistic", "mlp"], default="logistic",
                    help="Specify which model to use.")
parser.add_argument("--shadow", type=int, default=0,
                    help="Number of shadow models to train for MIA experiment (0 = skip shadow phase).")
parser.add_argument(
    "--replace_prob",
    type=float,
    default=0.5,
    help="Probability of replacing the original token with the noisy one."
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=1.0,
    help="Privacy budget parameter for differential privacy."
)
args = parser.parse_args()

# Preprocess and store data if needed
# preprocess_and_store_data()
preprocess_and_store_noisy_decoded_embeddings(replace_prob=args.replace_prob, epsilon=args.epsilon)


# Load data
# preprocess_and_store_data()
# preprocess_and_store_data()
# Check if Qdrant data is empty
# if is_qdrant_data_empty(QDRANT_DATA_PATH):
#     print("Qdrant data is empty. Running preprocessing...")

# else:
#     print("Qdrant data found. Loading data...")

# Load train and test data from Qdrant
# X_train, y_train = load_data_from_qdrant(TRAIN_COLLECTION_NAME)
X_train, y_train = load_data_from_qdrant(SPARSE_TRAIN_COLLECTION_NAME)
# X_test, y_test = load_data_from_qdrant(TEST_COLLECTION_NAME)
X_test, y_test = load_data_from_qdrant(SPARSE_TEST_COLLECTION_NAME)


input_dim = X_train.shape[1]
num_classes = len(set(y_train))

def train_and_save(model, X_tr, y_tr, X_te, y_te, tag):
    model.train(X_tr, y_tr)
    probs_train = model.predict_proba(X_tr)
    probs_test = model.predict_proba(X_te)
    np.save(f"{tag}_train_probs.npy", probs_train)
    np.save(f"{tag}_nontrain_probs.npy", probs_test)
    np.save(f"{tag}_train_labels.npy", np.ones(len(y_tr)))
    np.save(f"{tag}_nontrain_labels.npy", np.zeros(len(y_te)))
    if tag.startswith("logistic") or tag.startswith("dp_logistic_noise0.5"):  # pick which dp noise level to use
        np.save("target_model_probs.npy", probs_test)
        np.save("target_model_labels.npy", np.zeros(len(y_te)))  # or use y_te if you want ground-truth labels

    metrics = model.evaluate(X_te, y_te)
    print("Accuracy:", metrics["accuracy"])
    print("Classification Report:", metrics["classification_report"])
    model.save(f"{tag}_model.joblib")

if args.shadow > 0:
    print(f"Training {args.shadow} shadow models...")
    for i in range(args.shadow):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        split = int(0.5 * len(indices))
        shadow_train_idx, shadow_nontrain_idx = indices[:split], indices[split:]
        shadow_X_train, shadow_y_train = X_train[shadow_train_idx], y_train[shadow_train_idx]
        shadow_X_nontrain, shadow_y_nontrain = X_train[shadow_nontrain_idx], y_train[shadow_nontrain_idx]

        shadow_model = LogisticModel()
        train_and_save(shadow_model, shadow_X_train, shadow_y_train, shadow_X_nontrain, shadow_y_nontrain, f"shadow_model_{i}")

else:
    if args.model == "logistic":
        print("Using Logistic Regression baseline.")
        model = LogisticModel()
    elif args.model == "mlp":
        print("Using non-DP MLP baseline.")
        model = MLPModel(input_dim=input_dim, num_classes=num_classes, hidden=256, batch_size=64, epochs=50)
    elif args.model == "dp_logistic":
        print("Running DP-MLP (DP-SGD) experiments with different noise multipliers.")
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for noise in noise_levels:
            print(f"\n--- Training with noise_multiplier = {noise} ---")
            model = DPLogisticModel(input_dim=input_dim, num_classes=num_classes, noise_multiplier=noise,
                                    max_grad_norm=1.0, batch_size=64, epochs=50)
            train_and_save(model, X_train, y_train, X_test, y_test, f"dp_logistic_noise{noise}")
        exit()
    else:
        raise ValueError(f"Unknown model flag: {args.model}")

    # Single model run
    train_and_save(model, X_train, y_train, X_test, y_test, args.model)
