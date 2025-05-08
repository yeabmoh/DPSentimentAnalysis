import os
import argparse
import numpy as np
from data.scripts.load_data import load_data_from_qdrant
from data.scripts.sparsify import preprocess_and_store_noisy_decoded_embeddings
from models.logistic import LogisticModel
from models.dp_logistic import DPLogisticModel
from models.mlp import MLPModel
from sklearn.preprocessing import StandardScaler


QDRANT_DATA_PATH = os.path.abspath("qdrant_data/collections")
SPARSE_TRAIN_COLLECTION_NAME = "tweet_sentiment_noisy_decoded_vectors_train"
SPARSE_TEST_COLLECTION_NAME = "tweet_sentiment_noisy_decoded_vectors_test"

parser = argparse.ArgumentParser(description="Run sentiment analysis across epsilon and replace_prob grid.")
parser.add_argument("--model", type=str, choices=["logistic", "dp_logistic", "mlp"], default="logistic",
                    help="Specify which model to use.")
parser.add_argument("--shadow", type=int, default=0,
                    help="Number of shadow models to train for MIA experiment (0 = skip shadow phase).")
args = parser.parse_args()

epsilons = np.logspace(0, 1, 5)  

replace_probs = np.arange(0.0, 1.01, 0.2)

# Load test data once (no need to reload every loop)
# X_test, y_test = load_data_from_qdrant(SPARSE_TEST_COLLECTION_NAME)
# xtest, ytest = load_data_from_qdrant(SPARSE_TRAIN_COLLECTION_NAME)

def train_and_save(model, X_tr, y_tr, X_te, y_te, tag):
    model.train(X_tr, y_tr)
    probs_train = model.predict_proba(X_tr)
    probs_test = model.predict_proba(X_te)
    np.save(f"{tag}_train_probs.npy", probs_train)
    np.save(f"{tag}_nontrain_probs.npy", probs_test)
    np.save(f"{tag}_train_labels.npy", np.ones(len(y_tr)))
    np.save(f"{tag}_nontrain_labels.npy", np.zeros(len(y_te)))
    if tag.startswith("logistic"):
        np.save("target_non_dp_probs.npy", np.vstack([probs_train, probs_test]))
        np.save("target_non_dp_labels.npy", np.concatenate([np.ones(len(y_tr)), np.zeros(len(y_te))]))
    metrics = model.evaluate(X_te, y_te)
    print("Accuracy:", metrics["accuracy"])
    print("Classification Report:", metrics["classification_report"])
    model.save(f"{tag}_model.joblib")
    

for epsilon in epsilons:
    for replace_prob in replace_probs:
        print(f"\n=== Running with epsilon={epsilon:.1f}, replace_prob={replace_prob:.1f} ===")

        # Preprocess with current epsilon and replace_prob
        preprocess_and_store_noisy_decoded_embeddings(replace_prob=replace_prob, epsilon=epsilon)

        # Load train data after preprocessing
        X_train, y_train = load_data_from_qdrant(SPARSE_TRAIN_COLLECTION_NAME)
        X_test, y_test = load_data_from_qdrant(SPARSE_TEST_COLLECTION_NAME)
        classes, counts = np.unique(y_train, return_counts=True)
        print("Class distribution:", dict(zip(classes, counts)))
        print("Feature mean/std:", X_train.mean(), X_train.std())
        input_dim = X_train.shape[1]
        num_classes = len(set(y_train))

        if args.shadow > 0:
            print(f"Training {args.shadow} shadow models for eps={epsilon:.1f}, replace_prob={replace_prob:.1f}...")
            for i in range(args.shadow):
                indices = np.arange(len(X_train))
                np.random.shuffle(indices)
                split = int(0.5 * len(indices))
                shadow_train_idx, shadow_nontrain_idx = indices[:split], indices[split:]
                shadow_X_train, shadow_y_train = X_train[shadow_train_idx], y_train[shadow_train_idx]
                shadow_X_nontrain, shadow_y_nontrain = X_train[shadow_nontrain_idx], y_train[shadow_nontrain_idx]

                if args.model == "logistic":
                    shadow_model = LogisticModel()
                elif args.model == "mlp":
                    shadow_model = MLPModel(input_dim=input_dim, num_classes=num_classes, hidden=256, batch_size=64, epochs=50)
                elif args.model == "dp_logistic":
                    shadow_model = DPLogisticModel(input_dim=input_dim, num_classes=num_classes, noise_multiplier=0.5,
                                                   max_grad_norm=1.0, batch_size=64, epochs=50)
                else:
                    raise ValueError(f"Unknown model: {args.model}")

                tag = f"shadow_model_{i}_eps{epsilon:.1f}_replace{replace_prob:.1f}"
                train_and_save(shadow_model, shadow_X_train, shadow_y_train, shadow_X_nontrain, shadow_y_nontrain, tag)

        else:
            if args.model == "logistic":
                print("Using Logistic Regression baseline.")
                model = LogisticModel()
            elif args.model == "mlp":
                print("Using non-DP MLP baseline.")
                model = MLPModel(input_dim=input_dim, num_classes=num_classes, hidden=256, batch_size=64, epochs=50)
            elif args.model == "dp_logistic":
                print("Using DP Logistic Regression baseline.")
                model = DPLogisticModel(input_dim=input_dim, num_classes=3, noise_multiplier=0.5,
                                        max_grad_norm=1.0, batch_size=256, epochs=50)
            else:
                raise ValueError(f"Unknown model: {args.model}")

            tag = f"{args.model}_eps{epsilon:.1f}_replace{replace_prob:.1f}"
            train_and_save(model, X_train, y_train, X_test, y_test, tag)
