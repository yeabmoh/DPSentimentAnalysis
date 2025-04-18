import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any

class LogisticModel:
    def __init__(self, **kwargs):
        """
        Initialize the Logistic Regression model with configurable parameters.
        """
        self.model = LogisticRegression(**kwargs)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the logistic regression model.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        print(X_train[0:100])
        print(y_train[0:100])
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.

        Args:
            X_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): Test labels.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def save(self, filepath: str):
        """
        Save the trained model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        from joblib import dump
        dump(self.model, filepath)

    def load(self, filepath: str):
        """
        Load a trained model from a file.

        Args:
            filepath (str): Path to the saved model.
        """
        from joblib import load
        self.model = load(filepath)