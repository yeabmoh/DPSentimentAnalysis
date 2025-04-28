import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any


class LogisticModel:
    """
    Non-private baseline: Scikit-learn pipeline
       StandardScaler  ➔  Multinomial Logistic Regression
    """
    def __init__(self, **kwargs):
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        """
        Build a pipeline: StandardScaler ➔ Multinomial LogisticRegression.

        """
        lr_params = dict(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,         
        )
        lr_params.update(kwargs)   

        logistic = LogisticRegression(**lr_params)

        self.model = Pipeline(
            steps=[("scaler", StandardScaler()), ("clf", logistic)]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        y_pred = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str):
        from joblib import dump
        dump(self.model, filepath)

    def load(self, filepath: str):
        from joblib import load
        self.model = load(filepath)
