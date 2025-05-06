# dp_logistic.py  (overwrite the whole file)
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any
from models.logistic import LogisticModel


class DPLogisticModel(LogisticModel):
    """
    Differentially-Private tiny MLP (1 hidden layer) trained with DP-SGD.
    """

    def __init__(
        self,
        noise_multiplier: float = 0.6,
        max_grad_norm: float = 1.0,
        batch_size: int = 64,
        epochs: int = 50,
        use_class_weights: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)    
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_class_weights = use_class_weights


    def _initialize_model(self, input_dim: int, num_classes: int, **kwargs):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._initialize_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)))

        if self.use_class_weights:
            counts = np.bincount(y_train)
            weights = 1.0 / torch.tensor(counts, dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        opt = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        engine = PrivacyEngine(accountant="rdp")
        self.model, opt, loader = engine.make_private(
            module=self.model,
            optimizer=opt,
            data_loader=loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        best, bad, patience = float("inf"), 0, 5
        for ep in range(self.epochs):
            ep_loss = 0.0
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            sched.step()

            print(f"Epoch {ep+1:02d}/{self.epochs}, loss={ep_loss:.3f}")
            if ep_loss < best:
                best, bad = ep_loss, 0
            else:
                bad += 1
                if bad == patience:
                    print("Early stopping triggered.")
                    break

        eps = engine.accountant.get_epsilon(delta=1e-5)
        print(f"Training complete.  Privacy budget: ε = {eps:.2f}")

    # -----------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        self.model.eval()
        X_ts = torch.tensor(X_test, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(X_ts, batch_size=self.batch_size)

        preds = []
        with torch.no_grad():
            for xb in loader:
                out = self.model(xb)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())

        return {
            "accuracy": accuracy_score(y_test, preds),
            "classification_report": classification_report(y_test, preds, output_dict=True),
        }

    # keep .predict() if you use it elsewhere
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_ts = torch.tensor(X, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(X_ts, batch_size=self.batch_size)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                preds.extend(torch.argmax(self.model(xb), dim=1).cpu().numpy())
        return np.array(preds)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return softmax probabilities from the DP logistic model.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(X_tensor, batch_size=self.batch_size)
        probs = []

        with torch.no_grad():
            for xb in loader:
                logits = self.model(xb)
                softmax_probs = torch.nn.functional.softmax(logits, dim=1)
                probs.append(softmax_probs.cpu().numpy())

        return np.vstack(probs)

