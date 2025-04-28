# models/mlp.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any

class MLPModel:
    def __init__(self, input_dim: int = 768, num_classes: int = 3,
                 hidden: int = 512, batch_size: int = 64,
                 epochs: int = 50, use_class_weights: bool = False):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_class_weights = use_class_weights
        self.model = None

    def _init_net(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True)

        self._init_net()

        if self.use_class_weights:
            counts = np.bincount(y_train)
            w = 1.0 / torch.tensor(counts, dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=w)
        else:
            criterion = nn.CrossEntropyLoss()

        opt  = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        self.model.train()
        for ep in range(self.epochs):
            ep_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            sched.step()
            print(f"Epoch {ep+1:02d}/{self.epochs}, loss={ep_loss:.3f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        self.model.eval()
        X_ts = torch.tensor(X_test, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(X_ts, batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for xb in loader:
                preds.extend(torch.argmax(self.model(xb), dim=1).cpu().numpy())

        return {
            "accuracy": accuracy_score(y_test, preds),
            "classification_report": classification_report(y_test, preds, output_dict=True),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_ts = torch.tensor(X, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(X_ts, batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for xb in loader:
                preds.extend(torch.argmax(self.model(xb), dim=1).cpu().numpy())
        return np.array(preds)

    def save(self, filepath: str):
        import torch
        metadata = dict(
            input_dim = next(self.model.parameters()).shape[1],
            num_classes = self.model[-1].out_features
        )
        torch.save({"state_dict": self.model.state_dict(),
                    "meta": metadata}, filepath)

    def load(self, filepath: str):
        import torch
        checkpoint = torch.load(filepath, map_location="cpu")
        meta = checkpoint["meta"]
        self._init_net()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
