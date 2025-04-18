from models.logistic import LogisticModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any
from opacus.accountants import RDPAccountant


class DPLogisticModel(LogisticModel):
    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0, batch_size: int = 32, epochs: int = 10, **kwargs):
        """
        Initialize the DP-Logistic Regression model with DP-SGD parameters.

        Args:
            noise_multiplier (float): Noise multiplier for DP-SGD.
            max_grad_norm (float): Maximum gradient norm for DP-SGD.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            **kwargs: Additional arguments for LogisticModel.
        """
        super().__init__(**kwargs)
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs

    def _initialize_model(self, input_dim: int, num_classes: int, **kwargs):
        """
        Initialize the PyTorch logistic regression model.

        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
        """
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the logistic regression model using DP-SGD.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use long for multi-class labels

        # Create a DataLoader
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the model with the correct input dimension and number of classes
        self._initialize_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)))

        # Define the optimizer and loss function
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Attach the PrivacyEngine
        privacy_engine = PrivacyEngine(accountant="rdp")  # Use "rdp" for Rényi Differential Privacy
        self.model, optimizer, data_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        # Train the model
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in data_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        # Calculate and print the privacy budget
        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        print(f"Trained with DP-SGD. Privacy budget: ε = {epsilon:.2f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.

        Args:
            X_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): Test labels.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        # Convert test data to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create a DataLoader for evaluation
        dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        # Evaluate the model
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in data_loader:
                outputs = self.model(X_batch)
                y_pred.extend(torch.argmax(outputs, dim=1).numpy())

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained DP model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        # Convert input data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Create a DataLoader for prediction
        dataset = torch.utils.data.TensorDataset(X_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        # Predict using the model
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch in data_loader:
                outputs = self.model(X_batch[0]).view(-1)
                y_pred.extend((outputs > 0.5).int().numpy())
        return np.array(y_pred)