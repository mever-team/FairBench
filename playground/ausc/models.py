import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TorchModel:
    def __init__(self, input_dim, sensitive, lr=0.01, epochs=100, batch_size=32):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.sensitive = torch.tensor(sensitive, dtype=torch.float32).view(-1, 1)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                S = self.sensitive[: len(batch_y)]
                Sc = 1 - S
                l = self.loss_fn(predictions[S.bool()], batch_y[S.bool()])
                lc = self.loss_fn(predictions[Sc.bool()], batch_y[Sc.bool()])
                loss = l + lc + torch.abs(l - lc)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()
        return (predictions > 0.5).astype(
            int
        )  # Convert probabilities to binary predictions
