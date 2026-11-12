"""Optional CPU PyTorch tabular regressors: residual MLP and feature-token Transformer."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                   nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width))

    def forward(self, x):
        return x + self.block(x)


class FeatureTransformer(nn.Module):
    def __init__(self, features, width, layers, heads, dropout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, width) * .02)
        self.bias = nn.Parameter(torch.randn(features, width) * .02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        return self.head(self.encoder(tokens).mean(dim=1)).squeeze(-1)


class TorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, architecture="transformer", width=32, layers=2, heads=4,
                 dropout=.1, epochs=100, batch_size=128, learning_rate=.001,
                 weight_decay=.0001, random_state=42):
        self.architecture = architecture
        self.width = width
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        if self.epochs < 1 or self.batch_size < 1 or self.layers < 1 or self.width < 1:
            raise ValueError("epochs, batch_size, layers и width должны быть > 0.")
        if self.heads < 1 or self.width % self.heads:
            raise ValueError("width должен делиться на heads.")
        torch.manual_seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        if self.architecture == "transformer":
            self.network_ = FeatureTransformer(X.shape[1], self.width, self.layers, self.heads, self.dropout)
        elif self.architecture == "resnet":
            self.network_ = nn.Sequential(nn.Linear(X.shape[1], self.width),
                                          *[ResidualBlock(self.width, self.dropout) for _ in range(self.layers)],
                                          nn.LayerNorm(self.width), nn.Linear(self.width, 1), nn.Flatten(0))
        else:
            raise ValueError("Неизвестная нейросетевая архитектура.")
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X.copy()),
                                             torch.from_numpy(y.copy())), batch_size=self.batch_size,
                                             shuffle=True, generator=generator)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_curve_ = []
        self.network_.train()
        for _ in range(self.epochs):
            total = 0.
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.network_(xb), yb)
                if not torch.isfinite(loss):
                    raise ValueError("Нейросеть разошлась; уменьшите learning_rate.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 1.)
                optimizer.step()
                total += loss.item() * len(xb)
            self.loss_curve_.append(total / len(X))
        self.network_.eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "network_")
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Число признаков отличается от обучения.")
        with torch.inference_mode():
            return np.concatenate([self.network_(torch.from_numpy(X[i:i + self.batch_size].copy())).numpy()
                                   for i in range(0, len(X), self.batch_size)])
