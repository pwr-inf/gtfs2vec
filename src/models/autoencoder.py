"""Simple autoencoder model."""
from typing import Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class SimpleAutoEncoder(pl.LightningModule):
    def __init__(self, n_features: int, n_hidden: int = 48, emb_size: int = 64, sparsity_lambda: Union[float, None] = None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, emb_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if self.sparsity_lambda is not None:
            loss = loss + self.sparsity_lambda * torch.mean(torch.abs(z))
        self.log('train_loss', loss)
        return loss
