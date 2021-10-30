"""Generic functions related with models used in thesis."""
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def train_model_unsupervised(
    features: pd.DataFrame,
    model: torch.nn.Module.__class__,
    n_hidden: int = 48,
    emb_size: int = 64,
    sparsity_lambda: Union[float, None] = None
) -> torch.nn.Module:
    """Generic wraper for lightning trainer on unsupervised model.

    Args:
        features (pd.DataFrame): df with features
        model (torch.nn.Module.__class__): model class

    Returns:
        torch.nn.Module: trained model
    """
    X = features.to_numpy().astype(np.float32)
    model = model(
        n_features=X.shape[1], 
        n_hidden=n_hidden,
        emb_size=emb_size,
        sparsity_lambda=sparsity_lambda
    )
    x_dataloader = DataLoader(X, batch_size=25, num_workers=4)
    trainer = pl.Trainer(max_epochs=10)

    trainer.fit(model, x_dataloader)

    return model
