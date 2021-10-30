"""Methods for embedding calculations and clustering."""
import numpy as np
import pandas as pd
import torch
from sklearn.base import ClusterMixin


def calculate_embeddings(
    data: pd.DataFrame,
    model: torch.nn.Module
) -> np.ndarray:
    X = data.to_numpy().astype(np.float32)
    emb = model(torch.Tensor(X)).detach().numpy()

    return emb


def calculate_clusters_from_embeddings(
    embeddings: np.ndarray,
    model: ClusterMixin
) -> np.ndarray:
    if len(embeddings.shape) == 1:
        clusters = model.predict(embeddings.reshape(1, -1))
    else:
        clusters = model.predict(embeddings)

    return clusters


def calculate_clusters(data: pd.DataFrame, emb_model: torch.nn.Module, cluster_model: ClusterMixin) -> np.ndarray:
    emb = calculate_embeddings(data, emb_model)
    clusters = calculate_clusters_from_embeddings(emb, cluster_model)

    return clusters
