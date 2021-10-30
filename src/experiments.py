"""Main script to run experiments."""
import os
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from src.configurations import (CITIES_SIGSPATIAL, CLUSTERS_SWAPS,
                                DENSE_ENCODER_LAYERS, MAX_CLUSTERS,
                                MIN_CLUSTERS, SPARSE_ENCODER_LAYERS)
from src.data.feeds import load_data
from src.data.processing import normalize_data
from src.models.autoencoder import SimpleAutoEncoder
from src.models.embeddings import calculate_embeddings
from src.models.utils import train_model_unsupervised
from src.settings import MODELS_DIRECTORY, REPORTS_DIRECTORY
from src.visualization.maps import plot_clusters


def draw_dendrogram(
    emb: pd.DataFrame,
    path: str,
    normalization: str,
    version: str
):
    """Draws dendrogram for clustering.

    Args:
        emb (pd.DataFrame): embeddings
        path (str): dir save path
        normalization (str): normalization type
    """
    fig = plt.figure(figsize=(15, 10))
    plt.title(f"Clusters dendrogram on embeddings")
    # plt.title(f"Clusters dendrogram on embeddings - {version} - {normalization} norm")
    _ = dendrogram(linkage(emb, method='ward', metric='euclidean'), p=3, truncate_mode='level')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'dendrogram.png'), facecolor='white', transparent=False)
    plt.close(fig)


def plot_hour_features_boxplot(
    raw_df: pd.DataFrame,
    path: str,
    normalization: str,
    version: str,
    clusters: int
):
    p = os.path.join(path, str(clusters))
    Path(p).mkdir(parents=True, exist_ok=True)

    trips_columns = [f'trips_at_{i}' for i in range(6, 23)]
    directions_columns = [f'directions_at_{i}' for i in range(6, 23)]
    for cluster in range(clusters):
        fig = px.box(
            raw_df[raw_df['cluster'] == cluster],
            y=trips_columns,
            title=f'Trips per hour for cluster {cluster} - division for {clusters} clusters'  
        )
        fig.update_yaxes(range=[0, 600])
        fig.write_image(os.path.join(p, f'trips_cluster_{cluster}.png'), engine='orca')

        fig = px.box(
            raw_df[raw_df['cluster'] == cluster],
            y=directions_columns,
            title=f'Directions per hour for cluster {cluster} - division for {clusters} clusters'  
        )
        fig.update_yaxes(range=[0, 70])
        fig.write_image(os.path.join(p, f'directions_cluster_{cluster}.png'), engine='orca')


def run_clustering(
    emb: pd.DataFrame,
    raw_df: pd.DataFrame,
    path: str,
    normalization: str,
    version: str,
    cities: List[str]
):
    """Run clustering and prepare plots and maps.

    Args:
        emb (pd.DataFrame): embeddings
        raw_df (pd.DataFrame): raw features
        path (str): dir save path
        normalization (str): normalization type
        cities (List[str]): cities list
    """
    raw_df['sum_trips'] = raw_df[[f'trips_at_{h}' for h in range(6, 23)]].apply(sum, axis=1)

    save_df = raw_df.copy()

    p_plots = os.path.join(path, 'scatter_plots')
    b_plots = os.path.join(path, 'box_plots')
    p_maps = os.path.join(path, 'maps')

    Path(p_plots).mkdir(parents=True, exist_ok=True)
    Path(p_maps).mkdir(parents=True, exist_ok=True)
    Path(b_plots).mkdir(parents=True, exist_ok=True)

    for n in range(MIN_CLUSTERS, MAX_CLUSTERS):
        clusters = AgglomerativeClustering(n_clusters=n, linkage='ward').fit_predict(emb)
        raw_df['cluster'] = clusters

        if n in CLUSTERS_SWAPS.keys():
            raw_df['cluster'].replace(CLUSTERS_SWAPS[n], inplace=True)

        raw_df['cluster_dis'] = raw_df['cluster'].apply(str)

        save_df[f'cluster_{n}'] = raw_df['cluster']

        fig = px.scatter(
            raw_df.sort_values(by='cluster'), 
            x='sum_trips', y='directions_whole_day',
            opacity=0.5,
            color='cluster_dis',
            color_discrete_sequence=px.colors.qualitative.Set1,
            title=f'Trips vs directions scatter - {n} clusters'
            # title=f'Trips vs directions scatter - {version} - {normalization} norm - {n} clusters'
        )
        fig.write_image(os.path.join(p_plots, f'whole_day_{n}.png'), engine='orca')

        plot_hour_features_boxplot(raw_df, b_plots, normalization, version, n)

        if n <= 9:
            for city in cities:
                m = plot_clusters(
                    city, raw_df, n_clusters=9, 
                    save_html_path=os.path.join(p_maps, str(n)),
                    label=f'Clusters in {city} - {n} clusters'
                    # label=f'Clusters in {city} - {version} - {normalization} norm - {n} clusters'
                )

    raw_df.drop(columns=['cluster', 'cluster_dis', 'sum_trips'], inplace=True)

    save_df.to_csv(os.path.join(path, 'clusters.csv'))


def run_with_normalization(
    model: SimpleAutoEncoder, 
    features: pd.DataFrame, 
    raw: pd.DataFrame, 
    normalization: str,
    version: str,
    path_prefix: str, 
    cities: List[str]
):
    """Run all experiments for single normalization type.

    Args:
        model (SimpleAutoEncoder): emb model
        features (pd.DataFrame): normalized features
        raw (pd.DataFrame): raw features
        normalization (str): nozmalization type
        path_prefix (str): dir save path
        cities (List[str]): cities list
    """
    emb = calculate_embeddings(features, model)

    p = os.path.join(path_prefix, f'{normalization}_norm')
    Path(p).mkdir(parents=True, exist_ok=True)

    draw_dendrogram(emb, p, normalization, version)
    run_clustering(emb, raw, p, normalization, version, cities)


def load_model(features: pd.DataFrame, normalization: str, version: str) -> SimpleAutoEncoder:
    """Load model from cache, or train.

    Args:
        features (pd.DataFrame): features
        normalization (str): normalization type
        version (str): experiments group (PL/EU)

    Returns:
        SimpleAutoEncoder: emb model
    """
    p = os.path.join(MODELS_DIRECTORY, f'{version}_{normalization}.pth')

    if os.path.exists(p):
        model = torch.load(p)
    elif 'sparse_reg' in version:
        model = train_model_unsupervised(
            features, SimpleAutoEncoder,
            n_hidden=SPARSE_ENCODER_LAYERS[0], 
            emb_size=SPARSE_ENCODER_LAYERS[1],
            sparsity_lambda=0.1
        )
        torch.save(model, p)
    elif 'sparse' in version:
        model = train_model_unsupervised(
            features, SimpleAutoEncoder, 
            n_hidden=SPARSE_ENCODER_LAYERS[0], 
            emb_size=SPARSE_ENCODER_LAYERS[1]
        )
        torch.save(model, p)
    elif 'dense' in version:
        model = train_model_unsupervised(
            features, SimpleAutoEncoder, 
            n_hidden=DENSE_ENCODER_LAYERS[0], 
            emb_size=DENSE_ENCODER_LAYERS[1]
        )
        torch.save(model, p)
    else:
        model = train_model_unsupervised(features, SimpleAutoEncoder)
        torch.save(model, p)

    return model


def run_for_cities(cities: List[str], version: str):
    """Run all experiments for given cities group.

    Args:
        cities (List[str]): cities
        version (str): group name
    """
    raw_df = load_data(cities=cities)
    features_local, features_global = normalize_data(raw_df, column_groups=None,  agg_column='city', drop_columns=['directions_whole_day'])
    
    features_local = features_local.drop(columns=['city'])
    features_global = features_global.drop(columns=['city'])

    models = {
        # 'local': load_model(features_local, 'local', version),
        'global': load_model(features_global, 'global', version)
    }

    p = os.path.join(REPORTS_DIRECTORY, version)

    # run_with_normalization(models['local'], features_local, raw_df, 'local', version, p, cities)
    run_with_normalization(models['global'], features_global, raw_df, 'global', version, p, cities)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    run_for_cities(CITIES_SIGSPATIAL, 'SIGSPATIAL')
