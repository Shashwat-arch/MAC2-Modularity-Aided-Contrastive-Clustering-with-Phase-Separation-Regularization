"""
MAC2: Modularity-Aided Contrastive Clustering with Phase-Separation
Regularization.

Modular package layout:
    mac2/
        device.py         -- device selection, shared constants
        metrics.py         -- clustering_metrics, run_kmeans dispatch
        preprocessing.py   -- property-weighted adjacency, diffusion
        model.py            -- AMGC model (encoder + all losses)
        evaluation.py       -- multi-method clustering evaluation
        checkpointing.py    -- best-embedding tracking across runs
        datasets.py         -- dataset registry + loaders (incl. OGB)
        train.py            -- run_experiment training loop
"""
from .device import device, EPS
from .metrics import clustering_metrics
from .model import AMGC
from .preprocessing import (
    encode_adjacency_with_properties, norm_adj, compute_diffusion_matrix,
    get_clustering_coeff, compute_feature_homophily,
)
from .evaluation import (
    evaluate_all_clustering_methods, print_clustering_table, run_kmeans,
)
from .checkpointing import save_best_embedding_if_better
from .datasets import DATASET_CFG, DATASET_INFO, load_dataset
from .train import run_experiment

__all__ = [
    'device', 'EPS', 'clustering_metrics', 'AMGC',
    'encode_adjacency_with_properties', 'norm_adj', 'compute_diffusion_matrix',
    'get_clustering_coeff', 'compute_feature_homophily',
    'evaluate_all_clustering_methods', 'print_clustering_table', 'run_kmeans',
    'save_best_embedding_if_better',
    'DATASET_CFG', 'DATASET_INFO', 'load_dataset',
    'run_experiment',
]
