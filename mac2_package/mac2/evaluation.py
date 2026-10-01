"""Clustering evaluation: KMeans dispatch (GPU/CPU) + multi-method comparison."""
import numpy as np
from sklearn.cluster import KMeans, Birch

from .device import device as _default_device
from .metrics import clustering_metrics

try:
    from kmeans_cuda import kmeans as _gpu_kmeans
    HAS_GPU_KMEANS = True
except ImportError:
    HAS_GPU_KMEANS = False

try:
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import NearestNeighbors
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


def run_kmeans(X, num_clusters, use_device=None, seed=42,
              gpu_threshold=20000, kmeans_batch_size=20000):
    """
    X: torch.Tensor or np.ndarray, shape (N, D).
    Returns: np.ndarray of cluster assignments, shape (N,).

    kmeans_batch_size bounds the pairwise-distance tensor's memory
    footprint inside kmeans_cuda (N_batch x K x D floats) — lower it if
    you hit OOM on high-dimensional embeddings or large K.

    Note: the GPU path runs a single init (matches how MAGI itself calls
    this routine), trading sklearn's n_init=20 restart-robustness for
    speed at scale.
    """
    if use_device is None:
        use_device = _default_device
    n = X.shape[0]

    if HAS_GPU_KMEANS and use_device.type == 'cuda' and n > gpu_threshold:
        import torch
        X_t = X if torch.is_tensor(X) else torch.from_numpy(np.asarray(X))
        labels, _ = _gpu_kmeans(
            X=X_t, num_clusters=num_clusters, distance='euclidean',
            batch_size=kmeans_batch_size, tol=1e-4, device=use_device,
            tqdm_flag=False, seed=seed,
        )
        return labels.numpy()

    import torch
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    km = KMeans(n_clusters=num_clusters, n_init=20, random_state=seed)
    return km.fit_predict(X_np)


def evaluate_all_clustering_methods(embeddings, y_true, true_K,
                                    knn_k=10, birch_threshold=0.5,
                                    random_state=42, use_device=None):
    if hasattr(embeddings, 'detach'):
        emb = embeddings.detach().cpu().numpy()
    else:
        emb = np.asarray(embeddings)
    y_true = np.asarray(y_true)
    results = {}

    def score(labels, name):
        K_pred = len(np.unique(labels[labels >= 0]))
        cm = clustering_metrics(y_true, labels)
        acc, f1, *_ = cm.clusteringAcc()
        results[name] = {'ACC': float(acc), 'NMI': float(cm.NMI()),
                         'ARI': float(cm.ARI()), 'F1': float(f1),
                         'K_pred': int(K_pred)}

    score(run_kmeans(emb, true_K, use_device=use_device, seed=random_state), 'KMeans')

    perm = np.random.RandomState(random_state).permutation(len(emb))
    inv_perm = np.argsort(perm)
    birch = Birch(threshold=birch_threshold, branching_factor=50, n_clusters=None)
    labels_shuffled = birch.fit_predict(emb[perm])
    score(labels_shuffled[inv_perm], 'BIRCH')

    if HAS_LEIDEN:
        knn = NearestNeighbors(n_neighbors=knn_k + 1).fit(emb)
        _, idx = knn.kneighbors(emb)
        src = np.repeat(np.arange(len(emb)), knn_k)
        dst = idx[:, 1:].flatten()
        edges = list(zip(src.tolist(), dst.tolist()))
        g = ig.Graph(n=len(emb), edges=edges, directed=False)
        g.simplify()
        part = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, seed=random_state)
        score(np.array(part.membership), 'Leiden')

    return results


def print_clustering_table(results, dataset_name, true_K):
    print(f"\n  ── Clustering comparison on {dataset_name} (true K = {true_K}) ──")
    header = f"  {'Method':<10} {'K_pred':>7} {'ACC':>8} {'NMI':>8} {'ARI':>8} {'F1':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in results.items():
        print(f"  {name:<10} {m['K_pred']:>7d} {m['ACC']:>8.4f} "
              f"{m['NMI']:>8.4f} {m['ARI']:>8.4f} {m['F1']:>8.4f}")
