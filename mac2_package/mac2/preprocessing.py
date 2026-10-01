"""Property-weighted adjacency (PEO), normalization, and PPR diffusion."""
import os
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import degree

from .device import EPS

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False


def get_clustering_coeff(edge_index, num_nodes, dataset_name,
                         cache_dir='./cc_cache', max_edges_for_exact=100_000_000):
    """
    Local clustering coefficient per node.

    Two memory safeguards are applied unconditionally, before anything
    touches igraph, because they're free wins regardless of graph size:
      - drop self-loops (contribute nothing to a triangle count, and
        to_undirected(add_remaining_self_loops(...)) upstream adds one
        per node)
      - keep only one direction per undirected edge (row < col) — the
        upstream pipeline stores both (i,j) and (j,i), so this halves
        the edge count before it ever reaches Python-object territory
      - torch.unique() dedupes any remaining repeated pairs cheaply
        (vectorized sort, not Python-object comparison) — this also
        replaces the old post-construction g.simplify() call, which used
        to do this cleanup only *after* the bloated edge set was already
        fully materialized.

    The actual OOM risk at Reddit/ogbn-* scale is NOT igraph's transitivity
    algorithm (efficient compiled C) — it's edge_index.tolist() building
    ~2 * n_edges individual boxed Python int/list objects to hand to it.
    We pass a numpy array directly instead (falling back to .tolist() only
    if an older igraph build rejects that). Above max_edges_for_exact
    (post-filtering), we skip exact computation entirely rather than gamble
    on that construction step, and fall back to a neutral constant — this
    degrades the property-weighted operator to a degree+homophily-only
    reliability signal for graphs that large; worth flagging explicitly if
    such a run feeds into the paper's scalability claim.
    """
    cache_path = os.path.join(cache_dir, f"{dataset_name}_cc.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    row, col = edge_index[0].cpu(), edge_index[1].cpu()
    keep = row < col
    row, col = row[keep], col[keep]
    edge_pairs = torch.unique(torch.stack([row, col], dim=1), dim=0)
    n_edges = edge_pairs.size(0)

    if HAS_IGRAPH and n_edges <= max_edges_for_exact:
        edges_np = edge_pairs.numpy()
        try:
            g = ig.Graph(n=num_nodes, edges=edges_np, directed=False)
        except TypeError:
            g = ig.Graph(n=num_nodes, edges=edges_np.tolist(), directed=False)
        cc_list = g.transitivity_local_undirected(mode="zero")
        cc = torch.tensor(cc_list, dtype=torch.float)
    elif HAS_IGRAPH:
        print(f"  [warn] {dataset_name}: {n_edges:,} unique edges exceeds "
              f"max_edges_for_exact={max_edges_for_exact:,} — skipping exact "
              f"clustering-coefficient computation to avoid an OOM during "
              f"graph construction. Falling back to a neutral constant "
              f"(0.0); degree + homophily still carry the property "
              f"weighting for this dataset.")
        cc = torch.zeros(num_nodes, dtype=torch.float)
    else:
        print("  [warn] igraph not found — falling back to networkx for "
              "clustering-coefficient computation. This will not scale to "
              "Reddit/ogbn-* graphs; run `pip install python-igraph "
              "--break-system-packages`.")
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_pairs.tolist())
        cc_dict = nx.clustering(G)
        cc = torch.tensor([cc_dict[i] for i in range(num_nodes)], dtype=torch.float)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cc, cache_path)
    return cc


def compute_feature_homophily(edge_index, features, num_nodes,
                              dataset_name=None, cache_dir='./cc_cache',
                              chunked=None, edge_chunk_size=None):
    if dataset_name is not None:
        cache_path = os.path.join(cache_dir, f"{dataset_name}_homophily.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path)

    dev = features.device
    edge_index = edge_index.to(dev)
    row, col = edge_index[0], edge_index[1]
    num_edges = row.size(0)
    d = features.size(1)
    x_norm = F.normalize(features, p=2, dim=1)

    if chunked is None:
        est_bytes = 2 * num_edges * d * 4
        chunked = est_bytes > 1_000_000_000

    if chunked:
        if edge_chunk_size is None:
            target_bytes = 1_000_000_000
            edge_chunk_size = max(1, target_bytes // (d * 4 * 2))
        edge_cos = torch.empty(num_edges, device=dev)
        for start in range(0, num_edges, edge_chunk_size):
            end = min(start + edge_chunk_size, num_edges)
            edge_cos[start:end] = (
                x_norm[row[start:end]] * x_norm[col[start:end]]
            ).sum(dim=1)
    else:
        edge_cos = (x_norm[row] * x_norm[col]).sum(dim=1)

    h_sum = torch.zeros(num_nodes, device=dev)
    h_cnt = torch.zeros(num_nodes, device=dev)
    h_sum.scatter_add_(0, row, edge_cos)
    h_cnt.scatter_add_(0, row, torch.ones_like(edge_cos))
    h = torch.where(h_cnt > 0, h_sum / h_cnt, torch.zeros_like(h_sum))
    h = (h + 1.0) / 2.0

    if dataset_name is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(h.cpu(), cache_path)
    return h


def encode_adjacency_with_properties(edge_index, features, num_nodes,
                                     dataset_name, dev, beta=1.0,
                                     chunked=None, edge_chunk_size=None,
                                     cc_max_edges=100_000_000,
                                     drop_signal=None):
    edge_index = edge_index.to(dev)
    row, col = edge_index[0], edge_index[1]
    deg = degree(row, num_nodes, dtype=torch.float).to(dev)
    deg_norm = deg / (deg.max() + EPS)
    cc = get_clustering_coeff(edge_index, num_nodes, dataset_name,
                              max_edges_for_exact=cc_max_edges).to(dev)
    homophily = compute_feature_homophily(
        edge_index, features.to(dev), num_nodes,
        dataset_name=dataset_name, chunked=chunked,
        edge_chunk_size=edge_chunk_size,
    ).to(dev)

    # ── Property-weighting ablation: zero out one signal before stacking.
    #    Kept as a post-hoc zeroing (not skipping the computation above) so
    #    p_i stays 3-dimensional regardless of drop_signal -- avoids touching
    #    the edge_prop_sum / edge_prop_norm shape logic below. ──
    if drop_signal == 'degree':
        deg_norm = torch.zeros_like(deg_norm)
    elif drop_signal == 'clustering':
        cc = torch.zeros_like(cc)
    elif drop_signal == 'homophily':
        homophily = torch.zeros_like(homophily)
    elif drop_signal is not None:
        raise ValueError(
            f"Unknown drop_signal: {drop_signal!r}, expected one of "
            f"{{None, 'degree', 'clustering', 'homophily'}}"
        )

    props = torch.stack([deg_norm, cc, homophily], dim=1)
    edge_prop_sum = (props[row] + props[col]).abs().sum(dim=1)
    edge_prop_norm = edge_prop_sum / (edge_prop_sum.max() + EPS)
    edge_weights = (1.0 - beta) * 1.0 + beta * edge_prop_norm
    return torch.sparse_coo_tensor(
        edge_index, edge_weights, (num_nodes, num_nodes), device=dev
    ).coalesce()


def norm_adj(weighted_adj, use_device):
    adj = weighted_adj.coalesce()
    idx = adj.indices()
    vals = adj.values()
    n = adj.size(0)
    deg = torch.zeros(n, device=use_device).scatter_add_(0, idx[0], vals)
    dinv = deg.pow(-0.5)
    dinv[dinv == float('inf')] = 0.0
    return torch.sparse_coo_tensor(
        idx, dinv[idx[0]] * vals * dinv[idx[1]], adj.size(), device=use_device
    ).coalesce()


def compute_diffusion_matrix(norm_A, X, niter=5, alpha=0.2, use_device='cpu'):
    """PPR diffusion: S = Sum_{k=0}^{K} alpha(1-alpha)^k A^k X"""
    X = X.to(use_device)
    S = alpha * X.clone()
    Ak = X.clone()
    for k in range(1, niter + 1):
        Ak = torch.sparse.mm(norm_A, Ak)
        S = S + alpha * ((1 - alpha) ** k) * Ak
    return S
