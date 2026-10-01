"""
Dataset registry and loaders.

OGB datasets (ogbn-arxiv, ogbn-products, ogbn-papers100M) load via
ogb.nodeproppred.PygNodePropPredDataset -- the PyG-native OGB loader
(OGB explicitly documents this as one of three official loader
variants: PyG, DGL, and library-agnostic). Verified scale, current as
of this writing:
    ogbn-arxiv:      169,343 nodes / 1,166,243 edges / 128 features / 40 classes
    ogbn-products:   2,449,029 nodes / 61,859,140 edges / 100 features / 47 classes
    ogbn-papers100M: 111,059,956 nodes / 1,615,685,872 edges / 100 features / 172 classes
                     -- OGB's own README warns the raw download is 57GB and
                     the constructed PyG graph object is ~79GB. See the
                     module-level docstring warning in load_dataset() below.

Install: `pip install ogb --break-system-packages`. First load of each
dataset downloads and caches under `root/OGB/`.
"""
import os
import contextlib
import torch

try:
    from ogb.nodeproppred import PygNodePropPredDataset
    HAS_OGB = True
except ImportError:
    HAS_OGB = False


DATASET_CFG = {
    'Cora': (2708, 10), 'CiteSeer': (3327, 5), 'Pubmed': (4096, 10),
    'Photo': (4096, 20), 'Computers': (4096, 10), 'CS': (4096, 5),
    'Physics': (4096, 5), 'Roman-empire': (4096, 5),
    'Amazon-ratings': (4096, 5), 'Minesweeper': (4096, 5),
    'Tolokers': (4096, 5), 'Questions': (4096, 5), 'Texas': (183, 5),
    'Wisconsin': (251, 5), 'Cornell': (183, 5), 'Squirrel': (4096, 5),
    'Chameleon': (2277, 5), 'Flickr': (4096, 5),
    # Large-graph entries -- see hyperparameter guidance in the paper's
    # supplementary material / your notes for how these were chosen.
    'ogbn-arxiv': (8192, 2), 'ogbn-products': (8192, 1),
    'ogbn-papers100M': (8192, 1), 'Reddit': (8192, 2),
}

DATASET_INFO = {
    'Cora': ('planetoid', 'Cora'), 'CiteSeer': ('planetoid', 'CiteSeer'),
    'Pubmed': ('planetoid', 'Pubmed'), 'Photo': ('amazon', 'Photo'),
    'Computers': ('amazon', 'Computers'), 'CS': ('coauthor', 'CS'),
    'Physics': ('coauthor', 'Physics'),
    'Roman-empire': ('heterophilous', 'Roman-empire'),
    'Amazon-ratings': ('heterophilous', 'Amazon-ratings'),
    'Minesweeper': ('heterophilous', 'Minesweeper'),
    'Tolokers': ('heterophilous', 'Tolokers'),
    'Questions': ('heterophilous', 'Questions'),
    'Texas': ('webkb', 'Texas'), 'Wisconsin': ('webkb', 'Wisconsin'),
    'Cornell': ('webkb', 'Cornell'), 'Squirrel': ('wikipedia', 'squirrel'),
    'Chameleon': ('wikipedia', 'chameleon'), 'Flickr': ('flickr', 'Flickr'),
    'ogbn-arxiv': ('ogb', 'ogbn-arxiv'), 'ogbn-products': ('ogb', 'ogbn-products'),
    'ogbn-papers100M': ('ogb', 'ogbn-papers100M'), 'Reddit': ('reddit', 'Reddit'),
}


@contextlib.contextmanager
def _ogb_load_compat():
    """
    PyTorch 2.6 flipped torch.load's default to weights_only=True. OGB's
    PygNodePropPredDataset.__init__ and get_idx_split() both call
    torch.load(...) internally with no weights_only argument, so they
    inherit the new strict default and reject PyG's own Data/DataEdgeAttr
    globals — this is inside ogb's code, not ours, so there's no kwarg to
    pass through from here.

    Scoped monkey-patch: only affects torch.load calls made while this
    context is active (i.e. only during OGB dataset construction),
    restored immediately after. weights_only=False is safe here because
    the file being loaded is OGB's own processed cache of data downloaded
    directly from OGB's CDN, not an untrusted third-party checkpoint.
    """
    orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return orig_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        yield
    finally:
        torch.load = orig_load


def load_dataset(name, root='./data'):
    from torch_geometric.datasets import (
        Planetoid, Amazon, Coauthor, HeterophilousGraphDataset,
        WebKB, WikipediaNetwork, Flickr, Reddit)

    info = DATASET_INFO.get(name)
    if info is None:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_INFO.keys())}")
    kind, pyg_name = info

    if name == 'ogbn-papers100M':
        print(f"  [WARNING] {name}: raw download ~57GB, constructed PyG graph "
              f"object ~79GB (per OGB's own README). This pipeline builds "
              f"AX and the PPR diffusion view over the FULL graph each run — "
              f"that is very unlikely to fit even a strong single-machine "
              f"setup. Consider ogbn-products (2.4M nodes) as the realistic "
              f"large-scale target instead, unless you have a high-RAM / "
              f"multi-GPU or distributed setup and have adapted the "
              f"preprocessing to a sampled/batched path.")

    print(f"Loading {name} [{kind}]…")
    split_idx = None  # only populated for OGB, which has no native masks

    if kind == 'planetoid':
        ds = Planetoid(root=root, name=pyg_name)
    elif kind == 'amazon':
        ds = Amazon(root=root, name=pyg_name)
    elif kind == 'coauthor':
        ds = Coauthor(root=root, name=pyg_name)
    elif kind == 'heterophilous':
        ds = HeterophilousGraphDataset(root=root, name=pyg_name)
    elif kind == 'webkb':
        ds = WebKB(root=root, name=pyg_name)
    elif kind == 'wikipedia':
        ds = WikipediaNetwork(root=root, name=pyg_name)
    elif kind == 'flickr':
        ds = Flickr(root=root)
    elif kind == 'reddit':
        ds = Reddit(root=os.path.join(root, 'Reddit'))
    elif kind == 'ogb':
        if not HAS_OGB:
            raise ImportError(
                "ogb is not installed. Run `pip install ogb --break-system-packages` "
                "to load OGB datasets (ogbn-arxiv / ogbn-products / ogbn-papers100M)."
            )
        with _ogb_load_compat():
            ds = PygNodePropPredDataset(root=os.path.join(root, 'OGB'), name=pyg_name)
            split_idx = ds.get_idx_split()
    else:
        raise ValueError(f"Unhandled dataset kind '{kind}' for '{name}'.")

    data = ds[0]
    num_classes = getattr(ds, 'num_classes', None)
    if num_classes is None:
        num_classes = int(data.y.view(-1).max().item()) + 1

    if data.y.dim() == 2:
        data.y = data.y.view(-1)

    if split_idx is not None:
        n = data.num_nodes
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True

    for attr in ['train_mask', 'val_mask', 'test_mask']:
        mask = getattr(data, attr, None)
        if mask is not None and mask.dim() == 2:
            setattr(data, attr, mask[:, 0])
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        n = data.num_nodes
        idx = torch.randperm(n)
        tr, va = int(0.6 * n), int(0.2 * n)
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        data.train_mask[idx[:tr]] = True
        data.val_mask[idx[tr:tr+va]] = True
        data.test_mask[idx[tr+va:]] = True

    print(f"  Nodes:{data.num_nodes} | Edges:{data.num_edges} "
          f"| Features:{data.num_features} | Classes:{num_classes}")
    return data, num_classes
