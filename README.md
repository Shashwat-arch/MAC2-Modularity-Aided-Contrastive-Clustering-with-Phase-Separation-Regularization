# MAC² — Modularity Aided Contrastive Clustering

MAC² is a deep graph clustering (DGC) framework that combines a local
contrastive objective with a global modularity objective. Our framework addresses the
under-confident cluster assignment problem which arises from the naive combinination 
of the two objectives (modularity and contrastive) through the proposed phase-separation regularizer adapted from the
Ginzburg–Landau energy.

## Overview

Local objectives (contrastive) capture fine-grained, instance-level
structure but miss the bigger picture; global objectives (modularity)
capture overall community structure but are blind to local patterns.
Naively combining the two reduces, but does not eliminate, cluster
fragmentation and under-confident assignments. MAC² addresses these challenges with the following three components:

1. **Property Encoding Operator (PEO)** — reweights the adjacency
   matrix by structural reliability (node degree, clustering
   coefficient, feature homophily), up-weighting likely intra-cluster
   edges before training begins.
2. **Dual local–global encoder** — combines a one-hop aggregation view
   with a multi-step personalized PageRank (PPR) diffusion view to
   produce node embeddings.
3. **Combined objective with phase-separation regularizer** —
   trains on contrastive + modularity losses, plus a regularizer that
   provably drives ambiguous cluster assignments toward full
   confidence.

## Installation

```bash
git clone <repository-url>
cd mac2
pip install -r requirements.txt
```

Requires PyTorch and PyTorch Geometric. See `requirements.txt` for the
full dependency list, including `torch-scatter`, `torch-sparse`, and
`torch-cluster` (install via the PyG wheel index matching your
PyTorch/CUDA version).

## Quick start

```python
from mac2.train import run_experiment

results = run_experiment(
    dataset_name='Cora',
    epochs=100,
    gl_operator='weighted',
    lambda_gl=0.1,
)
```
To run individual datasets:

```bash
python run.py
```

To run the full benchmark sweep across all supported datasets:

```bash
python run_all_datasets.py
```

This is ledger-based and resumable — results are saved incrementally,
and reruns skip any (dataset, config) pair already completed.

## Repository structure

```
mac2/
├── __init__.py
├── device.py          # device selection (CPU/GPU)
├── preprocessing.py    # PEO, adjacency normalization, diffusion views
├── model.py            # encoder, cluster assignment head, loss functions
├── train.py             # training loop, epsilon-annealing schedule
├── evaluation.py        # clustering metrics (ACC, NMI, ARI, F1)
├── checkpointing.py     # best-model checkpoint selection and saving
├── metrics.py            # clustering_metrics class
├── datasets.py           # dataset loading (Planetoid, Amazon, Coauthor, OGB, etc.)
run.py                   # single dataset/config entry point
run_all_datasets.py       # full benchmark sweep across all datasets
requirements.txt
```

## Supported datasets

**Standard benchmarks:** Cora, CiteSeer, Pubmed, Amazon-Photo,
Amazon-Computers, Coauthor-CS, Coauthor-Physics

**Large-scale:** ogbn-arxiv, Reddit

## Results summary

Across nine real-world benchmark datasets, MAC² improves clustering
performance by 10.2% on average (NMI) over the strongest prior method,
and scales to graphs where ~83% of baseline methods run out of memory.
See the paper for complete per-dataset results and ablation studies.

## Citation

If you use this code, please cite:

```bibtex
@article{mac2,
  title={MAC$^2$: Modularity Aided Contrastive Clustering},
  author={TODO},
  journal={TODO},
  year={TODO}
}
```

## Contact

For questions, open an issue or contact TODO.
