"""
Run MAC2 across all datasets, ledger-based and resumable -- same pattern
as your other sweep scripts (encoder sweep, property ablation): writes
results after every dataset, skips completed entries on rerun, never
lets one dataset's failure kill the whole run.

DEFAULT SCOPE: the 7 original benchmarks + the 4 heterophilous sets +
the 3 WebKB sets + the 2 Wikipedia sets + Flickr, ogbn-arxiv,
ogbn-products, and Reddit -- everything that's realistic on a single
strong machine.

ogbn-papers100M is DELIBERATELY EXCLUDED from the default run. Per OGB's
own numbers, the raw download is ~57GB and the constructed PyG graph
object is ~79GB, and this pipeline builds AX + the full PPR diffusion
view over the entire graph every run (no neighbor sampling). Unless you
have very high RAM (200GB+) or have separately adapted preprocessing to
a sampled path, this will not fit -- see the warning already emitted by
mac2.datasets.load_dataset() if you try it directly. To opt in anyway,
add 'ogbn-papers100M' to DATASETS below explicitly.

Per-dataset hyperparameters below are split into two tiers:
  - TUNED: the 7 original benchmarks, values taken directly from your
    actual tuning sweeps (walk-length, K_d, lambda_gl, beta) earlier in
    this project.
  - DEFAULT: everything else (heterophilous/webkb/wikipedia/Flickr,
    ogbn-arxiv, ogbn-products, Reddit) -- reasonable starting points
    extrapolated from the tuned datasets' patterns (moderate walk
    length, diffusion_iters=5, small lambda_gl, beta=0.5), NOT verified
    by an actual sweep on these specific datasets. Treat these as a
    starting point to tune, not a substitute for tuning -- flagged
    explicitly in the ledger output so this distinction isn't lost.
"""
import os
import json
import torch
import numpy as np

from mac2 import run_experiment, DATASET_CFG

RESULTS_PATH = './all_datasets_results.json'

# ── Hyperparameter tiers ────────────────────────────────────────────────
TUNED_HPARAMS = {
    # dataset: (walk_length, diffusion_iters, lambda_gl, beta)
    # Taken directly from your actual tuning sweeps.
    'Cora':      dict(walk_length=5,  diffusion_iters=5, lambda_gl=0.1,  beta=0.75),
    'CiteSeer':  dict(walk_length=2,  diffusion_iters=5, lambda_gl=0.1,  beta=0.0),
    'Pubmed':    dict(walk_length=5,  diffusion_iters=10, lambda_gl=0.5, beta=0.0),
    'Photo':     dict(walk_length=3,  diffusion_iters=5, lambda_gl=0.1,  beta=1.0),
    'Computers': dict(walk_length=5,  diffusion_iters=5, lambda_gl=0.1,  beta=0.0),
    'CS':        dict(walk_length=5,  diffusion_iters=10, lambda_gl=0.01, beta=0.0),
    'Physics':   dict(walk_length=5,  diffusion_iters=5, lambda_gl=0.1,  beta=0.0),
}

DEFAULT_HPARAMS = dict(walk_length=5, diffusion_iters=5, lambda_gl=0.1, beta=0.5)

DATASETS = [
    # Tuned (7 original benchmarks)
    'Cora', 'CiteSeer', 'Pubmed', 'Photo', 'Computers', 'CS', 'Physics',
    # Heterophilous
    'Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions',
    # WebKB
    'Texas', 'Wisconsin', 'Cornell',
    # Wikipedia
    'Squirrel', 'Chameleon',
    # Misc
    'Flickr',
    # Large-scale (realistic on a strong single machine)
    'ogbn-arxiv', 'ogbn-products', 'Reddit',
    # 'ogbn-papers100M',  # <-- opt in explicitly, see module docstring
]

EPOCHS_BY_SCALE = {
    # Large graphs get more gradient steps per epoch (many more batches),
    # so fewer epochs is a reasonable starting point -- not verified by
    # a convergence sweep on these specific datasets, adjust if your own
    # runs show under/over-training.
    'ogbn-arxiv': 50, 'ogbn-products': 30, 'Reddit': 30,
    # 'ogbn-papers100M': 20,
}
DEFAULT_EPOCHS = 100


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_all(datasets=None, seed=42, output_dir='/kaggle/working'):
    datasets = datasets or DATASETS
    ledger = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            ledger = json.load(f)

    total = len(datasets)
    for n, dataset_name in enumerate(datasets, 1):
        if dataset_name in ledger:
            print(f"[{n}/{total}] SKIP (done): {dataset_name}")
            continue

        tuned = dataset_name in TUNED_HPARAMS
        hp = TUNED_HPARAMS.get(dataset_name, DEFAULT_HPARAMS)
        epochs = EPOCHS_BY_SCALE.get(dataset_name, DEFAULT_EPOCHS)

        print(f"\n[{n}/{total}] RUN: {dataset_name}  "
              f"({'TUNED' if tuned else 'DEFAULT (not yet tuned)'} hyperparameters)")
        set_seed(seed)
        try:
            res = run_experiment(
                dataset_name,
                epochs=epochs,
                walk_length=hp['walk_length'],
                diffusion_iters=hp['diffusion_iters'],
                lambda_gl=hp['lambda_gl'],
                beta=hp['beta'],
                gl_operator='weighted',
                output_dir=output_dir,
            )
            res['hparam_tier'] = 'tuned' if tuned else 'default'
            ledger[dataset_name] = res
        except Exception as e:
            print(f"  !! ERROR on {dataset_name}: {e}")
            ledger[dataset_name] = {'ERROR': repr(e)}

        with open(RESULTS_PATH, 'w') as f:
            json.dump(ledger, f, indent=2)
        print(f"  -> saved to {RESULTS_PATH}")

    print(f"\nAll requested datasets complete. Results at: {RESULTS_PATH}")
    return ledger


if __name__ == '__main__':
    run_all()
