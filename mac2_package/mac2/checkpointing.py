"""
Best-embedding checkpointing ACROSS repeated runs.

Scope: same dataset, different seeds / lambda_gl / beta / gl_operator.
Criterion: val KMeans ACC (the same `best_acc` already computed for
in-run checkpoint selection — no new metric introduced).
Policy: overwrite-if-better. A JSON sidecar per dataset tracks the best
ACC seen so far across ALL calls to run_experiment for that dataset;
the .npy is only rewritten when the current run beats it, so a worse
rerun (different seed, different lambda_gl) can never clobber your
best checkpoint.
"""
import os
import json
import numpy as np


def save_best_embedding_if_better(dataset_name, current_acc, current_emb,
                                  config, out_dir='/kaggle/working'):
    if current_emb is None:
        print(f"  [checkpoint] {dataset_name}: no embedding to save "
              f"(best_emb is None — eval_interval was never reached).")
        return False

    os.makedirs(out_dir, exist_ok=True)
    registry_path = os.path.join(out_dir, f'best_registry_{dataset_name}.json')
    emb_path = os.path.join(out_dir, f'best_embedding_{dataset_name}.npy')

    prev_best = -1.0
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            prev_best = json.load(f).get('best_acc', -1.0)

    if current_acc <= prev_best:
        print(f"  [checkpoint] {dataset_name}: run ACC {current_acc:.4f} did not "
              f"beat best-so-far {prev_best:.4f} — embedding NOT overwritten.")
        return False

    emb = current_emb.numpy() if hasattr(current_emb, 'numpy') else np.asarray(current_emb)
    np.save(emb_path, emb)
    record = {'dataset': dataset_name, 'best_acc': float(current_acc), **config}
    with open(registry_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"  [checkpoint] {dataset_name}: NEW best ACC {current_acc:.4f} "
          f"(prev {prev_best:.4f}) -> saved {emb_path}")
    return True
