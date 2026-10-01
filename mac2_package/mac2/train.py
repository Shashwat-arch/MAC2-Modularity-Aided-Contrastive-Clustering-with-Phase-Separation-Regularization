"""Training loop: run_experiment ties together preprocessing, model, and eval."""
import os
import gc
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_remaining_self_loops

from .device import device, EPS
from .metrics import clustering_metrics
from .preprocessing import encode_adjacency_with_properties, norm_adj, compute_diffusion_matrix
from .model import AMGC
from .evaluation import evaluate_all_clustering_methods, print_clustering_table, run_kmeans
from .checkpointing import save_best_embedding_if_better
from .datasets import DATASET_CFG, load_dataset


def run_experiment(dataset_name, epochs=100, hidden_dim=256, walk_length=3,
                   lr=0.01, log_interval=10, eval_interval=5, num_workers=2,
                   data_root='./data', diffusion_iters=5, beta=1.0,
                   gl_operator='weighted', lambda_gl=0.1, eps_start=1.0,
                   eps_end=0.05, gl_warmup=10, lambda_balance=0.5,
                   use_c=True, use_m=True, use_gl=True,
                   conv_nmi_interval=2,
                   cc_max_edges=100_000_000,
                   loss_chunk_size=5_000_000, loss_chunk_threshold=1_000_000,
                   drop_signal=None,
                   output_dir='/kaggle/working'):

    _use_gl = (lambda_gl > 0.0) if use_gl is None else use_gl

    if device.type == 'cuda':
        print(f"  [mem] CUDA allocated before run: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB / "
              f"reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    print("\n" + "="*70)
    print(f"  DATASET : {dataset_name}  | GL operator: {gl_operator} "
          f"| lambda_gl: {lambda_gl} | use_c={use_c} use_m={use_m} use_gl={_use_gl}"
          f"{f' | drop_signal={drop_signal}' if drop_signal else ''}")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)

    data, num_classes = load_dataset(dataset_name, data_root)

    y = data.y.view(-1)
    val_idx = data.val_mask.nonzero().view(-1).tolist()
    test_idx = data.test_mask.nonzero().view(-1).tolist()

    data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
    edge_index = data.edge_index
    num_nodes = data.x.size(0)

    batch_size, walks_per_node = DATASET_CFG.get(dataset_name, (4096, 5))
    print(f"  Config  : batch={batch_size} | diff_iters={diffusion_iters} "
          f"| walks={walks_per_node} | device={device}")

    # ── Preprocessing ────────────────────────────────────────────────────
    print("  Encoding adjacency…")
    w_adj = encode_adjacency_with_properties(
        edge_index, data.x, num_nodes, dataset_name, device, beta=beta,
        cc_max_edges=cc_max_edges, drop_signal=drop_signal)
    norm_A = norm_adj(w_adj, device)

    w_adj_c = w_adj.coalesce()
    if gl_operator == 'weighted':
        gl_op_index = w_adj_c.indices()
        gl_op_weight = w_adj_c.values()
    else:
        gl_op_index = edge_index.to(device)
        gl_op_weight = None

    print("  Computing AX…")
    AX = torch.sparse.mm(norm_A, data.x.to(device))

    print(f"  Computing SX (PPR K={diffusion_iters})…")
    SX = compute_diffusion_matrix(norm_A, data.x, niter=diffusion_iters, use_device=device)
    del norm_A, w_adj, w_adj_c
    gc.collect()

    # ── Model ────────────────────────────────────────────────────────────
    model = AMGC(
        input_dim=AX.size(-1),
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        edge_index=edge_index,
        walk_length=walk_length,
        context_size=walk_length,
        walks_per_node=walks_per_node,
        num_clusters=16,
        use_device=device,
        loss_chunk_size=loss_chunk_size,
        loss_chunk_threshold=loss_chunk_threshold,
    ).to(device)
    model.gl_op_index = gl_op_index
    model.gl_op_weight = gl_op_weight

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mapping = torch.zeros(num_nodes, dtype=torch.int64, device=device)

    best_acc, best_epoch, best_emb = -1, -1, None

    # ── Training loop ────────────────────────────────────────────────────
    print("\n  Training…")
    model.train()
    t0 = time.time()

    history = {'epoch': [], 'loss_total': [], 'loss_c': [], 'loss_m': [],
              'loss_gl': [], 'entropy': [], 'nmi_epoch': [], 'nmi_at_epoch': []}

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        ep_loss = ep_c = ep_m = ep_gl = 0.0
        nb = 0

        if epoch <= gl_warmup:
            eps_t, lam_gl_t = eps_start, 0.0
        else:
            prog = (epoch - gl_warmup) / max(1, epochs - gl_warmup)
            eps_t = eps_end + 0.5 * (eps_start - eps_end) * (1 + np.cos(np.pi * prog))
            lam_gl_t = lambda_gl

        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)

            unique = torch.unique(torch.cat([pos_rw, neg_rw], dim=-1))
            mapping.scatter_(0, unique, torch.arange(unique.size(0), device=device))

            model.update_B(F.embedding(unique, AX), F.embedding(unique, SX), unique)

            Z = model.get_embedding(AX, SX)
            total, lc, lm, w, lgl = model.combined_loss_ablation(
                pos_rw, neg_rw, mapping, Z,
                use_c=use_c, use_m=use_m, use_gl=_use_gl,
                eps=eps_t, lambda_gl=lam_gl_t, lambda_balance=lambda_balance)

            total.backward()
            optimizer.step()

            ep_loss += total.item(); ep_c += lc.item()
            ep_m += lm.item(); ep_gl += float(lgl); nb += 1

            if i % 10 == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        history['epoch'].append(epoch)
        history['loss_total'].append(ep_loss / nb)
        history['loss_c'].append(ep_c / nb)
        history['loss_m'].append(ep_m / nb)
        history['loss_gl'].append(ep_gl / nb)

        model.eval()
        with torch.no_grad():
            Z_full = model.get_embedding(AX, SX)
            S_full = model.get_cluster_assignments(Z_full)
            ent = -(S_full.clamp_min(EPS) * S_full.clamp_min(EPS).log()).sum(dim=1).mean().item()
        history['entropy'].append(ent)
        model.train()

        if epoch % log_interval == 0:
            print(f"  Ep {epoch:03d} ({time.time()-t_ep:.1f}s) | Loss {ep_loss/nb:.4f} "
                  f"[C:{ep_c/nb:.4f} M:{ep_m/nb:.4f} GL:{ep_gl/nb:.4f}] "
                  f"eps:{eps_t:.3f} W:[{w[0]:.3f},{w[1]:.3f}]")

        need_checkpoint_eval = (epoch % eval_interval == 0)
        need_history_eval = (epoch % conv_nmi_interval == 0)
        if need_checkpoint_eval or need_history_eval:
            model.eval()
            with torch.no_grad():
                pred = run_kmeans(Z_full.detach()[val_idx], num_classes,
                                  use_device=device, seed=42)
                cm = clustering_metrics(y[val_idx].cpu().numpy(), pred)

                if need_history_eval:
                    history['nmi_epoch'].append(cm.NMI())
                    history['nmi_at_epoch'].append(epoch)

                if need_checkpoint_eval:
                    acc = cm.clusteringAcc()[0]
                    if acc > best_acc:
                        best_acc, best_epoch = acc, epoch
                        best_emb = Z_full.detach().cpu()
                        print(f"  >>> Best Val ACC: {acc:.4f} (epoch {epoch}) <<<")
                    else:
                        print(f"  Val ACC: {acc:.4f}  (best {best_acc:.4f})")
                del pred
            model.train()
            gc.collect()

        del Z_full, S_full

    train_time = time.time() - t0
    print(f"\n  Done in {train_time:.1f}s | Best epoch:{best_epoch} | Best ACC:{best_acc:.4f}")

    history_path = f'{output_dir}/history_{dataset_name}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")

    print("\n  ── TEST RESULTS ──")
    emb_test = best_emb.numpy()[test_idx]
    y_test = y[test_idx].cpu().numpy()

    cluster_results = evaluate_all_clustering_methods(
        embeddings=emb_test, y_true=y_test, true_K=num_classes, use_device=device)
    print_clustering_table(cluster_results, dataset_name, num_classes)

    if drop_signal is None:
        save_best_embedding_if_better(
            dataset_name, best_acc, best_emb,
            config={'lambda_gl': float(lambda_gl), 'beta': float(beta),
                    'gl_operator': gl_operator, 'best_epoch': int(best_epoch),
                    'use_c': use_c, 'use_m': use_m, 'use_gl': _use_gl,
                    'hidden_dim': hidden_dim, 'walk_length': walk_length},
            out_dir=output_dir)

    del data, AX, SX, model, optimizer, loader, mapping, best_emb
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    out = {'dataset': dataset_name, 'best_epoch': int(best_epoch),
           'train_time_s': float(train_time), 'num_classes': int(num_classes),
           'gl_operator': gl_operator, 'lambda_gl': float(lambda_gl),
           'beta': float(beta), 'use_c': use_c, 'use_m': use_m, 'use_gl': _use_gl,
           'drop_signal': drop_signal}
    for method, metrics_d in cluster_results.items():
        for k, v in metrics_d.items():
            out[f"{method}_{k}"] = v
    return out
