"""AMGC model: encoder + contrastive/modularity/Allen-Cahn losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree

from .device import EPS

try:
    import torch_cluster
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None


class AMGC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, edge_index,
                 walk_length=3, context_size=3, walks_per_node=1,
                 p=1.0, q=1.0, num_negative_samples=1,
                 num_clusters=16, use_device='cpu',
                 loss_chunk_size=5_000_000, loss_chunk_threshold=1_000_000):
        super().__init__()
        if random_walk is None:
            raise ImportError('torch-cluster is required.')

        N = maybe_num_nodes(edge_index)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to('cpu')

        assert walk_length >= context_size
        self.hidden_dim = hidden_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p, self.q = p, q
        self.num_negative_samples = num_negative_samples
        self.num_clusters = num_clusters
        self.clus = None
        self.embedding = None
        self.use_device = use_device

        self.w1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w1.bias.data.fill_(0.0)
        self.w2.bias.data.fill_(0.0)
        self.iden = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.cluster_layer = nn.Linear(hidden_dim, num_clusters)
        self.loss_w_contrastive = nn.Parameter(torch.tensor(1.0))
        self.loss_w_modularity = nn.Parameter(torch.tensor(1.0))
        self.edge_index = edge_index.to(use_device)
        self.num_nodes = N

        # ── memory-bounded edge-reduction settings for modularity/GL loss ──
        self.loss_chunk_size = loss_chunk_size
        self.loss_chunk_threshold = loss_chunk_threshold

        # ── Allen–Cahn / Ginzburg–Landau operator (set after preprocessing) ──
        self.gl_op_index = None
        self.gl_op_weight = None

    # ── Embedding ────────────────────────────────────────────────────────
    def get_embedding(self, X1, X2):
        return normalize(self.w1(X1) + self.w2(X2) + self.iden)

    def update_B(self, X1, X2, unique):
        self.embedding = normalize(
            self.w1(X1) + self.w2(X2) + F.embedding(unique, self.iden)
        )

    def get_cluster_assignments(self, Z):
        return F.softmax(self.cluster_layer(Z), dim=1)

    # ── Modularity loss ──────────────────────────────────────────────────
    def _chunked_trace_SAS(self, S, row, col):
        """
        Exact (S[row] * S[col]).sum() — same value, same gradient — but
        computed in edge-dimension chunks via gradient checkpointing, so
        only one chunk's (chunk_size, K) intermediate is ever resident on
        GPU at once, instead of all n_edges of them simultaneously.
        """
        n = row.size(0)
        if n <= self.loss_chunk_threshold:
            return (S[row] * S[col]).sum()

        def _fn(S_in, r, c):
            return (S_in[r] * S_in[c]).sum()

        total = S.new_zeros(())
        for start in range(0, n, self.loss_chunk_size):
            end = min(start + self.loss_chunk_size, n)
            total = total + checkpoint(_fn, S, row[start:end], col[start:end],
                                       use_reentrant=False)
        return total

    def modularity_loss(self, Z, S):
        """
        Sparse DMoN-style modularity loss.
            L = -1/(2m) * tr(S^T B S),   B = A - dd^T / 2m
        """
        ei = self.edge_index
        n = Z.size(0)
        row, col = ei[0], ei[1]
        deg = degree(row, n, dtype=torch.float).to(Z.device)
        m = ei.size(1) / 2.0
        trace_SAS = self._chunked_trace_SAS(S, row, col)
        Std = S.t() @ deg
        trace_SddS = (Std * Std).sum()
        return -(trace_SAS - trace_SddS / (2.0 * m)) / (2.0 * m)

    # ── Allen–Cahn / Ginzburg–Landau loss ────────────────────────────────
    def _sparse_dirichlet(self, S, op_index, op_weight=None):
        """
        tr(S^T L S) = sum_{(i,j) in E} w_ij * ||S_i - S_j||^2
        Edge-gather form — never materializes a dense (N, N) tensor.
        Symmetric edge_index double-counts undirected edges -> /2.
        """
        row, col = op_index[0], op_index[1]
        n = row.size(0)

        if n <= self.loss_chunk_threshold:
            diff = S[row] - S[col]
            sq = (diff * diff).sum(dim=1)
            if op_weight is not None:
                sq = sq * op_weight
            return sq.sum() / 2.0

        total = S.new_zeros(())
        for start in range(0, n, self.loss_chunk_size):
            end = min(start + self.loss_chunk_size, n)
            w_chunk = op_weight[start:end] if op_weight is not None else None

            def _fn(S_in, r, c, w=w_chunk):
                diff = S_in[r] - S_in[c]
                sq = (diff * diff).sum(dim=1)
                if w is not None:
                    sq = sq * w
                return sq.sum()

            total = total + checkpoint(_fn, S, row[start:end], col[start:end],
                                       use_reentrant=False)
        return total / 2.0

    def _double_well(self, S):
        """
        W(s_i) = prod_k ||s_i - e_k||^2, mean over nodes.
        Zero iff every row is one-hot; maximal at the simplex barycenter.
        """
        s_sq = (S * S).sum(dim=1, keepdim=True)
        per_k = s_sq - 2.0 * S + 1.0
        per_k = per_k.clamp_min(EPS)
        W = per_k.prod(dim=1)
        return W.mean()

    def _balance_entropy(self, S):
        """Collapse guard: negative entropy of the mean cluster distribution."""
        p = S.mean(dim=0).clamp_min(EPS)
        return (p * p.log()).sum()

    def allen_cahn_loss(self, S, eps, op_index, op_weight=None,
                        lambda_balance=0.5):
        """
        L_GL = (eps/2)*Dirichlet + (1/eps)*double_well + lambda_balance*balance
        """
        n_edges = op_index.size(1) / 2.0
        dirichlet = self._sparse_dirichlet(S, op_index, op_weight) / (n_edges + EPS)
        well = self._double_well(S)
        balance = self._balance_entropy(S)
        return (eps / 2.0) * dirichlet + (1.0 / eps) * well + lambda_balance * balance

    # ── Samplers ─────────────────────────────────────────────────────────
    def loader(self, **kwargs):
        return DataLoader(range(self.adj.sparse_size(0)),
                          collate_fn=self.sample, **kwargs)

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rptr, col, _ = self.adj.csr()
        rw = random_walk(rptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        return torch.cat([rw[:, j:j+self.context_size]
                          for j in range(1+self.walk_length+1-self.context_size)], dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        if self.clus is None:
            rw = torch.randint(self.adj.sparse_size(0),
                               (batch.size(0), self.walk_length * self.num_negative_samples))
        else:
            rw = torch.empty(batch.size(0),
                             self.walk_length * self.num_negative_samples, dtype=torch.long)
            for i in range(self.clus.max() + 1):
                cluster = (self.clus[batch.view(-1)] == i).nonzero(as_tuple=True)[0].tolist()
                neg_ind = (self.clus != i).nonzero(as_tuple=True)[0]
                rw[cluster] = neg_ind[torch.randint(neg_ind.size(0),
                    (len(cluster), self.walk_length * self.num_negative_samples))]
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)
        return torch.cat([rw[:, j:j+self.context_size]
                          for j in range(1+self.walk_length+1-self.context_size)], dim=0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    # ── Contrastive loss ─────────────────────────────────────────────────
    def contrastive_loss(self, pos_rw, neg_rw, mapping):
        pos_rw = F.embedding(pos_rw.view(-1), mapping.view(-1, 1)).view(pos_rw.size())
        neg_rw = F.embedding(neg_rw.view(-1), mapping.view(-1, 1)).view(neg_rw.size())

        s, r = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        hs = F.embedding(s, self.embedding).view(pos_rw.size(0), 1, self.hidden_dim)
        hr = F.embedding(r.view(-1), self.embedding).view(pos_rw.size(0), -1, self.hidden_dim)
        pos_l = torch.logsumexp((hs * hr).sum(-1), dim=-1)

        s, r = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        hs = F.embedding(s, self.embedding).view(neg_rw.size(0), 1, self.hidden_dim)
        hr = F.embedding(r.view(-1), self.embedding).view(neg_rw.size(0), -1, self.hidden_dim)
        neg_l = torch.logsumexp((hs * hr).sum(-1), dim=-1)
        neg_l = torch.logsumexp(torch.cat([neg_l.view(-1, 1), pos_l.view(-1, 1)], dim=-1), dim=-1)
        return -torch.mean(torch.exp(pos_l - neg_l))

    # ── Combined loss (contrastive + modularity + Allen–Cahn) ─────────────
    def combined_loss(self, pos_rw, neg_rw, mapping, Z,
                      eps=1.0, lambda_gl=0.0, lambda_balance=0.5):
        lc = self.contrastive_loss(pos_rw, neg_rw, mapping)
        S = self.get_cluster_assignments(Z)
        lm = self.modularity_loss(Z, S)

        w = F.softmax(torch.stack([self.loss_w_contrastive,
                                   self.loss_w_modularity]), dim=0)
        base = w[0] * lc + w[1] * lm

        if lambda_gl > 0.0 and self.gl_op_index is not None:
            lgl = self.allen_cahn_loss(S, eps, self.gl_op_index,
                                       self.gl_op_weight, lambda_balance)
            total = base + lambda_gl * lgl
            return total, lc, lm, w, lgl.detach()

        zero = torch.zeros((), device=Z.device)
        return base, lc, lm, w, zero

    # ── Ablation-aware combined loss (used by run_experiment) ─────────────
    def combined_loss_ablation(self, pos_rw, neg_rw, mapping, Z,
                               use_c, use_m, use_gl,
                               eps=1.0, lambda_gl=0.1, lambda_balance=0.5):
        lc = self.contrastive_loss(pos_rw, neg_rw, mapping)
        S = self.get_cluster_assignments(Z)
        lm = self.modularity_loss(Z, S)

        if use_c and use_m:
            w = F.softmax(torch.stack([self.loss_w_contrastive,
                                       self.loss_w_modularity]), dim=0)
            base = w[0] * lc + w[1] * lm
        elif use_c:
            base = lc
            w = torch.tensor([1.0, 0.0], device=Z.device)
        elif use_m:
            base = lm
            w = torch.tensor([0.0, 1.0], device=Z.device)
        else:
            base = torch.zeros((), device=Z.device)
            w = torch.tensor([0.0, 0.0], device=Z.device)

        if use_gl and self.gl_op_index is not None:
            lgl = self.allen_cahn_loss(S, eps, self.gl_op_index,
                                       self.gl_op_weight, lambda_balance)
            gl_weight = lambda_gl if (use_c or use_m) else 1.0
            total = base + gl_weight * lgl
            return total, lc.detach(), lm.detach(), w, lgl.detach()

        zero = torch.zeros((), device=Z.device)
        return base, lc.detach(), lm.detach(), w, zero
