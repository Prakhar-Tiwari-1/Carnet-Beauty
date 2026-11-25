"""Collaborative-filtering models on the liked-review matrix, trained on GPU when available.

All models are fitted on reviews strictly before a cutoff, over every skincare product
a client liked (4-5 stars, not sampled). They score the routine catalogue.

EASE     : closed-form linear item-item model (Steck, 2019). B = -P / diag(P), P = (X'X + lambda I)^-1
RP3beta  : 3-step random walk user->item->user->item, item popularity penalised by deg^beta
iALS     : implicit alternating least squares (Hu, Koren, Volinsky, 2008), confidence 1 + alpha
BPR-MF   : matrix factorisation trained on pairwise ranking loss (Rendle et al., 2009)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from sillage.config import load_config
from sillage.data.splits import history
from sillage.device import torch_device


class LikedMatrix:
    """Binary user x item matrix of liked reviews before the cutoff."""

    def __init__(self, reviews: pd.DataFrame, cutoff: pd.Timestamp):
        min_rating = load_config()["positive_min_rating"]
        h = history(reviews, cutoff)
        liked = h[(h["rating"] >= min_rating) & ~h["incentivised"]][["author_id", "product_id"]].drop_duplicates()
        self.items = np.sort(liked["product_id"].unique())
        self.item_idx = pd.Series(np.arange(len(self.items)), index=self.items)
        users = pd.Categorical(liked["author_id"])
        self.user_idx = pd.Series(np.arange(len(users.categories)), index=users.categories)
        self.rows = users.codes.astype(np.int64)
        self.cols = self.item_idx.loc[liked["product_id"]].to_numpy()
        self.X = sparse.csr_matrix((np.ones(len(liked), np.float32), (self.rows, self.cols)),
                                   shape=(len(self.user_idx), len(self.items)))

    def user_rows(self, authors) -> np.ndarray:
        """Row index per author (-1 when the author liked nothing before the cutoff)."""
        return self.user_idx.reindex(authors).fillna(-1).astype(int).to_numpy()

    def product_cols(self, products) -> np.ndarray:
        return self.item_idx.reindex(products).fillna(-1).astype(int).to_numpy()

    def user_vectors(self, authors) -> sparse.csr_matrix:
        r = self.user_rows(authors)
        out = sparse.lil_matrix((len(r), len(self.items)), dtype=np.float32)
        for u, row in enumerate(r):
            if row >= 0:
                out[u] = self.X[row]
        return out.tocsr()


def _scatter(scores_items: np.ndarray, m: LikedMatrix, products: np.ndarray) -> np.ndarray:
    """Item-indexed scores (U x items) -> catalogue order (U x P); unknown products score 0."""
    cols = m.product_cols(products)
    out = np.zeros((scores_items.shape[0], len(products)), np.float32)
    known = cols >= 0
    out[:, known] = scores_items[:, cols[known]]
    return out


def _to_torch(x: sparse.csr_matrix) -> torch.Tensor:
    return torch.tensor(x.toarray(), device=torch_device())


# ---- EASE ---------------------------------------------------------------------

class EASE:
    def __init__(self, lam: float):
        self.lam = lam

    def fit(self, m: LikedMatrix) -> "EASE":
        dev = torch_device()
        Xt = torch.sparse_coo_tensor(np.vstack([m.rows, m.cols]), torch.ones(len(m.rows)),
                                     size=m.X.shape, device=dev)
        G = torch.sparse.mm(Xt.t(), Xt.to_dense()).double()
        G += self.lam * torch.eye(G.shape[0], device=dev, dtype=G.dtype)
        P = torch.linalg.inv(G)
        B = -P / torch.diag(P)
        B.fill_diagonal_(0)
        self.B, self.m = B.float(), m
        return self

    def score(self, authors, products) -> np.ndarray:
        U = _to_torch(self.m.user_vectors(authors))
        return _scatter((U @ self.B).cpu().numpy(), self.m, products)


# ---- RP3beta -----------------------------------------------------------------

class RP3beta:
    def __init__(self, alpha: float, beta: float, topk: int = 200):
        self.alpha, self.beta, self.topk = alpha, beta, topk

    def fit(self, m: LikedMatrix) -> "RP3beta":
        dev = torch_device()
        X = torch.tensor(m.X.toarray(), device=dev)
        du = X.sum(1, keepdim=True).clamp(min=1)
        di = X.sum(0, keepdim=True).clamp(min=1)
        P_ui = (X / du) ** self.alpha
        P_iu = (X / di).t() ** self.alpha
        W = P_iu @ P_ui
        W = W / di.pow(self.beta)
        W.fill_diagonal_(0)
        if self.topk < W.shape[1]:
            thresh = torch.topk(W, self.topk, dim=1).values[:, -1:]
            W = torch.where(W >= thresh, W, torch.zeros_like(W))
        self.W, self.m = W, m
        return self

    def score(self, authors, products) -> np.ndarray:
        U = _to_torch(self.m.user_vectors(authors))
        return _scatter((U @ self.W).cpu().numpy(), self.m, products)


# ---- iALS ---------------------------------------------------------------------

class IALS:
    def __init__(self, factors: int, reg: float, alpha: float, iterations: int = 15, seed: int = 0):
        self.factors, self.reg, self.alpha, self.iterations, self.seed = factors, reg, alpha, iterations, seed

    def _solve(self, fixed: torch.Tensor, target_idx: torch.Tensor, fixed_idx: torch.Tensor, n_target: int,
               chunk: int = 20000) -> torch.Tensor:
        """Exact least-squares update of all target factors given fixed factors."""
        dev, f = fixed.device, fixed.shape[1]
        base = fixed.t() @ fixed + self.reg * torch.eye(f, device=dev)
        out = torch.zeros((n_target, f), device=dev)
        # Deterministic accumulation: pad each target's interactions into a batch and use
        # batched matmul (fixed summation order) instead of atomic scatter-adds.
        order = torch.argsort(target_idx, stable=True)
        t_sorted, f_sorted = target_idx[order], fixed_idx[order]
        lengths = torch.bincount(t_sorted, minlength=n_target).cpu().numpy()
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        cap = 40_000_000 // f                      # max padded rows x factors per chunk
        lo = 0
        while lo < n_target:
            hi, longest = lo, 0
            while hi < n_target and max(longest, lengths[hi]) * (hi - lo + 1) <= max(cap, lengths[hi]) and hi - lo < chunk:
                longest = max(longest, lengths[hi])
                hi += 1
            n = hi - lo
            A = base.expand(n, f, f).clone()
            b = torch.zeros((n, f), device=dev)
            if longest > 0:
                s, e = int(starts[lo]), int(starts[hi - 1] + lengths[hi - 1])
                t_loc = t_sorted[s:e] - lo
                pos = torch.arange(e - s, device=dev) - torch.tensor(starts[lo:hi] - s, device=dev)[t_loc]
                Y = torch.zeros((n, int(longest), f), device=dev)
                Y[t_loc, pos] = fixed[f_sorted[s:e]]
                A += self.alpha * torch.bmm(Y.transpose(1, 2), Y)
                b += (1 + self.alpha) * Y.sum(1)
            out[lo:hi] = torch.linalg.solve(A, b)
            lo = hi
        return out

    def fit(self, m: LikedMatrix) -> "IALS":
        dev = torch_device()
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        n_u, n_i = m.X.shape
        U = (torch.randn((n_u, self.factors), generator=g) * 0.01).to(dev)
        V = (torch.randn((n_i, self.factors), generator=g) * 0.01).to(dev)
        r = torch.tensor(m.rows, device=dev)
        c = torch.tensor(m.cols, device=dev)
        for _ in range(self.iterations):
            U = self._solve(V, r, c, n_u)
            V = self._solve(U, c, r, n_i)
        self.V, self.m = V, m
        return self

    def score(self, authors, products) -> np.ndarray:
        """Fold-in: solve each client's factors from her liked items (no retraining)."""
        dev = torch_device()
        X = self.m.user_vectors(authors).tocoo()
        u = self._solve(self.V, torch.tensor(X.row, device=dev, dtype=torch.long),
                        torch.tensor(X.col, device=dev, dtype=torch.long), X.shape[0])
        return _scatter((u @ self.V.t()).cpu().numpy(), self.m, products)


# ---- BPR-MF -------------------------------------------------------------------

class BPR:
    def __init__(self, factors: int, lr: float, weight_decay: float, epochs: int = 30, batch: int = 16384, seed: int = 0):
        self.factors, self.lr, self.wd, self.epochs, self.batch, self.seed = factors, lr, weight_decay, epochs, batch, seed

    def fit(self, m: LikedMatrix) -> "BPR":
        dev = torch_device()
        torch.manual_seed(self.seed)
        n_u, n_i = m.X.shape
        U = torch.nn.Embedding(n_u, self.factors).to(dev)
        V = torch.nn.Embedding(n_i, self.factors).to(dev)
        bias = torch.nn.Embedding(n_i, 1).to(dev)
        for e in (U, V):
            torch.nn.init.normal_(e.weight, std=0.01)
        torch.nn.init.zeros_(bias.weight)
        opt = torch.optim.Adam(list(U.parameters()) + list(V.parameters()) + list(bias.parameters()), lr=self.lr)
        rows = torch.tensor(m.rows, device=dev)
        cols = torch.tensor(m.cols, device=dev)
        gen = torch.Generator(device=dev).manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(rows), device=dev, generator=gen)
            for s in range(0, len(perm), self.batch):
                idx = perm[s:s + self.batch]
                u, i = rows[idx], cols[idx]
                j = torch.randint(0, n_i, (len(idx),), device=dev, generator=gen)
                eu = U(u)
                x = (eu * (V(i) - V(j))).sum(1) + (bias(i) - bias(j)).squeeze(1)
                reg = self.wd * (eu.pow(2).sum(1) + V(i).pow(2).sum(1) + V(j).pow(2).sum(1)).mean()
                loss = -torch.nn.functional.logsigmoid(x).mean() + reg
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.U, self.V, self.b, self.m = U.weight.detach(), V.weight.detach(), bias.weight.detach().squeeze(1), m
        return self

    def score(self, authors, products) -> np.ndarray:
        r = torch.tensor(self.m.user_rows(authors), device=self.U.device)
        known = r >= 0
        s = torch.zeros((len(r), len(self.V)), device=self.U.device)
        s[known] = self.U[r[known]] @ self.V.t() + self.b
        s[~known] = self.b
        return _scatter(s.cpu().numpy(), self.m, products)
