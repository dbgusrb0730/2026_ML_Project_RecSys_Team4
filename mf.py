"""Matrix factorization models for rating prediction and ranking."""

import time

import numpy as np

from evaluation import rmse


class MatrixFactorization:
    """Model-based collaborative filtering: biased matrix factorization.

    r(u,i) ~ mu + b_u + b_i + p_u . q_i
    Loss: sum of squared errors on observed ratings + reg * (|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2)
    Optimizer: mini-batch stochastic gradient descent.
    """
    predicts_ratings = True

    def __init__(self, n_factors=32, lr=0.01, reg=0.05, n_epochs=30, batch_size=1024, patience=2, seed=0):
        self.n_factors, self.lr, self.reg = n_factors, lr, reg
        self.n_epochs, self.batch_size, self.patience, self.seed = n_epochs, batch_size, patience, seed

    def fit(self, u, i, r, shape, val=None):
        """val: optional (u, i, r) validation arrays used for early stopping."""
        rng = np.random.default_rng(self.seed)
        n_users, n_items = shape
        self.mu = float(r.mean())
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        self.bu = np.zeros(n_users, np.float32)
        self.bi = np.zeros(n_items, np.float32)
        self.history, best, best_rmse, bad = [], None, np.inf, 0
        lr, reg = self.lr, self.reg
        for epoch in range(1, self.n_epochs + 1):
            perm = rng.permutation(len(r))                               # shuffle once per epoch
            for s in range(0, len(r), self.batch_size):
                idx = perm[s:s + self.batch_size]
                uu, ii, rr = u[idx], i[idx], r[idx]
                pu, qi = self.P[uu], self.Q[ii]
                err = rr - (self.mu + self.bu[uu] + self.bi[ii] + (pu * qi).sum(1))
                # Gradient step on the regularized squared error. np.add.at accumulates correctly
                # when the same user/item appears several times in one batch.
                np.add.at(self.bu, uu, lr * (err - reg * self.bu[uu]))
                np.add.at(self.bi, ii, lr * (err - reg * self.bi[ii]))
                np.add.at(self.P, uu, lr * (err[:, None] * qi - reg * pu))
                np.add.at(self.Q, ii, lr * (err[:, None] * pu - reg * qi))
            entry = {"epoch": epoch, "train_rmse": rmse(r, self.predict(u, i))}
            if val is not None:
                entry["val_rmse"] = rmse(val[2], self.predict(val[0], val[1]))
            self.history.append(entry)
            if val is None:
                continue
            if entry["val_rmse"] < best_rmse - 1e-4:
                best_rmse, bad = entry["val_rmse"], 0
                best = (self.P.copy(), self.Q.copy(), self.bu.copy(), self.bi.copy(), epoch)
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best is not None:                                             # restore best epoch
            self.P, self.Q, self.bu, self.bi, self.best_epoch = best
        else:
            self.best_epoch = self.n_epochs
        return self

    def predict(self, u, i):
        pred = self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)
        return np.clip(pred, 1, 5)

    def scores(self, users):
        pred = self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T
        return np.clip(pred, 1, 5).astype(np.float32)


class FixedMF(MatrixFactorization):
    """Problem 1 fix: users/items without any training rating get p_u = 0 / q_i = 0 (and b = 0).

    In the base class, such rows keep their random N(0, 0.1) initial vectors, because SGD never
    touches them, and scores() adds p_u . (random q_i) to their predictions. Here their prediction
    becomes mu + b_u (+ b_i), the best guess without information.

    With n_factors=0 this is the bias-only model mu + b_u + b_i (problem 3), trained by the same
    SGD and L2 code of the parent class.
    """

    def fit(self, u, i, r, shape, val=None):
        # np.bincount(x, minlength=n): counts how often each integer 0..n-1 appears in x
        # (length n even if the largest values never appear). > 0 -> "has at least one training rating".
        self.seen_users = np.bincount(u, minlength=shape[0]) > 0
        self.seen_items = np.bincount(i, minlength=shape[1]) > 0
        # The parent fit calls self.predict for the per-epoch train/val RMSE, so the override below
        # also removes the random-vector noise from the early-stopping decision. SGD updates do not
        # use predict, so the training path itself is unchanged.
        self._sgd_fit(u, i, r, shape, val)
        # Biases of unseen rows start at 0 and are never updated; verify that before zeroing vectors.
        assert not self.bu[~self.seen_users].any() and not self.bi[~self.seen_items].any()
        self.P[~self.seen_users] = 0
        self.Q[~self.seen_items] = 0
        return self

    def _sgd_fit(self, u, i, r, shape, val):
        """The training loop: the unchanged parent fit (recsys.py:200-241). SplitRegMF replaces it."""
        MatrixFactorization.fit(self, u, i, r, shape, val)

    def predict(self, u, i):
        seen = self.seen_users[u] & self.seen_items[i]
        pred = self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1) * seen
        return np.clip(pred, 1, 5)

    def raw_scores(self, users):
        """scores() without the [1, 5] clip, used for ranking only (clipping-tie diagnostic)."""
        return (self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T).astype(np.float32)


class SplitRegMF(FixedMF):
    """FixedMF with separate L2 strengths for the biases and the latent vectors (experiment B, approved addition).

    Loss: sum of squared errors + reg_bias * (b_u^2 + b_i^2) + reg_factor * (|p_u|^2 + |q_i|^2)
    In recsys.MatrixFactorization one `reg` does both, so a large reg also shrinks the biases of MF;
    then "MF worse than bias only" cannot tell whether the latent factors or the squeezed biases are
    to blame. With reg_bias == reg_factor this class gives exactly the same result as FixedMF.
    """

    def __init__(self, n_factors=32, lr=0.01, reg_bias=0.05, reg_factor=0.05, n_epochs=30, batch_size=1024,
                 patience=2, seed=0):
        super().__init__(n_factors, lr, reg_factor, n_epochs, batch_size, patience, seed)
        self.reg_bias, self.reg_factor = reg_bias, reg_factor

    def _sgd_fit(self, u, i, r, shape, val=None):
        # Copied from recsys.MatrixFactorization.fit (recsys.py:200-241, team base code) line by line.
        # Only change: the regularization terms of the bias updates use reg_bias (recsys.py:220-221),
        # those of the vector updates use reg_factor (recsys.py:222-223).
        rng = np.random.default_rng(self.seed)
        n_users, n_items = shape
        self.mu = float(r.mean())
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        self.bu = np.zeros(n_users, np.float32)
        self.bi = np.zeros(n_items, np.float32)
        self.history, best, best_rmse, bad = [], None, np.inf, 0
        lr, reg_b, reg_f = self.lr, self.reg_bias, self.reg_factor
        for epoch in range(1, self.n_epochs + 1):
            perm = rng.permutation(len(r))                               # shuffle once per epoch
            for s in range(0, len(r), self.batch_size):
                idx = perm[s:s + self.batch_size]
                uu, ii, rr = u[idx], i[idx], r[idx]
                pu, qi = self.P[uu], self.Q[ii]
                err = rr - (self.mu + self.bu[uu] + self.bi[ii] + (pu * qi).sum(1))
                # np.add.at(a, idx, v): a[idx] += v, adding every value even when idx repeats
                # (a plain a[idx] += v would keep only one update per repeated user/item in the batch).
                np.add.at(self.bu, uu, lr * (err - reg_b * self.bu[uu]))
                np.add.at(self.bi, ii, lr * (err - reg_b * self.bi[ii]))
                np.add.at(self.P, uu, lr * (err[:, None] * qi - reg_f * pu))
                np.add.at(self.Q, ii, lr * (err[:, None] * pu - reg_f * qi))
            entry = {"epoch": epoch, "train_rmse": rmse(r, self.predict(u, i))}
            if val is not None:
                entry["val_rmse"] = rmse(val[2], self.predict(val[0], val[1]))
            self.history.append(entry)
            if val is None:
                continue
            if entry["val_rmse"] < best_rmse - 1e-4:
                best_rmse, bad = entry["val_rmse"], 0
                best = (self.P.copy(), self.Q.copy(), self.bu.copy(), self.bi.copy(), epoch)
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best is not None:                                             # restore best epoch
            self.P, self.Q, self.bu, self.bi, self.best_epoch = best
        else:
            self.best_epoch = self.n_epochs


class BPRMF:
    """Same factor structure as the MSE model (P, Q, b_i), trained with the BPR ranking loss.

    Score   : x_ui = b_i + p_u . q_i                 (b_u and mu do not change a user's ranking, so they are left out)
    Triples : (u, i, j), i = a positive item of u, j = an item u did not rate, drawn uniformly
    Loss    : L = - sum ln sigma(x_ui - x_uj) + reg * (|p_u|^2 + |q_i|^2 + |q_j|^2 + b_i^2 + b_j^2)
    Update  : with x = x_ui - x_uj and z = 1 - sigma(x) = 1 / (1 + e^x)   (gradient ascent on ln sigma)
              p_u += lr * (z * (q_i - q_j) - reg * p_u)
              q_i += lr * (z * p_u - reg * q_i)
              q_j += lr * (-z * p_u - reg * q_j)
              b_i += lr * (z - reg * b_i)
              b_j += lr * (-z - reg * b_j)
    One epoch = one triple per positive pair, in mini-batches of batch_size (np.add.at, as recsys.py:220-223).
    Users without any positive item get p_u = 0 (they are never updated; same principle as FixedMF).
    """
    predicts_ratings = False

    def __init__(self, n_factors=32, lr=0.05, reg=0.01, n_epochs=40, batch_size=1024, eval_every=2, seed=0):
        self.n_factors, self.lr, self.reg, self.n_epochs = n_factors, lr, reg, n_epochs
        self.batch_size, self.eval_every, self.seed = batch_size, eval_every, seed

    def _negatives(self, rng, u, keys, n_items):
        """One uniformly drawn unrated item per positive pair. keys: sorted u * n_items + item of rated pairs.
        rng.integers(0, n, size): uniform integers in 0..n-1. np.searchsorted(keys, x): insertion positions,
        used to test membership of x in the sorted array keys."""
        j = rng.integers(0, n_items, len(u))
        while True:
            k = u.astype(np.int64) * n_items + j
            pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
            bad = keys[pos] == k
            if not bad.any():
                return j
            j[bad] = rng.integers(0, n_items, int(bad.sum()))

    def fit(self, u, i, shape, rated, val_fn=None, eval_triples=None, stop_patience=None, stop_at=None):
        """u, i: positive pairs. rated: (users, items) of all rated pairs (excluded from negatives).
        val_fn(model) -> (hits, recall) on validation, evaluated every eval_every epochs; the best epoch is restored.
        eval_triples: optional fixed (u, i, j) arrays; after every epoch the BPR loss on them is recorded as
        'eval_loss' (no sampling noise; uses no random numbers, so training is unchanged). Used by E-1 (#32).
        stop_patience: stop when (current epoch - best epoch) >= stop_patience at an evaluation epoch (E_night, #40).
        stop_at: Unix time (time.time()); after the epoch that reaches it, evaluate once more if needed and stop.
        self.stopped_by records why training ended: "max_epochs", "no_improvement" or "deadline"."""
        rng = np.random.default_rng(self.seed)
        n_users, n_items = shape
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        self.bi = np.zeros(n_items, np.float32)
        self.has_pos = np.bincount(u, minlength=n_users) > 0
        keys = np.unique(rated[0].astype(np.int64) * n_items + rated[1])
        self.history, best, lr, reg = [], None, self.lr, self.reg
        for epoch in range(1, self.n_epochs + 1):
            perm = rng.permutation(len(u))
            jj_all = self._negatives(rng, u[perm], keys, n_items)
            loss = 0.0
            for s in range(0, len(u), self.batch_size):
                idx = perm[s:s + self.batch_size]
                uu, ii, jj = u[idx], i[idx], jj_all[s:s + self.batch_size]
                pu, qi, qj = self.P[uu], self.Q[ii], self.Q[jj]
                x = self.bi[ii] - self.bi[jj] + (pu * (qi - qj)).sum(1)
                # np.logaddexp(0, -x) = ln(1 + e^-x) = -ln sigma(x), computed without overflow
                loss += float(np.logaddexp(0, -x).sum())
                z = (1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))).astype(np.float32)
                bi_i, bi_j = self.bi[ii], self.bi[jj]
                np.add.at(self.P, uu, lr * (z[:, None] * (qi - qj) - reg * pu))
                np.add.at(self.Q, ii, lr * (z[:, None] * pu - reg * qi))
                np.add.at(self.Q, jj, lr * (-z[:, None] * pu - reg * qj))
                np.add.at(self.bi, ii, lr * (z - reg * bi_i))
                np.add.at(self.bi, jj, lr * (-z - reg * bi_j))
            entry = {"epoch": epoch, "bpr_loss": loss / len(u)}
            if eval_triples is not None:
                eu, ei, ej = eval_triples
                xe = self.bi[ei] - self.bi[ej] + (self.P[eu] * (self.Q[ei] - self.Q[ej])).sum(1)
                entry["eval_loss"] = float(np.logaddexp(0, -xe).mean())
            deadline_hit = stop_at is not None and time.time() >= stop_at
            if val_fn is not None and (epoch % self.eval_every == 0 or epoch == self.n_epochs or deadline_hit):
                self.P[~self.has_pos] = 0
                hits, recall = val_fn(self)
                entry.update({"val_hits": hits, "val_recall": recall})
                if best is None or (hits, recall) > best[0]:
                    best = ((hits, recall), self.P.copy(), self.Q.copy(), self.bi.copy(), epoch)
                if stop_patience is not None and epoch - best[4] >= stop_patience:
                    self.history.append(entry)
                    self.stopped_by = "no_improvement"
                    break
            self.history.append(entry)
            if deadline_hit:
                self.stopped_by = "deadline"
                break
        else:
            self.stopped_by = "max_epochs"
        if best is not None:
            _, self.P, self.Q, self.bi, self.best_epoch = best
        else:
            self.best_epoch = len(self.history)                 # = n_epochs unless stopped by stop_at
        self.P[~self.has_pos] = 0
        assert not self.P[~self.has_pos].any()
        return self

    def scores(self, users):
        return (self.bi[None, :] + self.P[users] @ self.Q.T).astype(np.float32)

    raw_scores = scores
