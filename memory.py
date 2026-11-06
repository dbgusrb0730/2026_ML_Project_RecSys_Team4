"""User-neighborhood models for memory-based collaborative filtering."""

import numpy as np
from scipy.sparse import csr_matrix


class ImprovedUserKNN:
    def __init__(self, k=500, popularity_power=0.0, shrink=5.0):
        self.k, self.popularity_power, self.shrink = k, popularity_power, shrink

    def fit(self, u, i, r, shape, similarity_data=None, neighbor_mask=None):
        self.shape = shape
        self.mu = float(r.mean())
        self.counts = np.bincount(i, minlength=shape[1]).astype(np.float32)
        uc = np.bincount(u, minlength=shape[0])
        self.bu = np.zeros(shape[0], np.float32)
        self.bi = np.zeros(shape[1], np.float32)
        # Regularized user/item mean corrections; these are baseline statistics,
        # not latent factors. All inputs here are training ratings only.
        for _ in range(5):
            self.bi = (np.bincount(i, weights=r-self.mu-self.bu[u], minlength=shape[1]) /
                       (self.counts+10)).astype(np.float32)
            self.bu = (np.bincount(u, weights=r-self.mu-self.bi[i], minlength=shape[0]) /
                       (uc+10)).astype(np.float32)
        self.B = csr_matrix((np.ones_like(r), (u, i)), shape=shape)
        self.C = csr_matrix((r-self.mu-self.bu[u]-self.bi[i], (u, i)), shape=shape)
        liked = r >= 4
        self.L = csr_matrix((np.ones(liked.sum(), np.float32), (u[liked], i[liked])), shape=shape)
        su, si, sr = (u, i, r) if similarity_data is None else similarity_data
        positive = sr >= 4
        A = csr_matrix((np.ones(positive.sum(), np.float32),
                        (su[positive], si[positive])), shape=shape)
        norm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
        self.A = A.multiply((1/np.maximum(norm, 1e-9))[:, None]).tocsr()
        self.AT = self.A.T.tocsr()
        self.neighbor_mask = neighbor_mask
        return self

    def baseline_scores(self, users):
        """Ablation: identical fitted biases, but no neighbor residual."""
        users = np.asarray(users, dtype=int)
        return np.clip(self.mu + self.bu[users, None] + self.bi[None, :], 1, 5).astype(np.float32)

    def score_pair(self, users):
        """Bound similarity memory even when called by the team's Hybrid."""
        users = np.asarray(users, dtype=int)
        rating = np.empty((len(users), self.shape[1]), np.float32)
        rank = np.empty_like(rating)
        for start in range(0, len(users), 32):
            rating[start:start+32], rank[start:start+32] = self._score_pair(users[start:start+32])
        return rating, rank

    def _score_pair(self, users):
        users = np.asarray(users)
        sim = (self.A[users] @ self.AT).toarray()
        sim[np.arange(len(users)), users] = 0
        if self.neighbor_mask is not None:
            sim[:, ~self.neighbor_mask] = 0
        k = min(self.k, self.shape[0]-1)
        nb = np.argpartition(-sim, k-1, axis=1)[:, :k]
        weights = np.take_along_axis(sim, nb, axis=1)
        W = csr_matrix((weights.ravel(),
                        (np.repeat(np.arange(len(users)), k), nb.ravel())),
                       shape=(len(users), self.shape[0]))
        W.eliminate_zeros()
        den = (W @ self.B).toarray()
        residual = (W @ self.C).toarray()
        rating = self.mu + self.bu[users, None] + self.bi[None, :] + residual/(den+self.shrink)
        votes = (W @ self.L).toarray()
        rank = votes / np.maximum(self.counts, 1)[None, :]**self.popularity_power
        # Only unsupported items fall back to popularity, below every supported item.
        fallback = -1.0 + 0.5*self.counts/max(float(self.counts.max()), 1)
        rank = np.where(votes > 0, rank, fallback[None, :])
        return np.clip(rating, 1, 5).astype(np.float32), rank.astype(np.float32)

    def scores(self, users):
        """Rating predictions compatible with the original Hybrid.scores interface."""
        return self.score_pair(users)[0]

    def ranking_scores(self, users):
        return self.score_pair(users)[1]


class UserKNN:
    """Memory-based collaborative filtering: user-based k-nearest neighbors.

    Similarity: cosine similarity between mean-centered rating vectors (adjusted cosine).
    Prediction: r(u,i) = mean(u) + sum_v sim(u,v) * (r(v,i) - mean(v)) / (sum_v |sim(u,v)| + shrink)
    over the k most similar users v. Only positive similarities are used. `shrink` pulls predictions
    based on few or weak neighbors back toward the user's mean.
    """
    predicts_ratings = True

    def __init__(self, k=50, shrink=1.0, chunk=256):
        self.k, self.shrink, self.chunk = k, shrink, chunk

    def fit(self, u, i, r, shape, neighbor_mask=None):
        """neighbor_mask: optional boolean array over users; only these users may act as neighbors."""
        n_users, _ = shape
        counts = np.bincount(u, minlength=n_users)
        sums = np.bincount(u, weights=r, minlength=n_users)
        self.user_mean = np.where(counts > 0, sums / np.maximum(counts, 1), r.mean()).astype(np.float32)
        self.C = csr_matrix((r - self.user_mean[u], (u, i)), shape=shape)   # centered ratings
        self.B = csr_matrix((np.ones_like(r), (u, i)), shape=shape)          # 1 where a rating exists
        self.norm = np.sqrt(np.asarray(self.C.multiply(self.C).sum(axis=1)).ravel()) + 1e-9
        self.neighbor_mask = neighbor_mask
        return self

    def scores(self, users):
        users = np.asarray(users)
        out = np.empty((len(users), self.C.shape[1]), dtype=np.float32)
        CT = self.C.T.tocsr()
        for s in range(0, len(users), self.chunk):
            batch = users[s:s + self.chunk]
            sim = (self.C[batch] @ CT).toarray()                         # dot products with all users
            sim /= self.norm[batch][:, None] * self.norm[None, :]       # -> cosine similarity
            sim[np.arange(len(batch)), batch] = 0                        # a user is not their own neighbor
            if self.neighbor_mask is not None:
                sim[:, ~self.neighbor_mask] = 0
            top = np.argpartition(-sim, self.k, axis=1)[:, :self.k]      # k largest, unordered
            for row, user in enumerate(batch):
                nb = top[row]
                w = np.maximum(sim[row, nb], 0)                          # positive similarities only
                num = self.C[nb].T @ w                                   # weighted rating deviations
                den = self.B[nb].T @ w                                   # total weight per item
                out[s + row] = self.user_mean[user] + num / (den + self.shrink)
        return np.clip(out, 1, 5)
