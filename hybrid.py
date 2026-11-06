"""Weighted combinations of fitted rating and ranking models."""

import numpy as np


def zscore_rows(scores, eligible=None):
    """Per-user (row-wise) z-score standardization so different ranking scales can be blended."""
    scores = np.asarray(scores, dtype=np.float32)
    if eligible is None:
        mean = scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
    else:
        mask = np.broadcast_to(np.asarray(eligible, dtype=bool), scores.shape)
        count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        mean = np.where(mask, scores, 0.0).sum(axis=1, keepdims=True) / count
        var = np.where(mask, (scores - mean) ** 2, 0.0).sum(axis=1, keepdims=True) / count
        std = np.sqrt(var)
    return ((scores - mean) / np.maximum(std, 1e-8)).astype(np.float32)


class Hybrid:
    """Weighted rating hybrid: alpha * MF rating + (1 - alpha) * kNN rating."""
    predicts_ratings = True

    def __init__(self, mf, knn, alpha):
        self.mf, self.knn, self.alpha = mf, knn, alpha

    def scores(self, users, knn_scores=None, mf_scores=None):
        mf = self.mf.scores(users) if mf_scores is None else mf_scores
        knn = self.knn.scores(users) if knn_scores is None else knn_scores
        return self.alpha * mf + (1 - self.alpha) * knn


class RankingHybrid:
    """Weighted ranking hybrid over per-user z-scored BPRMF and ImprovedUserKNN vote scores."""
    predicts_ratings = False

    def __init__(self, bpr, knn, alpha, candidates=None):
        self.bpr, self.knn, self.alpha, self.candidates = bpr, knn, alpha, candidates

    def scores(self, users, knn_z=None, bpr_z=None):
        if bpr_z is None:
            bpr_z = zscore_rows(self.bpr.scores(users), self.candidates)
        if knn_z is None:
            knn_z = zscore_rows(self.knn.ranking_scores(users), self.candidates)
        return self.alpha * bpr_z + (1 - self.alpha) * knn_z

