"""Weighted combination of two fitted rating models."""


class Hybrid:
    """Weighted hybrid: alpha * MF prediction + (1 - alpha) * kNN prediction."""
    predicts_ratings = True

    def __init__(self, mf, knn, alpha):
        self.mf, self.knn, self.alpha = mf, knn, alpha

    def scores(self, users, knn_scores=None, mf_scores=None):
        mf = self.mf.scores(users) if mf_scores is None else mf_scores
        knn = self.knn.scores(users) if knn_scores is None else knn_scores
        return self.alpha * mf + (1 - self.alpha) * knn
