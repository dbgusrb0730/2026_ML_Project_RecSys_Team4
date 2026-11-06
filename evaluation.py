"""Shared evaluation for kNN, MF and Hybrid. No training or tuning here."""
import numpy as np


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def top_items(scores, eligible, k):
    """Exact top-k; ties resolved by smaller encoded item index; never fill masked items."""
    ids = np.flatnonzero(eligible & np.isfinite(scores))
    take = min(k, len(ids))
    if not take:
        return ids
    values = scores[ids]
    cutoff = np.partition(values, len(ids)-take)[len(ids)-take]
    higher = ids[values > cutoff]
    ties = ids[values == cutoff][:take-len(higher)]
    chosen = np.concatenate((higher, ties))
    return chosen[np.lexsort((chosen, -scores[chosen]))]


class MetricAccumulator:
    def __init__(self, k=10, threshold=4):
        self.k, self.threshold = k, threshold
        self.se, self.n_ratings, self.n_test_ratings = 0., 0, 0
        self.rows = []

    def add(self, scores, users, known, test, candidates=None, rating_scores=None):
        scores = np.asarray(scores)
        if not np.isfinite(scores).all():
            raise ValueError('Model returned non-finite scores')
        T = test[users].tocsr()
        self.n_test_ratings += T.nnz
        K = known[users].tocsr()
        if rating_scores is not None:
            coo = T.tocoo()
            err = np.asarray(rating_scores, dtype=np.float64)[coo.row, coo.col]-coo.data
            self.se += float(err @ err)
            self.n_ratings += len(err)
        for row, uid in enumerate(users):
            eligible = np.ones(scores.shape[1], bool) if candidates is None else candidates.copy()
            eligible[K.indices[K.indptr[row]:K.indptr[row+1]]] = False
            ids = top_items(scores[row], eligible, self.k)
            lo, hi = T.indptr[row:row+2]
            relevant = T.indices[lo:hi][T.data[lo:hi] >= self.threshold]
            hits = int(np.isin(ids, relevant).sum())
            p = hits/self.k  # fixed Precision@K, even if fewer than K candidates
            r = hits/len(relevant) if len(relevant) else 0.
            f = 2*p*r/(p+r) if p+r else 0.
            self.rows.append((int(uid), len(relevant), hits, p, r, f))

    def result(self):
        rows = np.asarray(self.rows, dtype=float).reshape(-1, 6)
        relevant = rows[:, 1] > 0
        def metrics(a):
            return dict(zip(('precision','recall','f1'), a[:, 3:6].mean(0).tolist())) if len(a) else dict(precision=None,recall=None,f1=None)
        return {'rmse': (self.se/self.n_ratings)**0.5 if self.n_ratings else None,
                'n_test_ratings': self.n_test_ratings,
                'n_users_all': len(rows), 'n_users_relevant': int(relevant.sum()),
                'all_users': metrics(rows), 'relevant_users': metrics(rows[relevant])}


def evaluate_model(scores_fn, users, known, test, candidates=None, predicts_ratings=True, chunk=32):
    """MF/Hybrid integration: the SAME scores determine RMSE and ranking."""
    acc = MetricAccumulator()
    for s in range(0, len(users), chunk):
        batch = users[s:s+chunk]
        scores = scores_fn(batch)
        acc.add(scores, batch, known, test, candidates, scores if predicts_ratings else None)
    return acc.result()
