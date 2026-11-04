"""Cross-domain product recommendation: from Software to Video Games (ML term project, Team 4).

One file, organized in sections:
  0. Configuration      paths and experiment settings
  1. Data               loading, per-user time split, ID -> index encoding
  2. Models             GlobalMean, Popularity, UserKNN, MatrixFactorization, Hybrid
  3. Evaluation         RMSE and Precision/Recall/F1@K
  4. Data exploration   statistics and EDA figures
  5. Experiments        Software (tuning), Video Games, cross-domain
  6. Result figures
  7. Main               runs everything and writes outputs/

Usage:  python recsys.py
"""
import gzip
import json
import os
import time

import matplotlib
matplotlib.use("Agg")                     # draw figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ==================================================================================================
# 0. Configuration
# ==================================================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")

SEED = 42
N_EVAL_USERS = 10_000          # users sampled per experiment for evaluation (same sample for every model)
TOP_K = 10
THRESHOLD = 4.0                # rating >= 4 means the user liked the item
MAX_EPOCHS = 50
MF_GRID = [(f, lr, reg) for f in (16, 64) for lr in (0.01, 0.02) for reg in (0.02, 0.05, 0.1)]  # (n_factors, learning rate, L2)
KNN_GRID = (20, 50, 100)
ALPHAS = np.round(np.linspace(0, 1, 11), 1)
CROSS_TARGET_FRAC = 0.2        # share of the shared users whose Video Games ratings are hidden

COLORS = {"Global mean": "#9e9e9e", "Popularity": "#bdbdbd", "User kNN": "#4c72b0", "MF": "#dd8452", "Hybrid": "#55a868"}


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ==================================================================================================
# 1. Data: rating files have the columns user_id, parent_asin (item ID), rating (1-5), timestamp (ms)
# ==================================================================================================
def load_ratings(category):
    """Read one category's 5-core rating file into a DataFrame."""
    return pd.read_csv(os.path.join(DATA_DIR, f"{category}.csv"))


def load_titles(category, asins):
    """Return {item_id: title} for the given item IDs, read from the metadata file.

    The metadata file is gzip-compressed JSON Lines: one JSON object per line.
    """
    wanted, titles = set(asins), {}
    with gzip.open(os.path.join(DATA_DIR, f"meta_{category}.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["parent_asin"] in wanted:
                titles[item["parent_asin"]] = item.get("title") or "(no title)"
    return titles


def time_split(df, val_frac=0.1, test_frac=0.2):
    """Per-user chronological split.

    For every user, the oldest ratings go to train, the next ones to validation and the most
    recent ones to test (at least one rating each). This mimics predicting a user's future
    ratings from their past ratings.
    """
    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    pos = df.groupby("user_id").cumcount().to_numpy()          # 0 = oldest rating of the user
    n = df.groupby("user_id")["rating"].transform("size").to_numpy()
    n_test = np.maximum(1, np.round(n * test_frac)).astype(int)
    n_val = np.maximum(1, np.round(n * val_frac)).astype(int)
    part = np.where(pos >= n - n_test, 2, np.where(pos >= n - n_test - n_val, 1, 0))
    return df[part == 0], df[part == 1], df[part == 2]


class Encoder:
    """Maps string user/item IDs to consecutive integer indices (rows/columns of the rating matrix)."""

    def __init__(self, user_ids, item_ids):
        self.users = pd.Index(pd.unique(user_ids))
        self.items = pd.Index(pd.unique(item_ids))

    @property
    def shape(self):
        return len(self.users), len(self.items)

    def arrays(self, df):
        """DataFrame -> (user index array, item index array, rating array)."""
        u = self.users.get_indexer(df["user_id"])
        i = self.items.get_indexer(df["parent_asin"])
        return u, i, df["rating"].to_numpy(np.float32)

    def matrix(self, df):
        """DataFrame -> sparse user x item rating matrix (missing ratings are not stored)."""
        u, i, r = self.arrays(df)
        return csr_matrix((r, (u, i)), shape=self.shape)


# ==================================================================================================
# 2. Models: every model has fit(...) and scores(users) -> (len(users) x n_items) predicted scores
# ==================================================================================================
class GlobalMean:
    """Predicts the average training rating for every user-item pair."""
    predicts_ratings = True

    def fit(self, u, i, r, shape):
        self.mu, self.n_items = float(r.mean()), shape[1]
        return self

    def scores(self, users):
        return np.full((len(users), self.n_items), self.mu, dtype=np.float32)


class Popularity:
    """Ranks items by how many training ratings they received (same list for everyone)."""
    predicts_ratings = False

    def fit(self, u, i, r, shape):
        self.counts = np.bincount(i, minlength=shape[1]).astype(np.float32)
        return self

    def scores(self, users):
        return np.tile(self.counts, (len(users), 1))


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


class Hybrid:
    """Weighted hybrid: alpha * MF prediction + (1 - alpha) * kNN prediction."""
    predicts_ratings = True

    def __init__(self, mf, knn, alpha):
        self.mf, self.knn, self.alpha = mf, knn, alpha

    def scores(self, users, knn_scores=None, mf_scores=None):
        mf = self.mf.scores(users) if mf_scores is None else mf_scores
        knn = self.knn.scores(users) if knn_scores is None else knn_scores
        return self.alpha * mf + (1 - self.alpha) * knn


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


# ==================================================================================================
# 3. Evaluation
# ==================================================================================================
def evaluate(scores_fn, users, known, test, k=10, threshold=4.0, candidates=None, chunk=500):
    """Evaluate one model on the given users.

    scores_fn  : function(users) -> (len(users) x n_items) predicted scores
    known      : sparse user x item matrix of items the user already rated (excluded from top-K)
    test       : sparse user x item matrix of held-out ratings (the answers)
    threshold  : a held-out item rated >= threshold counts as relevant (the user liked it)
    candidates : optional boolean mask over items; only these items may be recommended
    Returns RMSE over all held-out ratings, and Precision@K / Recall@K / F1@K averaged over users
    that have at least one relevant held-out item.
    """
    users = np.asarray(users)
    se, n, prec, rec, f1 = 0.0, 0, [], [], []
    for s in range(0, len(users), chunk):
        batch = users[s:s + chunk]
        S = scores_fn(batch)

        T = test[batch].tocoo()                                  # held-out ratings of this batch
        se += float(((S[T.row, T.col] - T.data) ** 2).sum())
        n += len(T.data)

        K = known[batch].tocoo()
        S[K.row, K.col] = -np.inf                                # never recommend already-rated items
        if candidates is not None:
            S[:, ~candidates] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]               # indices of the K highest scores

        T = test[batch].tocsr()
        for row in range(len(batch)):
            lo, hi = T.indptr[row], T.indptr[row + 1]
            relevant = set(T.indices[lo:hi][T.data[lo:hi] >= threshold])
            if not relevant:
                continue
            hits = len(relevant.intersection(top[row]))
            p, r = hits / k, hits / len(relevant)
            prec.append(p)
            rec.append(r)
            f1.append(2 * p * r / (p + r) if hits else 0.0)
    return {
        "rmse": float(np.sqrt(se / n)),
        "precision": float(np.mean(prec)),
        "recall": float(np.mean(rec)),
        "f1": float(np.mean(f1)),
        "n_users": len(f1),
    }


def evaluate_hybrid(mf, knn, params, *args, **kwargs):
    """Hybrid metrics: RMSE with the alpha tuned for rating error, top-K metrics with the alpha tuned for F1."""
    by_rating = evaluate(Hybrid(mf, knn, params["alpha"]).scores, *args, **kwargs)
    by_rank = evaluate(Hybrid(mf, knn, params["alpha_rank"]).scores, *args, **kwargs)
    return {"rmse": by_rating["rmse"], **{k: by_rank[k] for k in ("precision", "recall", "f1", "n_users")}}


def predictions_at(model, users, R):
    """Model predictions and true values for every stored rating of `users` in sparse matrix R."""
    preds, actual = [], []
    for s in range(0, len(users), 500):
        batch = users[s:s + 500]
        T = R[batch].tocoo()
        preds.append(model.scores(batch)[T.row, T.col])
        actual.append(T.data)
    return np.concatenate(preds), np.concatenate(actual)


# ==================================================================================================
# 4. Data exploration
# ==================================================================================================
def describe(df):
    per_user = df.groupby("user_id").size()
    per_item = df.groupby("parent_asin").size()
    users, items = len(per_user), len(per_item)
    return {
        "ratings": len(df), "users": users, "items": items,
        "density": len(df) / (users * items),
        "mean_rating": float(df["rating"].mean()),
        "rating_share": {int(k): float(v) for k, v in df["rating"].value_counts(normalize=True).sort_index().items()},
        "median_ratings_per_user": float(per_user.median()),
        "median_ratings_per_item": float(per_item.median()),
        "max_ratings_per_item": int(per_item.max()),
        "first_date": str(pd.to_datetime(df["timestamp"].min(), unit="ms").date()),
        "last_date": str(pd.to_datetime(df["timestamp"].max(), unit="ms").date()),
    }


def run_eda(datasets):
    stats = {name: describe(df) for name, df in datasets.items()}
    names = list(datasets)
    stats["shared_users"] = len(set(datasets[names[0]]["user_id"]) & set(datasets[names[1]]["user_id"]))
    stats["shared_items"] = len(set(datasets[names[0]]["parent_asin"]) & set(datasets[names[1]]["parent_asin"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, 6)
    for j, name in enumerate(names):
        share = [stats[name]["rating_share"].get(v, 0) * 100 for v in x]
        ax.bar(x + (j - 0.5) * 0.38, share, width=0.38, label=name)
    ax.set(xlabel="Rating", ylabel="Share of ratings (%)", title="Rating distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eda_rating_distribution.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in [(axes[0], "user_id", "Ratings per user"), (axes[1], "parent_asin", "Ratings per item")]:
        for name in names:
            counts = datasets[name].groupby(col).size()
            bins = np.logspace(0, np.log10(counts.max()) + 0.01, 40)
            ax.hist(counts, bins=bins, histtype="step", linewidth=1.8, label=name)
        ax.set(xscale="log", yscale="log", xlabel=label + " (log)", ylabel="Count (log)", title=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eda_long_tail.png"), dpi=150)
    plt.close(fig)
    return stats


# ==================================================================================================
# 5. Experiments
# ==================================================================================================
def in_domain(name, df, params=None):
    """Time-split one dataset, tune (if params is None) on validation, evaluate all models on test."""
    log(f"[{name}] splitting")
    train, val, test = time_split(df)
    enc = Encoder(df["user_id"], df["parent_asin"])
    u, i, r = enc.arrays(train)
    val_arrays = enc.arrays(val)
    R_train, R_val, R_test = enc.matrix(train), enc.matrix(val), enc.matrix(test)
    rng = np.random.default_rng(SEED)
    eval_users = np.sort(rng.choice(enc.shape[0], min(N_EVAL_USERS, enc.shape[0]), replace=False))
    out = {"split": {"train": len(train), "val": len(val), "test": len(test)}, "eval_users": len(eval_users)}

    if params is None:
        tuning = {"mf": [], "knn": [], "alpha": []}
        best = None
        for n_factors, lr, reg in MF_GRID:
            mf = MatrixFactorization(n_factors, lr, reg, MAX_EPOCHS, seed=SEED).fit(u, i, r, enc.shape, val=val_arrays)
            v = min(h["val_rmse"] for h in mf.history)
            tuning["mf"].append({"n_factors": n_factors, "lr": lr, "reg": reg, "val_rmse": v, "best_epoch": mf.best_epoch})
            log(f"[{name}] MF factors={n_factors} lr={lr} reg={reg}: val RMSE {v:.4f} (epoch {mf.best_epoch})")
            if best is None or v < best[0]:
                best = (v, n_factors, lr, reg, mf)
        _, n_factors, lr, reg, mf = best
        # Same model without L2 regularization, run for all epochs, to show overfitting.
        mf_noreg = MatrixFactorization(n_factors, lr, 0.0, MAX_EPOCHS, patience=MAX_EPOCHS, seed=SEED)
        mf_noreg.fit(u, i, r, enc.shape, val=val_arrays)
        out["learning_curves"] = {"reg": reg, "with_l2": mf.history, "without_l2": mf_noreg.history}

        knn_preds = {}
        for k in KNN_GRID:
            knn = UserKNN(k).fit(u, i, r, enc.shape)
            pred, actual = predictions_at(knn, eval_users, R_val)
            knn_preds[k] = pred
            tuning["knn"].append({"k": k, "val_rmse": rmse(actual, pred)})
            log(f"[{name}] kNN k={k}: val RMSE {tuning['knn'][-1]['val_rmse']:.4f}")
        k = min(tuning["knn"], key=lambda t: t["val_rmse"])["k"]

        mf_pred, actual = predictions_at(mf, eval_users, R_val)
        for a in ALPHAS:
            tuning["alpha"].append({"alpha": float(a), "val_rmse": rmse(actual, a * mf_pred + (1 - a) * knn_preds[k])})
        alpha = min(tuning["alpha"], key=lambda t: t["val_rmse"])["alpha"]

        # Rating error and top-K quality can prefer different weights, so the hybrid gets a second
        # alpha chosen by validation F1@K. Scores are computed once and mixed for every alpha.
        S_mf = mf.scores(eval_users)
        S_knn = UserKNN(k).fit(u, i, r, enc.shape).scores(eval_users)
        pos = lambda b: np.searchsorted(eval_users, b)                  # user id -> row in S_mf/S_knn
        tuning["alpha_rank"] = []
        for a in ALPHAS:
            m = evaluate(lambda b, a=a: a * S_mf[pos(b)] + (1 - a) * S_knn[pos(b)],
                         eval_users, R_train, R_val, TOP_K, THRESHOLD)
            tuning["alpha_rank"].append({"alpha": float(a), "val_f1": m["f1"]})
        alpha_rank = max(tuning["alpha_rank"], key=lambda t: t["val_f1"])["alpha"]
        del S_mf, S_knn
        params = {"n_factors": n_factors, "lr": lr, "reg": reg, "best_epoch": mf.best_epoch, "k": k,
                  "alpha": alpha, "alpha_rank": alpha_rank}
        out["tuning"] = tuning
        log(f"[{name}] chosen params {params}")
    else:
        mf = MatrixFactorization(params["n_factors"], params["lr"], params["reg"], MAX_EPOCHS, seed=SEED)
        mf.fit(u, i, r, enc.shape, val=val_arrays)

    knn = UserKNN(params["k"]).fit(u, i, r, enc.shape)
    models = {
        "Global mean": GlobalMean().fit(u, i, r, enc.shape),
        "Popularity": Popularity().fit(u, i, r, enc.shape),
        "User kNN": knn,
        "MF": mf,
    }
    known = R_train + R_val                                           # everything before the test period
    out["results"] = {}
    for model_name, model in models.items():
        out["results"][model_name] = evaluate(model.scores, eval_users, known, R_test, TOP_K, THRESHOLD)
        log(f"[{name}] test {model_name}: {out['results'][model_name]}")
    out["results"]["Hybrid"] = evaluate_hybrid(mf, knn, params, eval_users, known, R_test, TOP_K, THRESHOLD)
    log(f"[{name}] test Hybrid: {out['results']['Hybrid']}")
    out["params"] = params
    return out


def cross_domain(source, target, params, source_name="Software", target_name="Video_Games"):
    """Hide the target-domain ratings of some shared users and predict them from their source ratings.

    Source and target ratings are put in one user x item matrix. Shared users that are not hidden
    ("bridge users") have ratings in both domains, which links the two item sets.
    """
    shared = np.array(sorted(set(source["user_id"]) & set(target["user_id"])))
    rng = np.random.default_rng(SEED)
    hidden = set(rng.choice(shared, int(len(shared) * CROSS_TARGET_FRAC), replace=False))
    is_hidden = target["user_id"].isin(hidden)
    target_train, target_test = target[~is_hidden], target[is_hidden]
    joint_train = pd.concat([source, target_train])

    enc = Encoder(pd.concat([source["user_id"], target["user_id"]]),
                  pd.concat([source["parent_asin"], target["parent_asin"]]))
    target_items = enc.items.isin(target["parent_asin"].unique())
    u, i, r = enc.arrays(joint_train)
    tu, ti, tr = enc.arrays(target_train)
    has_target = np.zeros(enc.shape[0], bool)
    has_target[np.unique(tu)] = True                                   # only these can be kNN neighbors
    test_users = np.sort(enc.users.get_indexer(list(hidden)))
    log(f"[cross] shared users {len(shared)}, hidden {len(hidden)}, bridge {len(shared) - len(hidden)}")

    mf = MatrixFactorization(params["n_factors"], params["lr"], params["reg"], params["best_epoch"], seed=SEED)
    mf.fit(u, i, r, enc.shape)
    knn = UserKNN(params["k"]).fit(u, i, r, enc.shape, neighbor_mask=has_target)
    models = {
        "Global mean": GlobalMean().fit(tu, ti, tr, enc.shape),       # mean of target-domain ratings
        "Popularity": Popularity().fit(tu, ti, tr, enc.shape),        # most popular target items
        "User kNN": knn,
        "MF": mf,
    }
    R_test = enc.matrix(target_test)
    no_history = csr_matrix(enc.shape, dtype=np.float32)              # no target-domain history
    out = {"shared_users": len(shared), "hidden_users": len(hidden), "test_ratings": len(target_test), "results": {}}
    for name, model in models.items():
        out["results"][name] = evaluate(model.scores, test_users, no_history, R_test, TOP_K, THRESHOLD,
                                        candidates=target_items)
        log(f"[cross] {name}: {out['results'][name]}")
    out["results"]["Hybrid"] = evaluate_hybrid(mf, knn, params, test_users, no_history, R_test, TOP_K, THRESHOLD,
                                               candidates=target_items)
    log(f"[cross] Hybrid: {out['results']['Hybrid']}")

    # Example for the slides: the hidden user with the most hits in the hybrid's top-10.
    S = Hybrid(mf, knn, params["alpha_rank"]).scores(test_users)
    S[:, ~target_items] = -np.inf
    top = np.argsort(-S, axis=1)[:, :TOP_K]
    liked = [set(R_test[uidx].indices[R_test[uidx].data >= THRESHOLD]) for uidx in test_users]
    hits = [len(l.intersection(t)) for l, t in zip(liked, top)]
    row = int(np.argmax(hits))
    user = enc.users[test_users[row]]
    history = source[source["user_id"] == user].sort_values("rating", ascending=False).head(5)
    recs = [enc.items[j] for j in top[row]]
    actual = dict(zip(target_test.loc[target_test["user_id"] == user, "parent_asin"],
                      target_test.loc[target_test["user_id"] == user, "rating"]))
    src_titles = load_titles(source_name, history["parent_asin"])
    tgt_titles = load_titles(target_name, recs)
    out["example"] = {
        "software_history": [{"title": src_titles.get(a, a), "rating": float(x)}
                             for a, x in zip(history["parent_asin"], history["rating"])],
        "recommended_games": [{"title": tgt_titles.get(a, a), "actual_rating": actual.get(a)} for a in recs],
        "hits": hits[row],
    }
    return out


# ==================================================================================================
# 6. Result figures
# ==================================================================================================
def plot_results(results):
    for metric, label, models in [("rmse", "RMSE (lower is better)", ["Global mean", "User kNN", "MF", "Hybrid"]),
                                  ("f1", "F1@10 (higher is better)", ["Popularity", "User kNN", "MF", "Hybrid"])]:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        exps = list(results)
        width = 0.8 / len(models)
        for j, m in enumerate(models):
            vals = [results[e]["results"][m][metric] for e in exps]
            bars = ax.bar(np.arange(len(exps)) + (j - (len(models) - 1) / 2) * width, vals, width, label=m, color=COLORS[m])
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xticks(np.arange(len(exps)), exps)
        ax.set(ylabel=label, title=label.split(" (")[0] + " by experiment")
        ax.legend(fontsize=8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
        ax.margins(y=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"results_{metric}.png"), dpi=150)
        plt.close(fig)


def plot_tuning(sw):
    curves = sw["learning_curves"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, style, label in [("without_l2", "--", "no L2"), ("with_l2", "-", f"L2 = {curves['reg']}")]:
        h = curves[key]
        ep = [e["epoch"] for e in h]
        ax.plot(ep, [e["train_rmse"] for e in h], style, color="#4c72b0", label=f"train ({label})")
        ax.plot(ep, [e["val_rmse"] for e in h], style, color="#dd8452", label=f"validation ({label})")
    ax.set(xlabel="Epoch", ylabel="RMSE", title="MF learning curves on Software")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_learning_curves.png"), dpi=150)
    plt.close(fig)

    t = sw["tuning"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([x["k"] for x in t["knn"]], [x["val_rmse"] for x in t["knn"]], "o-", color=COLORS["User kNN"])
    axes[0].set(xlabel="Number of neighbors k", ylabel="Validation RMSE", title="kNN: choosing k")
    axes[1].plot([x["alpha"] for x in t["alpha"]], [x["val_rmse"] for x in t["alpha"]], "o-", color=COLORS["Hybrid"],
                 label="RMSE (left)")
    axes[1].set(xlabel="alpha (0 = only kNN, 1 = only MF)", ylabel="Validation RMSE", title="Hybrid: choosing alpha")
    right = axes[1].twinx()                                           # second y-axis for F1
    right.plot([x["alpha"] for x in t["alpha_rank"]], [x["val_f1"] for x in t["alpha_rank"]], "s--",
               color=COLORS["MF"], label="F1@10 (right)")
    right.set_ylabel("Validation F1@10")
    lines = axes[1].get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    axes[1].legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tuning_knn_alpha.png"), dpi=150)
    plt.close(fig)


# ==================================================================================================
# 7. Main
# ==================================================================================================
def main():
    os.makedirs(FIG, exist_ok=True)
    start = time.time()
    software, games = load_ratings("Software"), load_ratings("Video_Games")
    report = {"config": {"seed": SEED, "eval_users": N_EVAL_USERS, "top_k": TOP_K, "threshold": THRESHOLD,
                         "max_epochs": MAX_EPOCHS}}
    report["eda"] = run_eda({"Software": software, "Video Games": games})
    log("EDA done")

    experiments = {}
    experiments["Software"] = in_domain("Software", software)
    params = experiments["Software"]["params"]
    experiments["Video Games"] = in_domain("Video Games", games, params)
    experiments["Cross-domain"] = cross_domain(software, games, params)
    report["experiments"] = experiments

    plot_results(experiments)
    plot_tuning(experiments["Software"])
    report["runtime_minutes"] = round((time.time() - start) / 60, 1)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    log(f"done in {report['runtime_minutes']} min -> outputs/results.json, outputs/figures/")


if __name__ == "__main__":
    main()
