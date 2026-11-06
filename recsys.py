"""Shared settings, rating encoding, and comparison baselines for main.py."""

import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_EVAL_USERS = 10_000
TOP_K = 10
THRESHOLD = 4.0
MAX_EPOCHS = 50
CROSS_TARGET_FRAC = 0.2


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


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
