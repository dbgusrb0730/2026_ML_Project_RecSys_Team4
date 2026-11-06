"""Shared data partitions for the model study and unified experiments."""

import numpy as np
import pandas as pd

import recsys
from recsys import Encoder, N_EVAL_USERS, SEED, time_split


D_VAL_FRAC = 0.1                         # hidden-val share of the shared users (hidden-test: recsys.CROSS_TARGET_FRAC)


def make_split(df):
    """Same split, encoding and evaluation users as recsys.in_domain (recsys.py:391-401, 460).

    recsys.py:394  train, val, test = time_split(df)
    recsys.py:395  enc = Encoder(df["user_id"], df["parent_asin"])   (all users/items of train+val+test)
    recsys.py:396-398  training arrays, validation arrays, sparse matrices R_train / R_val / R_test
    recsys.py:399-400  rng = np.random.default_rng(SEED); eval_users = sorted sample of N_EVAL_USERS users
    recsys.py:460  known = R_train + R_val   (items excluded from the top-K at test time)
    Added here: train+val arrays for the refit of problem 2.
    """
    train, val, test = time_split(df)
    enc = Encoder(df["user_id"], df["parent_asin"])
    # np.random.default_rng(seed): numpy's random number generator object; the same seed gives
    # the same numbers. rng.choice(n, size, replace=False): `size` distinct integers from 0..n-1.
    rng = np.random.default_rng(SEED)
    eval_users = np.sort(rng.choice(enc.shape[0], min(N_EVAL_USERS, enc.shape[0]), replace=False))
    R_train, R_val, R_test = enc.matrix(train), enc.matrix(val), enc.matrix(test)
    return {
        "enc": enc, "shape": enc.shape, "eval_users": eval_users,
        "train": enc.arrays(train), "val": enc.arrays(val), "test": enc.arrays(test),
        "trval": enc.arrays(pd.concat([train, val])),
        "R_train": R_train, "R_val": R_val, "R_test": R_test, "known": R_train + R_val,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }


def make_cross_split(sw, vg, val_frac=D_VAL_FRAC):
    """Users of the cross-domain experiment.

    hidden-test: exactly as recsys.cross_domain (recsys.py:477-479): sorted shared users,
                 default_rng(SEED).choice(shared, int(n * CROSS_TARGET_FRAC), replace=False)
    hidden-val : default_rng(SEED + 1) choice among the remaining shared users (val_frac of all shared users)
    bridge     : the other shared users, in the order of a default_rng(SEED + 2) permutation, so that the
                 users kept at a smaller bridge fraction are a subset of those kept at a larger one
    Encoder over all users and items of both domains, as recsys.py:484-485.
    """
    shared = np.array(sorted(set(sw["user_id"]) & set(vg["user_id"])))
    hidden_test = set(np.random.default_rng(SEED).choice(shared, int(len(shared) * recsys.CROSS_TARGET_FRAC), replace=False))
    rest = np.array(sorted(set(shared) - hidden_test))
    hidden_val = set(np.random.default_rng(SEED + 1).choice(rest, int(len(shared) * val_frac), replace=False))
    bridge = np.array(sorted(set(rest) - hidden_val))
    bridge = bridge[np.random.default_rng(SEED + 2).permutation(len(bridge))]
    enc = Encoder(pd.concat([sw["user_id"], vg["user_id"]]), pd.concat([sw["parent_asin"], vg["parent_asin"]]))
    vg_items = enc.items.isin(vg["parent_asin"].unique())
    test_df, val_df = vg[vg["user_id"].isin(hidden_test)], vg[vg["user_id"].isin(hidden_val)]
    return {"sw": sw, "vg": vg, "enc": enc, "shape": enc.shape, "vg_items": vg_items, "shared": shared,
            "hidden_test": hidden_test, "hidden_val": hidden_val, "bridge": bridge,
            "test": enc.arrays(test_df), "val": enc.arrays(val_df), "R_test": enc.matrix(test_df),
            "test_users": np.sort(enc.users.get_indexer(list(hidden_test)))}


def cross_train(c, frac):
    """Training data at one bridge fraction, with the D-1 leakage checks."""
    keep = set(c["bridge"][:int(round(frac * len(c["bridge"])))])
    dropped = set(c["bridge"]) - keep
    vg_train = c["vg"][~c["vg"]["user_id"].isin(c["hidden_test"] | c["hidden_val"] | dropped)]
    joint = pd.concat([c["sw"], vg_train])
    # D-1: no VG rating of a hidden-test / hidden-val user is in the training data ...
    leak = int(vg_train["user_id"].isin(c["hidden_test"] | c["hidden_val"]).sum())
    assert leak == 0, f"leakage: {leak} VG ratings of hidden users in training"
    # ... while their Software history is fully there (that is what is transferred).
    hidden = c["hidden_test"] | c["hidden_val"]
    sw_hidden = int(c["sw"]["user_id"].isin(hidden).sum())
    assert int(joint["user_id"].isin(hidden).sum()) == sw_hidden
    u, i, r = c["enc"].arrays(joint)
    item_seen = np.bincount(i, minlength=c["shape"][1]) > 0
    unseen_vg = c["vg_items"] & ~item_seen
    tu, ti, _ = c["test"]
    info = {"bridge_frac": frac, "n_bridge": len(keep), "n_bridge_vg_ratings": int(vg_train["user_id"].isin(keep).sum()),
            "train_ratings": len(r), "vg_train_ratings": len(vg_train), "leak_vg_ratings_of_hidden": leak,
            "hidden_users_software_ratings": sw_hidden, "vg_items_without_training": int(unseen_vg.sum()),
            "hidden_test_ratings_on_those_items": int(unseen_vg[ti].sum())}
    return (u, i, r), c["enc"].arrays(vg_train), unseen_vg, info
