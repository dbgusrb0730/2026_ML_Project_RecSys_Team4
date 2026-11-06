"""MF study (Team 4, 이후윤): training and evaluation experiments for matrix factorization.

Builds on the team base code `recsys.py` WITHOUT modifying it (decision D2 in ../MF_PLAN.md).
Data split, Encoder, evaluation and the base MatrixFactorization are imported from recsys.py,
so the evaluation protocol stays identical to the team code.

Implemented in this file (experiments A-E of ../MF_PLAN.md; process in EXPERIMENT_LOG.md):
  A1  synthetic-data recovery test: data generated from known P, Q; can the SGD recover hidden ratings?
  A2  fixes for problems 1-3 and before/after comparison on Software
        problem 1: items/users with no training rating keep random vectors  -> FixedMF
        problem 2: final model never learns from validation ratings        -> fit_select_refit
        problem 3: no bias-only baseline                                    -> FixedMF(n_factors=0)
      + clipping-tie diagnostic (approved addition): scores clipped to [1, 5] create ties at 5.0
  A3  timing per epoch and per evaluation -> scale of experiment B
  B1, B1x, B2  tuning on Software (λ, split bias/factor regularization SplitRegMF, k × reg_factor)
  C   evaluation on Software and Video Games, activity buckets, popularity of recommendations
  D, D5  cross-domain Software -> Video Games (bridge users, bootstrap, PCA, user-bias shift)
  E1, E  BPR-loss MF (BPRMF): synthetic test, tuning, comparison, VG and cross-domain

Usage:
  .venv/bin/python mf_study.py --exp A1      # A1 A2 A3 A B1 B1x B2 B C D D5 E1 E
Outputs go to outputs/mf/ (results.json, *.csv, figures/*.png at 150 dpi).
"""
import argparse
import json
import os
import shutil   # shutil.copy2(src, dst): copy a file (with its timestamps); used to archive outputs
import time

import matplotlib
matplotlib.use("Agg")                     # draw figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix   # sparse matrix (used for an empty 'known' matrix)

from archive import recsys
from archive.recsys import (SEED, N_EVAL_USERS, TOP_K, THRESHOLD, MAX_EPOCHS, Encoder, GlobalMean,
                    MatrixFactorization, Popularity, evaluate, load_ratings, log, rmse, time_split)
from mf import BPRMF, FixedMF, SplitRegMF
from splits import D_VAL_FRAC, cross_train, make_cross_split, make_split

OUT = os.path.join(recsys.ROOT, "outputs", "mf")
FIG = os.path.join(OUT, "figures")
RESULTS = os.path.join(OUT, "results.json")

# Experiment A-2 settings: the defaults of recsys.MatrixFactorization (inside the team grid MF_GRID).
# Not tuned here on purpose; A-2 isolates the effect of the fixes. Tuning is experiment B.
A2_PARAMS = {"n_factors": 32, "lr": 0.01, "reg": 0.05}
PATIENCE = 2                               # same as the recsys.MatrixFactorization default used by in_domain
CLIP_MAX = 5.0                             # upper clip of MatrixFactorization.predict/scores (recsys.py:245, 249)

# Colors: categorical slots 1, 2 of the reference palette (dataviz skill); gray for reference lines.
C_MF, C_BIAS, C_REF = "#2a78d6", "#eb6834", "#8a8984"


# ==================================================================================================
# Models
# ==================================================================================================
class TrackHidden:
    """Mixin for A-1 figures only: records the hidden-set RMSE after every epoch.

    recsys.MatrixFactorization.fit calls self.predict(val[0], val[1]) once per epoch (recsys.py:226).
    When that call is made with the tracked validation array (same object), the hidden RMSE of the
    current parameters is appended to hidden_curve. Nothing is chosen from it (selection stays on val),
    and SGD does not call predict, so training is unchanged.
    """

    def track(self, val_u, hidden):
        self._track, self.hidden_curve = (val_u, hidden), []
        return self

    def predict(self, u, i):
        pred = super().predict(u, i)
        t = getattr(self, "_track", None)
        if t is not None and u is t[0]:
            hu, hi, hr = t[1]
            self.hidden_curve.append(rmse(hr, super().predict(hu, hi)))
        return pred


class TrackedMF(TrackHidden, MatrixFactorization):
    pass


class TrackedFixedMF(TrackHidden, FixedMF):
    pass


def new_model(cls, params, n_epochs, patience=PATIENCE):
    return cls(params["n_factors"], params["lr"], params["reg"], n_epochs, patience=patience, seed=SEED)


def timed_fit(model, u, i, r, shape, val=None):
    # time.perf_counter(): wall-clock timer with the highest available resolution, in seconds.
    t0 = time.perf_counter()
    model.fit(u, i, r, shape, val=val)
    model.fit_seconds = time.perf_counter() - t0
    model.epochs_run = len(model.history)
    return model


def fit_select_refit(cls, params, s, patience=PATIENCE):
    """Problem 2 fix.

    Step 1: train on train, choose the best epoch by validation RMSE (early stopping, as in recsys).
            With patience = MAX_EPOCHS all epochs run and the parent fit restores the val-best epoch.
    Step 2: retrain on train+val with the same settings for exactly best_epoch epochs (no validation).
    Test data is never used for any choice. Returns (step-1 model, step-2 model).
    """
    m1 = timed_fit(new_model(cls, params, MAX_EPOCHS, patience), *s["train"], s["shape"], val=s["val"])
    m2 = timed_fit(new_model(cls, params, m1.best_epoch), *s["trval"], s["shape"])
    m2.best_epoch = m1.best_epoch
    return m1, m2


# ==================================================================================================
# Data
# ==================================================================================================
def make_synthetic(n_users=1000, n_warm=500, n_cold=20, k=5, density=0.1, sigma=0.2, cold_per_item=50, seed=SEED):
    """Ratings from a known model: r = mu + b_u + b_i + p_u . q_i + noise, noise ~ N(0, sigma^2).

    Warm items: a random `density` share of the user x item cells is observed and split at random
    into train 80% / val 10% / hidden 10%. Cold items (the last n_cold columns) only have hidden
    ratings, like items that appear only in the test period (problem 1).
    Ratings are kept continuous (not rounded) so the noise level is exactly sigma.
    """
    rng = np.random.default_rng(seed)
    n_items = n_warm + n_cold
    s = (0.25 / k) ** 0.25                     # entry std so that std(p_u . q_i) = sqrt(k) * s^2 = 0.5
    mu = 3.0
    bu, bi = rng.normal(0, 0.2, n_users), rng.normal(0, 0.2, n_items)
    P, Q = rng.normal(0, s, (n_users, k)), rng.normal(0, s, (n_items, k))

    def rate(u, i):
        return (mu + bu[u] + bi[i] + (P[u] * Q[i]).sum(1) + rng.normal(0, sigma, len(u))).astype(np.float32)

    wu, wi = np.nonzero(rng.random((n_users, n_warm)) < density)        # observed warm cells
    perm = rng.permutation(len(wu))
    a, b = int(0.8 * len(wu)), int(0.9 * len(wu))
    cu = np.concatenate([rng.choice(n_users, cold_per_item, replace=False) for _ in range(n_cold)])
    ci = np.repeat(np.arange(n_warm, n_items), cold_per_item)
    split = {name: (wu[idx], wi[idx]) for name, idx in
             [("train", perm[:a]), ("val", perm[a:b]), ("hidden", perm[b:])]}
    split["cold"] = (cu, ci)
    data = {name: (u, i, rate(u, i)) for name, (u, i) in split.items()}
    config = {"n_users": n_users, "n_warm_items": n_warm, "n_cold_items": n_cold, "k_true": k, "density": density,
              "sigma": sigma, "mu": mu, "bias_std": 0.2, "pq_entry_std": s, "cold_ratings_per_item": cold_per_item,
              "sizes": {name: len(d[2]) for name, d in data.items()}}
    return data, (n_users, n_items), config


# ==================================================================================================
# Diagnostics
# ==================================================================================================
def topk_diagnostics(model, users, known, unseen_items, chunk=500):
    """For the clipped scores used by recsys.evaluate (same exclusion of known items):

    at_cap     : per user, number of candidate items (not in known) whose score equals the clip max 5.0
    unseen_top : share of top-K slots filled by items with no training rating (problem 1)
    """
    at_cap, unseen_hits = [], 0
    for s in range(0, len(users), chunk):
        batch = users[s:s + chunk]
        S = model.scores(batch)
        K = known[batch].tocoo()
        S[K.row, K.col] = -np.inf
        at_cap.append((S >= CLIP_MAX).sum(1))
        # np.argpartition(-S, k, axis=1)[:, :k]: indices of the k largest scores per row (unordered),
        # without sorting the whole row. Same call as recsys.evaluate (recsys.py:297).
        top = np.argpartition(-S, TOP_K, axis=1)[:, :TOP_K]
        unseen_hits += int(unseen_items[top].sum())
    at_cap = np.concatenate(at_cap)
    return {"cap_items_mean": float(at_cap.mean()), "cap_items_median": float(np.median(at_cap)),
            "cap_items_max": int(at_cap.max()), "share_users_cap_ge_10": float((at_cap >= TOP_K).mean()),
            "share_users_cap_ge_1": float((at_cap >= 1).mean()),
            "unseen_item_share_in_top10": unseen_hits / (len(users) * TOP_K)}


# ==================================================================================================
# Output helpers
# ==================================================================================================
def save_results(key, obj):
    """Update one top-level key of outputs/mf/results.json, keeping the other experiments' results."""
    report = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            report = json.load(f)
    report[key] = obj
    # json.dump(obj, f, indent=2, ensure_ascii=False, default=float): write as JSON text;
    # indent pretty-prints, ensure_ascii=False keeps Korean readable, default=float converts numpy numbers.
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)


def load_results():
    if not os.path.exists(RESULTS):
        return {}
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


# ==================================================================================================
# Experiment A-1: synthetic-data recovery
# ==================================================================================================
def run_A1():
    data, shape, config = make_synthetic()
    sigma = config["sigma"]
    params_mf = {"n_factors": config["k_true"], "lr": 0.01, "reg": 0.02}
    params_bias = {**params_mf, "n_factors": 0}
    # patience = max_epochs: no early stop; all 300 epochs run and the parent fit restores the epoch
    # with the lowest validation RMSE. Run 1 used patience 5 and stopped in the plateau where the
    # latent vectors are still near 0 (EXPERIMENT_LOG.md #4, #5; decision D8).
    max_epochs = patience = 300
    log(f"[A1] synthetic data {config['sizes']}, sigma={sigma}")

    gm = GlobalMean().fit(*data["train"], shape)
    models = {
        "Bias only (k=0)": new_model(TrackedFixedMF, params_bias, max_epochs, patience),
        "Base MF (recsys)": new_model(TrackedMF, params_mf, max_epochs, patience),
        "FixedMF": new_model(TrackedFixedMF, params_mf, max_epochs, patience),
    }
    for m in models.values():
        m.track(data["val"][0], data["hidden"])
    rows = [{"model": "Global mean", "train_rmse": rmse(data["train"][2], np.full(len(data["train"][2]), gm.mu)),
             **{f"{p}_rmse": rmse(data[p][2], np.full(len(data[p][2]), gm.mu)) for p in ("val", "hidden", "cold")},
             "best_epoch": None, "epochs_run": None}]
    for name, m in models.items():
        timed_fit(m, *data["train"], shape, val=data["val"])
        m._track = None                                    # stop recording after training
        row = {"model": name, **{f"{p}_rmse": rmse(data[p][2], m.predict(data[p][0], data[p][1]))
                                 for p in ("train", "val", "hidden", "cold")},
               "best_epoch": m.best_epoch, "epochs_run": m.epochs_run}
        rows.append(row)
        log(f"[A1] {name}: {row}")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "A1_synthetic.csv"), index=False)

    by = {r["model"]: r for r in rows}
    fixed = models["FixedMF"]
    n_warm = config["n_warm_items"]
    checks = {
        "hidden_rmse_le_2sigma": by["FixedMF"]["hidden_rmse"] <= 2 * sigma and by["Base MF (recsys)"]["hidden_rmse"] <= 2 * sigma,
        "mf_le_0.7x_bias_only": by["FixedMF"]["hidden_rmse"] <= 0.7 * by["Bias only (k=0)"]["hidden_rmse"]
                                and by["Base MF (recsys)"]["hidden_rmse"] <= 0.7 * by["Bias only (k=0)"]["hidden_rmse"],
        "cold_fixed_le_base": by["FixedMF"]["cold_rmse"] <= by["Base MF (recsys)"]["cold_rmse"],
        "cold_rows_zero": bool(not fixed.Q[n_warm:].any() and not fixed.bi[n_warm:].any()),
        "cold_rows_random_in_base": bool(models["Base MF (recsys)"].Q[n_warm:].any()),
    }
    checks["passed"] = all(checks[c] for c in ("hidden_rmse_le_2sigma", "mf_le_0.7x_bias_only",
                                                "cold_fixed_le_base", "cold_rows_zero"))
    # Consistency: the hidden RMSE recorded at the best epoch equals the one of the restored model.
    for name, m in models.items():
        checks[f"hidden_curve_matches_restored[{name}]"] = bool(
            abs(m.hidden_curve[m.best_epoch - 1] - by[name]["hidden_rmse"]) < 1e-6)
    log(f"[A1] checks {checks}")
    run1_stop = run1_stop_epoch()
    plot_A1(models, sigma, by, run1_stop)
    result = {"config": config, "params_mf": params_mf, "max_epochs": max_epochs, "patience": patience,
              "table": rows, "checks": checks, "run1_stop_epoch": run1_stop,
              "history": {name: [{**e, "hidden_rmse": m.hidden_curve[j]} for j, e in enumerate(m.history)]
                          for name, m in models.items()}}
    return result


def run1_stop_epoch():
    """Last epoch of the base MF in A-1 run 1 (patience 5), read from its archived table."""
    path = os.path.join(OUT, "archive", "A1_run1_patience5", "A1_synthetic.csv")
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path).set_index("model")
    return int(t.loc["Base MF (recsys)", "epochs_run"]), int(t.loc["Base MF (recsys)", "best_epoch"])


def plot_A1(models, sigma, by, run1_stop):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    m = models["Base MF (recsys)"]
    ep = [e["epoch"] for e in m.history]
    ax.plot(ep, [e["train_rmse"] for e in m.history], "-", color=C_MF, lw=2, label="MF k=5: train")
    ax.plot(ep, [e["val_rmse"] for e in m.history], "--", color=C_MF, lw=2, label="MF k=5: validation")
    ax.plot(ep, m.hidden_curve, ":", color=C_MF, lw=2, label="MF k=5: hidden")
    best = m.best_epoch
    ax.plot([best], [m.history[best - 1]["val_rmse"]], "o", ms=8, color=C_MF, mec="white", mew=2,
            label=f"lowest validation (epoch {best}); hidden RMSE {by['Base MF (recsys)']['hidden_rmse']:.3f}")
    ax.axhline(by["Bias only (k=0)"]["hidden_rmse"], color=C_BIAS, lw=2,
               label=f"bias only (k=0): hidden RMSE {by['Bias only (k=0)']['hidden_rmse']:.3f}")
    if run1_stop is not None:
        ax.axvline(run1_stop[1], color=C_REF, lw=1, ls="-.")
        ax.text(run1_stop[1] + 3, 0.62, f"run 1 (patience 5) kept epoch {run1_stop[1]},\nstopped after epoch {run1_stop[0]}",
                fontsize=8, color="#52514e", va="top")
    for y, txt in [(sigma, f"noise σ = {sigma}"), (2 * sigma, f"pass line 2σ = {2 * sigma:.1f}")]:
        ax.axhline(y, color=C_REF, lw=1, ls=":")
        ax.text(ep[-1], y + 0.004, txt, va="bottom", ha="right", fontsize=8, color="#52514e")
    ax.set(xlabel="Epoch", ylabel="RMSE", ylim=(0.1, 0.65),
           title="A-1: recovery of synthetic ratings (true rank 5, 300 epochs, no early stop)")
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.83))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_A1_synthetic.png"), dpi=150)
    plt.close(fig)


# ==================================================================================================
# Experiment A-2: fixes for problems 1-3, before/after on Software
# ==================================================================================================
def run_A2(df=None):
    """df: optional ratings DataFrame (only for smoke tests on a subsample); default = Software."""
    log("[A2] loading Software")
    s = make_split(load_ratings("Software") if df is None else df)
    users, known, R_test = s["eval_users"], s["known"], s["R_test"]
    n_users, n_items = s["shape"]
    log(f"[A2] split {s['sizes']}, shape {s['shape']}, eval users {len(users)}")

    # A-2a: items/users without training ratings
    tr_items = np.bincount(s["train"][1], minlength=n_items) > 0
    tv_items = np.bincount(s["trval"][1], minlength=n_items) > 0
    tr_users = np.bincount(s["train"][0], minlength=n_users) > 0
    tv_users = np.bincount(s["trval"][0], minlength=n_users) > 0
    test_i = s["test"][1]
    eval_test = R_test[users].tocoo()
    unseen = {
        "items_total": n_items, "users_total": n_users,
        "items_without_train": int((~tr_items).sum()), "items_without_trainval": int((~tv_items).sum()),
        "users_without_train": int((~tr_users).sum()), "users_without_trainval": int((~tv_users).sum()),
        "test_ratings_on_items_without_train": int((~tr_items[test_i]).sum()),
        "test_ratings_on_items_without_trainval": int((~tv_items[test_i]).sum()),
        "test_ratings_total": len(test_i),
        "eval_user_test_ratings_on_items_without_train": int((~tr_items[eval_test.col]).sum()),
        "eval_user_test_ratings": len(eval_test.data),
    }
    log(f"[A2] unseen {unseen}")

    # Fit all variants (same settings, same seed; selection by validation only)
    bias_params = {**A2_PARAMS, "n_factors": 0}
    log("[A2] fitting base MF (train -> val epoch -> train+val)")
    base1, base2 = fit_select_refit(MatrixFactorization, A2_PARAMS, s)
    log("[A2] fitting FixedMF")
    fix1, fix2 = fit_select_refit(FixedMF, A2_PARAMS, s)
    log("[A2] fitting bias-only")
    bias1, bias2 = fit_select_refit(FixedMF, bias_params, s)
    # Plateau check (decision D9): same FixedMF, but patience = MAX_EPOCHS so that all 50 epochs run
    # and the val-best epoch is restored before the train+val refit. Epoch cap and init are unchanged.
    log("[A2] fitting FixedMF with patience = MAX_EPOCHS")
    long1, long2 = fit_select_refit(FixedMF, A2_PARAMS, s, patience=MAX_EPOCHS)
    for name, m in [("base", base1), ("fixed", fix1), ("bias", bias1), ("fixed_p50", long1)]:
        log(f"[A2] {name}: best epoch {m.best_epoch}, {m.epochs_run} epochs in {m.fit_seconds:.1f}s")

    # A-2a check: after fit, unseen rows are exactly 0 in FixedMF (both steps)
    zero_check = {}
    for name, m, seen_i, seen_u in [("FixedMF_train", fix1, tr_items, tr_users), ("FixedMF_trainval", fix2, tv_items, tv_users)]:
        zero_check[name] = {"q_zero": bool(not m.Q[~seen_i].any()), "b_i_zero": bool(not m.bi[~seen_i].any()),
                            "p_zero": bool(not m.P[~seen_u].any()), "b_u_zero": bool(not m.bu[~seen_u].any())}
    zero_check["BaseMF_train_q_nonzero"] = bool(base1.Q[~tr_items].any())   # random vectors remain in base
    zero_check["BaseMF_train_b_i_zero"] = bool(not base1.bi[~tr_items].any())
    log(f"[A2] zero check {zero_check}")

    rows_spec = [
        # (label, model, fix1, fix2, data, unseen-item mask of its training data, step-1 model or None)
        ("Global mean (reference)", GlobalMean().fit(*s["train"], s["shape"]), "-", "-", "train", None, None),
        ("Popularity (reference)", Popularity().fit(*s["train"], s["shape"]), "-", "-", "train", None, None),
        ("Bias only, train", bias1, "yes", "no", "train", ~tr_items, bias1),
        ("Bias only, train+val refit", bias2, "yes", "yes", "train+val", ~tv_items, bias1),
        ("Base MF (before fixes)", base1, "no", "no", "train", ~tr_items, base1),
        ("MF + fix 1", fix1, "yes", "no", "train", ~tr_items, fix1),
        ("MF + fix 2 (base class refit)", base2, "no", "yes", "train+val", ~tv_items, base1),
        ("MF + fixes 1,2 (after fixes)", fix2, "yes", "yes", "train+val", ~tv_items, fix1),
        (f"MF + fixes 1,2, patience {MAX_EPOCHS}", long2, "yes", "yes", "train+val", ~tv_items, long1),
    ]
    rows, diagnostics = [], {}
    for label, m, f1_, f2_, data, unseen_mask, step1 in rows_spec:
        t0 = time.perf_counter()
        res = evaluate(m.scores, users, known, R_test, TOP_K, THRESHOLD)
        eval_sec = time.perf_counter() - t0
        row = {"model": label, "fix1": f1_, "fix2": f2_, "trained_on": data, "ranking_scores": "clipped",
               **{k: res[k] for k in ("rmse", "precision", "recall", "f1", "n_users")},
               "val_rmse": None, "best_epoch": None, "epoch_cap": "", "eval_seconds": round(eval_sec, 2)}
        if not m.predicts_ratings:          # Popularity scores are rating counts: RMSE is meaningless
            row["rmse"] = None
        if step1 is not None:
            row["val_rmse"] = min(h["val_rmse"] for h in step1.history)
            row["best_epoch"] = step1.best_epoch
            row["epoch_cap"] = "상한 도달" if step1.best_epoch == MAX_EPOCHS else ""
            diagnostics[label] = topk_diagnostics(m, users, known, unseen_mask)
            log(f"[A2] diag {label}: {diagnostics[label]}")
        rows.append(row)
        log(f"[A2] test {label}: {res}")

    # Clipping-tie fix rows: RMSE from clipped scores, top-K from unclipped scores
    # (two evaluate calls, same pattern as recsys.evaluate_hybrid, recsys.py:319-323).
    for label, m, step1 in [("Bias only, refit + rank by unclipped", bias2, bias1),
                            ("MF + fixes 1,2 + rank by unclipped", fix2, fix1)]:
        by_rating = evaluate(m.scores, users, known, R_test, TOP_K, THRESHOLD)
        by_rank = evaluate(m.raw_scores, users, known, R_test, TOP_K, THRESHOLD)
        rows.append({"model": label, "fix1": "yes", "fix2": "yes", "trained_on": "train+val", "ranking_scores": "unclipped",
                     "rmse": by_rating["rmse"], **{k: by_rank[k] for k in ("precision", "recall", "f1", "n_users")},
                     "val_rmse": min(h["val_rmse"] for h in step1.history), "best_epoch": step1.best_epoch,
                     "epoch_cap": "상한 도달" if step1.best_epoch == MAX_EPOCHS else "", "eval_seconds": None})
        log(f"[A2] test {label}: {rows[-1]}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "A2_before_after.csv"), index=False)
    pd.DataFrame([{"model": k, **v} for k, v in diagnostics.items()]).to_csv(
        os.path.join(OUT, "A2_clip_diagnostics.csv"), index=False)

    # A-2f: learning curves of every step-1 (train-only, val-monitored) fit, and plateau diagnosis.
    curves = {"Base MF, patience 2": base1, "FixedMF, patience 2": fix1, "Bias only, patience 2": bias1,
              f"FixedMF, patience {MAX_EPOCHS}": long1}
    pd.DataFrame([{"model": name, **e} for name, m in curves.items() for e in m.history]).to_csv(
        os.path.join(OUT, "A2_learning_curves.csv"), index=False)
    plateau = plateau_diagnosis(curves, bias1, long1)
    pd.DataFrame([{"model": k, **v} for k, v in plateau.items()]).to_csv(
        os.path.join(OUT, "A2_plateau_diagnosis.csv"), index=False)
    log(f"[A2] plateau {plateau}")
    plot_A2_curves(curves)

    fits = {name: {"best_epoch": m.best_epoch, "epochs_run": m.epochs_run, "fit_seconds": m.fit_seconds,
                   "seconds_per_epoch": m.fit_seconds / m.epochs_run}
            for name, m in [("base_train", base1), ("base_trainval", base2), ("fixed_train", fix1),
                            ("fixed_trainval", fix2), ("bias_train", bias1), ("bias_trainval", bias2),
                            ("fixed_p50_train", long1), ("fixed_p50_trainval", long2)]}
    return {"params": A2_PARAMS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seed": SEED,
            "split": s["sizes"], "shape": list(s["shape"]), "eval_users": len(users),
            "unseen": unseen, "zero_check": zero_check, "table": rows, "clip_diagnostics": diagnostics,
            "plateau": plateau, "fits": fits, "history": {name: m.history for name, m in curves.items()}}


PLATEAU_TOL = 0.002     # val RMSE gap below which an MF fit is "at the bias-only level"


def plateau_diagnosis(curves, bias, long):
    """Did an early-stopped MF fit stop in the plateau (still at the bias-only level)?

    gap_to_bias      : best val RMSE of the fit - best val RMSE of the bias-only fit (negative = MF better)
    at_bias_level    : gap_to_bias > -PLATEAU_TOL (MF not clearly better than bias only)
    stopped_early    : the fit ended before MAX_EPOCHS
    later_gain_in_p50: how much lower the patience-50 run's best val RMSE is than this fit's best
                       (> PLATEAU_TOL means training further would have helped)
    plateau_stop     : stopped_early and at_bias_level and later_gain_in_p50 > PLATEAU_TOL
    """
    bias_best = min(h["val_rmse"] for h in bias.history)
    long_best = min(h["val_rmse"] for h in long.history)
    out = {}
    for name, m in curves.items():
        best = min(h["val_rmse"] for h in m.history)
        d = {"best_epoch": m.best_epoch, "epochs_run": m.epochs_run, "best_val_rmse": best,
             "gap_to_bias": best - bias_best, "at_bias_level": bool(best - bias_best > -PLATEAU_TOL),
             "stopped_early": bool(m.epochs_run < MAX_EPOCHS), "later_gain_in_p50": best - long_best,
             "epoch_cap": "상한 도달" if m.best_epoch == MAX_EPOCHS else ""}
        d["plateau_stop"] = bool(d["stopped_early"] and d["at_bias_level"] and d["later_gain_in_p50"] > PLATEAU_TOL
                                 and m.n_factors > 0)
        out[name] = d
    return out


def plot_A2_curves(curves):
    colors = {"Base MF, patience 2": "#1baf7a", "FixedMF, patience 2": C_MF, "Bias only, patience 2": C_BIAS,
              f"FixedMF, patience {MAX_EPOCHS}": "#6250d6"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, key, title in [(axes[0], "train_rmse", "Train RMSE"), (axes[1], "val_rmse", "Validation RMSE")]:
        for name, m in curves.items():
            ep = [e["epoch"] for e in m.history]
            ax.plot(ep, [e[key] for e in m.history], "-", color=colors[name], lw=2, marker="o", ms=3, label=name)
            if key == "val_rmse":
                ax.plot([m.best_epoch], [m.history[m.best_epoch - 1]["val_rmse"]], "o", ms=8,
                        color=colors[name], mec="white", mew=2)
        ax.set(xlabel="Epoch", ylabel="RMSE", title=f"A-2 Software: {title} (k=32, lr 0.01, λ 0.05)")
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].legend(fontsize=8, frameon=False, title="big dot = val-best epoch", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_A2_learning_curves.png"), dpi=150)
    plt.close(fig)


# ==================================================================================================
# Experiment A-3: timing
# ==================================================================================================
def run_A3(factors=(0, 32, 128), n_epochs=3):
    log("[A3] loading Software")
    s = make_split(load_ratings("Software"))
    rows = []
    for k in factors:
        # patience = n_epochs so that early stopping never cuts the timed run short
        m = timed_fit(new_model(FixedMF, {**A2_PARAMS, "n_factors": k}, n_epochs, patience=n_epochs),
                      *s["train"], s["shape"], val=s["val"])
        t0 = time.perf_counter()
        evaluate(m.scores, s["eval_users"], s["known"], s["R_test"], TOP_K, THRESHOLD)
        eval_sec = time.perf_counter() - t0
        rows.append({"n_factors": k, "epochs": m.epochs_run, "fit_seconds": m.fit_seconds,
                     "seconds_per_epoch": m.fit_seconds / m.epochs_run, "evaluate_seconds": eval_sec,
                     "train_ratings": len(s["train"][2]), "eval_users": len(s["eval_users"])})
        log(f"[A3] {rows[-1]}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "A3_timing.csv"), index=False)
    return {"note": "seconds_per_epoch includes the per-epoch train/val RMSE computed inside fit",
            "machine": "MacBook Air (Apple M2, 8GB)", "table": rows}


# ==================================================================================================
# Experiment B: tuning on Software (λ first at k=32, then k at the chosen λ)
# ==================================================================================================
B_MAX_EPOCHS = 100                       # approved scale: train all epochs, restore the val-best epoch
B_LR = 0.01
B_K = 32                                 # k of the λ sweep
B_LAMBDAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
B1X_FACTOR_REGS = (2.0, 5.0, 10.0)          # B1x: extension of the split series (EXPERIMENT_LOG #14)
B2_FACTORS = (8, 16, 32, 64, 128)             # B2 grid (decision of EXPERIMENT_LOG #14)
B2_FACTOR_REGS = (0.2, 0.5, 1.0, 2.0)
C_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")   # sequential blue steps 250/400/550/700 (dataviz)
TIE = 1e-4                               # val RMSE differences below this count as a tie
C_SPLIT = "#1baf7a"                      # categorical slot 3 (aqua) for the split-regularization series

# Series of the λ sweep:
#   bias  : bias only (k=0), reg = λ                       -> λ_b* = val-best λ
#   joint : FixedMF k=32, one reg = λ for biases and vectors (as in recsys)
#   split : SplitRegMF k=32, reg_bias = λ_b*, reg_factor = λ
def make_mf(series, k, lam, lam_b, n_epochs, patience):
    if series == "split":
        return SplitRegMF(k, B_LR, reg_bias=lam_b, reg_factor=lam, n_epochs=n_epochs, patience=patience, seed=SEED)
    return FixedMF(k, B_LR, lam, n_epochs, patience=patience, seed=SEED)


def fit_and_record(s, series, k, lam, lam_b=None, max_epochs=B_MAX_EPOCHS):
    """Train on train (all epochs, val-best epoch restored) and return (row, history) with the values of
    design section 3. F1@10 is on validation (known = R_train, answers = R_val, as recsys.py:440-441)
    and is recorded only; it is never used for any choice."""
    m = timed_fit(make_mf(series, k, lam, lam_b, max_epochs, patience=max_epochs), *s["train"], s["shape"], val=s["val"])
    best = m.history[m.best_epoch - 1]
    f = evaluate(m.scores, s["eval_users"], s["R_train"], s["R_val"], TOP_K, THRESHOLD)
    # np.linalg.norm(X, axis=1): Euclidean length of every row of X. Only users/items with a training
    # rating are averaged (the others are 0 in FixedMF).
    norm_p = float(np.linalg.norm(m.P[m.seen_users], axis=1).mean()) if k > 0 else 0.0
    norm_q = float(np.linalg.norm(m.Q[m.seen_items], axis=1).mean()) if k > 0 else 0.0
    row = {"series": series, "n_factors": k, "lambda": lam, "reg_bias": lam if series != "split" else lam_b,
           "reg_factor": lam if k > 0 else None, "best_epoch": m.best_epoch,
           "epoch_cap": "상한 도달" if m.best_epoch == max_epochs else "", "val_rmse": best["val_rmse"],
           "train_rmse": best["train_rmse"], "mean_norm_p": norm_p, "mean_norm_q": norm_q,
           "val_f1": f["f1"], "val_hits": int(round(f["precision"] * TOP_K * f["n_users"])), "val_f1_users": f["n_users"],
           "fit_seconds": m.fit_seconds, "reused_from": ""}
    log(f"[B] {series} k={k} λ={lam}: best epoch {m.best_epoch}, val {row['val_rmse']:.5f}, train {row['train_rmse']:.4f}, "
        f"|p| {norm_p:.3f}, |q| {norm_q:.3f}, val hits {row['val_hits']}, {m.fit_seconds:.1f}s")
    return row, m.history


def add_gaps(rows, bias_rows, lam_b):
    """gap_vs_bias_best: val - val of the best bias-only model (λ_b*), the main comparison.
    gap_vs_bias_same_lambda: val - val of the bias-only model with the same bias regularization
    (joint: λ, split: λ_b*), for reference."""
    bias_val = {r["lambda"]: r["val_rmse"] for r in bias_rows}
    for r in rows:
        r["gap_vs_bias_best"] = r["val_rmse"] - bias_val[lam_b]
        r["gap_vs_bias_same_lambda"] = r["val_rmse"] - bias_val[r["reg_bias"]]


def pick(rows, key_order):
    """Lowest val RMSE; rows within TIE of the best are ordered by key_order (simpler model first)."""
    best = min(r["val_rmse"] for r in rows)
    return sorted([r for r in rows if r["val_rmse"] < best + TIE], key=key_order)[0]


def run_B1(df=None, max_epochs=B_MAX_EPOCHS):
    log("[B1] loading Software")
    s = make_split(load_ratings("Software") if df is None else df)
    rows, curves = [], []

    def record(series, k, lam, lam_b=None):
        row, hist = fit_and_record(s, series, k, lam, lam_b, max_epochs)
        rows.append(row)
        curves.extend({"series": series, "n_factors": k, "lambda": lam, **e} for e in hist)
        return row

    bias_rows = [record("bias", 0, lam) for lam in B_LAMBDAS]
    lam_b = pick(bias_rows, lambda r: -r["lambda"])["lambda"]          # tie -> larger λ
    log(f"[B1] λ_b* = {lam_b}")
    joint_rows = [record("joint", B_K, lam) for lam in B_LAMBDAS]
    split_rows = [record("split", B_K, lam, lam_b) for lam in B_LAMBDAS]
    add_gaps(bias_rows + joint_rows + split_rows, bias_rows, lam_b)

    # Real-data check of SplitRegMF: reg_factor = λ_b* is the same model as joint λ = λ_b*.
    same = [r for r in split_rows if r["lambda"] == lam_b][0], [r for r in joint_rows if r["lambda"] == lam_b][0]
    split_equals_joint = bool(same[0]["val_rmse"] == same[1]["val_rmse"] and same[0]["best_epoch"] == same[1]["best_epoch"])

    # λ*: best (series, λ) over both k=32 series; tie -> joint (team model) first, then larger λ.
    chosen = pick(joint_rows + split_rows, lambda r: (r["series"] != "joint", -r["lambda"]))
    lam_star, series_star = chosen["lambda"], chosen["series"]
    boundary = {"lambda_star_at_upper_end": lam_star == max(B_LAMBDAS), "lambda_star_at_zero": lam_star == 0.0,
                "lambda_b_star_at_upper_end": lam_b == max(B_LAMBDAS), "lambda_b_star_at_zero": lam_b == 0.0}
    log(f"[B1] chosen series={series_star} λ*={lam_star} (val {chosen['val_rmse']:.5f}), λ_b*={lam_b}, "
        f"boundary {boundary}, split==joint at λ_b*: {split_equals_joint}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "B1_lambda_sweep.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "B_learning_curves.csv"), index=False)
    plot_B_lambda(bias_rows, joint_rows, split_rows, lam_b, chosen)
    plot_B_curves(pd.DataFrame(curves), [("joint", B_K, 0.0, None, C_BIAS, "no L2 (λ = 0)"),
                                         (series_star, B_K, lam_star, chosen["best_epoch"], C_MF,
                                          f"chosen: {series_star}, λ = {lam_star}")])
    return {"max_epochs": max_epochs, "lr": B_LR, "k": B_K, "lambdas": list(B_LAMBDAS), "seed": SEED,
            "lambda_b_star": lam_b, "lambda_star": lam_star, "series_star": series_star, "chosen": chosen,
            "boundary": boundary, "split_equals_joint_at_lambda_b": split_equals_joint, "table": rows}


def upsert_csv(path, new_rows, keys=("series", "n_factors", "lambda")):
    """Add rows to a CSV file, replacing existing rows with the same keys (re-runs never duplicate)."""
    new = pd.DataFrame(new_rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        drop = set(zip(*[new[k] for k in keys]))
        old = old[[key not in drop for key in zip(*[old[k] for k in keys])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def b_results(*names):
    report = load_results().get("B", {})
    missing = [n for n in names if n not in report]
    if missing:
        raise SystemExit(f"results for {missing} not found in outputs/mf/results.json: run them first")
    return [report[n] for n in names]


def run_B1x(df=None, max_epochs=B_MAX_EPOCHS):
    """Extension of the B1 split series: k=32, reg_bias = λ_b*, reg_factor ∈ B1X_FACTOR_REGS.
    Purpose: does the split series approach the best bias-only model from above?
    Also records the validation hits of Popularity (same validation F1 protocol as fit_and_record)."""
    (b1,) = b_results("B1")
    lam_b = b1["lambda_b_star"]
    bias_rows = [r for r in b1["table"] if r["series"] == "bias"]
    log(f"[B1x] loading Software; split k={B_K}, reg_bias={lam_b}, reg_factor {B1X_FACTOR_REGS}")
    s = make_split(load_ratings("Software") if df is None else df)
    rows, curves = [], []
    for rf in B1X_FACTOR_REGS:
        row, hist = fit_and_record(s, "split", B_K, rf, lam_b, max_epochs)
        rows.append(row)
        curves.extend({"series": "split", "n_factors": B_K, "lambda": rf, **e} for e in hist)
    add_gaps(rows, bias_rows, lam_b)
    all_nonneg = bool(all(r["gap_vs_bias_best"] >= 0 for r in rows))

    pop = Popularity().fit(*s["train"], s["shape"])
    f = evaluate(pop.scores, s["eval_users"], s["R_train"], s["R_val"], TOP_K, THRESHOLD)
    pop_row = {"model": "Popularity (reference)", "trained_on": "train", "val_f1": f["f1"],
               "val_precision": f["precision"], "val_recall": f["recall"],
               "val_hits": int(round(f["precision"] * TOP_K * f["n_users"])), "val_f1_users": f["n_users"]}
    log(f"[B1x] Popularity on validation: {pop_row}")
    pd.DataFrame([pop_row]).to_csv(os.path.join(OUT, "B1_val_reference.csv"), index=False)

    upsert_csv(os.path.join(OUT, "B1_lambda_sweep.csv"), rows)
    upsert_csv(os.path.join(OUT, "B_learning_curves.csv"), curves, keys=("series", "n_factors", "lambda", "epoch"))
    table = pd.read_csv(os.path.join(OUT, "B1_lambda_sweep.csv")).to_dict("records")
    split_all = sorted([r for r in table if r["series"] == "split"], key=lambda r: r["lambda"])
    plot_B_lambda(bias_rows, [r for r in b1["table"] if r["series"] == "joint"], split_all, lam_b,
                  pick(split_all + [r for r in b1["table"] if r["series"] == "joint"],
                       lambda r: (r["series"] != "joint", -r["lambda"])))
    log(f"[B1x] gaps vs best bias only: {[round(r['gap_vs_bias_best'], 6) for r in rows]}, all >= 0: {all_nonneg}")
    return {"max_epochs": max_epochs, "k": B_K, "reg_bias": lam_b, "factor_regs": list(B1X_FACTOR_REGS),
            "table": rows, "all_gaps_nonnegative": all_nonneg, "popularity_val": pop_row}


def run_B2(df=None, max_epochs=B_MAX_EPOCHS):
    """Grid k × reg_factor for the split series (reg_bias = λ_b*), then the final test of 4 models.
    Starts only if B1x found no MF setting below the best bias-only model (approved condition)."""
    b1, b1x = b_results("B1", "B1x")
    if not b1x["all_gaps_nonnegative"]:
        raise SystemExit("[B2] B1x has a split setting below the best bias-only model: stop and report")
    lam_b = b1["lambda_b_star"]
    bias_row = [r for r in b1["table"] if r["series"] == "bias" and r["lambda"] == lam_b][0]
    k32 = {r["lambda"]: r for r in b1["table"] + b1x["table"] if r["series"] == "split" and r["n_factors"] == B_K}
    log(f"[B2] loading Software; grid k {B2_FACTORS} × reg_factor {B2_FACTOR_REGS}, reg_bias={lam_b}")
    s = make_split(load_ratings("Software") if df is None else df)

    rows, curves = [], []
    for k in B2_FACTORS:
        for rf in B2_FACTOR_REGS:
            if k == B_K:                                  # same settings and seed as B1/B1x: reuse
                rows.append({**k32[rf], "reused_from": "B1" if rf in B_LAMBDAS else "B1x"})
                continue
            row, hist = fit_and_record(s, "split", k, rf, lam_b, max_epochs)
            rows.append(row)
            curves.extend({"series": "split", "n_factors": k, "lambda": rf, **e} for e in hist)
    for r in rows:
        r["gap_vs_bias_best"] = r["val_rmse"] - bias_row["val_rmse"]
        r["gap_vs_bias_same_lambda"] = r["gap_vs_bias_best"]          # bias regularization is λ_b* everywhere
    edge = lambda r: ("경계" if r["n_factors"] in (min(B2_FACTORS), max(B2_FACTORS))
                      or r["lambda"] in (min(B2_FACTOR_REGS), max(B2_FACTOR_REGS)) else "")

    # Best reg_factor for every k (tie -> larger reg_factor), and the overall choice (tie -> smaller k).
    best_k = []
    for k in B2_FACTORS:
        r = pick([x for x in rows if x["n_factors"] == k], lambda x: -x["lambda"])
        best_k.append({"n_factors": k, "best_reg_factor": r["lambda"], "val_rmse": r["val_rmse"],
                       "gap_vs_bias_best": r["gap_vs_bias_best"], "best_epoch": r["best_epoch"],
                       "epoch_cap": r["epoch_cap"], "mean_norm_p": r["mean_norm_p"], "mean_norm_q": r["mean_norm_q"],
                       "val_hits": r["val_hits"], "train_rmse": r["train_rmse"],
                       "reg_factor_at_edge": "경계" if r["lambda"] in (min(B2_FACTOR_REGS), max(B2_FACTOR_REGS)) else ""})
    chosen = pick(rows, lambda x: (x["n_factors"], -x["lambda"]))
    for r in rows:
        r["chosen"] = r is chosen
        r["boundary"] = edge(r) if r is chosen else ""
    k_star, rf_star = chosen["n_factors"], chosen["lambda"]
    log(f"[B2] chosen k*={k_star} reg_factor*={rf_star} (val {chosen['val_rmse']:.5f}, gap {chosen['gap_vs_bias_best']:+.6f}, "
        f"best epoch {chosen['best_epoch']}) {edge(chosen)}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "B2_k_sweep.csv"), index=False)
    pd.DataFrame(best_k).to_csv(os.path.join(OUT, "B2_k_best.csv"), index=False)
    upsert_csv(os.path.join(OUT, "B_learning_curves.csv"), curves, keys=("series", "n_factors", "lambda", "epoch"))

    # Final test: the chosen MF and the three reference models (all settings chosen on validation).
    # Refit on train+val for the val-best number of epochs (problem 2 fix).
    finals = [
        ("Global mean", GlobalMean().fit(*s["train"], s["shape"]), "train", None),
        ("Popularity", Popularity().fit(*s["train"], s["shape"]), "train", None),
        (f"Bias only (λ_b*={lam_b})", make_mf("bias", 0, lam_b, None, bias_row["best_epoch"], 1), "train+val", bias_row),
        (f"MF split (k*={k_star}, reg_factor*={rf_star}, reg_bias={lam_b})",
         make_mf("split", k_star, rf_star, lam_b, chosen["best_epoch"], 1), "train+val", chosen),
    ]
    test_rows = []
    for name, m, data, sel in finals:
        if data == "train+val":
            timed_fit(m, *s["trval"], s["shape"])
        res = evaluate(m.scores, s["eval_users"], s["known"], s["R_test"], TOP_K, THRESHOLD)
        test_rows.append({"model": name, "trained_on": data, "rmse": res["rmse"] if m.predicts_ratings else None,
                          "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
                          "hits": int(round(res["precision"] * TOP_K * res["n_users"])), "n_users": res["n_users"],
                          "best_epoch": sel["best_epoch"] if sel else None, "val_rmse": sel["val_rmse"] if sel else None,
                          "boundary": edge(sel) if sel is chosen else ""})
        log(f"[B2] test {test_rows[-1]}")
    pd.DataFrame(test_rows).to_csv(os.path.join(OUT, "B_final_test.csv"), index=False)

    plot_B_k(rows, best_k, bias_row["val_rmse"], chosen, lam_b)
    plot_B_curves(pd.read_csv(os.path.join(OUT, "B_learning_curves.csv")),
                  [("joint", B_K, 0.0, None, C_BIAS, f"no L2 (k={B_K}, λ = 0)"),
                   ("split", k_star, rf_star, chosen["best_epoch"], C_MF,
                    f"chosen: k={k_star}, reg_factor={rf_star}, reg_bias={lam_b}")])
    return {"max_epochs": max_epochs, "reg_bias": lam_b, "factors": list(B2_FACTORS), "factor_regs": list(B2_FACTOR_REGS),
            "k_star": k_star, "reg_factor_star": rf_star, "chosen": chosen, "boundary": edge(chosen),
            "best_by_k": best_k, "table": rows, "final_test": test_rows, "popularity_val": b1x["popularity_val"]}


def _style(ax):
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_B_lambda(bias_rows, joint_rows, split_rows, lam_b, chosen):
    lams = sorted({r["lambda"] for r in joint_rows + split_rows})
    pos = {v: n for n, v in enumerate(lams)}          # equal spacing: λ = 0 rules out a log axis
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for rows, color, label in [(joint_rows, C_MF, f"MF k={B_K}, one λ for biases and vectors"),
                               (split_rows, C_SPLIT, f"MF k={B_K}, bias λ fixed at λ_b*={lam_b}, vector λ varies")]:
        wide = rows is joint_rows                     # one-λ series drawn wide underneath: both stay visible
        top.plot([pos[r["lambda"]] for r in rows], [r["val_rmse"] for r in rows], "-o", color=color,
                 lw=4 if wide else 1.5, ms=8 if wide else 5, label=label)
    bias_best = [r for r in bias_rows if r["lambda"] == lam_b][0]["val_rmse"]
    top.axhline(bias_best, color=C_BIAS, lw=2, label=f"bias only, best λ_b*={lam_b} (val {bias_best:.4f})")
    top.plot([pos[chosen["lambda"]]], [chosen["val_rmse"]], "o", ms=12, mfc="none", mec="#0b0b0b", mew=1.5,
             label=f"lowest: {chosen['series']}, λ={chosen['lambda']} (val {chosen['val_rmse']:.4f})")
    top.set(ylabel="Validation RMSE (val-best epoch)", title="B-2: regularization sweep on Software (k=32, lr 0.01)")
    top.legend(fontsize=8, frameon=False)
    for rows, color, name in [(joint_rows, C_MF, "one λ"), (split_rows, C_SPLIT, "split λ")]:
        lw, ms = (4, 8) if rows is joint_rows else (1.5, 5)
        x = [pos[r["lambda"]] for r in rows]
        bot.plot(x, [r["mean_norm_p"] for r in rows], "-o", color=color, lw=lw, ms=ms, label=f"mean |p_u| ({name})")
        bot.plot(x, [r["mean_norm_q"] for r in rows], "--s", color=color, lw=lw, ms=ms, label=f"mean |q_i| ({name})")
    bot.set(xlabel="λ (vector regularization; one-λ series: also the bias regularization)", ylabel="Mean vector length")
    bot.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])
    bot.legend(fontsize=8, frameon=False, ncol=2)
    for ax in (top, bot):
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_lambda_sweep.png"), dpi=150)
    plt.close(fig)


def plot_B_k(rows, best_k, bias_best_val, chosen, lam_b):
    x = {k: n for n, k in enumerate(B2_FACTORS)}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for rf, color in zip(B2_FACTOR_REGS, C_RAMP):     # ordered magnitude -> one-hue ramp, light to dark
        rr = sorted([r for r in rows if r["lambda"] == rf], key=lambda r: r["n_factors"])
        ax.plot([x[r["n_factors"]] for r in rr], [r["val_rmse"] for r in rr], "-o", color=color, lw=2, ms=5,
                label=f"reg_factor = {rf:g}")
    ax.plot([x[b["n_factors"]] for b in best_k], [b["val_rmse"] for b in best_k], "-", color="#0b0b0b", lw=1,
            label="best reg_factor for each k")
    ax.plot([x[b["n_factors"]] for b in best_k], [b["val_rmse"] for b in best_k], "o", ms=11, mfc="none",
            mec="#0b0b0b", mew=1.5)
    ax.axhline(bias_best_val, color=C_BIAS, lw=2, label=f"bias only, best λ_b*={lam_b} (val {bias_best_val:.4f})")
    ax.plot([x[chosen["n_factors"]]], [chosen["val_rmse"]], "*", ms=14, color="#0b0b0b",
            label=f"chosen: k={chosen['n_factors']}, reg_factor={chosen['lambda']:g} (val {chosen['val_rmse']:.5f})")
    ax.set_xticks(range(len(B2_FACTORS)), [str(k) for k in B2_FACTORS])
    ax.set(xlabel="Number of latent factors k", ylabel="Validation RMSE (val-best epoch)",
           title=f"B-1: k × vector regularization on Software (bias λ = {lam_b}, lr 0.01)")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_k_sweep.png"), dpi=150)
    plt.close(fig)


def plot_B_curves(c, lines):
    """c: learning-curve DataFrame. lines: (series, k, λ, best_epoch or None, color, label); the dot marks
    best_epoch (the epoch restored by the 1e-4 improvement rule), or the curve minimum if None."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for series, k, lam, best, color, label in lines:
        h = c[(c.series == series) & (c.n_factors == k) & (c["lambda"] == lam)].sort_values("epoch").set_index("epoch")
        ax.plot(h.index, h.train_rmse, "--", color=color, lw=2, label=f"{label}: train")
        ax.plot(h.index, h.val_rmse, "-", color=color, lw=2, label=f"{label}: validation")
        b = int(best) if best is not None else int(h.val_rmse.idxmin())
        ax.plot([b], [h.val_rmse[b]], "o", ms=8, color=color, mec="white", mew=2)
    ax.set(xlabel="Epoch", ylabel="RMSE", title="B-4: MF learning curves on Software (lr 0.01; dot = val-best epoch)")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_learning_curves.png"), dpi=150)
    plt.close(fig)


# ==================================================================================================
# Experiment C: evaluation and analysis (Software, Video Games). Decision D16.
# ==================================================================================================
C_MAX_EPOCHS = 100                       # VG epoch reselection (policy 2): same scale as experiment B
C_BUCKETS = [("4-5", 4, 5), ("6-10", 6, 10), ("11+", 11, None)]
C_BUCKET_NOTE = ("Users are grouped by their number of train+val ratings (the data the final model was trained on). "
                 "Data are 5-core and time_split puts at least one rating in test, so the minimum is 4: "
                 "buckets 4-5 / 6-10 / 11+. RMSE over all test ratings of the users in the bucket.")
ANALYSIS = "분석용(선택 대상 아님)"
C_COLORS = {"bias": C_BIAS, "selected": C_MF, "team": C_SPLIT, "overfit": "#eda100", "overfit100": "#6250d6"}


def c_models():
    """The models of experiment C with their Software epochs (read from the A/B results, not retyped).

    key, label, constructor(n_epochs), Software epochs, analysis flag
    """
    report = load_results()
    b1, b2 = report["B"]["B1"], report["B"]["B2"]
    row = lambda series, lam: [r for r in b1["table"] if r["series"] == series and r["lambda"] == lam][0]
    lam_b = b1["lambda_b_star"]
    k_s, rf_s, ep_s = b2["k_star"], b2["reg_factor_star"], b2["chosen"]["best_epoch"]
    return [
        ("bias", f"Bias only (λ_b*={lam_b})", lambda n: FixedMF(0, B_LR, lam_b, n, seed=SEED),
         row("bias", lam_b)["best_epoch"], False),
        ("selected", f"Selected MF (k={k_s}, reg_factor {rf_s:g}, reg_bias {lam_b})",
         lambda n: SplitRegMF(k_s, B_LR, reg_bias=lam_b, reg_factor=rf_s, n_epochs=n, seed=SEED), ep_s, False),
        ("team", "Team default MF (k=32, λ=0.05)", lambda n: FixedMF(32, B_LR, 0.05, n, seed=SEED),
         row("joint", 0.05)["best_epoch"], False),
        ("overfit", "Overfit MF (k=32, λ=0)", lambda n: FixedMF(32, B_LR, 0.0, n, seed=SEED),
         row("joint", 0.0)["best_epoch"], False),
        ("overfit100", "Overfit MF (k=32, λ=0, 100 epochs) [analysis, not selected]", lambda n: FixedMF(32, B_LR, 0.0, n, seed=SEED),
         C_MAX_EPOCHS, True),
    ]


def c_fit_dataset(name, s, models, policy, max_epochs=C_MAX_EPOCHS):
    """Final models (train+val refit) of one dataset under one epoch policy.

    policy "sw_epochs"    : train+val for the Software epochs (policy 1)
    policy "vg_val_epochs": train on train for max_epochs (val-best restored), then train+val for that epoch (policy 2)
    The analysis model always uses its fixed 100 epochs.
    """
    fitted = {}
    for key, label, make, sw_ep, analysis in models:
        best_val = None
        if policy == "vg_val_epochs" and not analysis:
            m1 = make(max_epochs)
            m1.patience = max_epochs                                     # train all epochs, restore val-best
            timed_fit(m1, *s["train"], s["shape"], val=s["val"])
            epochs, best_val = m1.best_epoch, m1.history[m1.best_epoch - 1]["val_rmse"]
        else:
            epochs = sw_ep
        m = timed_fit(make(epochs), *s["trval"], s["shape"])
        fitted[key] = {"model": m, "label": label, "epochs": epochs, "val_rmse": best_val, "analysis": analysis,
                       "epoch_cap": "상한 도달" if (policy == "vg_val_epochs" and not analysis and epochs == max_epochs) else ""}
        log(f"[C] {name} {policy} {label}: {epochs} epochs, train+val fit {m.fit_seconds:.1f}s")
    return fitted


def top10(score_fn, users, known, chunk=500):
    """Top-K exactly as recsys.evaluate (recsys.py:293-297): known items set to -inf, then np.argpartition.
    Also returns the top-K scores (to count items at the clip maximum)."""
    tops, vals = [], []
    for s in range(0, len(users), chunk):
        batch = users[s:s + chunk]
        S = score_fn(batch)
        K = known[batch].tocoo()
        S[K.row, K.col] = -np.inf
        top = np.argpartition(-S, TOP_K, axis=1)[:, :TOP_K]
        tops.append(top)
        # np.take_along_axis(S, idx, axis=1): picks S[row, idx[row, j]] for every row (the scores of the top-K)
        vals.append(np.take_along_axis(S, top, axis=1))
    return np.concatenate(tops), np.concatenate(vals)


def count_hits(top, users, R_test):
    """Total number of top-K items that are relevant held-out items (rating >= THRESHOLD), as in recsys.evaluate."""
    T = R_test[users].tocsr()
    hits = 0
    for row in range(len(users)):
        lo, hi = T.indptr[row], T.indptr[row + 1]
        relevant = set(T.indices[lo:hi][T.data[lo:hi] >= THRESHOLD])
        hits += len(relevant.intersection(top[row]))
    return hits


def rec_popularity(name, s, fitted, pops):
    """C-3: popularity and b_i of the recommended items, for clipped scores (main) and unclipped (diagnosis)."""
    users, known = s["eval_users"], s["known"]
    item_pop = np.bincount(s["trval"][1], minlength=s["shape"][1])      # train+val ratings per item
    b_ref = fitted["bias"]["model"].bi                                    # common b_i reference: bias-only model
    entries = [(k, f["label"], f["model"], "clipped", f["analysis"]) for k, f in fitted.items()]
    entries += [(k, p["label"], p["model"], "counts", False) for k, p in pops.items()]
    entries += [(k, fitted[k]["label"], fitted[k]["model"], "unclipped (diagnosis)", False)
                for k in ("bias", "selected", "team", "overfit", "overfit100")]   # D16 + approved extension (#20)
    rows = []
    for key, label, m, ranking, analysis in entries:
        fn = m.raw_scores if ranking.startswith("unclipped") else m.scores
        top, vals = top10(fn, users, known)
        rated = m.predicts_ratings
        rows.append({"dataset": name, "model": label, "key": key, "ranking": ranking,
                     "note": ANALYSIS if analysis else ("진단" if ranking.startswith("unclipped") else ""),
                     "mean_pop": float(item_pop[top].mean()), "median_pop": float(np.median(item_pop[top])),
                     "mean_bi_ref": float(b_ref[top].mean()),
                     "mean_bi_own": float(m.bi[top].mean()) if rated else None,
                     "share_top10_at_cap": float((vals >= CLIP_MAX).mean()) if rated else None,
                     "distinct_items": int(len(np.unique(top))), "hits": count_hits(top, users, s["R_test"]),
                     "n_users": len(users)})
        log(f"[C3] {name} {label} [{ranking}]: pop mean {rows[-1]['mean_pop']:.1f} median {rows[-1]['median_pop']:.0f}, "
            f"b_ref {rows[-1]['mean_bi_ref']:+.3f}, at cap {rows[-1]['share_top10_at_cap']}, hits {rows[-1]['hits']}")
    return rows


def rmse_by_activity(name, s, fitted):
    """C-2: test RMSE by number of train+val ratings per user (all test users, and the evaluation users)."""
    n_tv = np.bincount(s["trval"][0], minlength=s["shape"][0])
    assert n_tv[np.unique(s["test"][0])].min() >= C_BUCKETS[0][1], "a test user has fewer train+val ratings than 4"
    tu, ti, tr = s["test"]
    is_eval = np.isin(tu, s["eval_users"])
    rows = []
    for key, f in fitted.items():
        pred = f["model"].predict(tu, ti)
        for scope, mask in [("all", np.ones(len(tu), bool)), ("eval", is_eval)]:
            for bname, lo, hi in C_BUCKETS:
                cnt = n_tv[tu]
                sel = mask & (cnt >= lo) & ((cnt <= hi) if hi is not None else True)
                rows.append({"dataset": name, "model": f["label"], "key": key, "note": ANALYSIS if f["analysis"] else "",
                             "users": scope, "bucket": bname, "n_users": int(len(np.unique(tu[sel]))),
                             "n_ratings": int(sel.sum()), "rmse": rmse(tr[sel], pred[sel])})
    bias = {(r["dataset"], r["users"], r["bucket"]): r["rmse"] for r in rows if r["key"] == "bias"}
    for r in rows:
        r["diff_vs_bias"] = r["rmse"] - bias[(r["dataset"], r["users"], r["bucket"])]
    return rows


def hypothesis(rows, ranking):
    """C-3 hypothesis on one dataset, with every model ranked the same way (ranking: 'clipped' or
    'unclipped (diagnosis)'): the overfit MF recommends more popular items, and the RMSE-optimal models
    (selected MF, bias only) recommend items with larger b_i (common reference: bias-only model)."""
    get = lambda key: [r for r in rows if r["key"] == key and r["ranking"] == ranking][0]
    out = {"models": {k: {"mean_pop": get(k)["mean_pop"], "median_pop": get(k)["median_pop"],
                          "mean_bi_ref": get(k)["mean_bi_ref"], "share_top10_at_cap": get(k)["share_top10_at_cap"]}
                      for k in ("bias", "selected", "team", "overfit", "overfit100")}}
    for over in ("overfit", "overfit100"):                       # overfit100 is the analysis model
        o = get(over)
        for key in ("selected", "bias"):
            r = get(key)
            out[f"{over}_vs_{key}"] = {"overfit_more_popular": bool(o["mean_pop"] > r["mean_pop"]),
                                       "larger_bi_than_overfit": bool(r["mean_bi_ref"] > o["mean_bi_ref"])}
    out["supported"] = bool(all(out[f"overfit_vs_{k}"]["overfit_more_popular"] and out[f"overfit_vs_{k}"]["larger_bi_than_overfit"]
                                for k in ("selected", "bias")))
    out["supported_with_analysis_model"] = bool(all(
        out[f"overfit100_vs_{k}"]["overfit_more_popular"] and out[f"overfit100_vs_{k}"]["larger_bi_than_overfit"]
        for k in ("selected", "bias")))
    return out


def run_C(sw_df=None, vg_df=None, max_epochs=C_MAX_EPOCHS):
    """sw_df, vg_df: optional DataFrames for smoke tests; the reproduction asserts run only on the full data."""
    full = sw_df is None and vg_df is None
    models = c_models()
    if not full:                          # smoke test: short epochs
        models = [(k, l, mk, min(ep, 2), a) for k, l, mk, ep, a in models]
    log("[C] loading Software and Video Games")
    data = {"Software": make_split(load_ratings("Software") if sw_df is None else sw_df),
            "Video Games": make_split(load_ratings("Video_Games") if vg_df is None else vg_df)}
    runs = {("Software", "sw_epochs"): c_fit_dataset("Software", data["Software"], models, "sw_epochs", max_epochs),
            ("Video Games", "vg_val_epochs"): c_fit_dataset("Video Games", data["Video Games"], models, "vg_val_epochs", max_epochs),
            ("Video Games", "sw_epochs"): c_fit_dataset("Video Games", data["Video Games"], models, "sw_epochs", max_epochs)}
    pops = {name: {"pop_trval": {"model": Popularity().fit(*s["trval"], s["shape"]), "label": "Popularity (train+val)"},
                   "pop_train": {"model": Popularity().fit(*s["train"], s["shape"]), "label": "Popularity (train, reference)"}}
            for name, s in data.items()}

    # C-1
    c1 = []
    for (name, policy), fitted in runs.items():
        s = data[name]
        entries = [(f["label"], f["model"], f["epochs"], f["val_rmse"], f["epoch_cap"],
                    ANALYSIS if f["analysis"] else ("민감도" if policy == "sw_epochs" and name == "Video Games" else ""))
                   for f in fitted.values()]
        if policy == "sw_epochs":                                       # Popularity has no epochs: once per dataset
            entries += [(p["label"], p["model"], None, None, "", "참고" if k == "pop_train" else "")
                        for k, p in pops[name].items()]
        for label, m, epochs, val, cap, note in entries:
            res = evaluate(m.scores, s["eval_users"], s["known"], s["R_test"], TOP_K, THRESHOLD)
            c1.append({"dataset": name, "epoch_policy": policy if m.predicts_ratings else "-", "model": label,
                       "note": note, "rmse": res["rmse"] if m.predicts_ratings else None,
                       "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
                       "hits": int(round(res["precision"] * TOP_K * res["n_users"])), "n_users": res["n_users"],
                       "epochs": epochs, "val_rmse_vg": val, "epoch_cap": cap})
            log(f"[C1] {name} {policy} {label}: rmse {c1[-1]['rmse']}, f1 {res['f1']:.6f}, hits {c1[-1]['hits']}")
    pd.DataFrame(c1).to_csv(os.path.join(OUT, "C1_test.csv"), index=False)

    repro = None
    if full:
        # Reproduction: identical settings and seed as experiments A and B must give identical test RMSE.
        report = load_results()
        b2_test = {r["model"]: r["rmse"] for r in report["B"]["B2"]["final_test"]}
        a2_team = [r for r in report["A"]["A2"]["table"] if r["model"] == "MF + fixes 1,2 (after fixes)"][0]["rmse"]
        sw = {r["model"]: r["rmse"] for r in c1 if r["dataset"] == "Software" and r["epoch_policy"] == "sw_epochs"}
        lab = {k: l for k, l, _, _, _ in models}
        repro = {"bias": [sw[lab["bias"]], [v for k, v in b2_test.items() if k.startswith("Bias only")][0]],
                 "selected": [sw[lab["selected"]], [v for k, v in b2_test.items() if k.startswith("MF split")][0]],
                 "team": [sw[lab["team"]], a2_team]}
        log(f"[C] reproduction (C vs A/B): {repro}")
        for key, (now, before) in repro.items():
            assert now == before, f"reproduction failed for {key}: {now} != {before}"

    # C-2 (main epoch policy for VG) and C-3 (both datasets)
    main_runs = {"Software": runs[("Software", "sw_epochs")], "Video Games": runs[("Video Games", "vg_val_epochs")]}
    c2 = [r for name, f in main_runs.items() for r in rmse_by_activity(name, data[name], f)]
    pd.DataFrame(c2).to_csv(os.path.join(OUT, "C2_by_activity.csv"), index=False)
    c3 = [r for name, f in main_runs.items() for r in rec_popularity(name, data[name], f, pops[name])]
    pd.DataFrame(c3).to_csv(os.path.join(OUT, "C3_rec_popularity.csv"), index=False)
    # Consistency: hits counted on our top-10 (clipped scores) equal the hits of recsys.evaluate in C-1.
    for r in c3:
        if r["ranking"] in ("clipped", "counts"):
            policy = "sw_epochs" if r["dataset"] == "Software" else "vg_val_epochs"
            c1_hits = [x["hits"] for x in c1 if x["dataset"] == r["dataset"] and x["model"] == r["model"]
                       and x["epoch_policy"] in (policy, "-")][0]
            assert r["hits"] == c1_hits, f"C-3 hits {r['hits']} != C-1 hits {c1_hits} for {r['model']}"
    hyp = {name: {"clipped": hypothesis([r for r in c3 if r["dataset"] == name], "clipped"),
                  "unclipped": hypothesis([r for r in c3 if r["dataset"] == name], "unclipped (diagnosis)")}
           for name in main_runs}
    log(f"[C3] hypothesis {hyp}")
    plot_C_activity(c2)
    plot_C_popularity(c3)
    return {"models": [{"key": k, "label": l, "software_epochs": ep, "analysis": a} for k, l, _, ep, a in models],
            "vg_epochs": {f["label"]: f["epochs"] for f in runs[("Video Games", "vg_val_epochs")].values()},
            "bucket_definition": C_BUCKET_NOTE, "buckets": [b[0] for b in C_BUCKETS],
            "split_sizes": {n: s["sizes"] for n, s in data.items()}, "reproduction": repro,
            "c1": c1, "c2": c2, "c3": c3, "hypothesis": hyp}


def plot_C_activity(c2):
    rows = [r for r in c2 if r["users"] == "all"]
    keys = list(dict.fromkeys(r["key"] for r in rows))
    labels = {r["key"]: r["model"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, name in zip(axes, ("Software", "Video Games")):
        x = np.arange(len(C_BUCKETS))
        w = 0.8 / len(keys)
        for j, key in enumerate(keys):
            vals = [[r for r in rows if r["dataset"] == name and r["key"] == key and r["bucket"] == b][0]["rmse"]
                    for b, _, _ in C_BUCKETS]
            ax.bar(x + (j - (len(keys) - 1) / 2) * w, vals, w * 0.92, color=C_COLORS[key],
                   hatch="//" if key == "overfit100" else None, edgecolor="white", label=labels[key])
        nus = [[r for r in rows if r["dataset"] == name and r["key"] == "bias" and r["bucket"] == b][0]
               for b, _, _ in C_BUCKETS]
        ax.set_xticks(x, [f"{b}\n{r['n_users']:,} users\n{r['n_ratings']:,} ratings" for (b, _, _), r in zip(C_BUCKETS, nus)],
                      fontsize=8.5)
        ax.set(xlabel="train+val ratings per user", title=f"C-2 {name}: test RMSE by user activity")
        _style(ax)
    axes[0].set_ylabel("Test RMSE (all test users)")
    axes[0].set_ylim(bottom=0.8)
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_by_activity.png"), dpi=150)
    plt.close(fig)


C_SHORT = {"bias": "Bias only", "selected": "Selected MF (k=16)", "team": "Team default MF (k=32, λ=0.05)",
           "overfit": "Overfit MF (k=32, λ=0, val-best)", "overfit100": "Overfit MF (λ=0, 100 ep) [analysis]",
           "pop_trval": "Popularity (train+val)", "pop_train": "Popularity (train) [reference]"}


def plot_C_popularity(c3):
    """Dot plot (a log axis would distort bar lengths): filled dot = mean, hollow dot = median."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), gridspec_kw={"width_ratios": [1.3, 1]})
    for row_axes, name in zip(axes, ("Software", "Video Games")):
        rows = [r for r in c3 if r["dataset"] == name]
        y = np.arange(len(rows))[::-1]
        names = [C_SHORT[r["key"]] + (" [unclipped, diagnosis]" if r["ranking"].startswith("unclipped") else "")
                 for r in rows]
        colors = ["#8a8984" if r["key"].startswith("pop") else C_COLORS[r["key"]] for r in rows]
        ax = row_axes[0]
        ax.scatter([r["mean_pop"] for r in rows], y, c=colors, s=60, zorder=3, label="mean")
        ax.scatter([r["median_pop"] for r in rows], y, facecolors="none", edgecolors=colors, s=60, lw=1.5, zorder=3,
                   label="median")
        ax.hlines(y, [min(r["mean_pop"], r["median_pop"]) for r in rows],
                  [max(r["mean_pop"], r["median_pop"]) for r in rows], colors=colors, lw=1.5)
        ax.set_xscale("log")
        ax.set_yticks(y, names, fontsize=8)
        ax.set(xlabel="train+val ratings of recommended items (log)", title=f"C-3 {name}: popularity of the top-10")
        from matplotlib.lines import Line2D                         # legend markers in neutral ink, not a series color
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#52514e", label="mean"),
                           Line2D([], [], marker="o", ls="", mfc="none", color="#52514e", label="median")],
                  fontsize=8, frameon=False, loc="lower right")
        ax = row_axes[1]
        ax.scatter([r["mean_bi_ref"] for r in rows], y, c=colors, s=60, zorder=3)
        ax.axvline(0, color="#52514e", lw=1)
        ax.set_yticks(y, [""] * len(rows))
        ax.set(xlabel="mean b_i (bias-only model, common reference)", title=f"C-3 {name}: item bias of the top-10")
        for a in row_axes:
            a.grid(axis="x", color="#e5e4e0", lw=0.8)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_rec_popularity.png"), dpi=150)
    plt.close(fig)


# ==================================================================================================
# Experiment D: cross-domain Software -> Video Games. Decisions D15, D17 (EXPERIMENT_LOG #22).
# ==================================================================================================
D_MAX_EPOCHS = 100
D_BRIDGE_FRACS = (0.0, 0.25, 0.5, 1.0)
D_BOOT = 1000                            # bootstrap resamples of the hidden-test users
D_FINAL_NOTE = ("The model whose epoch was chosen on the hidden-val users' VG RMSE is used as the final model, without "
                "retraining on train+val: adding the hidden-val users' VG ratings to training would turn them into "
                "bridge users and break the definition of the bridge fraction (especially 0%).")


def d_models():
    """(key, label, constructor, Software epochs, analysis). Software epochs are for the sensitivity rows."""
    sw_ep = {k: ep for k, _, _, ep, _ in c_models()}
    return [
        ("bias", "Bias only (λ_b*=0.05)", lambda n: FixedMF(0, B_LR, 0.05, n, seed=SEED), sw_ep["bias"], False),
        ("selected", "Selected MF (k=16, reg_factor 0.5, reg_bias 0.05)",
         lambda n: SplitRegMF(16, B_LR, reg_bias=0.05, reg_factor=0.5, n_epochs=n, seed=SEED), sw_ep["selected"], False),
        ("team", "Team default MF (k=32, λ=0.05)", lambda n: FixedMF(32, B_LR, 0.05, n, seed=SEED), sw_ep["team"], False),
        ("overfit", "Overfit MF (k=32, λ=0) [analysis, not selected]", lambda n: FixedMF(32, B_LR, 0.0, n, seed=SEED),
         None, True),
    ]


def masked(fn, mask):
    """Score function restricted to candidate items (others -inf), as evaluate(candidates=...) does (recsys.py:295-296)."""
    def g(users):
        S = fn(users)
        S[:, ~mask] = -np.inf
        return S
    return g


def per_user_stats(m, c, ranking):
    """Per hidden-test user: squared-error sum, rating count, hits in the top-10 (for the bootstrap)."""
    users = c["test_users"]
    pos = {u: j for j, u in enumerate(users)}
    tu, ti, tr = c["test"]
    row = np.array([pos[u] for u in tu])
    if hasattr(m, "predict"):
        err = m.predict(tu, ti) - tr
    elif m.predicts_ratings:                               # GlobalMean: the same mean for every pair
        err = np.full(len(tr), m.mu, dtype=np.float32) - tr
    else:                                                  # Popularity: no rating prediction
        err = np.zeros(len(tr))
    se = np.bincount(row, weights=err.astype(np.float64) ** 2, minlength=len(users))
    n = np.bincount(row, minlength=len(users))
    fn = m.raw_scores if ranking == "unclipped" else m.scores
    no_history = csr_matrix(c["shape"], dtype=np.float32)
    top, _ = top10(masked(fn, c["vg_items"]), users, no_history)
    T = c["R_test"][users].tocsr()
    hits = np.zeros(len(users), int)
    for j in range(len(users)):
        lo, hi = T.indptr[j], T.indptr[j + 1]
        hits[j] = len(set(T.indices[lo:hi][T.data[lo:hi] >= THRESHOLD]).intersection(top[j]))
    return se, n, hits, float(err.mean())


def bootstrap_ci(a, b, kind, n_boot=D_BOOT, seed=SEED + 3):
    """95% percentile interval of (model a - model b) over users resampled with replacement.
    kind 'rmse': a, b = (se, n) per user; kind 'hits': a, b = hits per user.
    np.random.default_rng(seed).integers(0, n, (B, n)): B rows of n random user positions (resampling)."""
    n_users = len(a[0]) if kind == "rmse" else len(a)
    idx = np.random.default_rng(seed).integers(0, n_users, (n_boot, n_users))
    if kind == "rmse":
        f = lambda x: np.sqrt(x[0][idx].sum(1) / x[1][idx].sum(1))
        d = f(a) - f(b)
    else:
        d = a[idx].sum(1) - b[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])            # np.percentile: values below which 2.5% / 97.5% of d fall
    return float(lo), float(hi)


def run_D(sw_df=None, vg_df=None, max_epochs=D_MAX_EPOCHS):
    full = sw_df is None and vg_df is None
    log("[D] loading Software and Video Games")
    c = make_cross_split(load_ratings("Software") if sw_df is None else sw_df,
                         load_ratings("Video_Games") if vg_df is None else vg_df)
    log(f"[D] shared {len(c['shared'])}, hidden-test {len(c['hidden_test'])}, hidden-val {len(c['hidden_val'])}, "
        f"bridge {len(c['bridge'])}")
    models = d_models()
    no_history = csr_matrix(c["shape"], dtype=np.float32)
    rows, curves, infos, stats, fitted_100 = [], [], [], {}, {}

    def add_rows(frac, policy, key, label, m, epochs, val_rmse, analysis):
        rankings = ("clipped", "unclipped") if hasattr(m, "raw_scores") else (("constant",) if m.predicts_ratings else ("counts",))
        for ranking in rankings:
            fn = m.raw_scores if ranking == "unclipped" else m.scores
            res = evaluate(fn, c["test_users"], no_history, c["R_test"], TOP_K, THRESHOLD, candidates=c["vg_items"])
            se, n, hits, mean_err = per_user_stats(m, c, ranking)
            assert int(hits.sum()) == int(round(res["precision"] * TOP_K * res["n_users"])), "hits mismatch"
            stats[(frac, policy, key, ranking)] = (se, n, hits)
            rows.append({"bridge_frac": frac, "epoch_policy": policy, "model": label, "key": key,
                         "note": ANALYSIS if analysis else ("민감도" if policy == "sw_epochs" else ""),
                         "ranking": ranking, "rmse": res["rmse"] if m.predicts_ratings else None,
                         "mean_error": mean_err if m.predicts_ratings else None,
                         "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
                         "hits": int(hits.sum()), "n_users": res["n_users"], "epochs": epochs, "val_rmse": val_rmse,
                         "epoch_cap": "상한 도달" if epochs == max_epochs and policy == "val_users" else ""})
            log(f"[D] frac {frac} {policy} {label} [{ranking}]: rmse {rows[-1]['rmse']}, mean err {rows[-1]['mean_error']}, "
                f"hits {rows[-1]['hits']}")

    for frac in D_BRIDGE_FRACS:
        train, vg_train, unseen_vg, info = cross_train(c, frac)
        infos.append(info)
        log(f"[D] frac {frac}: {info}")
        for key, label, make, _, analysis in models:
            m = make(max_epochs)
            m.patience = max_epochs                          # train all epochs, restore the hidden-val best epoch
            timed_fit(m, *train, c["shape"], val=c["val"])
            # unseen VG items are zero in FixedMF / SplitRegMF
            assert not m.Q[unseen_vg].any() and not m.bi[unseen_vg].any()
            curves.extend({"bridge_frac": frac, "key": key, **e} for e in m.history)
            add_rows(frac, "val_users", key, label, m, m.best_epoch, m.history[m.best_epoch - 1]["val_rmse"], analysis)
            if frac == 1.0:
                fitted_100[key] = m
        tu, ti, tr = vg_train
        add_rows(frac, "val_users", "gm", "Global mean (VG)", GlobalMean().fit(tu, ti, tr, c["shape"]), None, None, False)
        add_rows(frac, "val_users", "pop", "Popularity (VG)", Popularity().fit(tu, ti, tr, c["shape"]), None, None, False)

    # Sensitivity at 100% bridges: team way, Software best epochs, no validation.
    train, _, _, _ = cross_train(c, 1.0)
    for key, label, make, sw_ep, analysis in models:
        if sw_ep is None:
            continue
        m = timed_fit(make(sw_ep), *train, c["shape"])
        add_rows(1.0, "sw_epochs", key, label, m, sw_ep, None, analysis)

    # MF - bias differences with bootstrap intervals (same fraction, policy and ranking).
    for r in rows:
        r.update({"diff_rmse_vs_bias": None, "ci_low_rmse": None, "ci_high_rmse": None,
                  "diff_hits_vs_bias": None, "ci_low_hits": None, "ci_high_hits": None})
        b = stats.get((r["bridge_frac"], r["epoch_policy"], "bias", r["ranking"]))
        if r["key"] in ("selected", "team", "overfit") and b is not None:
            a = stats[(r["bridge_frac"], r["epoch_policy"], r["key"], r["ranking"])]
            bias_row = [x for x in rows if x["bridge_frac"] == r["bridge_frac"] and x["epoch_policy"] == r["epoch_policy"]
                        and x["key"] == "bias" and x["ranking"] == r["ranking"]][0]
            r["diff_rmse_vs_bias"] = r["rmse"] - bias_row["rmse"]
            r["ci_low_rmse"], r["ci_high_rmse"] = bootstrap_ci(a[:2], b[:2], "rmse")
            r["diff_hits_vs_bias"] = r["hits"] - bias_row["hits"]
            r["ci_low_hits"], r["ci_high_hits"] = bootstrap_ci(a[2], b[2], "hits")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "D_cross_domain.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "D_learning_curves.csv"), index=False)
    plot_D_bridge(rows)
    pca = plot_D_pca(fitted_100, c)
    return {"split": {"shared": len(c["shared"]), "hidden_test": len(c["hidden_test"]), "hidden_val": len(c["hidden_val"]),
                      "bridge": len(c["bridge"]), "hidden_test_vg_ratings": len(c["test"][2]),
                      "hidden_val_vg_ratings": len(c["val"][2])},
            "final_model_note": D_FINAL_NOTE, "bridge_fracs": list(D_BRIDGE_FRACS), "max_epochs": max_epochs,
            "bootstrap": {"resamples": D_BOOT, "seed": SEED + 3, "unit": "hidden-test user"},
            "data_by_fraction": infos, "pca": pca, "table": rows}


def plot_D_bridge(rows):
    main = [r for r in rows if r["epoch_policy"] == "val_users"]
    fr = list(D_BRIDGE_FRACS)
    x = np.arange(len(fr))
    keys = [("bias", C_BIAS), ("selected", C_MF), ("team", C_SPLIT), ("overfit", C_COLORS["overfit"])]
    get = lambda key, ranking, f: [r for r in main if r["key"] == key and r["ranking"] == ranking and r["bridge_frac"] == f][0]
    label = {"bias": "Bias only", "selected": "Selected MF (k=16)", "team": "Team default MF (k=32)",
             "overfit": "Overfit MF λ=0 [analysis]"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for key, color in keys:
        ax.plot(x, [get(key, "clipped", f)["rmse"] for f in fr], "-o", color=color, lw=2, ms=6, label=label[key])
    ax.plot(x, [get("gm", "constant", f)["rmse"] for f in fr], ":", color=C_REF, lw=2, label="Global mean (VG)")
    ax.set(ylabel="RMSE on hidden-test users (VG)", title="D-3: RMSE by share of bridge users")
    ax = axes[1]
    w = 0.25
    for j, (key, color) in enumerate(keys[1:]):
        rr = [get(key, "clipped", f) for f in fr]
        d = np.array([r["diff_rmse_vs_bias"] for r in rr])
        err = np.array([[v - r["ci_low_rmse"] for v, r in zip(d, rr)], [r["ci_high_rmse"] - v for v, r in zip(d, rr)]])
        ax.errorbar(x + (j - 1) * w, d, yerr=err, fmt="o", color=color, ms=6, capsize=4, lw=1.5, label=label[key])
    ax.axhline(0, color=C_BIAS, lw=2, label="bias only (0)")
    ax.set(ylabel="RMSE difference: model − bias only", title="MF − bias only, 95% bootstrap interval")
    ax = axes[2]
    for key, color in keys:
        ax.plot(x, [get(key, "clipped", f)["hits"] for f in fr], "-o", color=color, lw=2, ms=6, label=f"{label[key]} (clipped)")
        ax.plot(x, [get(key, "unclipped", f)["hits"] for f in fr], "--s", color=color, lw=1.5, ms=5, label=f"{label[key]} (unclipped)")
    ax.plot(x, [get("pop", "counts", f)["hits"] for f in fr], ":", color=C_REF, lw=2, label="Popularity (VG)")
    ax.set(ylabel="Hits in top-10 (hidden-test users)", title="D-3: top-10 hits by share of bridge users")
    for ax in axes:
        ax.set_xticks(x, [f"{int(f * 100)}%" for f in fr])
        ax.set_xlabel("Share of bridge users kept")
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_bridge_users.png"), dpi=150)
    plt.close(fig)


def plot_D_pca(fitted, c):
    """D-4: item vectors of the 100%-bridge models projected on their first two principal components."""
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, key, title in [(axes[0], "selected", "Selected MF (k=16)"), (axes[1], "team", "Team default MF (k=32)")]:
        m = fitted[key]
        Q = m.Q[m.seen_items].astype(np.float64)
        dom = c["vg_items"][m.seen_items]
        Qc = Q - Q.mean(0)
        # np.linalg.svd(X, full_matrices=False): X = U S V^T; rows of V^T are the principal directions
        _, S, Vt = np.linalg.svd(Qc, full_matrices=False)
        Z = Qc @ Vt[:2].T
        evr = (S[:2] ** 2 / (S ** 2).sum()).tolist()
        cent = {"Software": Z[~dom].mean(0).tolist(), "Video Games": Z[dom].mean(0).tolist()}
        out[key] = {"explained_variance_ratio_pc1_pc2": evr, "centroids": cent,
                    "mean_norm_q": {"Software": float(np.linalg.norm(Q[~dom], axis=1).mean()),
                                    "Video Games": float(np.linalg.norm(Q[dom], axis=1).mean())}}
        # Draw the points in a random order, so that neither domain hides the other (fixed seed).
        order = np.random.default_rng(SEED).permutation(len(Z))
        ax.scatter(Z[order, 0], Z[order, 1], s=2, alpha=0.3, c=np.where(dom[order], C_BIAS, C_MF), rasterized=True)
        for color, name in [(C_MF, "Software items"), (C_BIAS, "Video Games items")]:
            ax.scatter([], [], s=12, color=color, label=name)
        ax.set(xlabel=f"PC1 ({evr[0]:.1%})", ylabel=f"PC2 ({evr[1]:.1%})", title=f"D-4: item vectors, {title}, 100% bridges")
        ax.legend(fontsize=8, frameon=False, markerscale=4)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_cross_pca.png"), dpi=150)
    plt.close(fig)
    return out


def run_D5():
    """D-5 (approved addition): how much of the cross-domain error comes from the transferred user bias b_u?

    Retrains the 100%-bridge models of experiment D for their recorded epochs (main: hidden-val epochs;
    sensitivity: Software epochs). Training for e epochs without validation gives exactly the epoch-e state
    of the validated run (same seed, and predict() uses no random numbers), checked by an assert on the RMSE.
    Then predicts the hidden-test users' VG ratings again with b_u = 0.
    """
    d = pd.read_csv(os.path.join(OUT, "D_cross_domain.csv"))
    sw, vg = load_ratings("Software"), load_ratings("Video_Games")
    c = make_cross_split(sw, vg)
    train, _, _, _ = cross_train(c, 1.0)
    ht = list(c["hidden_test"])
    stats = {"software_mean_all": float(sw["rating"].mean()),
             "software_mean_hidden_test": float(sw.loc[sw["user_id"].isin(ht), "rating"].mean()),
             "vg_mean_all": float(vg["rating"].mean()),
             "vg_mean_hidden_test": float(vg.loc[vg["user_id"].isin(ht), "rating"].mean()),
             "joint_train_mean": float(train[2].mean())}
    log(f"[D5] rating means {stats}")
    tu, ti, tr = c["test"]
    users = c["test_users"]
    rows = []
    for key, label, make, sw_ep, analysis in d_models():
        for policy in ("val_users", "sw_epochs"):
            rec = d[(d.bridge_frac == 1.0) & (d.epoch_policy == policy) & (d.key == key) & (d.ranking == "clipped")]
            if rec.empty:
                continue
            epochs = int(rec.epochs.iloc[0])
            m = timed_fit(make(epochs), *train, c["shape"])
            pred = m.predict(tu, ti)
            assert abs(rmse(tr, pred) - rec.rmse.iloc[0]) < 1e-6, f"D-5 reproduction failed for {key} {policy}"
            seen = m.seen_users[tu] & m.seen_items[ti]
            pred0 = np.clip(m.mu + m.bi[ti] + (m.P[tu] * m.Q[ti]).sum(1) * seen, 1, 5)     # b_u = 0
            rows.append({"model": label, "key": key, "epoch_policy": policy, "epochs": epochs,
                         "note": ANALYSIS if analysis else ("민감도" if policy == "sw_epochs" else ""),
                         "rmse": rmse(tr, pred), "mean_error": float((pred - tr).mean()),
                         "rmse_bu0": rmse(tr, pred0), "mean_error_bu0": float((pred0 - tr).mean()),
                         "mean_bu_hidden_test": float(m.bu[users].mean()), "mu": m.mu,
                         "mean_bi_hidden_test_items": float(m.bi[ti].mean()), "fit_seconds": m.fit_seconds})
            log(f"[D5] {label} {policy} ({epochs} ep): {rows[-1]}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "D5_bias_shift.csv"), index=False)
    return {"rating_means": stats, "table": rows,
            "note": "Models retrained for their recorded epochs; RMSE equals D_cross_domain.csv (assert)."}


# ==================================================================================================
# Experiment E: BPR-loss matrix factorization. Decision D20 (EXPERIMENT_LOG #27).
# Reference: S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme, "BPR: Bayesian Personalized
# Ranking from Implicit Feedback", UAI 2009. The loss and update rules below follow that paper
# (LearnBPR with uniform sampling); the code is written for this project.
# ==================================================================================================
def timed_bpr(model, *args, **kwargs):
    t0 = time.perf_counter()
    model.fit(*args, **kwargs)
    model.fit_seconds = time.perf_counter() - t0
    return model


def make_bpr_synthetic(n_users=1000, n_items=500, k=5, n_pos=40, n_hidden=5, scale=3.0, seed=SEED):
    """Implicit data from a known ranking: true score s_ui = b_i + p_u . q_i with
    b ~ N(0, (0.5 scale)^2), P, Q ~ N(0, 0.5^2 scale) (so p_u . q_i also grows by `scale`).
    scale = 1 was run 1 (EXPERIMENT_LOG #29): too noisy for the Gumbel draw (true-score AUC 0.708);
    scale = 3 is the approved setting (#30).
    Each user's n_pos positives are drawn without replacement with probability increasing in s_ui
    (Gumbel-top-k: the top n_pos of s_ui + Gumbel noise). n_hidden of them per user are held out."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 0.5 * scale, n_items)
    P, Q = rng.normal(0, 0.5 * np.sqrt(scale), (n_users, k)), rng.normal(0, 0.5 * np.sqrt(scale), (n_items, k))
    S = b[None, :] + P @ Q.T
    G = S + rng.gumbel(size=S.shape)                   # rng.gumbel: standard Gumbel noise
    pos = np.argpartition(-G, n_pos, axis=1)[:, :n_pos]
    hid_mask = np.zeros_like(pos, bool)
    for u in range(n_users):
        hid_mask[u, rng.choice(n_pos, n_hidden, replace=False)] = True
    uu = np.repeat(np.arange(n_users), n_pos).reshape(n_users, n_pos)
    train = (uu[~hid_mask], pos[~hid_mask])
    hidden = (uu[hid_mask], pos[hid_mask])
    is_pos = np.zeros((n_users, n_items), bool)
    is_pos[uu.ravel(), pos.ravel()] = True
    return train, hidden, is_pos, S, (n_users, n_items)


def auc(score, hidden, is_pos):
    """Mean over users of P(score of a hidden positive > score of an item the user never chose)."""
    out = []
    hu, hi = hidden
    for u in np.unique(hu):
        neg = score[u][~is_pos[u]]
        p = score[u][hi[hu == u]]
        out.append(float((p[:, None] > neg[None, :]).mean() + 0.5 * (p[:, None] == neg[None, :]).mean()))
    return float(np.mean(out))


E1_SEEDS = (SEED, SEED + 10)            # second seed: approved check that the criteria were not fitted to seed 42


def e1_one(seed, scale):
    """Train BPRMF(k=5) on one synthetic data set; return the summary row and the loss history.

    Loss criteria 1a-1c use the loss on a FIXED evaluation sample of triples (one uniform negative per
    training positive, drawn once with default_rng(seed + 1)), recomputed after every epoch (#32). The true-score
    ("oracle") loss is computed on the same sample. The loss measured during SGD is kept as a reference column.
    """
    train, hidden, is_pos, S, shape = make_bpr_synthetic(scale=scale, seed=seed)
    m = BPRMF(5, 0.05, 0.01, 50, seed=SEED)
    keys = np.unique(train[0].astype(np.int64) * shape[1] + train[1])
    ej = m._negatives(np.random.default_rng(seed + 1), train[0], keys, shape[1])
    triples = (train[0], train[1], ej)
    timed_bpr(m, *train, shape, rated=train, eval_triples=triples)
    pop = np.bincount(train[1], minlength=shape[1]).astype(float)
    a = {"bpr": auc(m.scores(np.arange(shape[0])), hidden, is_pos),
         "popularity": auc(np.tile(pop, (shape[0], 1)), hidden, is_pos), "oracle": auc(S, hidden, is_pos)}
    oracle_loss = float(np.logaddexp(0, -(S[train[0], train[1]] - S[train[0], ej])).mean())
    loss = [h["eval_loss"] for h in m.history]
    ratio = [y / x for x, y in zip(loss, loss[1:])]
    up = [(e + 2, r) for e, r in enumerate(ratio) if r > 1.01]
    checks = {"1a_loss_last_le_0.7_first": loss[-1] <= 0.7 * loss[0],
              "1b_loss_last_le_1.05_oracle_loss": loss[-1] <= 1.05 * oracle_loss,
              "1c_loss_never_up_more_than_1pct": not up,
              "2a_auc_ge_0.9_oracle": a["bpr"] >= 0.9 * a["oracle"],
              "2b_auc_minus_popularity_ge_0.05": a["bpr"] - a["popularity"] >= 0.05}
    train_loss = [h["bpr_loss"] for h in m.history]
    row = {"seed": seed, "scale": scale, "role": "test" if scale == 3.0 else "reference (run 1 data, not a pass criterion)",
           "loss_basis": "fixed evaluation triples", "loss_first": loss[0], "loss_last": loss[-1], "oracle_loss": oracle_loss,
           "auc_bpr": a["bpr"], "auc_popularity": a["popularity"], "auc_oracle": a["oracle"], **checks,
           "passed": bool(all(checks.values())), "1c_fail_epochs": " ".join(str(e) for e, _ in up),
           "1c_max_increase": float(max(ratio) - 1), "sgd_loss_first": train_loss[0], "sgd_loss_last": train_loss[-1],
           "fit_seconds": m.fit_seconds}
    log(f"[E1] seed {seed} scale {scale}: {row}")
    return row, [{"seed": seed, "scale": scale, **h} for h in m.history]


def e1_verdict(rows):
    """Decision rule fixed before the run (EXPERIMENT_LOG #32). Reproducibility is judged outside (diff of 2 runs)."""
    test = [r for r in rows if r["scale"] == 3.0]
    ok = lambda key: all(r[key] for r in test)
    if all(r["passed"] for r in test):
        return "pass"
    if not (ok("1b_loss_last_le_1.05_oracle_loss") and ok("2a_auc_ge_0.9_oracle") and ok("2b_auc_minus_popularity_ge_0.05")):
        return "stop: 1b, 2a or 2b failed"
    if not ok("1a_loss_last_le_0.7_first"):
        return "stop: 1a failed (case not in the rule, treated conservatively)"
    return "pass (implementation correct; 1c = SGD fluctuation after convergence)"


def run_E1():
    rows, curves = [], []
    for seed in E1_SEEDS:
        r, c = e1_one(seed, 3.0)
        rows.append(r)
        curves += c
    r, c = e1_one(SEED, 1.0)                                      # control: weak signal (run 1 data)
    rows.append(r)
    curves += c
    verdict = e1_verdict(rows)
    passed = verdict.startswith("pass")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E1_synthetic.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "E1_synthetic_curves.csv"), index=False)
    log(f"[E1] verdict (both seeds, scale 3, before the reproducibility diff): {verdict}")
    fig, (l, r_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {(SEED, 3.0): C_MF, (SEED + 10, 3.0): C_SPLIT, (SEED, 1.0): "#8a8984"}
    for row in rows:
        h = [c for c in curves if c["seed"] == row["seed"] and c["scale"] == row["scale"]]
        col = colors[(row["seed"], row["scale"])]
        name = f"seed {row['seed']}, signal ×{row['scale']:g}"
        l.plot([c["epoch"] for c in h], [c["eval_loss"] for c in h], "-", color=col, lw=2, label=name)
        l.axhline(row["oracle_loss"], color=col, lw=1, ls=":")
    l.set(xlabel="Epoch", ylabel="BPR loss on fixed evaluation triples",
          title="E-1: BPR loss (dotted = loss of the true scores)")
    l.legend(fontsize=8, frameon=False)
    y = np.arange(len(rows))
    w = 0.27
    for n, (key, col, lab_) in enumerate([("auc_bpr", C_MF, "BPR (k=5)"), ("auc_popularity", "#bdbdbd", "Popularity"),
                                          ("auc_oracle", "#52514e", "True scores")]):
        r_ax.barh(y + (1 - n) * w, [row[key] for row in rows], w * 0.9, color=col, label=lab_)
    r_ax.set_yticks(y, [f"seed {row['seed']}, ×{row['scale']:g}" for row in rows])
    r_ax.axvline(0.5, color="#52514e", lw=1, ls=":")
    r_ax.set(xlim=(0.4, 1.0), xlabel="AUC: hidden positives vs never-chosen items", title="E-1: ranking recovery")
    r_ax.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in (l, r_ax):
        _style(ax)
    r_ax.grid(axis="x", color="#e5e4e0", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_bpr_synthetic.png"), dpi=150)
    plt.close(fig)
    return {"config": {"n_users": 1000, "n_items": 500, "k_true": 5, "positives_per_user": 40, "hidden_per_user": 5,
                       "k": 5, "lr": 0.05, "reg": 0.01, "epochs": 50, "seeds": list(E1_SEEDS), "scale": 3.0},
            "rows": rows, "verdict": verdict, "passed": passed,
            "rule": "EXPERIMENT_LOG #32 (fixed before the run); reproducibility checked by a diff of two runs"}


E_FACTORS = (0, 16, 32, 64)
E_REGS = (0.001, 0.01, 0.1)
E_LR, E_LR_CHECK, E_MAX_EPOCHS = 0.05, 0.01, 40


def e_rank_stats(name, model_label, key, ranking, top, users, R_test, item_pop, b_ref):
    """Per-user hits / relevant counts and the C-3 recommendation statistics of one top-10."""
    T = R_test[users].tocsr()
    hits, nrel = np.zeros(len(users), int), np.zeros(len(users), int)
    for j in range(len(users)):
        lo, hi = T.indptr[j], T.indptr[j + 1]
        rel = set(T.indices[lo:hi][T.data[lo:hi] >= THRESHOLD])
        nrel[j] = len(rel)
        hits[j] = len(rel.intersection(top[j]))
    rec = {"dataset": name, "model": model_label, "key": key, "ranking": ranking,
           "mean_pop": float(item_pop[top].mean()), "median_pop": float(np.median(item_pop[top])),
           "mean_bi_ref": float(b_ref[top].mean()), "distinct_items": int(len(np.unique(top))), "hits": int(hits.sum())}
    return hits, nrel, rec


def e_metrics(hits, nrel):
    ok = nrel > 0
    p, r = hits[ok] / TOP_K, hits[ok] / nrel[ok]
    f1 = np.where(hits[ok] > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f1, ok


def e_evaluate(name, entries, users, known, R_test, item_pop, b_ref, candidates=None, ref_key="bpr", seed=SEED + 4):
    """entries: (key, label, score_fn, ranking, note). Returns test rows (with bootstrap intervals) and C-3 rows."""
    score_of = (lambda fn: masked(fn, candidates)) if candidates is not None else (lambda fn: fn)
    per, rows, recs = {}, [], []
    for key, label, fn, ranking, note in entries:
        top, _ = top10(score_of(fn), users, known)
        hits, nrel, rec = e_rank_stats(name, label, key, ranking, top, users, R_test, item_pop, b_ref)
        res = evaluate(fn, users, known, R_test, TOP_K, THRESHOLD, candidates=candidates)
        assert int(hits.sum()) == int(round(res["precision"] * TOP_K * res["n_users"])), f"hits mismatch {label}"
        per[(key, ranking)] = (hits, nrel)
        rec["note"] = note
        recs.append(rec)
        rows.append({"dataset": name, "model": label, "key": key, "ranking": ranking, "note": note,
                     "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
                     "hits": int(hits.sum()), "n_users": res["n_users"]})
    # Bootstrap over the users with at least one relevant item (the F1 users), 1,000 resamples.
    ok = next(iter(per.values()))[1] > 0
    idx = np.random.default_rng(seed).integers(0, int(ok.sum()), (D_BOOT, int(ok.sum())))
    ref = [k for k in per if k[0] == ref_key][0]
    for r in rows:
        hits, nrel = per[(r["key"], r["ranking"])]
        _, _, f1, _ = e_metrics(hits, nrel)
        h = hits[ok]
        r["f1_ci_low"], r["f1_ci_high"] = [float(v) for v in np.percentile(f1[idx].mean(1), [2.5, 97.5])]
        r["hits_ci_low"], r["hits_ci_high"] = [float(v) for v in np.percentile(h[idx].sum(1), [2.5, 97.5])]
        hr = per[ref][0][ok]
        d = hr[idx].sum(1) - h[idx].sum(1)
        r["diff_hits_ref_minus_model"] = int(hr.sum() - h.sum())
        r["diff_ci_low"], r["diff_ci_high"] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
        r["reference"] = ref_key
    return rows, recs


def run_E(sw_df=None, vg_df=None, max_epochs=E_MAX_EPOCHS, factors=E_FACTORS, regs=E_REGS):
    full = sw_df is None and vg_df is None
    report = load_results()
    # ---------------- Software: tuning on validation ----------------
    s = make_split(load_ratings("Software") if sw_df is None else sw_df)
    users = s["eval_users"]
    rated_tr = s["train"][:2]

    def val_fn(m):
        res = evaluate(m.scores, users, s["R_train"], s["R_val"], TOP_K, THRESHOLD)
        return int(round(res["precision"] * TOP_K * res["n_users"])), res["recall"]

    tuning, curves, models = [], [], {}

    def tune(k, reg, lr, positives="all", tag="grid"):
        u, i, r = s["train"]
        sel = np.ones(len(r), bool) if positives == "all" else r >= THRESHOLD
        m = timed_bpr(BPRMF(k, lr, reg, max_epochs, seed=SEED), u[sel], i[sel], s["shape"], rated=rated_tr, val_fn=val_fn)
        best = [h for h in m.history if h["epoch"] == m.best_epoch][0]
        row = {"n_factors": k, "reg": reg, "lr": lr, "positives": positives, "tag": tag, "best_epoch": m.best_epoch,
               "epoch_cap": "상한 도달" if m.best_epoch == max_epochs else "", "val_hits": best["val_hits"],
               "val_recall": best["val_recall"], "loss_first": m.history[0]["bpr_loss"],
               "loss_best": best["bpr_loss"], "users_without_positive": int((~m.has_pos).sum()), "fit_seconds": m.fit_seconds}
        tuning.append(row)
        curves.extend({"n_factors": k, "reg": reg, "lr": lr, "positives": positives, **h} for h in m.history)
        log(f"[E] tune {row}")
        return row

    for k in factors:
        for reg in regs:
            tune(k, reg, E_LR)
    order = lambda r: (-r["val_hits"], -r["val_recall"], r["n_factors"], -r["reg"])
    best_k0 = sorted([r for r in tuning if r["n_factors"] == 0], key=order)[0]
    best_mf = sorted([r for r in tuning if r["n_factors"] > 0], key=order)[0]
    lr_row = tune(best_mf["n_factors"], best_mf["reg"], E_LR_CHECK, tag="lr check")
    best_mf = sorted([best_mf, lr_row], key=order)[0]
    pos_b = tune(best_mf["n_factors"], best_mf["reg"], best_mf["lr"], positives=">=4", tag="positives (b)")
    pd.DataFrame(tuning).to_csv(os.path.join(OUT, "E_tuning.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "E_learning_curves.csv"), index=False)
    log(f"[E] chosen BPR {best_mf}, BPR k=0 {best_k0}")
    plot_E_tuning(tuning)

    # ---------------- Software: final models on train+val, test ----------------
    tv_u, tv_i, tv_r = s["trval"]

    def final_bpr(row, positives_sel):
        sel = np.ones(len(tv_r), bool) if positives_sel == "all" else tv_r >= THRESHOLD
        return timed_bpr(BPRMF(row["n_factors"], row["lr"], row["reg"], row["best_epoch"], seed=SEED),
                         tv_u[sel], tv_i[sel], s["shape"], rated=(tv_u, tv_i))

    bpr_a, bpr_0, bpr_b = final_bpr(best_mf, "all"), final_bpr(best_k0, "all"), final_bpr(pos_b, ">=4")
    assert not bpr_b.P[~bpr_b.has_pos].any()                    # (b): users without positives have p_u = 0
    cm = {k: (l, mk, ep) for k, l, mk, ep, _ in c_models()}
    mse = {k: timed_fit(cm[k][1](cm[k][2]), *s["trval"], s["shape"]) for k in ("bias", "selected", "overfit")}
    pop = Popularity().fit(*s["trval"], s["shape"])
    lab = lambda row, extra="": f"BPR (k={row['n_factors']}, λ={row['reg']}, lr={row['lr']}{extra})"
    entries = [("bpr", lab(best_mf), bpr_a.scores, "scores", ""),
               ("bpr_k0", lab(best_k0) + " [b_i only]", bpr_0.scores, "scores", ""),
               ("bpr_pos4", lab(pos_b, ", positives ≥4"), bpr_b.scores, "scores", "비교 행"),
               ("pop", "Popularity (train+val)", pop.scores, "counts", "")]
    for k in ("bias", "selected", "overfit"):
        entries += [(k, cm[k][0], mse[k].scores, "clipped", ""), (k, cm[k][0], mse[k].raw_scores, "unclipped", "진단")]
    item_pop = np.bincount(tv_i, minlength=s["shape"][1])
    sw_rows, sw_recs = e_evaluate("Software", entries, users, s["known"], s["R_test"], item_pop, mse["bias"].bi)
    if full:          # reproduction: MSE models give the hits of experiment C
        c3 = {(r["key"], r["ranking"]): r["hits"] for r in report["C"]["C"]["c3"] if r["dataset"] == "Software"}
        for r in sw_rows:
            if r["key"] in ("bias", "selected", "overfit"):
                want = c3[(r["key"], "clipped" if r["ranking"] == "clipped" else "unclipped (diagnosis)")]
                assert r["hits"] == want, f"E vs C hits differ for {r['model']} {r['ranking']}: {r['hits']} != {want}"
    rows, recs = list(sw_rows), list(sw_recs)

    # ---------------- E-3a: Video Games in-domain (settings fixed, epochs re-chosen on VG validation) ----------------
    v = make_split(load_ratings("Video_Games") if vg_df is None else vg_df)

    def vg_val_fn(m):
        res = evaluate(m.scores, v["eval_users"], v["R_train"], v["R_val"], TOP_K, THRESHOLD)
        return int(round(res["precision"] * TOP_K * res["n_users"])), res["recall"]

    vg_epochs = {}
    vg_bpr = {}
    for key, row in (("bpr", best_mf), ("bpr_k0", best_k0)):
        m1 = timed_bpr(BPRMF(row["n_factors"], row["lr"], row["reg"], max_epochs, seed=SEED), *v["train"][:2], v["shape"],
                       rated=v["train"][:2], val_fn=vg_val_fn)
        vg_epochs[key] = m1.best_epoch
        curves_vg = [{"dataset": "Video Games", "key": key, **h} for h in m1.history]
        curves.extend(curves_vg)
        vg_bpr[key] = timed_bpr(BPRMF(row["n_factors"], row["lr"], row["reg"], m1.best_epoch, seed=SEED),
                                *v["trval"][:2], v["shape"], rated=v["trval"][:2])
    c_vg = report["C"]["C"]["vg_epochs"] if full else {}
    vg_mse = {}
    for k in ("bias", "selected", "overfit"):
        ep = c_vg.get(cm[k][0], cm[k][2])
        vg_mse[k] = timed_fit(cm[k][1](ep), *v["trval"], v["shape"])
    entries = [("bpr", lab(best_mf), vg_bpr["bpr"].scores, "scores", ""),
               ("bpr_k0", lab(best_k0) + " [b_i only]", vg_bpr["bpr_k0"].scores, "scores", ""),
               ("pop", "Popularity (train+val)", Popularity().fit(*v["trval"], v["shape"]).scores, "counts", "")]
    for k in ("bias", "selected", "overfit"):
        entries += [(k, cm[k][0], vg_mse[k].scores, "clipped", ""), (k, cm[k][0], vg_mse[k].raw_scores, "unclipped", "진단")]
    vg_rows, vg_recs = e_evaluate("Video Games", entries, v["eval_users"], v["known"], v["R_test"],
                                  np.bincount(v["trval"][1], minlength=v["shape"][1]), vg_mse["bias"].bi)
    if full:
        c1 = {r["model"]: r["hits"] for r in report["C"]["C"]["c1"] if r["dataset"] == "Video Games" and r["epoch_policy"] == "vg_val_epochs"}
        for r in vg_rows:
            if r["key"] in ("bias", "selected", "overfit") and r["ranking"] == "clipped":
                assert r["hits"] == c1[r["model"]], f"E-3a vs C hits differ for {r['model']}"
    rows += vg_rows
    recs += vg_recs

    # ---------------- E-3b: cross-domain, 100% bridges (epochs chosen on hidden-val users) ----------------
    c = make_cross_split(load_ratings("Software") if sw_df is None else sw_df,
                         load_ratings("Video_Games") if vg_df is None else vg_df)
    train, vg_train, _, _ = cross_train(c, 1.0)
    no_hist = csr_matrix(c["shape"], dtype=np.float32)
    val_df = c["vg"][c["vg"]["user_id"].isin(c["hidden_val"])]
    R_val, val_users = c["enc"].matrix(val_df), np.sort(c["enc"].users.get_indexer(list(c["hidden_val"])))

    def cross_val_fn(m):
        res = evaluate(m.scores, val_users, no_hist, R_val, TOP_K, THRESHOLD, candidates=c["vg_items"])
        return int(round(res["precision"] * TOP_K * res["n_users"])), res["recall"]

    cross_bpr, cross_epochs = {}, {}
    for key, row in (("bpr", best_mf), ("bpr_k0", best_k0)):
        m = timed_bpr(BPRMF(row["n_factors"], row["lr"], row["reg"], max_epochs, seed=SEED), *train[:2], c["shape"],
                      rated=train[:2], val_fn=cross_val_fn)
        cross_bpr[key], cross_epochs[key] = m, m.best_epoch      # no retraining (same reason as D_FINAL_NOTE)
        curves.extend({"dataset": "Cross-domain", "key": key, **h} for h in m.history)
    d_csv = pd.read_csv(os.path.join(OUT, "D_cross_domain.csv")) if full else None
    dm = {k: (l, mk) for k, l, mk, _, _ in d_models()}
    cross_mse = {}
    for k in ("bias", "selected", "overfit"):
        ep = int(d_csv[(d_csv.bridge_frac == 1.0) & (d_csv.epoch_policy == "val_users") & (d_csv.key == k)].epochs.iloc[0]) if full else 2
        cross_mse[k] = timed_fit(dm[k][1](ep), *train, c["shape"])
    tu, ti, tr = vg_train
    entries = [("bpr", lab(best_mf), cross_bpr["bpr"].scores, "scores", ""),
               ("bpr_k0", lab(best_k0) + " [b_i only]", cross_bpr["bpr_k0"].scores, "scores", ""),
               ("pop", "Popularity (VG)", Popularity().fit(tu, ti, tr, c["shape"]).scores, "counts", "")]
    for k in ("bias", "selected", "overfit"):
        entries += [(k, dm[k][0], cross_mse[k].scores, "clipped", ""), (k, dm[k][0], cross_mse[k].raw_scores, "unclipped", "진단")]
    cross_rows, cross_recs = e_evaluate("Cross-domain (100% bridges)", entries, c["test_users"], no_hist, c["R_test"],
                                        np.bincount(train[1], minlength=c["shape"][1]), cross_mse["bias"].bi,
                                        candidates=c["vg_items"])
    if full:
        for r in cross_rows:
            if r["key"] in ("bias", "selected", "overfit"):
                want = d_csv[(d_csv.bridge_frac == 1.0) & (d_csv.epoch_policy == "val_users") & (d_csv.key == r["key"])
                             & (d_csv.ranking == r["ranking"])].hits.iloc[0]
                assert r["hits"] == want, f"E-3b vs D hits differ for {r['model']} {r['ranking']}"
    rows += cross_rows
    recs += cross_recs

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E_test.csv"), index=False)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "E_rec_popularity.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "E_learning_curves.csv"), index=False)
    plot_E_compare(rows, recs)
    return {"chosen": best_mf, "chosen_k0": best_k0, "positives_b": pos_b, "tuning": tuning,
            "vg_epochs": vg_epochs, "cross_epochs": cross_epochs, "test": rows, "rec_popularity": recs,
            "bootstrap": {"resamples": D_BOOT, "seed": SEED + 4, "unit": "user with at least one relevant test item",
                          "reference": "BPR (all positives)"},
            "final_model_note_cross": D_FINAL_NOTE}


# ---------------- Experiment E boundary extension (decision D23, rules in EXPERIMENT_LOG #36) ----------------
E_EXT_FACTORS, E_EXT_REGS, E_EXT_MAX_EPOCHS = (64, 128), (0.001, 0.01), 80
E_ARCHIVE = os.path.join(OUT, "archive", "E_run1_cap40")          # experiment E before the extension


def e_val_fn(users, known, answers, candidates=None):
    def f(m):
        res = evaluate(m.scores, users, known, answers, TOP_K, THRESHOLD, candidates=candidates)
        return int(round(res["precision"] * TOP_K * res["n_users"])), res["recall"]
    return f


def run_E_ext(sw_df=None, max_epochs=E_EXT_MAX_EPOCHS):
    """Stage 1: the 4 new cells on Software validation, and the final choice by the rule of #36."""
    old = pd.read_csv(os.path.join(E_ARCHIVE, "E_tuning.csv"))
    old_curves = pd.read_csv(os.path.join(E_ARCHIVE, "E_learning_curves.csv"))
    s = make_split(load_ratings("Software") if sw_df is None else sw_df)
    val_fn = e_val_fn(s["eval_users"], s["R_train"], s["R_val"])
    new, curves = [], []
    for k in E_EXT_FACTORS:
        for reg in E_EXT_REGS:
            m = timed_bpr(BPRMF(k, E_LR, reg, max_epochs, seed=SEED), *s["train"][:2], s["shape"], rated=s["train"][:2],
                          val_fn=val_fn)
            best = [h for h in m.history if h["epoch"] == m.best_epoch][0]
            new.append({"n_factors": k, "reg": reg, "lr": E_LR, "positives": "all", "tag": f"ext{max_epochs}",
                        "max_epochs": max_epochs, "best_epoch": m.best_epoch,
                        "epoch_cap": "상한 도달" if m.best_epoch == max_epochs else "", "val_hits": best["val_hits"],
                        "val_recall": best["val_recall"], "loss_first": m.history[0]["bpr_loss"],
                        "loss_best": best["bpr_loss"], "users_without_positive": int((~m.has_pos).sum()),
                        "fit_seconds": m.fit_seconds})
            curves += [{"n_factors": k, "reg": reg, "lr": E_LR, "positives": "all", "max_epochs": max_epochs, **h}
                       for h in m.history]
            log(f"[E_ext] {new[-1]}")
            # Determinism check: up to epoch 40 the 80-epoch run must repeat the 40-epoch run of experiment E.
            prev = old_curves[(old_curves.n_factors == k) & (old_curves["reg"] == reg) & (old_curves.lr == E_LR)
                              & (old_curves.positives == "all") & old_curves["dataset"].isna()]
            if len(prev) and sw_df is None:
                now = pd.DataFrame(m.history).set_index("epoch").loc[prev.epoch, "bpr_loss"].to_numpy()
                assert np.allclose(now, prev.bpr_loss.to_numpy(), rtol=0, atol=1e-12), "80-epoch run differs from E up to epoch 40"
    grid = old[old.tag == "grid"].assign(max_epochs=E_MAX_EPOCHS).to_dict("records")
    for r in grid:
        if r["n_factors"] == 64 and r["reg"] == 0.01:
            r["note"] = "40 에폭 상한 결과"
    pool = grid + new
    order = lambda r: (-r["val_hits"], -r["val_recall"], r["n_factors"], r["max_epochs"])     # rule 2 (+ last tie)
    chosen = sorted(pool, key=order)[0]
    previous = [r for r in grid if r["n_factors"] == 64 and r["reg"] == 0.01][0]
    same = chosen["max_epochs"] == previous["max_epochs"] and chosen["n_factors"] == previous["n_factors"] \
        and chosen["reg"] == previous["reg"]
    case = "A (final setting unchanged: no new test)" if same else "B (final setting changed: re-evaluate E-2, E-3a, E-3b)"
    log(f"[E_ext] chosen {chosen}; case {case}")
    table = pd.concat([old.assign(max_epochs=E_MAX_EPOCHS), pd.DataFrame(new)], ignore_index=True)
    table.loc[(table.tag == "grid") & (table.n_factors == 64) & (table["reg"] == 0.01), "note"] = "40 에폭 상한 결과"
    table.to_csv(os.path.join(OUT, "E_tuning.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "E_ext_learning_curves.csv"), index=False)
    plot_E_tuning(table.to_dict("records"))
    return {"rule": "EXPERIMENT_LOG #36", "new_cells": new, "chosen": chosen, "previous_final": previous,
            "case": case, "changed": not same}


def run_E_ext_eval(sw_df=None, vg_df=None, max_epochs=E_EXT_MAX_EPOCHS, report=None):
    """Stage 2 (case B only): the new final BPR on Software, VG and cross-domain (100% bridges).
    BPR k=0 is retrained with its recorded epochs (its per-user hits are needed for the bootstrap; hits are
    asserted equal to experiment E), Popularity is recounted (asserted), MSE rows are reused from experiment E."""
    report = load_results() if report is None else report
    ext, e_old = report["E"]["E_ext"], report["E"]["E"]
    if not ext["changed"]:
        raise SystemExit("[E_ext_eval] final setting unchanged (case A): no re-evaluation")
    ch, k0 = ext["chosen"], e_old["chosen_k0"]
    old_test = pd.read_csv(os.path.join(E_ARCHIVE, "E_test.csv"))
    old_recs = pd.read_csv(os.path.join(E_ARCHIVE, "E_rec_popularity.csv"))
    full = sw_df is None and vg_df is None
    lab = f"BPR (k={ch['n_factors']}, λ={ch['reg']}, lr={ch['lr']}, ≤{ch['max_epochs']} ep)"
    lab0 = f"BPR (k=0, λ={k0['reg']}, lr={k0['lr']}) [b_i only]"
    rows, recs, info = [], [], {}

    def bpr(k, reg, epochs, data, rated, val_fn=None):
        return timed_bpr(BPRMF(k, E_LR, reg, epochs, seed=SEED), *data, shape_now[0], rated=rated, val_fn=val_fn)

    def finish(name, entries, users, known, R_test, item_pop, b_ref, candidates=None):
        new_rows, new_recs = e_evaluate(name, entries, users, known, R_test, item_pop, b_ref, candidates=candidates)
        for r in new_rows:                                      # reproduction of the reused models
            if r["key"] in ("bpr_k0", "pop") and full:
                want = int(old_test[(old_test.dataset == name) & (old_test.key == r["key"])].hits.iloc[0])
                assert r["hits"] == want, f"{name} {r['key']}: {r['hits']} != {want}"
        bpr_hits = [r for r in new_rows if r["key"] == "bpr"][0]["hits"]
        reused = old_test[(old_test.dataset == name) & ~old_test.key.isin(["bpr", "bpr_k0", "pop"])].copy()
        reused["diff_hits_ref_minus_model"] = bpr_hits - reused["hits"]
        reused[["diff_ci_low", "diff_ci_high"]] = np.nan
        reused["note"] = reused["note"].fillna("").astype(str) + " | 재사용(E, 재학습 안 함), 차이 구간 없음"
        prev = old_test[(old_test.dataset == name) & (old_test.key == "bpr")].copy()
        prev["key"], prev["note"] = "bpr_cap40", "40 에폭 상한 결과 (이전 최종, 참고)"
        prev["model"] = prev["model"] + " [≤40 ep, previous final]"
        prev["diff_hits_ref_minus_model"] = bpr_hits - prev["hits"]           # new reference: the new final BPR
        prev[["diff_ci_low", "diff_ci_high"]] = np.nan                        # per-user hits of the old run not kept
        rows.extend(new_rows + reused.to_dict("records") + prev.to_dict("records"))
        reused_recs = old_recs[(old_recs.dataset == name) & ~old_recs.key.isin(["bpr", "bpr_k0", "pop"])].copy()
        prev_rec = old_recs[(old_recs.dataset == name) & (old_recs.key == "bpr")].copy()
        prev_rec["key"], prev_rec["note"] = "bpr_cap40", "40 에폭 상한 결과 (이전 최종, 참고)"
        prev_rec["model"] = prev_rec["model"] + " [≤40 ep, previous final]"
        recs.extend(new_recs + reused_recs.to_dict("records") + prev_rec.to_dict("records"))

    # Software: train+val refit for the chosen epoch
    s = make_split(load_ratings("Software") if sw_df is None else sw_df)
    shape_now = [s["shape"]]
    tv = s["trval"][:2]
    m = bpr(ch["n_factors"], ch["reg"], int(ch["best_epoch"]), tv, tv)
    m0 = bpr(0, k0["reg"], int(k0["best_epoch"]), tv, tv)
    bias_bi = FixedMF(0, B_LR, 0.05, [r for r in c_models() if r[0] == "bias"][0][3], seed=SEED).fit(*s["trval"], s["shape"]).bi
    info["Software"] = {"epochs": int(ch["best_epoch"]), "fit_seconds": m.fit_seconds}
    finish("Software", [("bpr", lab, m.scores, "scores", ""), ("bpr_k0", lab0, m0.scores, "scores", ""),
                        ("pop", "Popularity (train+val)", Popularity().fit(*s["trval"], s["shape"]).scores, "counts", "")],
           s["eval_users"], s["known"], s["R_test"], np.bincount(s["trval"][1], minlength=s["shape"][1]), bias_bi)

    # Video Games: epochs from VG validation (max 80), then train+val refit
    v = make_split(load_ratings("Video_Games") if vg_df is None else vg_df)
    shape_now[0] = v["shape"]
    m1 = bpr(ch["n_factors"], ch["reg"], max_epochs, v["train"][:2], v["train"][:2],
             e_val_fn(v["eval_users"], v["R_train"], v["R_val"]))
    ep_vg = m1.best_epoch
    m = bpr(ch["n_factors"], ch["reg"], ep_vg, v["trval"][:2], v["trval"][:2])
    m0 = bpr(0, k0["reg"], int(e_old["vg_epochs"]["bpr_k0"]) if full else 2, v["trval"][:2], v["trval"][:2])
    vg_bias_ep = report["C"]["C"]["vg_epochs"][[r for r in c_models() if r[0] == "bias"][0][1]] if full else 2
    bias_bi = FixedMF(0, B_LR, 0.05, vg_bias_ep, seed=SEED).fit(*v["trval"], v["shape"]).bi
    info["Video Games"] = {"epochs": ep_vg, "epoch_cap": "상한 도달" if ep_vg == max_epochs else "", "fit_seconds": m1.fit_seconds + m.fit_seconds}
    finish("Video Games", [("bpr", lab, m.scores, "scores", ""), ("bpr_k0", lab0, m0.scores, "scores", ""),
                           ("pop", "Popularity (train+val)", Popularity().fit(*v["trval"], v["shape"]).scores, "counts", "")],
           v["eval_users"], v["known"], v["R_test"], np.bincount(v["trval"][1], minlength=v["shape"][1]), bias_bi)

    # Cross-domain, 100% bridges: epochs from hidden-val users (max 80), no retraining (D_FINAL_NOTE)
    c = make_cross_split(load_ratings("Software") if sw_df is None else sw_df,
                         load_ratings("Video_Games") if vg_df is None else vg_df)
    shape_now[0] = c["shape"]
    train, vg_train, _, _ = cross_train(c, 1.0)
    no_hist = csr_matrix(c["shape"], dtype=np.float32)
    val_df = c["vg"][c["vg"]["user_id"].isin(c["hidden_val"])]
    val_users = np.sort(c["enc"].users.get_indexer(list(c["hidden_val"])))
    m = bpr(ch["n_factors"], ch["reg"], max_epochs, train[:2], train[:2],
            e_val_fn(val_users, no_hist, c["enc"].matrix(val_df), c["vg_items"]))
    m0 = bpr(0, k0["reg"], int(e_old["cross_epochs"]["bpr_k0"]) if full else 2, train[:2], train[:2])
    d = pd.read_csv(os.path.join(OUT, "D_cross_domain.csv")) if full else None
    bias_ep = int(d[(d.bridge_frac == 1.0) & (d.epoch_policy == "val_users") & (d.key == "bias")].epochs.iloc[0]) if full else 2
    bias_bi = FixedMF(0, B_LR, 0.05, bias_ep, seed=SEED).fit(*train, c["shape"]).bi
    info["Cross-domain (100% bridges)"] = {"epochs": m.best_epoch, "epoch_cap": "상한 도달" if m.best_epoch == max_epochs else "",
                                           "fit_seconds": m.fit_seconds}
    tu, ti, tr = vg_train
    finish("Cross-domain (100% bridges)",
           [("bpr", lab, m.scores, "scores", ""), ("bpr_k0", lab0, m0.scores, "scores", ""),
            ("pop", "Popularity (VG)", Popularity().fit(tu, ti, tr, c["shape"]).scores, "counts", "")],
           c["test_users"], no_hist, c["R_test"], np.bincount(train[1], minlength=c["shape"][1]), bias_bi,
           candidates=c["vg_items"])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E_test.csv"), index=False)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "E_rec_popularity.csv"), index=False)
    plot_E_compare(rows, recs)
    log(f"[E_ext_eval] epochs {info}")
    return {"chosen": ch, "epochs": info, "test": rows, "rec_popularity": recs,
            "note": "BPR k=0 retrained with recorded epochs (hits asserted), Popularity recounted (asserted), MSE rows reused"}


# ---------------- Experiment E_night: time-limited BPR search (decision D24, rules in EXPERIMENT_LOG #40) ----------------
E_NIGHT_CONFIGS = [  # (id, k, λ, lr), in the order of the rule; max 200 epochs each
    (1, 128, 0.01, 0.05), (2, 256, 0.01, 0.05), (3, 256, 0.003, 0.05), (4, 128, 0.01, 0.1),
    (5, 256, 0.01, 0.1), (6, 128, 0.03, 0.05), (7, 256, 0.03, 0.05), (8, 512, 0.01, 0.05)]
E_NIGHT_MAX_EPOCHS, E_NIGHT_EVAL_EVERY, E_NIGHT_PATIENCE = 200, 4, 20
E_NIGHT_SEC_PER_EPOCH_K128 = 3.3      # measured: E_ext k=128, 80 epochs in 260-272 s (incl. val every 2 epochs)
E_NIGHT_SW_TRAIN = 859_781            # Software train ratings of that measurement (time scales with data size)
E_TUNING_PATH = os.path.join(recsys.ROOT, "outputs", "mf", "E_tuning.csv")   # the 16 candidates of E + E_ext
KST = "Asia/Seoul"


def kst_now():
    # zoneinfo.ZoneInfo("Asia/Seoul"): the Korean time zone from the standard library, independent of the Mac's setting
    import datetime, zoneinfo
    return datetime.datetime.now(zoneinfo.ZoneInfo(KST))


def parse_kst(text):
    import datetime, zoneinfo
    # datetime.strptime(text, "%Y-%m-%d %H:%M"): parse "2026-11-05 06:30"; .timestamp() -> Unix time (seconds)
    return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=zoneinfo.ZoneInfo(KST)).timestamp()


def fmt_kst(ts):
    import datetime, zoneinfo
    return datetime.datetime.fromtimestamp(ts, zoneinfo.ZoneInfo(KST)).strftime("%Y-%m-%d %H:%M:%S")


def run_E_night(deadline=None, sw_df=None):
    """Time-limited BPR search on Software (rules: EXPERIMENT_LOG #40). Resumable: configurations already in
    E_night_tuning.csv are skipped. Ends with one test evaluation of the best of the E candidates + night cells."""
    if deadline is None:
        raise SystemExit('[E_night] --deadline "YYYY-MM-DD HH:MM" (Korean time) is required')
    deadline_ts = parse_kst(deadline)
    started = time.time()
    tuning_path, curves_path = os.path.join(OUT, "E_night_tuning.csv"), os.path.join(OUT, "E_night_curves.csv")
    s = make_split(load_ratings("Software") if sw_df is None else sw_df)
    n_train = len(s["train"][2])
    scale = n_train / E_NIGHT_SW_TRAIN
    sec_per_epoch = {k: E_NIGHT_SEC_PER_EPOCH_K128 * k / 128 * scale for k in (128, 256, 512)}   # updated by measurements
    overhead = 30 + 150 * scale                                     # final: data, BPR k=0, Popularity, test, bootstrap
    log(f"[E_night] start {fmt_kst(started)} KST, deadline {fmt_kst(deadline_ts)} KST, train ratings {n_train}")

    def est_fit(k, epochs):
        return epochs * sec_per_epoch[k]

    def est_final(k, epochs):                                       # train+val refit is ~1.17x the train data
        return epochs * sec_per_epoch.get(k, sec_per_epoch[128] * k / 128) * 1.17 + overhead

    done = pd.read_csv(tuning_path) if os.path.exists(tuning_path) else pd.DataFrame()
    done_ids = set(done["config_id"]) if len(done) else set()
    old = pd.read_csv(E_TUNING_PATH)
    old = old[old.tag.isin(["grid", "ext80"])].to_dict("records")  # the 16 candidates of E and E_ext
    order = lambda r: (-r["val_hits"], -r["val_recall"], r["n_factors"], r.get("seq", 0))
    for n, r in enumerate(old):
        r["seq"], r["source"] = n, "E" if r["tag"] == "grid" else "E_ext"
    val_fn = e_val_fn(s["eval_users"], s["R_train"], s["R_val"])
    skipped, ran = [], []
    for cid, k, reg, lr in E_NIGHT_CONFIGS:
        if cid in done_ids:
            log(f"[E_night] config {cid} already recorded: skipped (resume)")
            continue
        pool = old + (done.to_dict("records") if len(done) else [])
        best = sorted(pool, key=order)[0]
        reserve = max(est_final(int(best["n_factors"]), int(best["best_epoch"])), est_final(k, E_NIGHT_MAX_EPOCHS))
        now = time.time()
        need = est_fit(k, E_NIGHT_MAX_EPOCHS) + reserve
        if now + need > deadline_ts:
            reason = (f"would end at {fmt_kst(now + need)} KST > deadline (fit {est_fit(k, E_NIGHT_MAX_EPOCHS) / 60:.1f} min "
                      f"+ final reserve {reserve / 60:.1f} min)")
            skipped.append({"config_id": cid, "n_factors": k, "reg": reg, "lr": lr, "reason": reason})
            log(f"[E_night] config {cid} skipped: {reason}")
            continue
        stop_at = deadline_ts - reserve
        t0 = time.time()
        m = BPRMF(k, lr, reg, E_NIGHT_MAX_EPOCHS, eval_every=E_NIGHT_EVAL_EVERY, seed=SEED)
        m.fit(*s["train"][:2], s["shape"], rated=s["train"][:2], val_fn=val_fn, stop_patience=E_NIGHT_PATIENCE, stop_at=stop_at)
        sec = time.time() - t0
        best_h = [h for h in m.history if h["epoch"] == m.best_epoch][0]
        row = {"config_id": cid, "n_factors": k, "reg": reg, "lr": lr, "positives": "all", "tag": "night",
               "max_epochs": E_NIGHT_MAX_EPOCHS, "epochs_run": len(m.history), "best_epoch": m.best_epoch,
               "epoch_cap": "상한 도달" if m.best_epoch == E_NIGHT_MAX_EPOCHS else "", "stopped_by": m.stopped_by,
               "val_hits": best_h["val_hits"], "val_recall": best_h["val_recall"], "fit_seconds": sec,
               "sec_per_epoch": sec / len(m.history), "started_kst": fmt_kst(t0), "ended_kst": fmt_kst(time.time()),
               "seq": 100 + cid, "source": "E_night"}
        sec_per_epoch[k] = row["sec_per_epoch"]                     # measured speed replaces the estimate
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(tuning_path, index=False)                      # written after every configuration
        pd.DataFrame([{"config_id": cid, "n_factors": k, "reg": reg, "lr": lr, **h} for h in m.history]).to_csv(
            curves_path, mode="a", header=not os.path.exists(curves_path), index=False)
        ran.append(cid)
        log(f"[E_night] config {cid} done: {row}")
        del m
        if m_deadline_passed := (time.time() >= stop_at):
            log("[E_night] training deadline reached: no further configurations")
            for cid2, k2, reg2, lr2 in E_NIGHT_CONFIGS:
                if cid2 > cid and cid2 not in done_ids:
                    skipped.append({"config_id": cid2, "n_factors": k2, "reg": reg2, "lr": lr2,
                                    "reason": "training deadline (deadline − final reserve) reached"})
            break

    # ---------------- final: best of E candidates + night cells, one test evaluation ----------------
    pool = old + (done.to_dict("records") if len(done) else [])
    chosen = sorted(pool, key=order)[0]
    ck, creg, clr, cep = int(chosen["n_factors"]), float(chosen["reg"]), float(chosen["lr"]), int(chosen["best_epoch"])
    log(f"[E_night] final choice {chosen} at {fmt_kst(time.time())} KST")
    tv = s["trval"][:2]
    t0 = time.time()
    final = timed_bpr(BPRMF(ck, clr, creg, cep, seed=SEED), *tv, s["shape"], rated=tv)
    k0 = load_results()["E"]["E"]["chosen_k0"]
    m0 = timed_bpr(BPRMF(0, k0["lr"], k0["reg"], int(k0["best_epoch"]), seed=SEED), *tv, s["shape"], rated=tv)
    bias_bi = FixedMF(0, B_LR, 0.05, [r for r in c_models() if r[0] == "bias"][0][3], seed=SEED).fit(*s["trval"], s["shape"]).bi
    lab = f"BPR (k={ck}, λ={creg}, lr={clr}, {cep} ep) [{chosen.get('source', '')}]"
    rows, recs = e_evaluate("Software", [
        ("bpr", lab, final.scores, "scores", "E_night 최종"),
        ("bpr_k0", f"BPR (k=0, λ={k0['reg']}, lr={k0['lr']}) [b_i only]", m0.scores, "scores", ""),
        ("pop", "Popularity (train+val)", Popularity().fit(*s["trval"], s["shape"]).scores, "counts", "")],
        s["eval_users"], s["known"], s["R_test"], np.bincount(s["trval"][1], minlength=s["shape"][1]), bias_bi)
    if sw_df is None:            # reproduction of the reference models of experiment E
        e_test = pd.read_csv(os.path.join(E_ARCHIVE, "E_test.csv"))
        for r in rows:
            if r["key"] in ("bpr_k0", "pop"):
                want = int(e_test[(e_test.dataset == "Software") & (e_test.key == r["key"])].hits.iloc[0])
                assert r["hits"] == want, f"E_night reference {r['key']}: {r['hits']} != {want}"
    final_sec = time.time() - t0
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E_night_test.csv"), index=False)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "E_night_rec_popularity.csv"), index=False)
    ended = time.time()
    write_E_night_summary(deadline_ts, started, ended, done, skipped, ran, chosen, rows, recs, final_sec)
    log(f"[E_night] finished at {fmt_kst(ended)} KST (deadline {fmt_kst(deadline_ts)})")
    return {"deadline_kst": fmt_kst(deadline_ts), "started_kst": fmt_kst(started), "ended_kst": fmt_kst(ended),
            "ran_this_time": ran, "skipped": skipped, "night_cells": done.to_dict("records") if len(done) else [],
            "chosen": chosen, "test": rows, "rec_popularity": recs, "final_seconds": final_sec, "rule": "EXPERIMENT_LOG #40"}


def write_E_night_summary(deadline_ts, started, ended, done, skipped, ran, chosen, rows, recs, final_sec):
    lines = ["# E_night: 시간 제한 BPR 탐색 결과 요약", "",
             f"- 규칙: EXPERIMENT_LOG #40 (결정 D24). 데이터: Software, 긍정 정의 (a). 선택은 val 적중만, test는 최종 1회",
             f"- 시작 {fmt_kst(started)} KST, 종료 {fmt_kst(ended)} KST, deadline {fmt_kst(deadline_ts)} KST",
             f"- 이번 실행에서 돈 설정: {', '.join(str(c) for c in ran) if ran else '없음'}", "",
             "## 밤 탐색 설정 (val, 평가 사용자 10,000명)", "",
             "| # | k | λ | lr | val 적중 | Recall@10 | best_epoch | 실행 에폭 | 종료 이유 | 소요 (분) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in (done.sort_values("config_id").to_dict("records") if len(done) else []):
        cap = " (상한 도달)" if r.get("epoch_cap") == "상한 도달" else ""
        lines.append(f"| {r['config_id']} | {r['n_factors']} | {r['reg']} | {r['lr']} | {r['val_hits']} | {r['val_recall']:.4f} | "
                     f"{r['best_epoch']}{cap} | {r['epochs_run']} | {r['stopped_by']} | {r['fit_seconds'] / 60:.1f} |")
    lines += ["", "## 건너뛴 설정", ""]
    lines += [f"- #{x['config_id']} (k={x['n_factors']}, λ={x['reg']}, lr={x['lr']}): {x['reason']}" for x in skipped] or ["- 없음"]
    lines += ["", "## 최종 설정과 선택 이유", "",
              f"- 후보: 기존 E 16칸(격자 12 + 확장 4) + 밤 탐색 칸. 규칙: val 적중 최고 → 동률이면 Recall@10 → 작은 k → 먼저 기록된 칸",
              f"- **최종: k={chosen['n_factors']}, λ={chosen['reg']}, lr={chosen['lr']}, best_epoch {chosen['best_epoch']} "
              f"(출처 {chosen.get('source', '')}), val 적중 {chosen['val_hits']}, Recall@10 {chosen['val_recall']:.4f}**",
              "", "## Software test (1회, train+val 재학습, 부트스트랩 1,000회)", "",
              "| 모델 | 적중 [95%] | F1@10 [95%] | Recall@10 | BPR − 모델 [95%] | distinct_items | 평균 인기도 |",
              "|---|---|---|---|---|---|---|"]
    for r, c in zip(rows, recs):
        lines.append(f"| {r['model']} | {r['hits']} [{r['hits_ci_low']:.0f}, {r['hits_ci_high']:.0f}] | {r['f1']:.4f} "
                     f"[{r['f1_ci_low']:.4f}, {r['f1_ci_high']:.4f}] | {r['recall']:.4f} | {r['diff_hits_ref_minus_model']:+d} "
                     f"[{r['diff_ci_low']:.0f}, {r['diff_ci_high']:.0f}] | {c['distinct_items']} | {c['mean_pop']:.0f} |")
    lines += ["", f"- 최종 평가 소요: {final_sec / 60:.1f}분. 참고 기준(BPR k=0, Popularity)의 적중은 실험 E와 같은지 assert로 확인함(전체 데이터일 때)",
              "- VG·교차 도메인 재평가와 CHECKLIST 갱신은 하지 않음 (사용자 지시)"]
    with open(os.path.join(OUT, "E_night_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------- Experiment E_final: E-3a / E-3b with the final BPR of E_night (decision D26) ----------------
E_FINAL_WINDOW = 40          # D26 (3): val-hit range over the last 40 epochs of a night run (epoch >= epochs_run - 40)


def write_E_night_top_configs(n_top=5):
    """D26 (3), no training: the n_top best cells (E 16 + night 8, rule of #40) with their val hits and the range of
    val hits over the last E_FINAL_WINDOW epochs of their run (night cells only; E/E_ext runs were 40-80 epochs)."""
    night = pd.read_csv(os.path.join(OUT, "E_night_tuning.csv"))
    curves = pd.read_csv(os.path.join(OUT, "E_night_curves.csv"))
    old = pd.read_csv(os.path.join(OUT, "E_tuning.csv"))
    old = old[old.tag.isin(["grid", "ext80"])].assign(source=lambda d: np.where(d.tag == "grid", "E", "E_ext"))
    old["seq"] = np.arange(len(old))
    pool = pd.concat([old, night], ignore_index=True)
    pool = pool.sort_values(["val_hits", "val_recall", "n_factors", "seq"], ascending=[False, False, True, True])
    rows = []
    for rank, r in enumerate(pool.head(n_top).to_dict("records"), 1):
        row = {"rank": rank, "source": r["source"], "config_id": r.get("config_id"), "n_factors": int(r["n_factors"]),
               "reg": r["reg"], "lr": r["lr"], "best_epoch": int(r["best_epoch"]), "val_hits": int(r["val_hits"]),
               "val_recall": r["val_recall"]}
        if r["source"] == "E_night":
            c = curves[(curves.config_id == r["config_id"]) & curves.val_hits.notna()]
            start = int(r["epochs_run"]) - E_FINAL_WINDOW
            w = c[c.epoch >= start]
            row.update({"window_epochs": f"{start}-{int(r['epochs_run'])}", "window_n_evals": len(w),
                        "window_val_hits_min": int(w.val_hits.min()), "window_val_hits_max": int(w.val_hits.max()),
                        "window_val_hits_mean": float(w.val_hits.mean())})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E_night_top_configs.csv"), index=False)
    return rows


def run_E_final(sw_df=None, vg_df=None, max_epochs=E_NIGHT_MAX_EPOCHS, eval_every=E_NIGHT_EVAL_EVERY,
                patience=E_NIGHT_PATIENCE):
    """E-3a (Video Games) and E-3b (cross-domain, 100% bridges) for the final BPR chosen by E_night (D26 (1)).
    Epochs: validation hits every eval_every epochs, stop after patience epochs without improvement, max max_epochs.
    VG: refit on train+val with the chosen epoch, test once. Cross-domain: no refit (D_FINAL_NOTE).
    BPR k=0 is retrained with its recorded epochs and Popularity recounted (hits asserted equal to the previous
    E_test.csv), so the bootstrap intervals BPR − Popularity and BPR − k=0 can be computed.
    Software is not evaluated again: its rows come from E_night_test.csv (the single test of E_night).
    The previous final BPRs (k=128 ≤80 ep and k=64 ≤40 ep) and the MSE rows are reused as reference rows."""
    report = load_results()
    full = sw_df is None and vg_df is None
    ch, k0, e_old = report["E"]["E_night"]["chosen"], report["E"]["E"]["chosen_k0"], report["E"]["E"]
    ck, creg, clr = int(ch["n_factors"]), float(ch["reg"]), float(ch["lr"])
    lab = f"BPR (k={ck}, λ={creg}, lr={clr}) [E_night final]"
    lab0 = f"BPR (k=0, λ={k0['reg']}, lr={k0['lr']}) [b_i only]"

    # Keep the tables of the k=128 (≤80 ep) evaluation; a rerun reads them from the archive and never overwrites it.
    archive = os.path.join(OUT, "archive", "E_run2_ext80")
    if not os.path.exists(os.path.join(archive, "E_test.csv")):
        os.makedirs(archive, exist_ok=True)
        for f in ("E_test.csv", "E_rec_popularity.csv", os.path.join("figures", "mf_bpr_compare.png")):
            if os.path.exists(os.path.join(OUT, f)):
                shutil.copy2(os.path.join(OUT, f), os.path.join(archive, os.path.basename(f)))
    old_test = pd.read_csv(os.path.join(archive, "E_test.csv"))
    old_recs = pd.read_csv(os.path.join(archive, "E_rec_popularity.csv"))
    rows, recs, curves, info = [], [], [], {}

    def reused(name, bpr_hits, new_keys):
        """Rows of the archived table that are not recomputed: the previous final BPR becomes 'bpr_ext80'."""
        t = old_test[(old_test.dataset == name) & ~old_test.key.isin(new_keys)].copy()
        rc = old_recs[(old_recs.dataset == name) & ~old_recs.key.isin(new_keys)].copy()
        for d in (t, rc):
            prev = d.key == "bpr"
            d.loc[prev, "model"] = d.loc[prev, "model"] + " [previous final]"
            d.loc[prev, "key"] = "bpr_ext80"
            d["note"] = d["note"].fillna("").astype(str)
            d.loc[d.key.isin(["bpr_ext80", "bpr_cap40"]), "note"] = "이전 최종 (참고)"
            d["note"] = d["note"].where(d["note"].str.contains("재사용"), d["note"] + " | 재사용, 차이 구간 없음")
        t["diff_hits_ref_minus_model"] = bpr_hits - t["hits"]                 # reference: the new final BPR
        t[["diff_ci_low", "diff_ci_high"]] = np.nan                            # per-user hits of old runs not kept
        return t.to_dict("records"), rc.to_dict("records")

    def check_refs(name, new_rows):
        if not full:
            return
        for r in new_rows:
            if r["key"] in ("bpr_k0", "pop"):
                want = int(old_test[(old_test.dataset == name) & (old_test.key == r["key"])].hits.iloc[0])
                assert r["hits"] == want, f"E_final {name} {r['key']}: {r['hits']} != {want}"

    def choose_epoch(name, data, shape, val_fn):
        m = BPRMF(ck, clr, creg, max_epochs, eval_every=eval_every, seed=SEED)
        timed_bpr(m, *data, shape, rated=data, val_fn=val_fn, stop_patience=patience)
        curves.extend({"dataset": name, **h} for h in m.history)
        info[name] = {"best_epoch": m.best_epoch, "epochs_run": len(m.history), "stopped_by": m.stopped_by,
                      "epoch_cap": "상한 도달" if m.best_epoch == max_epochs else "", "select_seconds": m.fit_seconds}
        log(f"[E_final] {name}: {info[name]}")
        return m

    # ---------------- Software: rows of the single E_night test ----------------
    sw_new = pd.read_csv(os.path.join(OUT, "E_night_test.csv")).to_dict("records")
    sw_new_recs = pd.read_csv(os.path.join(OUT, "E_night_rec_popularity.csv")).to_dict("records")
    sw_hits = [r for r in sw_new if r["key"] == "bpr"][0]["hits"]
    t, rc = reused("Software", sw_hits, ["bpr_k0", "pop"])
    rows += sw_new + t
    recs += sw_new_recs + rc

    # ---------------- E-3a: Video Games ----------------
    v = make_split(load_ratings("Video_Games") if vg_df is None else vg_df)
    m1 = choose_epoch("Video Games", v["train"][:2], v["shape"], e_val_fn(v["eval_users"], v["R_train"], v["R_val"]))
    ep_vg = m1.best_epoch
    del m1
    tv = v["trval"][:2]
    m = timed_bpr(BPRMF(ck, clr, creg, ep_vg, seed=SEED), *tv, v["shape"], rated=tv)
    info["Video Games"]["refit_seconds"] = m.fit_seconds
    m0 = timed_bpr(BPRMF(0, k0["lr"], k0["reg"], int(e_old["vg_epochs"]["bpr_k0"]) if full else 2, seed=SEED),
                   *tv, v["shape"], rated=tv)
    vg_bias_ep = report["C"]["C"]["vg_epochs"][[r for r in c_models() if r[0] == "bias"][0][1]] if full else 2
    bias_bi = FixedMF(0, B_LR, 0.05, vg_bias_ep, seed=SEED).fit(*v["trval"], v["shape"]).bi
    new_rows, new_recs = e_evaluate(
        "Video Games", [("bpr", lab.replace("]", f", {ep_vg} ep]"), m.scores, "scores", "E_final"),
                        ("bpr_k0", lab0, m0.scores, "scores", ""),
                        ("pop", "Popularity (train+val)", Popularity().fit(*v["trval"], v["shape"]).scores, "counts", "")],
        v["eval_users"], v["known"], v["R_test"], np.bincount(v["trval"][1], minlength=v["shape"][1]), bias_bi)
    check_refs("Video Games", new_rows)
    t, rc = reused("Video Games", new_rows[0]["hits"], ["bpr_k0", "pop"])
    rows += new_rows + t
    recs += new_recs + rc
    del m, m0, v

    # ---------------- E-3b: cross-domain, 100% bridges (no refit) ----------------
    c = make_cross_split(load_ratings("Software") if sw_df is None else sw_df,
                         load_ratings("Video_Games") if vg_df is None else vg_df)
    train, vg_train, _, _ = cross_train(c, 1.0)
    no_hist = csr_matrix(c["shape"], dtype=np.float32)
    val_df = c["vg"][c["vg"]["user_id"].isin(c["hidden_val"])]
    val_users = np.sort(c["enc"].users.get_indexer(list(c["hidden_val"])))
    name = "Cross-domain (100% bridges)"
    m = choose_epoch(name, train[:2], c["shape"], e_val_fn(val_users, no_hist, c["enc"].matrix(val_df), c["vg_items"]))
    m0 = timed_bpr(BPRMF(0, k0["lr"], k0["reg"], int(e_old["cross_epochs"]["bpr_k0"]) if full else 2, seed=SEED),
                   *train[:2], c["shape"], rated=train[:2])
    d = pd.read_csv(os.path.join(OUT, "D_cross_domain.csv")) if full else None
    bias_ep = int(d[(d.bridge_frac == 1.0) & (d.epoch_policy == "val_users") & (d.key == "bias")].epochs.iloc[0]) if full else 2
    bias_bi = FixedMF(0, B_LR, 0.05, bias_ep, seed=SEED).fit(*train, c["shape"]).bi
    tu, ti, tr = vg_train
    new_rows, new_recs = e_evaluate(
        name, [("bpr", lab.replace("]", f", {m.best_epoch} ep]"), m.scores, "scores", "E_final"),
               ("bpr_k0", lab0, m0.scores, "scores", ""),
               ("pop", "Popularity (VG)", Popularity().fit(tu, ti, tr, c["shape"]).scores, "counts", "")],
        c["test_users"], no_hist, c["R_test"], np.bincount(train[1], minlength=c["shape"][1]), bias_bi,
        candidates=c["vg_items"])
    check_refs(name, new_rows)
    t, rc = reused(name, new_rows[0]["hits"], ["bpr_k0", "pop"])
    rows += new_rows + t
    recs += new_recs + rc

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E_test.csv"), index=False)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "E_rec_popularity.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "E_final_curves.csv"), index=False)
    plot_E_compare(rows, recs)
    top = write_E_night_top_configs()
    return {"chosen": ch, "epochs": info, "archive": os.path.relpath(archive, OUT), "test": rows, "rec_popularity": recs,
            "top_configs": top, "rule": "decision D26",
            "note": "Software rows from E_night_test.csv; BPR k=0 retrained, Popularity recounted (hits asserted); "
                    "MSE and previous-final rows reused without intervals"}


def plot_E_tuning(tuning):
    """Validation hits by λ; solid = grid with at most 40 epochs, dashed + hollow = extension with at most 80 epochs."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = {reg: n for n, reg in enumerate(E_REGS)}
    colors = {0: "#8a8984", 16: "#86b6ef", 32: "#3987e5", 64: "#1c5cab", 128: "#0d366b"}
    for tag, style, mfc in [("grid", "-o", None), ("ext", "--s", "none")]:
        rows = [r for r in tuning if str(r["tag"]).startswith(tag)]
        for k in sorted({r["n_factors"] for r in rows}):
            rr = sorted([r for r in rows if r["n_factors"] == k], key=lambda r: r["reg"])
            cap = rr[0].get("max_epochs", E_MAX_EPOCHS)
            cap = E_MAX_EPOCHS if cap != cap else int(cap)                       # NaN -> 40 (old rows)
            ax.plot([x[r["reg"]] for r in rr], [r["val_hits"] for r in rr], style, color=colors[k], lw=2, ms=7,
                    mfc=mfc if mfc else colors[k], label=f"k = {k}" + (" (b_i only)" if k == 0 else "") + f", ≤{cap} epochs")
    ax.set_xticks(range(len(E_REGS)), [str(v) for v in E_REGS])
    ax.set(xlabel="λ (L2)", ylabel="Validation hits in top-10", title=f"E: BPR tuning on Software validation (lr {E_LR})")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_bpr_tuning.png"), dpi=150)
    plt.close(fig)


def plot_E_compare(rows, recs):
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    color = {"bpr": "#6250d6", "bpr_cap40": "#b9b1f0", "bpr_ext80": "#9e94e6", "bpr_k0": "#e87ba4", "bpr_pos4": "#9085e9", "pop": "#8a8984", "bias": C_BIAS,
             "selected": C_MF, "overfit": C_COLORS["overfit"]}
    fig, axes = plt.subplots(len(datasets), 2, figsize=(13, 4.2 * len(datasets)), squeeze=False)
    for (left, right), name in zip(axes, datasets):
        rr = [r for r in rows if r["dataset"] == name]
        y = np.arange(len(rr))[::-1]
        names = [r["model"] + (" [unclipped]" if r["ranking"] == "unclipped" else "") for r in rr]
        cols = [color[r["key"]] for r in rr]
        h = np.array([r["hits"] for r in rr])
        err = np.array([h - [r["hits_ci_low"] for r in rr], [r["hits_ci_high"] for r in rr] - h])
        left.barh(y, h, color=cols, height=0.7)
        left.errorbar(h, y, xerr=err, fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
        left.set_yticks(y, names, fontsize=7.5)
        left.set(xlabel="Test hits in top-10 (bar), 95% bootstrap interval", title=f"E-2 {name}: top-10 hits")
        rc = [x for x in recs if x["dataset"] == name]
        right.scatter([x["mean_pop"] for x in rc], y, c=cols, s=55, zorder=3)
        right.scatter([x["median_pop"] for x in rc], y, facecolors="none", edgecolors=cols, s=55, lw=1.5, zorder=3)
        right.set_xscale("log")
        right.set_yticks(y, [""] * len(rr))
        right.set(xlabel="training ratings of recommended items (filled = mean, hollow = median; log)",
                  title=f"E-2 {name}: popularity of the top-10")
        for a in (left, right):
            a.grid(axis="x", color="#e5e4e0", lw=0.8)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mf_bpr_compare.png"), dpi=150)
    plt.close(fig)


# ==================================================================================================
# Main
# ==================================================================================================
def main():
    # argparse: command-line argument parser. add_argument("--exp", choices=[...], required=True)
    # defines the option --exp, which must be one of the listed values.
    parser = argparse.ArgumentParser(description="MF experiments (experiments A-E of MF_PLAN.md)")
    parser.add_argument("--exp", choices=["A1", "A2", "A3", "A", "B1", "B1x", "B2", "B", "C", "D", "D5", "E1", "E", "E_ext",
                                          "E_ext_eval", "E_night", "E_final"], required=True, help="experiment to run")
    # --deadline "YYYY-MM-DD HH:MM" in Korean time; used only by --exp E_night (EXPERIMENT_LOG #40)
    parser.add_argument("--deadline", default=None, help='E_night only: stop time in Korean time, e.g. "2026-11-05 06:30"')
    args = parser.parse_args()
    os.makedirs(FIG, exist_ok=True)
    runs = {"A1": run_A1, "A2": run_A2, "A3": run_A3, "B1": run_B1, "B1x": run_B1x, "B2": run_B2, "C": run_C, "D": run_D, "D5": run_D5,
            "E1": run_E1, "E": run_E, "E_ext": run_E_ext, "E_ext_eval": run_E_ext_eval,
            "E_night": lambda: run_E_night(deadline=args.deadline), "E_final": run_E_final}
    order = {"A": ["A1", "A3", "A2"], "B": ["B1", "B1x", "B2"]}.get(args.exp, [args.exp])
    for name in order:
        start = time.perf_counter()
        result = runs[name]()
        result["runtime_seconds"] = time.perf_counter() - start
        group = name[0]                                    # "A" or "B": top-level key in results.json
        report = load_results().get(group, {})
        report[name] = result
        save_results(group, report)
        log(f"[{name}] done in {result['runtime_seconds']:.1f}s -> {RESULTS}")


if __name__ == "__main__":
    main()
