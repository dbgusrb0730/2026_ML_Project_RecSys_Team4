"""Train and compare memory-based CF, matrix factorization, and their hybrid."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import recsys as base
from evaluation import MetricAccumulator
from hybrid import Hybrid
from memory import ImprovedUserKNN
from mf import SplitRegMF
from splits import cross_train, make_cross_split, make_split


ROOT = Path(base.ROOT)
MF_CONFIG = {"n_factors": 16, "lr": 0.01, "reg_bias": 0.05, "reg_factor": 0.5}
ALPHAS = tuple(i / 10 for i in range(11))
CHUNK = 32


def fit_mf(train, shape, epochs, val=None):
    model = SplitRegMF(**MF_CONFIG, n_epochs=epochs, patience=epochs, seed=base.SEED)
    return model.fit(*train, shape, val=val)


def select_alpha(mf, knn, users, validation):
    """Choose alpha by RMSE on observed validation ratings, in bounded batches."""
    blends = [Hybrid(mf, knn, alpha) for alpha in ALPHAS]
    squared_errors = np.zeros(len(blends), dtype=np.float64)
    n_ratings = 0
    for start in range(0, len(users), CHUNK):
        batch = users[start:start + CHUNK]
        observed = validation[batch].tocoo()
        if not observed.nnz:
            continue
        index = observed.row, observed.col
        mf_ratings = mf.scores(batch)[index]
        knn_ratings = knn.scores(batch)[index]
        for i, blend in enumerate(blends):
            pred = blend.scores(batch, knn_scores=knn_ratings, mf_scores=mf_ratings)
            error = pred.astype(np.float64) - observed.data
            squared_errors[i] += error @ error
        n_ratings += observed.nnz
    if not n_ratings:
        raise ValueError("Alpha selection requires at least one held-out validation rating")
    errors = np.sqrt(squared_errors / n_ratings)
    grid = [{"alpha_mf": alpha, "validation_rmse": float(error)}
            for alpha, error in zip(ALPHAS, errors)]
    return ALPHAS[int(np.argmin(errors))], grid


def compare(mf, knn, alpha, baseline_train, users, known, test, candidates=None):
    """Compare components, their blend and baselines on exactly the same test data."""
    blend = Hybrid(mf, knn, alpha)
    mean = base.GlobalMean().fit(*baseline_train, known.shape)
    popularity = base.Popularity().fit(*baseline_train, known.shape)
    names = ("Memory kNN (rating)", "MF (MSE)", "Hybrid", "Global mean", "Popularity")
    metrics = {name: MetricAccumulator(base.TOP_K, base.THRESHOLD) for name in names}
    for start in range(0, len(users), CHUNK):
        batch = users[start:start + CHUNK]
        mf_ratings = mf.scores(batch)
        knn_ratings = knn.scores(batch)
        hybrid = blend.scores(batch, knn_scores=knn_ratings, mf_scores=mf_ratings)
        scores = (knn_ratings, mf_ratings, hybrid, mean.scores(batch), popularity.scores(batch))
        for name, values in zip(names, scores):
            rating_scores = values if name != "Popularity" else None
            metrics[name].add(values, batch, known, test, candidates, rating_scores)
    return {name: accumulator.result() for name, accumulator in metrics.items()}


def in_domain(df, knn_config, max_epochs, eval_users, alpha=None):
    split = make_split(df)
    users = np.sort(np.random.default_rng(base.SEED).choice(
        split["eval_users"], min(eval_users, len(split["eval_users"])), replace=False))
    shape = split["shape"]
    mf = fit_mf(split["train"], shape, max_epochs, val=split["val"])
    best_epoch = mf.best_epoch
    selection = {"source": "Software validation"}
    if alpha is None:
        validation_users = np.sort(np.random.default_rng(142).choice(
            users, min(2000, len(users)), replace=False))
        knn = ImprovedUserKNN(**knn_config).fit(*split["train"], shape)
        alpha, grid = select_alpha(mf, knn, validation_users, split["R_val"])
        selection.update(grid=grid, n_validation_users=len(validation_users))
    final_train = split["trval"]
    mf = fit_mf(final_train, shape, best_epoch)
    knn = ImprovedUserKNN(**knn_config).fit(*final_train, shape)
    return {"alpha_mf": alpha, "alpha_selection": selection, "mf_epochs": best_epoch,
            "split": split["sizes"], "final_train_ratings": len(final_train[2]),
            "metrics": compare(mf, knn, alpha, final_train, users, split["known"], split["R_test"])}


def cross_domain(software, games, knn_config, max_epochs):
    split = make_cross_split(software, games)
    if not split["hidden_val"] or not split["hidden_test"]:
        raise ValueError("Cross-domain evaluation requires both validation and test users")
    train, game_train, _, info = cross_train(split, frac=1.0)
    shape, enc = split["shape"], split["enc"]
    eligible_neighbors = np.bincount(game_train[0], minlength=shape[0]) > 0
    knn = ImprovedUserKNN(**knn_config).fit(
        *train, shape, similarity_data=enc.arrays(software), neighbor_mask=eligible_neighbors)
    mf = fit_mf(train, shape, max_epochs, val=split["val"])
    u, i, r = split["val"]
    validation = csr_matrix((r, (u, i)), shape=shape)
    alpha, grid = select_alpha(mf, knn, np.unique(u), validation)
    # Keep validation users' game ratings hidden when moving from validation to test.
    metrics = compare(mf, knn, alpha, game_train, split["test_users"],
                      csr_matrix(shape, dtype=np.float32), split["R_test"], split["vg_items"])
    return {"alpha_mf": alpha, "mf_epochs": mf.best_epoch,
            "alpha_selection": {"source": "Cross-domain hidden validation users", "grid": grid},
            "split": {"shared_users": len(split["shared"]), "validation_users": len(split["hidden_val"]),
                      "test_users": len(split["hidden_test"]), **info}, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "hybrid")
    parser.add_argument("--eval-users", type=int, default=base.N_EVAL_USERS,
                        help="Cap evaluation users for each in-domain dataset")
    parser.add_argument("--max-epochs", type=int, default=base.MAX_EPOCHS,
                        help="MF epoch search limit, including for cross-domain validation")
    args = parser.parse_args()
    if not 1 <= args.eval_users <= base.N_EVAL_USERS or args.max_epochs < 1:
        parser.error(f"Require 1 <= --eval-users <= {base.N_EVAL_USERS} and --max-epochs >= 1")

    memory_selection = ROOT / "outputs" / "memory_reviewed" / "validation_selection.json"
    knn_config = json.loads(memory_selection.read_text(encoding="utf-8"))["config"]
    software = pd.read_csv(args.data_dir / "Software.csv")
    games = pd.read_csv(args.data_dir / "Video_Games.csv")
    base.log("Software: training and evaluating kNN, MF, and hybrid")
    sw = in_domain(software, knn_config, args.max_epochs, args.eval_users)
    base.log("Video Games: training and evaluating kNN, MF, and hybrid")
    vg = in_domain(games, knn_config, args.max_epochs, args.eval_users, alpha=sw["alpha_mf"])
    base.log("Cross-domain: training and evaluating kNN, MF, and hybrid")
    cross = cross_domain(software, games, knn_config, args.max_epochs)
    results = {"Software": sw, "Video Games": vg, "Cross-domain": cross}
    report = {"formula": "alpha * MF rating + (1 - alpha) * kNN rating",
              "selection_metric": "validation RMSE", "ranking_metric": "F1@10 over all evaluated users",
              "seed": base.SEED, "mf_config": MF_CONFIG, "knn_config": knn_config,
              "max_epochs": args.max_epochs, "in_domain_eval_users": args.eval_users, "results": results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for domain, experiment in results.items():
        for name, scores in experiment["metrics"].items():
            base.log(domain, name, "RMSE", scores["rmse"], "F1@10", scores["all_users"]["f1"])
        base.log(domain, "alpha (MF weight)", experiment["alpha_mf"])
    base.log("Saved", output)


if __name__ == "__main__":
    main()
