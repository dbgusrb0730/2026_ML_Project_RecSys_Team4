"""Train and compare memory-based CF, matrix factorization, and their hybrids (rating + ranking)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import recsys as base
from evaluation import MetricAccumulator
from hybrid import Hybrid, RankingHybrid, zscore_rows
from memory import ImprovedUserKNN
from mf import BPRMF, FixedMF, SplitRegMF
from splits import cross_train, make_cross_split, make_split


ROOT = Path(base.ROOT)
MF_CONFIG = {"n_factors": 16, "lr": 0.01, "reg_bias": 0.05, "reg_factor": 0.5}
BPR_CONFIGS = {
    "fast": {"n_factors": 64, "lr": 0.05, "reg": 0.01, "epochs": (40, 40, 16)},
    "full": {"n_factors": 512, "lr": 0.05, "reg": 0.01, "epochs": (188, 184, 16)},
}
BPR_K0_CONFIG = {"n_factors": 0, "lr": 0.05, "reg": 0.001, "epochs": (10, 22, 20)}
ALPHAS = tuple(i / 10 for i in range(11))
CHUNK = 32


def fit_mf(train, shape, epochs, val=None):
    model = SplitRegMF(**MF_CONFIG, n_epochs=epochs, patience=epochs, seed=base.SEED)
    return model.fit(*train, shape, val=val)


def fit_bias_only(train, shape, epochs, val=None):
    model = FixedMF(n_factors=0, lr=0.01, reg=0.05, n_epochs=epochs, patience=epochs, seed=base.SEED)
    return model.fit(*train, shape, val=val)


def fit_bpr(train, shape, config, epochs, label="BPR"):
    u, i, _ = train
    n_factors = config["n_factors"]
    base.log(f"  [{label}] start training: k={n_factors}, epochs={epochs}, ratings={len(u):,}")
    model = BPRMF(n_factors=n_factors, lr=config["lr"], reg=config["reg"],
                  n_epochs=epochs, eval_every=max(1, epochs // 5), seed=base.SEED)

    def progress_fn(m):
        ep = len(m.history) + 1
        loss = m._last_loss if hasattr(m, "_last_loss") else 0.0
        base.log(f"  [{label}] epoch {ep}/{epochs} done")
        return (ep, 0.0)

    return model.fit(u, i, shape, rated=(u, i), val_fn=progress_fn)


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


def select_ranking_alpha(bpr, knn, users, known, validation, candidates=None):
    """Choose ranking alpha by F1@10 on held-out validation data over z-scored BPR and kNN vote scores."""
    accumulators = [MetricAccumulator(base.TOP_K, base.THRESHOLD) for _ in ALPHAS]
    for start in range(0, len(users), CHUNK):
        batch = users[start:start + CHUNK]
        bpr_z = zscore_rows(bpr.scores(batch), candidates)
        knn_z = zscore_rows(knn.ranking_scores(batch), candidates)
        for idx, alpha in enumerate(ALPHAS):
            blended = alpha * bpr_z + (1.0 - alpha) * knn_z
            accumulators[idx].add(blended, batch, known, validation, candidates, rating_scores=None)
    results = [acc.result() for acc in accumulators]
    grid = [{"alpha_bpr": alpha,
             "validation_f1": float(res["all_users"]["f1"]),
             "validation_recall": float(res["all_users"]["recall"])}
            for alpha, res in zip(ALPHAS, results)]
    best_idx = max(range(len(ALPHAS)), key=lambda idx: (
        grid[idx]["validation_f1"], grid[idx]["validation_recall"], -abs(ALPHAS[idx] - 0.5)))
    return ALPHAS[best_idx], grid


def compare(mf, bias_only, bpr, bpr_k0, knn, alpha_rating, alpha_ranking,
            baseline_train, users, known, test, candidates=None):
    """Compare rating models, ranking models, both hybrids, and baselines on the exact same test data."""
    rating_blend = Hybrid(mf, knn, alpha_rating)
    ranking_blend = RankingHybrid(bpr, knn, alpha_ranking, candidates=candidates)
    mean = base.GlobalMean().fit(*baseline_train, known.shape)
    popularity = base.Popularity().fit(*baseline_train, known.shape)
    names = (
        "Memory kNN (rating)",
        "MF (MSE)",
        "Bias only (MSE)",
        "Hybrid (rating)",
        "Memory kNN (vote)",
        "MF (BPR)",
        "BPR (b_i only)",
        "Hybrid (ranking)",
        "Global mean",
        "Popularity",
    )
    rating_models = {"Memory kNN (rating)", "MF (MSE)", "Bias only (MSE)", "Hybrid (rating)", "Global mean"}
    metrics = {name: MetricAccumulator(base.TOP_K, base.THRESHOLD) for name in names}
    for start in range(0, len(users), CHUNK):
        batch = users[start:start + CHUNK]
        mf_ratings = mf.scores(batch)
        bias_ratings = bias_only.scores(batch)
        knn_ratings, knn_votes = knn.score_pair(batch)
        hybrid_ratings = rating_blend.scores(batch, knn_scores=knn_ratings, mf_scores=mf_ratings)

        bpr_scores = bpr.scores(batch)
        bpr_k0_scores = bpr_k0.scores(batch)
        bpr_z = zscore_rows(bpr_scores, candidates)
        knn_z = zscore_rows(knn_votes, candidates)
        hybrid_ranking = ranking_blend.scores(batch, knn_z=knn_z, bpr_z=bpr_z)

        scores = (
            knn_ratings,
            mf_ratings,
            bias_ratings,
            hybrid_ratings,
            knn_votes,
            bpr_scores,
            bpr_k0_scores,
            hybrid_ranking,
            mean.scores(batch),
            popularity.scores(batch),
        )
        for name, values in zip(names, scores):
            rating_scores = values if name in rating_models else None
            metrics[name].add(values, batch, known, test, candidates, rating_scores)
    return {name: accumulator.result() for name, accumulator in metrics.items()}


def in_domain(df, knn_config, max_epochs, eval_users, bpr_cfg, bpr_epochs, bpr_k0_epochs,
              alpha_rating=None, alpha_ranking=None):
    split = make_split(df)
    users = np.sort(np.random.default_rng(base.SEED).choice(
        split["eval_users"], min(eval_users, len(split["eval_users"])), replace=False))
    shape = split["shape"]
    base.log("  [MSE MF] selecting best epoch on validation...")
    mf = fit_mf(split["train"], shape, max_epochs, val=split["val"])
    best_epoch = mf.best_epoch
    bias_val = fit_bias_only(split["train"], shape, max_epochs, val=split["val"])
    bias_epoch = bias_val.best_epoch
    base.log(f"  [MSE MF] best_epoch={best_epoch}, bias_epoch={bias_epoch}")

    rating_selection = {"source": "Software validation"}
    ranking_selection = {"source": "Software validation"}
    if alpha_rating is None or alpha_ranking is None:
        validation_users = np.sort(np.random.default_rng(142).choice(
            users, min(2000, len(users)), replace=False))
        base.log("  [kNN] fitting on train for validation alpha selection...")
        knn_val = ImprovedUserKNN(**knn_config).fit(*split["train"], shape)
        if alpha_rating is None:
            base.log("  [Hybrid rating] selecting alpha_mf on 2,000 validation users...")
            alpha_rating, grid_r = select_alpha(mf, knn_val, validation_users, split["R_val"])
            rating_selection.update(grid=grid_r, n_validation_users=len(validation_users))
            base.log(f"  [Hybrid rating] selected alpha_mf={alpha_rating}")
        if alpha_ranking is None:
            bpr_val = fit_bpr(split["train"], shape, bpr_cfg, bpr_epochs, label="BPR-val")
            base.log("  [Hybrid ranking] selecting alpha_bpr on 2,000 validation users...")
            alpha_ranking, grid_rk = select_ranking_alpha(
                bpr_val, knn_val, validation_users, split["R_train"], split["R_val"])
            ranking_selection.update(grid=grid_rk, n_validation_users=len(validation_users))
            base.log(f"  [Hybrid ranking] selected alpha_bpr={alpha_ranking}")
            del bpr_val, knn_val

    final_train = split["trval"]
    base.log("  [Final refit] fitting MSE MF, Bias-only, BPR, and kNN on train+val...")
    mf = fit_mf(final_train, shape, best_epoch)
    bias_only = fit_bias_only(final_train, shape, bias_epoch)
    bpr = fit_bpr(final_train, shape, bpr_cfg, bpr_epochs, label="BPR-final")
    bpr_k0 = fit_bpr(final_train, shape, BPR_K0_CONFIG, bpr_k0_epochs, label="BPR-k0")
    knn = ImprovedUserKNN(**knn_config).fit(*final_train, shape)
    base.log(f"  [Evaluate] evaluating all 10 models on {len(users):,} test users...")
    return {
        "alpha_mf": alpha_rating,
        "alpha_bpr": alpha_ranking,
        "alpha_selection": rating_selection,
        "ranking_alpha_selection": ranking_selection,
        "mf_epochs": best_epoch,
        "bias_epochs": bias_epoch,
        "bpr_epochs": bpr_epochs,
        "bpr_k0_epochs": bpr_k0_epochs,
        "split": split["sizes"],
        "final_train_ratings": len(final_train[2]),
        "metrics": compare(mf, bias_only, bpr, bpr_k0, knn, alpha_rating, alpha_ranking,
                           final_train, users, split["known"], split["R_test"]),
    }


def cross_domain(software, games, knn_config, max_epochs, bpr_cfg, bpr_epochs, bpr_k0_epochs):
    split = make_cross_split(software, games)
    if not split["hidden_val"] or not split["hidden_test"]:
        raise ValueError("Cross-domain evaluation requires both validation and test users")
    train, game_train, _, info = cross_train(split, frac=1.0)
    shape, enc = split["shape"], split["enc"]
    eligible_neighbors = np.bincount(game_train[0], minlength=shape[0]) > 0
    base.log("  [Cross kNN] fitting on joint train with Software similarity...")
    knn = ImprovedUserKNN(**knn_config).fit(
        *train, shape, similarity_data=enc.arrays(software), neighbor_mask=eligible_neighbors)
    base.log("  [Cross MSE MF] fitting and selecting epoch on hidden-val users...")
    mf = fit_mf(train, shape, max_epochs, val=split["val"])
    bias_only = fit_bias_only(train, shape, max_epochs, val=split["val"])
    bpr = fit_bpr(train, shape, bpr_cfg, bpr_epochs, label="Cross-BPR")
    bpr_k0 = fit_bpr(train, shape, BPR_K0_CONFIG, bpr_k0_epochs, label="Cross-BPR-k0")

    u, i, r = split["val"]
    validation = csr_matrix((r, (u, i)), shape=shape)
    val_users = np.unique(u)
    no_known = csr_matrix(shape, dtype=np.float32)
    base.log("  [Cross Hybrid] selecting alpha_mf and alpha_bpr on 402 validation users...")
    alpha_rating, grid_r = select_alpha(mf, knn, val_users, validation)
    alpha_ranking, grid_rk = select_ranking_alpha(
        bpr, knn, val_users, no_known, validation, candidates=split["vg_items"])
    base.log(f"  [Cross Hybrid] selected alpha_mf={alpha_rating}, alpha_bpr={alpha_ranking}")

    # Keep validation users' game ratings hidden when moving from validation to test.
    base.log(f"  [Cross Evaluate] evaluating all 10 models on {len(split['test_users']):,} test users...")
    metrics = compare(mf, bias_only, bpr, bpr_k0, knn, alpha_rating, alpha_ranking,
                      game_train, split["test_users"], no_known, split["R_test"], split["vg_items"])
    return {
        "alpha_mf": alpha_rating,
        "alpha_bpr": alpha_ranking,
        "mf_epochs": mf.best_epoch,
        "bias_epochs": bias_only.best_epoch,
        "bpr_epochs": bpr_epochs,
        "bpr_k0_epochs": bpr_k0_epochs,
        "alpha_selection": {"source": "Cross-domain hidden validation users", "grid": grid_r},
        "ranking_alpha_selection": {"source": "Cross-domain hidden validation users", "grid": grid_rk},
        "split": {"shared_users": len(split["shared"]), "validation_users": len(split["hidden_val"]),
                  "test_users": len(split["hidden_test"]), **info},
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "hybrid")
    parser.add_argument("--eval-users", type=int, default=base.N_EVAL_USERS,
                        help="Cap evaluation users for each in-domain dataset")
    parser.add_argument("--max-epochs", type=int, default=base.MAX_EPOCHS,
                        help="MF epoch search limit, including for cross-domain validation")
    parser.add_argument("--bpr-mode", choices=("fast", "full"), default="fast",
                        help="'fast' uses BPR k=64 (40/40/16 ep, ~3 min); 'full' uses BPR k=512 (188/184/16 ep)")
    args = parser.parse_args()
    if not 1 <= args.eval_users <= base.N_EVAL_USERS or args.max_epochs < 1:
        parser.error(f"Require 1 <= --eval-users <= {base.N_EVAL_USERS} and --max-epochs >= 1")

    bpr_cfg = BPR_CONFIGS[args.bpr_mode]
    sw_bpr_ep, vg_bpr_ep, cross_bpr_ep = bpr_cfg["epochs"]
    sw_k0_ep, vg_k0_ep, cross_k0_ep = BPR_K0_CONFIG["epochs"]

    memory_selection = ROOT / "outputs" / "memory_reviewed" / "validation_selection.json"
    knn_config = json.loads(memory_selection.read_text(encoding="utf-8"))["config"]
    software = pd.read_csv(args.data_dir / "Software.csv")
    games = pd.read_csv(args.data_dir / "Video_Games.csv")
    base.log(f"Software: training and evaluating rating & ranking hybrids (BPR mode={args.bpr_mode})")
    sw = in_domain(software, knn_config, args.max_epochs, args.eval_users,
                   bpr_cfg, sw_bpr_ep, sw_k0_ep)
    base.log(f"Video Games: training and evaluating rating & ranking hybrids (BPR mode={args.bpr_mode})")
    vg = in_domain(games, knn_config, args.max_epochs, args.eval_users,
                   bpr_cfg, vg_bpr_ep, vg_k0_ep,
                   alpha_rating=sw["alpha_mf"], alpha_ranking=sw["alpha_bpr"])
    base.log(f"Cross-domain: training and evaluating rating & ranking hybrids (BPR mode={args.bpr_mode})")
    cross = cross_domain(software, games, knn_config, args.max_epochs,
                         bpr_cfg, cross_bpr_ep, cross_k0_ep)
    results = {"Software": sw, "Video Games": vg, "Cross-domain": cross}
    report = {
        "formula_rating": "alpha_mf * MF(MSE) rating + (1 - alpha_mf) * kNN rating",
        "formula_ranking": "alpha_bpr * zscore(BPRMF) + (1 - alpha_bpr) * zscore(kNN vote)",
        "selection_metric_rating": "validation RMSE",
        "selection_metric_ranking": "validation F1@10 over all evaluated users",
        "ranking_metric": "F1@10 over all evaluated users",
        "seed": base.SEED,
        "mf_config": MF_CONFIG,
        "bpr_mode": args.bpr_mode,
        "bpr_config": {k: v for k, v in bpr_cfg.items() if k != "epochs"},
        "bpr_k0_config": {k: v for k, v in BPR_K0_CONFIG.items() if k != "epochs"},
        "knn_config": knn_config,
        "max_epochs": args.max_epochs,
        "in_domain_eval_users": args.eval_users,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for domain, experiment in results.items():
        for name, scores in experiment["metrics"].items():
            base.log(domain, f"{name:22s}", "RMSE", scores["rmse"],
                     "F1@10", f"{scores['all_users']['f1']:.6f}",
                     "Recall@10", f"{scores['all_users']['recall']:.6f}")
        base.log(domain, "alpha_mf (rating)", experiment["alpha_mf"],
                 "alpha_bpr (ranking)", experiment["alpha_bpr"])
    base.log("Saved", output)


if __name__ == "__main__":
    main()
