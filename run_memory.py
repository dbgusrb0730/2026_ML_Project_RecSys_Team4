"""In-domain kNN experiments. Original recsys.py is imported without running main().

Quick check: python run_memory.py
Full evaluation: python run_memory.py --eval-users 10000
Games: python run_memory.py --category Video_Games --eval-users 10000
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import recsys as base


def main():
    parser = argparse.ArgumentParser(description="Run in-domain UserKNN only")
    parser.add_argument("--eval-users", type=int, default=100)
    parser.add_argument("--category", choices=["Software", "Video_Games"], default="Software")
    args = parser.parse_args()
    if args.eval_users <= 0:
        parser.error("--eval-users must be positive")
    transferred_k = None
    source_result = Path(__file__).parent / "outputs" / "memory_software_full_10000" / "results.json"
    if args.category == "Video_Games":
        if not source_result.is_file():
            parser.error("Run Software with --eval-users 10000 first.")
        source_report = json.loads(source_result.read_text(encoding="utf-8"))
        transferred_k = source_report["best_k"]
        if not isinstance(transferred_k, int) or transferred_k <= 0:
            parser.error("Software best_k must be a positive integer")
    started = time.time()
    stage = "full" if args.eval_users == base.N_EVAL_USERS else "preliminary"
    out = Path(__file__).parent / "outputs" / f"memory_{args.category.lower()}_{stage}_{args.eval_users}"
    out.mkdir(parents=True, exist_ok=True)
    base.log(f"Loading {args.category}; keeping the original time split and algorithms")
    df = base.load_ratings(args.category)
    train, val, test = base.time_split(df)
    enc = base.Encoder(df.user_id, df.parent_asin)
    u, i, r = enc.arrays(train)
    R_train, R_val, R_test = (enc.matrix(x) for x in (train, val, test))
    users = np.sort(np.random.default_rng(base.SEED).choice(
        enc.shape[0], min(args.eval_users, enc.shape[0]), replace=False))
    report = {
        "stage": stage, "category": args.category, "seed": base.SEED,
        "eval_users": len(users), "k_grid": list(base.KNN_GRID) if transferred_k is None else [],
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "notes": ["Original UserKNN and evaluation functions used unchanged.",
                  "All training users retained; only evaluation users sampled.",
                  "F1 averages users with at least one relevant held-out item.",
                  "Software validation chooses k; Video Games reuses it without tuning.",
                  "Preliminary runs are not final report results."],
        "tuning": [], "results": {},
    }
    for k in (base.KNN_GRID if transferred_k is None else ()):
        model = base.UserKNN(k, chunk=32).fit(u, i, r, enc.shape)
        pred, actual = base.predictions_at(model, users, R_val)
        score = base.rmse(actual, pred)
        report["tuning"].append({"k": k, "validation_rmse": score})
        base.log(f"k={k}: validation RMSE={score:.6f}")
    best_k = (min(report["tuning"], key=lambda x: x["validation_rmse"])["k"]
              if transferred_k is None else transferred_k)
    report["k_selection"] = "Software validation RMSE"
    if transferred_k is not None:
        report["k_source"] = str(source_result)
        base.log(f"Using k={best_k} selected on Software; no Video Games tuning")
    report["best_k"] = best_k
    known = R_train + R_val
    models = {
        "Global mean": base.GlobalMean().fit(u, i, r, enc.shape),
        "Popularity": base.Popularity().fit(u, i, r, enc.shape),
        "User kNN": base.UserKNN(best_k, chunk=32).fit(u, i, r, enc.shape),
    }
    for name, model in models.items():
        metrics = base.evaluate(model.scores, users, known, R_test,
                                base.TOP_K, base.THRESHOLD, chunk=32)
        if not model.predicts_ratings:
            metrics["rmse"] = None  # popularity counts are not predicted ratings
        report["results"][name] = metrics
        base.log(name, metrics)
    report["runtime_seconds"] = round(time.time() - started, 2)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["tuning"]:
        pd.DataFrame(report["tuning"]).to_csv(out / "k_comparison.csv", index=False)
    pd.DataFrame(report["results"]).T.to_csv(out / "test_metrics.csv", index_label="model")
    pd.DataFrame({"user_index": users, "user_id": enc.users[users]}).to_csv(
        out / "evaluation_users.csv", index=False)
    base.log(f"Done: {out}")


if __name__ == "__main__":
    main()
