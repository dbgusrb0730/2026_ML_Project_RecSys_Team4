"""Evaluate cross-domain UserKNN using the team's original joint-matrix approach.

Run: python run_memory_cross.py
No MF or hybrid models are fitted. The original recsys.py stays unchanged.
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import recsys as base


def main():
    started = time.time()
    root = Path(__file__).parent
    source_result = root / "outputs/memory_software_full_10000/results.json"
    params = json.loads(source_result.read_text(encoding="utf-8"))
    k = params["best_k"]
    base.log(f"Loading Software and Video_Games; using Software k={k}")
    source, target = base.load_ratings("Software"), base.load_ratings("Video_Games")
    shared = np.array(sorted(set(source.user_id) & set(target.user_id)))
    rng = np.random.default_rng(base.SEED)
    hidden = set(rng.choice(shared, int(len(shared) * base.CROSS_TARGET_FRAC), replace=False))
    is_hidden = target.user_id.isin(hidden)
    target_train, target_test = target[~is_hidden], target[is_hidden]
    joint_train = pd.concat([source, target_train])
    # Guard against accidentally exposing any evaluated user's game ratings.
    assert not set(target_train.user_id).intersection(hidden)
    assert set(target_test.user_id) == hidden
    enc = base.Encoder(pd.concat([source.user_id, target.user_id]),
                       pd.concat([source.parent_asin, target.parent_asin]))
    target_items = enc.items.isin(target.parent_asin.unique())
    u, i, r = enc.arrays(joint_train)
    tu, ti, tr = enc.arrays(target_train)
    has_target = np.zeros(enc.shape[0], bool)
    has_target[np.unique(tu)] = True
    users = np.sort(enc.users.get_indexer(list(hidden)))
    assert not has_target[users].any()
    R_test = enc.matrix(target_test)
    no_history = csr_matrix(enc.shape, dtype=np.float32)
    base.log(f"Shared users={len(shared)}; hidden={len(hidden)}; bridge={len(shared)-len(hidden)}")
    report = {
        "experiment": "Cross-domain", "seed": base.SEED, "best_k": k,
        "k_source": str(source_result), "shared_users": len(shared),
        "hidden_users": len(hidden), "bridge_users": len(shared) - len(hidden),
        "test_ratings": len(target_test), "results": {},
        "notes": [
            "Original recsys.py cross-domain split, UserKNN and evaluation preserved.",
            "Evaluated users have all Video Games ratings withheld, but retain Software history.",
            "Joint Software + non-hidden Video Games ratings determine centering and similarity norms.",
            "This differs from computing neighbor similarity on Software alone as in the proposal.",
            "Only users with training Video Games ratings may be neighbors.",
            "Only Video Games items are recommendation candidates.",
            "F1 averages users with at least one relevant test rating.",
            "This is a user holdout experiment, not a chronological cross-domain split.",
        ],
    }
    models = {
        "Global mean": base.GlobalMean().fit(tu, ti, tr, enc.shape),
        "Popularity": base.Popularity().fit(tu, ti, tr, enc.shape),
        "User kNN": base.UserKNN(k, chunk=32).fit(u, i, r, enc.shape, neighbor_mask=has_target),
    }
    for name, model in models.items():
        metrics = base.evaluate(model.scores, users, no_history, R_test,
                                base.TOP_K, base.THRESHOLD, candidates=target_items, chunk=32)
        if not model.predicts_ratings:
            metrics["rmse"] = None
        report["results"][name] = metrics
        base.log(name, metrics)
    report["runtime_seconds"] = round(time.time() - started, 2)
    out = root / "outputs/memory_cross_domain"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["results"]).T.to_csv(out / "test_metrics.csv", index_label="model")
    pd.DataFrame({"user_index": users, "user_id": enc.users[users]}).to_csv(
        out / "evaluation_users.csv", index=False)
    base.log(f"Done: {out}")


if __name__ == "__main__":
    main()
