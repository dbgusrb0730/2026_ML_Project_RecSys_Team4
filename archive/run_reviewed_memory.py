"""Reproducible memory-CF experiments, shared metrics and bias-only ablation."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from archive import recsys as base
from memory import ImprovedUserKNN
from evaluation import MetricAccumulator, evaluate_model

ROOT = Path(base.ROOT)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def prepare(df):
    train, val, test = base.time_split(df)
    enc = base.Encoder(df.user_id, df.parent_asin)
    users = np.sort(np.random.default_rng(42).choice(enc.shape[0], min(10000, enc.shape[0]), replace=False))
    return enc, enc.arrays(train), enc.matrix(train), enc.matrix(val), enc.matrix(test), users


def evaluate_improved(model, users, known, test, candidates=None, include_bias=True):
    names = ['Improved rating', 'Improved vote'] + (['Bias only'] if include_bias else [])
    acc = {name: MetricAccumulator() for name in names}
    for s in range(0, len(users), 32):
        batch = users[s:s+32]
        rating, vote = model.score_pair(batch)
        acc['Improved rating'].add(rating, batch, known, test, candidates, rating)
        # Votes are NOT rating estimates: no RMSE is assigned to this row.
        acc['Improved vote'].add(vote, batch, known, test, candidates)
        if include_bias:
            bias = model.baseline_scores(batch)
            acc['Bias only'].add(bias, batch, known, test, candidates, bias)
    return {name: a.result() for name, a in acc.items()}


def experiment(arrays, shape, users, known, test, config, candidates=None, similarity_data=None,
               neighbor_mask=None, baseline_arrays=None):
    model = ImprovedUserKNN(**config).fit(*arrays, shape, similarity_data=similarity_data, neighbor_mask=neighbor_mask)
    result = evaluate_improved(model, users, known, test, candidates)
    del model
    source = arrays if baseline_arrays is None else baseline_arrays
    for name, m in [('Global mean', base.GlobalMean()), ('Popularity', base.Popularity())]:
        m.fit(*source, shape)
        result[name] = evaluate_model(m.scores, users, known, test, candidates, m.predicts_ratings)
    # Historical k=20 is retained to compare against the supplied original version.
    original = base.UserKNN(20).fit(*arrays, shape, neighbor_mask=neighbor_mask)
    result['Original kNN'] = evaluate_model(original.scores, users, known, test, candidates)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'memory_reviewed')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dfs, fingerprints = {}, {}
    for category in ['Software','Video_Games']:
        path = args.data_dir/f'{category}.csv'
        fingerprints[category] = hashlib.sha256(path.read_bytes()).hexdigest()
        df = pd.read_csv(path)
        assert not df.duplicated(['user_id','parent_asin']).any(), 'Duplicate user-item ratings need an explicit policy'
        assert df.rating.between(1,5).all()
        assert df.groupby('user_id').size().min() >= 5
        dfs[category] = df
    sw, vg = dfs['Software'], dfs['Video_Games']
    report = {'environment': {'python':platform.python_version(),'numpy':np.__version__, 'pandas':pd.__version__, 'scipy':scipy.__version__},
              'data_sha256':fingerprints, 'seed':42, 'tuning_seed':142,
              'metric_policy':{'primary':'all_users; no relevant test item => precision/recall/F1=0',
                               'secondary':'relevant_users only, comparable to previous reports',
                               'rmse':'over all observed test ratings of sampled users; null for votes/popularity',
                               'ties':'smaller encoded item index first', 'k':10,'threshold':4},
              'limitations':['Previously inspected test sets reused; not a fresh holdout.',
                             'Original kNN uses joint cross-domain similarity; improved uses Software only.',
                             'Rating and vote variants are separate methods; do not combine their metrics into one row.',
                             'Only bias-residual ablation performed; other simultaneous changes not isolated.'],
              'tuning':[], 'results':{}}
    enc, arrays, train, val, test, users = prepare(sw)
    tuning_users = np.sort(np.random.default_rng(142).choice(users, min(2000,len(users)), replace=False))
    model = ImprovedUserKNN().fit(*arrays, enc.shape)
    for k in [100,500,2000]:
        for power in [0.,0.5]:
            model.k, model.popularity_power = k, power
            metrics = evaluate_improved(model, tuning_users, train, val, include_bias=False)
            report['tuning'].append({'k':k,'popularity_power':power,'metrics':metrics})
            base.log('validation', k, power, metrics['Improved vote']['all_users']['f1'])
    best = max(report['tuning'], key=lambda e:(e['metrics']['Improved vote']['all_users']['f1'], -e['metrics']['Improved rating']['rmse']))
    config = dict(k=best['k'], popularity_power=best['popularity_power'], shrink=5.)
    report['config'] = config
    report['selection'] = 'Software validation vote F1 over all sampled users; rating RMSE breaks ties. Same settings used for rating variant and bias ablation.'
    save(out/'validation_selection.json', report)
    del model
    report['results']['Software'] = experiment(arrays, enc.shape, users, train+val, test, config)
    base.log('Software complete')
    save(out/'results.partial.json', report)
    enc, arrays, train, val, test, users = prepare(vg)
    report['results']['Video Games'] = experiment(arrays, enc.shape, users, train+val, test, config)
    base.log('Video Games complete')
    save(out/'results.partial.json', report)
    shared = np.array(sorted(set(sw.user_id)&set(vg.user_id)))
    hidden = set(np.random.default_rng(42).choice(shared,int(len(shared)*.2),replace=False))
    target_train, target_test = vg[~vg.user_id.isin(hidden)], vg[vg.user_id.isin(hidden)]
    assert not set(target_train.user_id)&hidden
    joint = pd.concat([sw,target_train])
    enc = base.Encoder(pd.concat([sw.user_id,vg.user_id]),pd.concat([sw.parent_asin,vg.parent_asin]))
    users = np.sort(enc.users.get_indexer(sorted(hidden)))
    mask = enc.users.isin(target_train.user_id.unique())
    assert not mask[users].any()
    report['cross_split'] = {'shared_users':len(shared),'hidden_users':len(hidden), 'test_ratings':len(target_test),
                             'hidden_user_sha256':hashlib.sha256('\n'.join(sorted(hidden)).encode()).hexdigest()}
    report['results']['Cross-domain'] = experiment(enc.arrays(joint),enc.shape,users,csr_matrix(enc.shape),enc.matrix(target_test),config,
         candidates=enc.items.isin(vg.parent_asin.unique()),similarity_data=enc.arrays(sw),neighbor_mask=mask,baseline_arrays=enc.arrays(target_train))
    report['runtime_seconds'] = time.time()-started
    save(out/'results.json',report)
    base.log('All experiments complete', report['runtime_seconds'])


if __name__ == '__main__':
    main()
