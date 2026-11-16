"""Model-independent CPU audit of a completed full-data one-epoch run."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from training_schedule import query_batches, should_validate

from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, recall_score, roc_auc_score)


def audit(run, data):
    config = json.loads((run / 'config.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    proof = json.loads((run / 'gradient_proof.json').read_text())
    parameters = json.loads((run / 'parameters.json').read_text())
    assert result['status'] == 'complete'
    assert parameters['trainable'] == parameters['total'] == proof['trainable_parameters']
    assert proof['gradient_norm'] > 0 and proof['maximum_parameter_delta'] > 0
    assert not proof['tensors_without_grad']
    train = pd.read_parquet(data / 'train.parquet')
    val = pd.read_parquet(data / 'val.parquet')
    test = pd.read_parquet(data / 'test.parquet')
    assert config['train_row_ids'] == train.row_id.tolist()
    assert set(config['context_row_ids']) <= set(train.row_id)
    assert not set(train.row_id) & set(val.row_id)
    assert not set(train.row_id) & set(test.row_id)
    assert not set(val.row_id) & set(test.row_id)

    def read_prediction(path, truth):
        pred = pd.read_parquet(path)
        assert pred.row_id.is_unique and set(pred.row_id) == set(truth.row_id)
        pred = pred.set_index('row_id').loc[truth.row_id]
        np.testing.assert_array_equal(pred.y_true, truth.target)
        score = pred.p_positive.to_numpy()
        assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
        return truth.target.to_numpy(), score

    def independent_metrics(y, p, t):
        pred = p >= t
        return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC_AP=average_precision_score(y, p),
            MCC=matthews_corrcoef(y, pred), F1=f1_score(y, pred, zero_division=0),
            recall_positive=recall_score(y, pred, zero_division=0))

    primary = config['primary_validation_metric']
    budget = (len(train) + config['query'] - 1) // config['query']
    assert config['epochs'] == 1 and config['context'] == 1024
    assert result['optimizer_steps'] == budget
    assert result['completed_data_passes'] == 1
    expected_steps = [0] + [s for s in range(1, budget + 1) if should_validate(s, budget)]
    assert [h['step'] for h in history] == expected_steps
    coverage = np.zeros(len(train), dtype=np.int32)
    interval = np.zeros(len(train), dtype=np.int32)
    batches = iter(query_batches(len(train), config['query'], config['query_order_seed']))
    previous_step = 0
    for row in history:
        y, p = read_prediction(run / f"val_step_{row['step']:05d}.parquet", val)
        values = independent_metrics(y, p, row['validation']['threshold'])
        for key, value in values.items():
            assert np.isclose(value, row['validation'][key], rtol=0, atol=1e-12)
        if row['step'] > 0:
            t = row['training']
            for _ in range(row['step'] - previous_step):
                _, _, q, _ = next(batches)
                interval[q] += 1
                coverage[q] += 1
            assert t['steps'] == row['step'] - previous_step
            assert t['query_rows'] == int(interval.sum())
            assert t['unique_query_rows'] == int(np.count_nonzero(interval))
            assert t['min_query_exposure'] == int(interval.min())
            assert t['max_query_exposure'] == int(interval.max())
            interval.fill(0)
            previous_step = row['step']
    saved_coverage = pd.read_parquet(run / 'training_coverage.parquet').set_index('row_id').loc[train.row_id]
    np.testing.assert_array_equal(saved_coverage.query_count, coverage)
    assert np.all(coverage == 1)
    winner = max(history, key=lambda row: row['validation'][primary])
    assert winner['step'] == selection['best_step']
    assert winner['validation']['threshold'] == selection['threshold']
    assert winner['validation'][primary] == selection['best_validation_score']
    y, p = read_prediction(run / 'best_val_predictions.parquet', val)
    original = pd.read_parquet(run / f"val_step_{winner['step']:05d}.parquet")
    np.testing.assert_array_equal(p, original.set_index('row_id').loc[val.row_id].p_positive)
    # Verify threshold is globally optimal on validation; include the no-positive option.
    target_metric = config['threshold_metric']
    calculate = matthews_corrcoef if target_metric == 'MCC' else f1_score
    candidates = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
    optimum = max(calculate(y, p >= t) for t in candidates)
    assert np.isclose(calculate(y, p >= selection['threshold']), optimum, rtol=0, atol=1e-12)
    y, p = read_prediction(run / 'test_predictions.parquet', test)
    for key, value in independent_metrics(y, p, selection['threshold']).items():
        assert np.isclose(value, result['test'][key], rtol=0, atol=1e-12)
    assert confusion_matrix(y, p >= selection['threshold'], labels=[0, 1]).tolist() == result['test']['confusion_matrix']
    report = dict(status='passed', checks=['real labels and complete prediction coverage',
        'exactly one complete train pass, context 1024, validation at start and end', 'all parameters trainable with nonzero gradients and actual updates',
        'validation primary metric selects earliest best step', 'validation threshold global optimum',
        'independent test metrics match'], train_rows=len(train), val_rows=len(val), test_rows=len(test))
    (run / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    a = p.parse_args()
    audit(a.run.resolve(), a.data.resolve())
