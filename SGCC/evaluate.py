#!/usr/bin/env python3
"""Compute the four required SGCC metrics from external probability files."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             recall_score, roc_auc_score)

from rebuild_database import ROOT, digest, read_json


def aligned_scores(path, truth):
    predictions = pd.read_parquet(path) if path.suffix.lower() == '.parquet' else pd.read_csv(path)
    if not {'row_id', 'p_theft'} <= set(predictions.columns):
        raise ValueError('Prediction file must contain row_id and p_theft columns')
    if predictions.row_id.isna().any() or not predictions.row_id.is_unique:
        raise ValueError('Prediction row_id values must be present and unique')
    if len(predictions) != len(truth) or set(predictions.row_id) != set(truth.row_id):
        raise ValueError('Predictions must cover the complete corresponding split, without extra rows')
    # Ground truth is always read from the frozen data, never from predictions.
    p = predictions.set_index('row_id').loc[truth.row_id, 'p_theft'].to_numpy(dtype=float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('p_theft must contain finite probabilities in [0, 1]')
    return p


def validation_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-30)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--test-predictions', type=Path)
    parser.add_argument('--validation-predictions', type=Path)
    parser.add_argument('--validation-only', action='store_true', help='Select configurations by validation ROC-AUC without reading test labels')
    parser.add_argument('--threshold-mode', choices=['validation-f1', 'fixed'], default='validation-f1')
    parser.add_argument('--fixed-threshold', type=float, default=.5)
    parser.add_argument('--output', type=Path, help='Optional JSON destination; otherwise print only')
    args = parser.parse_args()
    if args.validation_only and args.validation_predictions is None:
        parser.error('--validation-only requires --validation-predictions')
    if not args.validation_only and args.test_predictions is None:
        parser.error('--test-predictions is required for final test evaluation')
    if args.threshold_mode == 'validation-f1' and args.validation_predictions is None:
        parser.error('--validation-predictions is required to select the threshold without test labels')
    if not 0 <= args.fixed_threshold <= 1:
        parser.error('--fixed-threshold must be in [0, 1]')
    manifest = read_json(args.data_dir / 'manifest.json')
    for name in (['val.parquet'] if args.validation_only else ['val.parquet', 'test.parquet']):
        if digest(args.data_dir / name) != manifest['files_sha256'][name]:
            raise ValueError(f'Frozen data changed: {name}')
    if args.validation_only:
        val = pd.read_parquet(args.data_dir / 'val.parquet', columns=['row_id', 'target'])
        val_scores = aligned_scores(args.validation_predictions, val)
        report = {'stage': 'validation_configuration_selection', 'metric': 'ROC_AUC',
                  'direction': 'maximize', 'rows': len(val),
                  'score': float(roc_auc_score(val.target, val_scores))}
        text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text, end='')
        return
    test = pd.read_parquet(args.data_dir / 'test.parquet', columns=['row_id', 'target'])
    test_scores = aligned_scores(args.test_predictions, test)
    if args.threshold_mode == 'validation-f1':
        val = pd.read_parquet(args.data_dir / 'val.parquet', columns=['row_id', 'target'])
        val_scores = aligned_scores(args.validation_predictions, val)
        threshold = validation_f1_threshold(val.target.to_numpy(), val_scores)
    else:
        threshold = args.fixed_threshold
    prediction = test_scores >= threshold
    report = {'positive_class': 'theft (target=1)', 'test_rows': len(test),
              'primary_validation_metric': 'ROC_AUC',
              'threshold_mode': args.threshold_mode, 'threshold': threshold,
              'metrics': {'ROC_AUC': float(roc_auc_score(test.target, test_scores)),
                          'PR_AUC_AP': float(average_precision_score(test.target, test_scores)),
                          'F1': float(f1_score(test.target, prediction, zero_division=0)),
                          'recall_theft': float(recall_score(test.target, prediction, zero_division=0))}}
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
