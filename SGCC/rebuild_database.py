#!/usr/bin/env python3
"""Rebuild SGCC raw-daily tables, the frozen 2048-row context, and SQLite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'reference'
SPLITS = ('train', 'val', 'test')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_source(path, recompute):
    protocol = read_json(REFERENCE / 'protocol.json')
    if digest(path) != protocol['source_csv_sha256']:
        raise ValueError('Raw CSV SHA256 differs from the frozen SGCC release')
    for name, expected in protocol['reference_files_sha256'].items():
        if digest(REFERENCE / name) != expected:
            raise ValueError(f'Frozen reference changed: {name}')
    raw = pd.read_csv(path)
    assert raw.shape == (42372, 1036) and raw.CONS_NO.is_unique
    assert raw.FLAG.value_counts().to_dict() == {0: 38757, 1: 3615}
    date_columns = [c for c in raw.columns if c not in ('CONS_NO', 'FLAG')]
    dates = pd.to_datetime(date_columns, format='%Y/%m/%d')
    order = np.argsort(dates)
    dates = dates[order]
    date_columns = [date_columns[i] for i in order]
    x = raw[date_columns]
    assert len(date_columns) == 1034 and dates.is_unique
    assert not (x < 0).any().any() and not np.isinf(x.to_numpy()).any()
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    columns = ['day_' + day.strftime('%Y_%m_%d') for day in dates]
    assert columns == read_json(REFERENCE / 'feature_columns.json')
    split_ids = {k: np.asarray(v, dtype=np.int64) for k, v in read_json(REFERENCE / 'split_indices.json').items()}
    context = read_json(REFERENCE / 'context_indices.json')
    context_ids = np.asarray(context['row_ids'], dtype=np.int64)
    y = raw.FLAG.to_numpy(dtype=np.int64)
    combined = np.concatenate([split_ids[s] for s in SPLITS])
    assert np.array_equal(np.sort(combined), np.arange(len(raw)))
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert set(groups[split_ids[left]]).isdisjoint(groups[split_ids[right]])
    assert len(context_ids) == len(set(context_ids)) == 2048
    assert set(context_ids).issubset(split_ids['train'])
    assert int(y[context_ids].sum()) == 175
    for split in SPLITS:
        assert len(split_ids[split]) == protocol['splits'][split]['rows']
        assert int(y[split_ids[split]].sum()) == protocol['splits'][split]['positive']
    if recompute:
        folds = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
                     .split(np.zeros((len(raw), 1)), y, groups))
        held = np.sort(np.concatenate([folds[i][1] for i in range(3)]))
        train = np.setdiff1d(np.arange(len(raw)), held)
        val_pos, test_pos = next(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=43)
                                 .split(np.zeros((len(held), 1)), y[held], groups[held]))
        calculated = dict(train=train, val=held[val_pos], test=held[test_pos])
        for split in SPLITS:
            assert np.array_equal(calculated[split], split_ids[split]), f'{split}: split algorithm drift; use pinned versions'
        computed_context, _ = train_test_split(train, train_size=2048, random_state=42, stratify=y[train])
        assert np.array_equal(computed_context, context_ids), 'Context selection or ordering changed'
        print('Recomputed split assignments and context exactly match frozen references', flush=True)
    frame = x.copy()
    frame.columns = columns
    frame.insert(0, 'row_id', np.arange(len(raw), dtype=np.int64))
    frame['target'] = y
    return protocol, raw, frame, groups, split_ids, context_ids, columns


def create_sqlite(path, raw, frame, groups, split_ids, context_ids, columns):
    connection = sqlite3.connect(path)
    try:
        if connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) < len(columns) + 6:
            raise RuntimeError('SQLite must support at least 1040 bind parameters per statement')
        fields = ', '.join(f'"{c}" REAL' for c in columns)
        connection.execute(f'''CREATE TABLE samples (
            row_id INTEGER PRIMARY KEY, customer_id TEXT NOT NULL UNIQUE,
            target INTEGER NOT NULL CHECK(target IN (0, 1)),
            split TEXT NOT NULL CHECK(split IN ('train', 'val', 'test')),
            context_order INTEGER UNIQUE,
            feature_group TEXT NOT NULL, {fields})''')
        membership = np.empty(len(frame), dtype=object)
        for split in SPLITS:
            membership[split_ids[split]] = split
        context_order = pd.array([pd.NA] * len(frame), dtype='Int64')
        context_order[context_ids] = np.arange(len(context_ids))
        sql_frame = frame[columns].copy()
        sql_frame.insert(0, 'row_id', frame.row_id)
        sql_frame.insert(1, 'customer_id', raw.CONS_NO)
        sql_frame.insert(2, 'target', frame.target)
        sql_frame.insert(3, 'split', membership)
        sql_frame.insert(4, 'context_order', context_order)
        # UInt64 hashes are stored as decimal text to avoid signed SQLite overflow.
        sql_frame.insert(5, 'feature_group', [str(int(g)) for g in groups])
        sql_frame.to_sql('samples', connection, if_exists='append', index=False, chunksize=256)
        connection.execute('CREATE INDEX samples_split_row ON samples(split, row_id)')
        connection.execute('CREATE INDEX samples_group ON samples(feature_group)')
        select_columns = ', '.join('"' + c + '"' for c in ['row_id'] + columns + ['target'])
        for split in SPLITS:
            connection.execute(f'''CREATE VIEW "{split}" AS SELECT {select_columns}
                                    FROM samples WHERE split='{split}' ORDER BY row_id''')
        connection.execute(f'''CREATE VIEW context_2048 AS SELECT {select_columns}
                               FROM samples WHERE context_order IS NOT NULL ORDER BY context_order''')
        connection.commit()
    finally:
        connection.close()


def verify(output, raw, frame, groups, split_ids, context_ids, columns):
    checks = []
    for split in SPLITS:
        expected = frame.iloc[split_ids[split]].reset_index(drop=True)
        part = pd.read_parquet(output / f'{split}.parquet')
        pd.testing.assert_frame_equal(part, expected, check_exact=True)
        csv = output / f'{split}.csv'
        if csv.exists():
            pd.testing.assert_frame_equal(pd.read_csv(csv), expected, check_exact=False, rtol=1e-12, atol=1e-9)
        checks.append(f'{split}: complete rows, labels, order, missingness and 1034 raw values match source')
    context = pd.read_parquet(output / 'context_2048.parquet')
    pd.testing.assert_frame_equal(context, frame.iloc[context_ids].reset_index(drop=True), check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(output / 'context_2048.csv'), context,
                                  check_exact=False, rtol=1e-12, atol=1e-9)
    checks.append('context: exactly 2048 train-only rows in frozen order; 175 positive and 1873 normal')
    with sqlite3.connect(f'file:{output / "sgcc.sqlite3"}?mode=ro', uri=True) as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        names = [r[1] for r in connection.execute('PRAGMA table_info(samples)')]
        assert names == ['row_id', 'customer_id', 'target', 'split', 'context_order', 'feature_group'] + columns
        assert connection.execute('SELECT count(*) FROM samples').fetchone()[0] == len(raw)
        assert connection.execute('SELECT count(DISTINCT customer_id) FROM samples').fetchone()[0] == len(raw)
        assert not connection.execute('SELECT feature_group FROM samples GROUP BY feature_group HAVING count(DISTINCT split)>1 LIMIT 1').fetchall()
        for split in SPLITS:
            ids = [r[0] for r in connection.execute(f'SELECT row_id FROM "{split}"')]
            assert np.array_equal(ids, split_ids[split])
        ids = [r[0] for r in connection.execute('SELECT row_id FROM context_2048')]
        assert np.array_equal(ids, context_ids)
        offset = 0
        for batch in pd.read_sql_query('SELECT * FROM samples ORDER BY row_id', connection, chunksize=256):
            end = offset + len(batch)
            assert np.array_equal(batch.row_id, np.arange(offset, end))
            assert np.array_equal(batch.customer_id, raw.CONS_NO.iloc[offset:end])
            assert np.array_equal(batch.target, frame.target.iloc[offset:end])
            np.testing.assert_allclose(batch[columns].to_numpy(dtype=float),
                                       frame[columns].iloc[offset:end].to_numpy(dtype=float),
                                       rtol=0, atol=0, equal_nan=True)
            offset = end
        assert offset == len(frame)
    checks += ['SQLite integrity, schema, all 42372 source rows and every raw feature checked',
               'no customer overlap or exact consumption-pattern overlap across splits',
               'SQLite views preserve frozen split and context order']
    return {'status': 'passed', 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-csv', type=Path, default=ROOT / 'raw/data.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--recompute-splits', action='store_true', help='Re-run the original seeds and require exact equality to frozen assignments')
    parser.add_argument('--overwrite', action='store_true', help='Replace generated data files after validation; raw source and reference files are untouched')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--no-split-csv', action='store_true', help='Skip full split CSVs; keep Parquet and context CSV')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output in (ROOT.resolve(), (ROOT / 'raw').resolve(), REFERENCE.resolve()):
        parser.error('Use a dedicated output directory, not the root/raw/reference directory')
    if not args.raw_csv.is_file():
        parser.error(f'Original SGCC CSV is missing: {args.raw_csv}')
    if not args.verify_only and output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error('Output directory exists. Use --verify-only, a new --output-dir, or --overwrite')
    protocol, raw, frame, groups, split_ids, context_ids, columns = load_source(args.raw_csv, args.recompute_splits)
    if args.verify_only:
        manifest = read_json(output / 'manifest.json')
        for name, expected in manifest['files_sha256'].items():
            assert digest(output / name) == expected, f'Generated file changed: {name}'
        report = verify(output, raw, frame, groups, split_ids, context_ids, columns)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sgcc_rebuild_', dir=output.parent) as temp:
        staging = Path(temp)
        for split in SPLITS:
            part = frame.iloc[split_ids[split]].reset_index(drop=True)
            part.to_parquet(staging / f'{split}.parquet', index=False)
            if not args.no_split_csv:
                part.to_csv(staging / f'{split}.csv', index=False)
            print(f'{split}: {len(part)} users, {int(part.target.sum())} theft', flush=True)
        context = frame.iloc[context_ids].reset_index(drop=True)
        context.to_parquet(staging / 'context_2048.parquet', index=False)
        context.to_csv(staging / 'context_2048.csv', index=False)
        assignments = pd.DataFrame({'row_id': frame.row_id, 'CONS_NO': raw.CONS_NO,
                                     'target': frame.target, 'feature_group': groups, 'split': ''})
        for split in SPLITS:
            assignments.loc[split_ids[split], 'split'] = split
        assignments['context_order'] = pd.array([pd.NA] * len(frame), dtype='Int64')
        assignments.loc[context_ids, 'context_order'] = np.arange(2048)
        assignments.to_csv(staging / 'customer_splits.csv', index=False)
        for name in ('feature_columns.json', 'context_indices.json'):
            shutil.copy2(REFERENCE / name, staging / name)
        print('Building SQLite single table and train/val/test/context views', flush=True)
        create_sqlite(staging / 'sgcc.sqlite3', raw, frame, groups, split_ids, context_ids, columns)
        print('Checking all rows and raw feature values against the original CSV', flush=True)
        validation = verify(staging, raw, frame, groups, split_ids, context_ids, columns)
        byte_checks = {}
        for split in SPLITS:
            actual = digest(staging / f'{split}.parquet')
            byte_checks[split] = actual == protocol['splits'][split]['expected_parquet_sha256']
        validation['original_parquet_byte_match'] = byte_checks
        if pyarrow.__version__ == protocol['reconstruction_environment']['pyarrow']:
            assert all(byte_checks.values()), 'Parquet bytes differ in the original pinned environment'
        write_json(staging / 'validation_report.json', validation)
        manifest = {'dataset': protocol['dataset'], 'version': protocol['version'],
                    'validation_protocol': protocol['validation_protocol'],
                    'source_csv_sha256': protocol['source_csv_sha256'],
                    'feature_count': 1034, 'feature_columns': columns,
                    'target': 'target', 'label_mapping': protocol['label_mapping'],
                    'splits': {s: {'rows': len(split_ids[s]), 'positive': int(frame.target.iloc[split_ids[s]].sum())} for s in SPLITS},
                    'context': {'rows': 2048, 'normal': 1873, 'positive': 175, 'order': 'reference/context_indices.json row_ids order'},
                    'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                    'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                                    'pyarrow': pyarrow.__version__, 'sqlite': sqlite3.sqlite_version},
                    'files_sha256': {p.name: digest(p) for p in sorted(staging.iterdir()) if p.is_file()}}
        write_json(staging / 'manifest.json', manifest)
        output.mkdir(parents=True, exist_ok=True)
        # Publish only fully validated files, leaving unrelated output files intact.
        for path in sorted(staging.iterdir(), key=lambda p: p.name == 'manifest.json'):
            os.replace(path, output / path.name)
    print(f'Rebuild complete: {output / "sgcc.sqlite3"}', flush=True)
    print('Full data verification passed; context=2048 (normal=1873, theft=175)', flush=True)


if __name__ == '__main__':
    main()
