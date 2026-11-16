"""Recreate mark4h's nine-feature measurement table from public peak artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.compat.pickle_compat import Unpickler
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
FEATURES = ['peak_count_Q13', 'abs_small_dist_to_min_mean_Q02', 'height_mean_Q02',
            'height_std_Q02', 'sawtooth_rmse_mean_Q02', 'peak_count_Q02',
            'ratio_next_mean_Q02', 'ratio_prev_mean_Q02', 'peak_count_total']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class DataFrameUnpickler(Unpickler):
    """Allow only data-construction globals required by the author's DataFrame."""
    allowed = {
        ('pandas.core.frame', 'DataFrame'),
        ('pandas.core.internals.managers', 'BlockManager'),
        ('pandas.core.internals.blocks', 'new_block'),
        ('pandas.core.internals', 'BlockManager'),
        ('pandas.core.indexes.base', '_new_Index'),
        ('pandas.core.indexes.base', 'Index'),
        ('pandas.core.indexes.numeric', 'Int64Index'),
        ('pandas.core.indexes.range', 'RangeIndex'),
        ('pandas.core.arrays.categorical', 'Categorical'),
        ('pandas.core.dtypes.dtypes', 'CategoricalDtype'),
        ('pandas._libs.arrays', '__pyx_unpickle_NDArrayBacked'),
        ('numpy.core.multiarray', '_reconstruct'),
        ('numpy.core.multiarray', 'scalar'),
        ('numpy._core.multiarray', '_reconstruct'),
        ('numpy', 'ndarray'), ('numpy', 'dtype'), ('builtins', 'slice'),
    }

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise ValueError(f'Unsupported pickle data constructor: {module}.{name}')
        return super().find_class(module, name)


def aggregate(peaks, meta):
    """Match the author's non-simplified process() function; never use target."""
    assert 'target' not in peaks.columns
    lookup = meta[['signal_id', 'id_measurement']]
    assert lookup.signal_id.is_unique
    p = peaks.merge(lookup, on='signal_id', how='left', validate='many_to_one')
    assert p.id_measurement.notna().all()
    # The author's winning notebook removes these likely spurious high peaks.
    p = p[~((p.ratio_next > .33333) & (p.height > 50))]
    result = pd.DataFrame(index=pd.Index(meta.id_measurement.unique(), name='id_measurement'))
    q02 = p[p.Q.isin([0, 2])].copy()
    q13 = p[p.Q.isin([1, 3])]
    result['peak_count_Q02'] = q02.groupby('id_measurement').px.count()
    result['peak_count_total'] = p.groupby('id_measurement').px.count()
    result['peak_count_Q13'] = q13.groupby('id_measurement').px.count()
    q02['abs_small_dist_to_min'] = np.abs(q02.small_dist_to_min)
    statistics = q02.groupby('id_measurement').agg({'height': ['mean', 'std'],
                   'ratio_prev': ['mean'], 'ratio_next': ['mean'],
                   'abs_small_dist_to_min': ['mean'], 'sawtooth_rmse': ['mean']})
    statistics.columns = ['_'.join(pair) + '_Q02' for pair in statistics.columns]
    result = result.join(statistics)
    return result[FEATURES]


def main():
    raw = HERE / 'raw'
    expected = {
        'metadata_train.csv': 'c7f4b00e867259870c5381b5da9ba23b878ae9b907e75140e70f953c611b053a',
        'train_peaks.pkl': '352a470df35d78cbd03b9c7f072b583ce0c3e1cec92ead3eab2370a1e0ba5100',
    }
    for filename, checksum in expected.items():
        assert digest(raw / filename) == checksum, f'Raw source checksum mismatch: {filename}'
    meta = pd.read_csv(raw / 'metadata_train.csv')
    assert meta.shape == (8712, 4), meta.shape
    assert meta.signal_id.is_unique
    assert meta.groupby('id_measurement').size().eq(3).all()
    assert set(meta.target) == {0, 1}
    with (raw / 'train_peaks.pkl').open('rb') as f:
        peaks = DataFrameUnpickler(f).load()
    assert isinstance(peaks, pd.DataFrame)
    print('peak table', peaks.shape, peaks.columns.tolist(), flush=True)
    assert {'signal_id', 'px', 'height', 'Q', 'ratio_next', 'ratio_prev',
            'small_dist_to_min', 'sawtooth_rmse'} <= set(peaks.columns)
    assert not {'target', 'id_measurement'} & set(peaks.columns)
    table = aggregate(peaks, meta)
    # Confirm equivalence to only the two reviewed aggregation functions in the author source.
    source = (HERE / 'author_solution.py').read_text()
    start = source.index('def process(peaks_df, meta_df, use_improved=True):')
    end = source.index('\n%%time\nX_train_full', start)
    namespace = {'pd': pd, 'np': np, 'USE_SIMPLIFIED_VERSION': False}
    exec(compile(source[start:end], 'reviewed_author_aggregation', 'exec'), namespace)
    joined = peaks.merge(meta[['signal_id', 'id_measurement']], on='signal_id', validate='many_to_one')
    reference = namespace['create_features'](meta, joined)[FEATURES]
    pd.testing.assert_frame_equal(table, reference, check_exact=False, atol=1e-12, rtol=1e-12)
    target = meta.groupby('id_measurement').target.max().reindex(table.index)
    table['target'] = target.to_numpy(dtype=np.int64)
    table.insert(0, 'row_id', table.index.to_numpy(dtype=np.int64))
    assert table.shape == (2904, 11)
    train, held = train_test_split(np.arange(len(table)), test_size=.3, random_state=42,
                                   stratify=table.target)
    val, test = train_test_split(held, test_size=.5, random_state=42, stratify=table.target.iloc[held])
    data = HERE / 'data'
    data.mkdir(exist_ok=True)
    table.to_csv(data / 'all_measurements.csv', index=False)
    meta.to_csv(data / 'original_phase_labels.csv', index=False)
    manifest = dict(dataset='VSB Power Line Fault Detection', feature_columns=FEATURES, seed=42,
                    source='https://www.kaggle.com/competitions/vsb-power-line-fault-detection',
                    feature_source='mark4h/vsb-1st-place-solution, version 3, public train_peaks.pkl output',
                    kernel_version=3, kernel_run_id=12111254, author_aggregation_equivalence='passed',
                    source_sha256={name: digest(raw / name) for name in ['metadata_train.csv', 'train_peaks.pkl']},
                    rows=2904, original_phase_rows=8712, target='any original positive phase in measurement',
                    label_mapping={'no labeled partial discharge in any phase': 0, 'at least one positive phase': 1},
                    total_positive=int(table.target.sum()), original_phase_positive=int(meta.target.sum()),
                    removed_from_inputs=['row_id', 'signal_id', 'id_measurement', 'phase', 'target'],
                    original_competition_test='20337 signals with no public ground truth; not used for scoring',
                    evaluation='internal held-out measurement split; not Kaggle private leaderboard score', splits={})
    for split, idx in [('train', train), ('val', val), ('test', test)]:
        part = table.iloc[idx].reset_index(drop=True)
        path = data / f'{split}.parquet'
        part.to_parquet(path, index=False)
        part.to_csv(data / f'{split}.csv', index=False)
        manifest['splits'][split] = {'rows': len(part), 'positive': int(part.target.sum()),
                                    'row_ids': part.row_id.tolist(), 'sha256': digest(path)}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert set(manifest['splits'][left]['row_ids']).isdisjoint(manifest['splits'][right]['row_ids'])
    save(data / 'metadata.json', manifest)
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'row_ids'} for k, v in manifest['splits'].items()}), flush=True)
    print('measurement positives', manifest['total_positive'], 'phase positives', manifest['original_phase_positive'], flush=True)


if __name__ == '__main__':
    main()
