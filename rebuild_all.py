"""Rebuild/verify SGCC and the fixed VSB nine-feature data package."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def verify_vsb():
    data = ROOT / 'vsb/data'
    meta = json.loads((data / 'metadata.json').read_text())
    phases = pd.read_csv(ROOT / 'vsb/raw/metadata_train.csv')
    truth = phases.groupby('id_measurement').target.max()
    parts = {}
    for name, spec in meta['splits'].items():
        path = data / f'{name}.parquet'
        assert digest(path) == spec['sha256'], f'VSB {name} checksum mismatch'
        frame = pd.read_parquet(path)
        assert frame.row_id.tolist() == spec['row_ids']
        assert len(frame) == spec['rows'] and int(frame.target.sum()) == spec['positive']
        np.testing.assert_array_equal(frame.target, truth.loc[frame.row_id])
        assert frame.columns.tolist() == ['row_id', *meta['feature_columns'], 'target']
        csv = pd.read_csv(data / f'{name}.csv')
        pd.testing.assert_frame_equal(csv, frame, check_exact=False, atol=1e-10, rtol=1e-10)
        parts[name] = set(frame.row_id)
    assert not parts['train'] & parts['val']
    assert not parts['train'] & parts['test']
    assert not parts['val'] & parts['test']
    assert set.union(*parts.values()) == set(truth.index)
    for filename, expected in meta['source_sha256'].items():
        assert digest(ROOT / 'vsb/raw' / filename) == expected
    print(json.dumps(dict(VSB='verified', rows=len(truth), positive=int(truth.sum()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--recompute-splits', action='store_true')
    a = parser.parse_args()
    command = [sys.executable, str(ROOT / 'SGCC/rebuild_database.py')]
    if a.verify_only:
        command.append('--verify-only')
    else:
        if a.overwrite:
            command.append('--overwrite')
        if a.recompute_splits:
            command.append('--recompute-splits')
    subprocess.run(command, check=True)
    if not a.verify_only:
        subprocess.run([sys.executable, str(ROOT / 'vsb/prepare_vsb.py')], check=True)
    verify_vsb()
