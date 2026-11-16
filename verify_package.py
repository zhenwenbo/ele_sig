"""Write or verify file SHA256 inventory for the dataset/method package."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'PACKAGE_MANIFEST.json'


def inventory():
    files = {}
    forbidden = {'best.pt', 'result.json', 'history.json', 'selection.json'}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path == MANIFEST or '__pycache__' in path.parts or '.git' in path.parts:
            continue
        assert path.name not in forbidden and path.suffix not in {'.log', '.ckpt'}, path
        assert 'predictions' not in path.name, path
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=digest)
    return files


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    a = p.parse_args()
    current = inventory()
    if a.write:
        MANIFEST.write_text(json.dumps(dict(package='SGCC-1034 and VSB-author-nine, data and full-finetuning method only',
            files=current), ensure_ascii=False, indent=2) + '\n')
    else:
        expected = json.loads(MANIFEST.read_text())['files']
        assert current == expected, 'Package files differ from PACKAGE_MANIFEST.json'
    print(json.dumps(dict(status='passed', files=len(current), total_bytes=sum(v['bytes'] for v in current.values()))))
