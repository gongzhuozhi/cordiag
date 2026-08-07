"""Source fingerprint cache gate for the cordiag regression suite."""
import hashlib, json, os
from pathlib import Path

SOURCE_FILES = ['cordiag/m1.py', 'cordiag/zpg.py', 'cordiag/tg.py', 'cordiag/calibration.py']

def compute_fingerprint(repo_root):
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        fpath = Path(repo_root) / rel
        if fpath.exists():
            with open(fpath, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()

def write_fingerprint(out_dir, repo_root):
    os.makedirs(out_dir, exist_ok=True)
    fp = compute_fingerprint(repo_root)
    with open(os.path.join(out_dir, '_source_sha256.json'), 'w') as f:
        json.dump({'sha256': fp}, f)
    return fp

def cache_valid(out_dir):
    fp_file = os.path.join(out_dir, '_source_sha256.json')
    if not os.path.exists(fp_file):
        return False
    with open(fp_file) as f:
        stored = json.load(f).get('sha256', '')
    repo_root = str(Path(__file__).resolve().parent.parent)
    return stored == compute_fingerprint(repo_root)
