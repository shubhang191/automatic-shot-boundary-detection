"""Read-only integrity/deserialization audit of the two bundled verifier models."""
from pathlib import Path
import sys
import json
import importlib.metadata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from legacy_python.integration_adapter import checked_bundle_path, load_bundle, modules
from sbd_app.config import sha256


def validate():
    modules()
    out = {'status': 'PASS', 'packages': {p: importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib','threadpoolctl']}, 'bundles': {}}
    for detector in ['autoshot','omnishotcut']:
        path = checked_bundle_path(detector)
        b = load_bundle(detector)
        models = [p['model'] for p in b['parts']] if 'parts' in b else b['models']
        out['bundles'][detector] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha256(path),
            keys=list(b), settings={k:v for k,v in b.items() if k not in ['parts','models']},
            models=[dict(type=type(m).__name__, feature_count=int(m.n_features_in_), iterations=int(m.n_iter_)) for m in models],
            features=[p['cols'] for p in b['parts']] if 'parts' in b else b['feats'])
    return out


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
