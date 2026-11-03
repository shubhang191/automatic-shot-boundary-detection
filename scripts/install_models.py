"""Copy exact user-supplied large weights into the source checkout. No untrusted loading."""
from pathlib import Path
import argparse
import hashlib
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbd_app.config import MODEL_PATHS, KNOWN_HASHES, sha256

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--autoshot", type=Path)
    p.add_argument("--omnishotcut", type=Path)
    args = p.parse_args()
    for key in ["autoshot", "omnishotcut"]:
        source = getattr(args, key)
        if source is not None:
            if sha256(source) != KNOWN_HASHES[key]:
                raise SystemExit(f"Wrong {key} bytes: expected the hash documented in models/models_manifest.json")
            dest = MODEL_PATHS[key]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            print(key, "verified and installed")
