from pathlib import Path
import hashlib
import os
import shutil
from .errors import SBDError

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(os.environ.get("SBD_CACHE", str(ROOT / ".cache"))).resolve()
MAX_BYTES = 500 * 1024 * 1024
MAX_SECONDS = 300
MAX_FRAMES = 18000
PREPROCESS_VERSION = "cfr-v1-libx264-crf18-height720-fps-rational-rgb48x27-bicubic"
DETECTOR_NAMES = {"tinycut": "Clean TinyCut S", "autoshot": "AutoShot", "omnishotcut": "OmniShotCut (provided checkpoint)"}
DETECTOR_NAMES['omnishotcut_legacy_v15'] = 'OmniShotCut Legacy v1.5'
KNOWN_HASHES = {
    "tinycut": "b3f52ff114a832094bd20005b408c60f4dc3ea2b1c1a7a129f390e6ade3fcdd0",
    "decoder": "1151c8bdcf0e5eef3b64c4d3ff90b601cda35117787d89d05fde7e711bac919e",
    "autoshot": "3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6",
    "omnishotcut": "5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa",
    "verifier": "2d939badcbd1f8b1d85659b4187a746ce73211a4018b8721ee28689b7f2ce89d",
    "omnishotcut_legacy_v15": "5bafe33d8d0dde16c6e24fc63dfc580da9d5731804d6e51e1fc8de21d4600999",
}
MODEL_PATHS = {
    "tinycut": ROOT / "models/tinycut/clean_tinycut_s_member.pkl",
    "decoder": ROOT / "models/tinycut/decoder_frozen.json",
    "autoshot": ROOT / "models/autoshot/ckpt_0_200_0.pth",
    "omnishotcut": ROOT / "models/omnishotcut/OmniShotCut_ckpt.pth",
    "verifier": ROOT / "third_party/sbdv2/models/generic_local.sbdm",
    "omnishotcut_legacy_v15": ROOT / "models/omnishotcut_legacy_v15/OmniShotCut_ckpt.pth",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_path(name):
    path = MODEL_PATHS[name]
    if not path.is_file():
        raise SBDError(f"Missing {name} asset: {path.name}. See docs/model_installation.md.")
    if sha256(path) != KNOWN_HASHES[name]:
        raise SBDError(f"SHA-256 mismatch for {name}: expected the frozen handoff artifact. Restore {path.name}.")
    return path


def executable(name):
    override = os.environ.get("SBD_" + name.upper())
    bundled = ROOT / "tools/ffmpeg/bin" / (name + (".exe" if os.name == "nt" else ""))
    found = override or (str(bundled) if bundled.is_file() else shutil.which(name))
    if not found or not Path(found).is_file():
        raise SBDError(f"{name} is missing. Run setup.bat or install FFmpeg/FFprobe on PATH.")
    return str(Path(found).resolve())
