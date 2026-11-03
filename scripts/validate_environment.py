from pathlib import Path
import importlib.metadata
import json
import platform
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbd_app.config import MODEL_PATHS, verified_path, sha256, executable
from sbd_app.verifier.build_status import library_path, binary_path
from sbd_app.video_io import run


def validate():
    report = dict(os=platform.platform(), python=platform.python_version(), architecture=platform.machine(), models={}, tools={}, packages={})
    failures = []
    if sys.version_info[:2] != (3, 13) or platform.architecture()[0] != "64bit":
        failures.append("Use 64-bit Python 3.13")
    for key in MODEL_PATHS:
        try:
            p = verified_path(key)
            report["models"][key] = dict(status="PASS", sha256=sha256(p), bytes=p.stat().st_size)
        except Exception as exc:
            report["models"][key] = dict(status="FAIL", reason=str(exc))
            failures.append(str(exc))
    for name in ["ffmpeg", "ffprobe"]:
        try:
            report["tools"][name] = run([executable(name), "-version"]).decode(errors="replace").splitlines()[0]
        except Exception as exc:
            failures.append(str(exc))
    for name in ["torch", "torchvision", "numpy", "scipy", "gradio", "opencv-python-headless"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing Python package {name}; run setup.bat")
    try:
        from sbd_app.verifier.sbdv2_adapter import SBDV2Verifier
        report["verifier"] = dict(status="PASS", info=SBDV2Verifier().info,
                                  dll_sha256=sha256(library_path()), cli_sha256=sha256(binary_path()))
    except Exception as exc:
        failures.append(str(exc))
    try:
        from verify_legacy_models import validate as validate_legacy
        report['legacy_python'] = validate_legacy()
    except Exception as exc:
        failures.append(f'Legacy Python validation: {exc}')
    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    return report


if __name__ == "__main__":
    report = validate()
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["failures"] else 0)
