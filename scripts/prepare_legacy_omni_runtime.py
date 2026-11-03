"""Build an isolated CPU import namespace from the unmodified pinned upstream source."""
from pathlib import Path
import shutil
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    vendor = ROOT / 'third_party/omnishotcut_legacy_v15'
    original = vendor / 'omnishotcut'
    runtime = vendor / 'sbd_omni_v15'
    changes = []
    for path in original.rglob('*.py'):
        rel = path.relative_to(original)
        text = path.read_text(encoding='utf-8')
        edited = text.replace('from omnishotcut.', 'from sbd_omni_v15.')
        if rel.as_posix() == '__init__.py':
            edited = '"""Offline CPU namespace; loading is handled by the application adapter."""\n'
        elif rel.as_posix() == 'engine.py':
            edited = edited.replace('.to("cuda")', '.to("cpu")')
        elif rel.as_posix() == 'architecture/backbone.py':
            edited = edited.replace('pretrained = is_main_process()', 'pretrained = False')
        out = runtime / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(edited, encoding='utf-8', newline='\n')
        if edited != text:
            changes.append(dict(path=rel.as_posix(), original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), runtime_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    report = ROOT / 'reports/legacy/omni_cpu_portability.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(source_commit='40ecf12', changes=changes,
        explanation=['Separate import namespace permits both Omni detectors in one process.',
                     'Use CPU tensors/model instead of CUDA.',
                     'Skip redundant ImageNet download; strict loading restores every backbone weight from the bundled checkpoint.',
                     'Bypass Hugging Face download wrapper; runtime is offline. Decoder and stitching code are unchanged.']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    prepare()
