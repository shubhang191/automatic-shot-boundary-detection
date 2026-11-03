"""Explicit, reviewed distribution file allowlist; never recurse a developer tree."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = 'SBD_Faculty_Release_Portable_Windows_v2'
PROHIBITED = {'.venv','venv','env','__pycache__','.pytest_cache','.mypy_cache','.cache','.git',
              '.ruff_cache','installation_logs','wheelhouse','build','dist','frame_cache','.env','.tox','.nox','site-packages'}

def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe distribution path: ' + name)
    if any(s.lower() in PROHIBITED or s.lower().startswith('.venv.') or s.lower().endswith('.egg-info') for s in p.parts):
        raise ValueError('Prohibited distribution file: ' + name)
    if p.suffix.lower() in {'.pyc','.pyo','.log','.zip'} or p.name.lower() in {'pyvenv.cfg','.env','python.exe','pythonw.exe'}:
        raise ValueError('Machine-specific distribution file: ' + name)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def verify(output):
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None, 'ZIP CRC failure'
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP names'
        for name in names:
            assert name.startswith(TOP + '/'), 'Unexpected top-level folder'
            safe_name(name[len(TOP)+1:])
        manifest = json.loads(z.read(TOP + '/SHA256_MANIFEST.json'))
        assert len(manifest) + 1 == len(names), 'Manifest/file count mismatch'
        for name, expected in manifest.items():
            data = z.read(TOP + '/' + name)
            assert len(data) == expected['bytes'] and digest(data) == expected['sha256'], name
    return dict(status='PASS', files=len(names), crc='PASS', hashes='PASS', prohibited_files=0)

def package(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    names = json.loads((ROOT/'distribution_files.json').read_text())
    assert len(names) == len(set(names)), 'Duplicate allowlist entries'
    manifest = {}
    for name in names:
        safe_name(name)
        path = ROOT/name
        assert path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink(), name
        assert path.is_file(), 'Missing required file: ' + name
        data = path.read_bytes()
        if path.suffix in {'.py','.ps1','.bat','.md','.json','.txt','.toml'}:
            text = data.decode('utf-8-sig',errors='replace')
            secret = r'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'
            assert not re.search(secret,text), 'Potential secret: ' + name
        manifest[name] = dict(bytes=len(data),sha256=digest(data))
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in names:
            z.write(ROOT/name,TOP+'/'+name)
        z.writestr(TOP+'/SHA256_MANIFEST.json',json.dumps(manifest,indent=2))
    result = verify(output)
    with output.open('rb') as stream:
        result.update(path=str(output),bytes=output.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
    print(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify: print(json.dumps(verify(a.output),indent=2))
    else: package(a.output)
