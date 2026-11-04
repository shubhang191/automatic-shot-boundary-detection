"""Explicit, reviewed distribution file allowlist; never recurse a developer tree."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import zipfile
import stat

ROOT = Path(__file__).resolve().parents[1]
TOP = '209_ShubhangShrivats'
PROHIBITED = {'.venv','venv','env','__pycache__','.pytest_cache','.mypy_cache','.cache','.git',
              '.ruff_cache','installation_logs','wheelhouse','build','dist','frame_cache','.env','.tox','.nox','site-packages'}

REQUIRED = {'app.py','setup.bat','run.bat','verify_installation.py','verify_installation.bat',
            'requirements.txt','requirements-lock.txt','THIRD_PARTY_NOTICES.md','PORTABLE_WINDOWS_README.md',
            'models/tinycut/clean_tinycut_s_member.pkl','models/tinycut/decoder_frozen.json',
            'models/autoshot/ckpt_0_200_0.pth','models/omnishotcut/OmniShotCut_ckpt.pth',
            'models/omnishotcut_legacy_v15/OmniShotCut_ckpt.pth','models/legacy_python/omni_verifier_v15.joblib',
            'models/legacy_python/verifier_v15_idea12.joblib','third_party/sbdv2/models/generic_local.sbdm',
            'third_party/sbdv2/bin/sbdv.dll','third_party/sbdv2/bin/sbdv.exe',
            'tools/ffmpeg/bin/ffmpeg.exe','tools/ffmpeg/bin/ffprobe.exe'}


def require(condition, message):
    if not condition: raise ValueError(message)


def checked_names(names):
    require(isinstance(names, list) and all(isinstance(n,str) for n in names), 'Expected a list of file paths')
    require(len(names)==len(set(names)), 'Duplicate file paths')
    require(len(names)==len({n.casefold() for n in names}), 'Case-insensitive file collisions')
    for n in names: safe_name(n)


def safe_name(name):
    require(isinstance(name,str) and bool(name), 'Empty or non-string path')
    p = PurePosixPath(name)
    require(p.as_posix()==name, 'Noncanonical distribution path: '+name)
    require(not re.search(r'[<>"|?*\x00-\x1f\x7f]',name), 'Invalid Windows filename: '+name)
    for part in p.parts:
        require(not part.endswith((' ','.')), 'Trailing dot/space in path: '+name)
        require(not re.fullmatch(r'(?i)(?:CON|PRN|AUX|NUL|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])(?:\..*)?',part), 'Reserved Windows filename: '+name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe distribution path: ' + name)
    if any(s.lower() in PROHIBITED or s.lower().startswith('.venv.') or s.lower().endswith('.egg-info') for s in p.parts):
        raise ValueError('Prohibited distribution file: ' + name)
    if p.suffix.lower() in {'.pyc','.pyo','.log','.zip'} or p.name.lower() in {'pyvenv.cfg','.env','python.exe','pythonw.exe'}:
        raise ValueError('Machine-specific distribution file: ' + name)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def verify(output, expected_top=None):
    with zipfile.ZipFile(output) as z:
        require(z.testzip() is None, 'ZIP CRC failure')
        infos=z.infolist(); names=[i.filename for i in infos]
        require(bool(names), 'Empty archive')
        require(len(names)==len(set(names)), 'Duplicate ZIP names')
        require(len(names)==len({n.casefold() for n in names}), 'Case-insensitive ZIP collision')
        tops={n.split('/')[0] for n in names}
        require(len(tops)==1, 'Expected one top-level folder')
        top=next(iter(tops));safe_name(top)
        if expected_top: require(top==expected_top, 'Unexpected top-level folder')
        prefix=top+'/'
        for info in infos:
            require(not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16), 'Directory/symlink ZIP member: '+info.filename)
            require(info.filename.startswith(prefix), 'Unsafe ZIP prefix')
            safe_name(info.filename[len(prefix):])
        def unique_object(pairs):
            result={}
            for k,v in pairs:
                require(k not in result, 'Duplicate manifest key: '+k)
                result[k]=v
            return result
        manifest=json.loads(z.read(prefix+'SHA256_MANIFEST.json'),object_pairs_hook=unique_object)
        require(isinstance(manifest,dict), 'Invalid manifest')
        checked_names(list(manifest))
        require(set(names)=={prefix+n for n in manifest}|{prefix+'SHA256_MANIFEST.json'}, 'Manifest member set mismatch')
        require(REQUIRED.issubset(manifest), 'Missing required files: '+str(sorted(REQUIRED-set(manifest))))
        for name,expected in manifest.items():
            require(isinstance(expected,dict) and isinstance(expected.get('bytes'),int) and expected['bytes']>=0 and isinstance(expected.get('sha256'),str), 'Malformed manifest record: '+name)
            data=z.read(prefix+name)
            require(len(data)==expected['bytes'] and digest(data)==expected['sha256'], 'Manifest integrity mismatch: '+name)
    return dict(status='PASS',files=len(names),crc='PASS',hashes='PASS',prohibited_files=0,top_level=top)

def package(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    names = json.loads((ROOT/'distribution_files.json').read_text())
    checked_names(names)
    require(REQUIRED.issubset(names), 'Required files omitted from allowlist: '+str(sorted(REQUIRED-set(names))))
    manifest = {}
    for name in names:
        safe_name(name)
        path = ROOT/name
        require(path.resolve().is_relative_to(ROOT.resolve()) and not any(p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()) for p in [path,*path.parents] if p!=ROOT and p.is_relative_to(ROOT)), 'Symlink/junction or external file: '+name)
        require(path.is_file(), 'Missing selected file: '+name)
        data = path.read_bytes()
        if path.suffix in {'.py','.ps1','.bat','.md','.json','.txt','.toml'}:
            text = data.decode('utf-8-sig',errors='replace')
            secret = r'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'
            require(not re.search(secret,text), 'Potential secret: '+name)
        manifest[name] = dict(bytes=len(data),sha256=digest(data))
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in names:
            z.write(ROOT/name,TOP+'/'+name)
        z.writestr(TOP+'/SHA256_MANIFEST.json',json.dumps(manifest,indent=2))
    result = verify(output, expected_top=TOP)
    with output.open('rb') as stream:
        result.update(path=str(output),bytes=output.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
    print(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify: print(json.dumps(verify(a.output),indent=2))
    else: package(a.output)
