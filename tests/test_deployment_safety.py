"""Deployment-only regressions; never replace or mock detector predictions."""
import hashlib,importlib.util,json,sys,zipfile,stat,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import package_portable as p
from build_verifier import select_artifact,validate_windows_binary

@pytest.mark.parametrize('name',['../a','/a','a//b','a/./b','a/../b','C:/a','a\\b','x.','x ','CON.txt','LPT1','a?b','a\x00b','.VENV/x','x/.venv.invalid-1/pyvenv.cfg','x/__pycache__/a.pyc'])
def test_invalid_path(name):
    with pytest.raises(ValueError):p.safe_name(name)

def fixture_zip(path,files,manifest=None):
    if manifest is None:manifest={n:dict(bytes=len(d),sha256=hashlib.sha256(d).hexdigest()) for n,d in files}
    with zipfile.ZipFile(path,'w') as z:
        for n,d in files:z.writestr('209_ShubhangShrivats/'+n,d)
        z.writestr('209_ShubhangShrivats/SHA256_MANIFEST.json',json.dumps(manifest))

def test_duplicate_and_case_collision(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'REQUIRED',{'app.py'})
    for files in [[('app.py',b'a'),('app.py',b'b')],[('app.py',b'a'),('APP.py',b'a')]]:
        path=tmp_path/'x.zip';fixture_zip(path,files)
        with pytest.raises(ValueError):p.verify(path)

def test_manifest_tamper_missing_and_symlink(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'REQUIRED',{'app.py'})
    path=tmp_path/'x.zip';fixture_zip(path,[('app.py',b'x')],{'app.py':{'bytes':1,'sha256':'0'*64}})
    with pytest.raises(ValueError):p.verify(path)
    fixture_zip(path,[('other.py',b'x')])
    with pytest.raises(ValueError):p.verify(path)
    info=zipfile.ZipInfo('209_ShubhangShrivats/app.py');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(path,'w') as z:z.writestr(info,b'target')
    with pytest.raises(ValueError):p.verify(path)

def test_output_preserved_and_optimized_safety(tmp_path):
    path=tmp_path/'existing.zip';path.write_bytes(b'keep')
    with pytest.raises(FileExistsError):p.package(path)
    assert path.read_bytes()==b'keep'
    code='import sys;sys.path.insert(0,sys.argv[1]);import package_portable as p;p.safe_name("../unsafe")'
    run=subprocess.run([sys.executable,'-O','-B','-c',code,str(ROOT/'scripts')],capture_output=True)
    assert run.returncode!=0 and b'ValueError' in run.stderr

def test_allowlist_case_and_parent_symlink(tmp_path,monkeypatch):
    with pytest.raises(ValueError):p.checked_names(['app.py','APP.py'])
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'REQUIRED',{'app.py'})
    (tmp_path/'distribution_files.json').write_text('["app.py"]')
    with pytest.raises(ValueError,match='Missing selected'):p.package(tmp_path/'out.zip')

def test_build_discovery_never_uses_import_lib_or_stale_subdir(tmp_path):
    (tmp_path/'old').mkdir();(tmp_path/'old/sbdv.dll').write_bytes(b'stale');(tmp_path/'libsbdv.dll.a').write_bytes(b'import')
    with pytest.raises(RuntimeError,match='fresh build output'):select_artifact(tmp_path,('sbdv.dll','libsbdv.dll'))
    (tmp_path/'libsbdv.dll').write_bytes(b'dll')
    assert select_artifact(tmp_path,('sbdv.dll','libsbdv.dll')).name=='libsbdv.dll'
    (tmp_path/'sbdv.dll').write_bytes(b'dll')
    with pytest.raises(RuntimeError):select_artifact(tmp_path,('sbdv.dll','libsbdv.dll'))
    with pytest.raises(RuntimeError):validate_windows_binary(tmp_path/'sbdv.dll')
