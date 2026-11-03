"""Installation diagnostics, distinct from real video inference tests."""
from pathlib import Path
import importlib,json,os,sys,tempfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from interpreter_probe import check
from local_cache import configure

def main():
    configure()
    rows=[]
    def record(name,action):
        try: rows.append(dict(check=name,status='PASS',detail=action()))
        except Exception as exc: rows.append(dict(check=name,status='FAIL',detail=str(exc)))
    def interpreter():
        result=check(True)
        if result['status']!='PASS':raise RuntimeError('; '.join(result['errors']))
        return result
    record('Windows / Python 3.13 x64 / local environment',interpreter)
    for module in ['torch','torchvision','numpy','scipy','cv2','PIL','gradio','sklearn','joblib','einops','matplotlib','ffmpeg']:
        record('Import '+module,lambda m=module:str(importlib.import_module(m).__file__))
    def release():
        from validate_environment import validate
        result=validate()
        if result['status']!='PASS':raise RuntimeError(json.dumps(result))
        return result
    record('Checkpoint hashes / Python bundles / native verifier / FFmpeg',release)
    def runtime():
        if not (ROOT/'app.py').is_file():raise RuntimeError('Missing app.py')
        cache=Path(os.environ.get('SBD_CACHE',str(ROOT/'.cache')))
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as td:(Path(td)/'write-test').write_text('ok')
        return str(cache.resolve())
    record('Entry point and writable runtime cache',runtime)
    rows.append(dict(check='Video inference',status='NOT TESTED',detail='Process a real video or run scripts/run_legacy_smoke_test.py'))
    print(json.dumps(rows,indent=2))
    return int(any(r['status']=='FAIL' for r in rows))
if __name__=='__main__':sys.exit(main())
