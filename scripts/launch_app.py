"""Launch only after diagnostics, with a short per-installation cache location."""
from pathlib import Path
import os,subprocess,sys
from local_cache import configure
ROOT=Path(__file__).resolve().parents[1]
def main():
    cache=configure()
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    from verify_installation import main as verify
    if verify():
        print('Installation verification failed. Run setup.bat and read installation_logs.')
        return 1
    print('Runtime cache: '+str(cache),flush=True)
    return subprocess.call([sys.executable,'-B',str(ROOT/'app.py'),*sys.argv[1:]])
if __name__=='__main__':sys.exit(main())
