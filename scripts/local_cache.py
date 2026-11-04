"""Keep generated filenames below Windows path limits without changing cache keys."""
from pathlib import Path
import hashlib,os,tempfile
ROOT=Path(__file__).resolve().parents[1]
def configure():
    if not os.environ.get('SBD_CACHE'):
        base=Path(os.environ.get('LOCALAPPDATA') or tempfile.gettempdir())
        install=hashlib.sha256(str(ROOT).casefold().encode('utf-8')).hexdigest()[:12]
        os.environ['SBD_CACHE']=str(base/'SBD'/'c'/install)
    return Path(os.environ['SBD_CACHE'])
if __name__=='__main__':print(configure())
