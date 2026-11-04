"""Small native ABI/model/operation self-test; not neural inference or an accuracy test."""
from pathlib import Path
import json,os,subprocess,tempfile
import numpy as np

def check(verifier):
    from sbd_app.config import ROOT,verified_path
    from sbd_app.verifier.build_status import binary_path
    from sbd_app.shot_schema import CandidateCut
    if verifier.model.L.sbdv_n_features()!=57:
        raise RuntimeError('Native feature API mismatch: expected 57-entry engine feature catalog (frozen model uses 30)')
    frames=np.zeros((24,27,48,3),dtype=np.uint8);frames[12:]=255
    cuts=[CandidateCut(11,12)]
    _,first,_=verifier.verify(frames,cuts)
    _,second,_=verifier.verify(frames,cuts)
    if first!=second or len(first)!=1 or not np.isfinite(first[0]['p_keep']) or not 0<=first[0]['p_keep']<=1:
        raise RuntimeError('Native verification self-test produced invalid/nondeterministic decisions')
    cache=Path(os.environ.get('SBD_CACHE',str(ROOT/'.cache')));cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-',dir=cache) as td:
        td=Path(td);frames.tofile(td/'frames.u8');(td/'cuts.txt').write_text('11 12\n')
        cli=json.loads(subprocess.check_output([str(binary_path()),'verify','--model',str(verified_path('verifier')),'--frames',str(td/'frames.u8'),'--cuts',str(td/'cuts.txt'),'--ram'],timeout=30))
    expected=[(d['a'],d['b'],d['p_keep'],d['status']=='KEEP') for d in first]
    actual=[(d['a'],d['b'],d['p_keep'],d['kept']) for d in cli['candidates']]
    if expected!=actual:raise RuntimeError('Native DLL/CLI self-test decision mismatch')
    return dict(status='PASS',engine_feature_catalog_count=57,decisions=first,DLL_CLI_equal=True)
