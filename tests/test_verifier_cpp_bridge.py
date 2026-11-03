import json
import subprocess
import numpy as np
import pytest
from sbd_app.config import verified_path
from sbd_app.video_io import decode_rgb
from sbd_app.verifier.sbdv2_adapter import SBDV2Verifier
from sbd_app.verifier.build_status import binary_path
from sbd_app.shot_schema import CandidateCut


@pytest.mark.parametrize("seed", [0,1,2,42])
def test_cpp_api_ram_stream_identical(seed,tmp_path):
    rng=np.random.default_rng(seed)
    frames=rng.integers(0,256,(160,27,48,3),np.uint8)
    frames[60:90]=frames[59]
    cuts=[CandidateCut(19,20),CandidateCut(59,60),CandidateCut(89,92),CandidateCut(139,140)]
    bridge=SBDV2Verifier()
    _,decisions,_=bridge.verify(frames,cuts)
    raw=tmp_path/'RGB frames.u8';frames.tofile(raw)
    proposals=tmp_path/'candidate cuts.txt';proposals.write_text('\n'.join(f'{c.a} {c.b}' for c in cuts))
    for mode in ['--ram','--stream']:
        result=json.loads(subprocess.check_output([str(binary_path()),'verify','--model',str(verified_path('verifier')),'--frames',str(raw),'--cuts',str(proposals),mode]))
        assert result['frames']==len(frames)
        assert [(d['a'],d['b'],d['status']=='KEEP') for d in decisions]==[(d['a'],d['b'],d['kept']) for d in result['candidates']]
        np.testing.assert_array_equal([d['p_keep'] for d in decisions],[d['p_keep'] for d in result['candidates']])


def test_real_video_no_new_candidates(sample,pipeline):
    result=pipeline.analyze(sample,'tinycut',True)
    assert {(d['a'],d['b']) for d in result['decisions']}=={(c['a'],c['b']) for c in result['raw_candidates']}
    assert result['verifier']['model']['trees']==60
    assert result['verifier']['model']['nodes']==1632


def test_strict_invalid():
    frames=np.zeros((10,27,48,3),np.uint8)
    model=SBDV2Verifier().model
    with pytest.raises(ValueError,match='outside'):model.verify(frames,cuts=[(9,10)])
    with pytest.raises(ValueError):SBDV2Verifier().verify(frames,[CandidateCut(0,10)])


def test_no_candidates():
    frames=np.zeros((10,27,48,3),np.uint8)
    kept,decisions,_=SBDV2Verifier().verify(frames,[])
    assert kept==[] and decisions==[]
