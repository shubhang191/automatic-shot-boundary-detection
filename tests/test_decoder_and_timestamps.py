import numpy as np
import pytest
from sbd_app.detectors.tinycut import TinyCutDetector
from sbd_app.detectors.autoshot import window_indices, official_scenes
from sbd_app.detectors.omnishotcut import convert_native
from sbd_app.detectors.base import import_file
from sbd_app.config import ROOT


@pytest.mark.parametrize("n", [1,2,49,50,51,99,100,101,151])
def test_autoshot_window_joins(n):
    upstream=import_file("sbd_as_utils_test",ROOT/"third_party/autoshot/utils.py")
    frames=np.arange(n).reshape(n,1,1,1).repeat(3,axis=-1)
    batches=list(upstream.get_batches(frames))
    ours=[frames[i] for i in window_indices(n)]
    for a,b in zip(batches,ours):np.testing.assert_array_equal(a,b)
    centers=np.concatenate([x[25:75] for x in ours])[:n]
    np.testing.assert_array_equal(centers,frames)


@pytest.mark.parametrize("p", [[0],[1],[0,0,0],[1,1,1],[0,1,0,0],[0,1,1,0,0,1,0],[0,0,1,1]])
def test_official_decoder(p):
    upstream=import_file("sbd_as_utils_decoder",ROOT/"third_party/autoshot/utils.py")
    assert official_scenes(p)==upstream.predictions_to_scenes(np.array(p)).tolist()


def test_frozen_tiny_decoder():
    model=TinyCutDetector()
    p=np.zeros(20,np.float32);p[4]=.95
    assert model.lib.decode(p,model.decoder)==[(3,4)]
    p[6]=.98
    assert model.lib.decode(p,model.decoder)==[(3,6)]
    assert model.lib.decode(np.zeros(20),model.decoder)==[]


def test_omni_half_open_and_transitions():
    cuts,clean=convert_native([[0,5],[5,8],[8,12]],["General","Dissolve","General"],["New_Start","Transition_Source","Transition_Target"],12)
    assert clean==[[0,4],[8,11]]
    assert [(c.a,c.b) for c in cuts]==[(4,8)]
    assert convert_native([[0,5],[5,10]],["General","General"],["New_Start","Hard_Cut"],10)[0][0].b==5
    with pytest.raises(ValueError):convert_native([[1,5]],["General"],["New_Start"],5)
