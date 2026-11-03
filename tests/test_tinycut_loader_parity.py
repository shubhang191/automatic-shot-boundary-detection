import json
import numpy as np
import pytest
from sbd_app.config import ROOT, KNOWN_HASHES, sha256, MODEL_PATHS
from sbd_app.detectors.tinycut import TinyCutDetector
from sbd_app.errors import SBDError, ModelError


def test_canonical_notebook39_fixture():
    fixture=ROOT/"tests/fixtures"
    with np.load(fixture/"tinycut_reference.npz",allow_pickle=False) as z:
        frames=z["frames"]; expected=z["probabilities"]
    model=TinyCutDetector()
    # Original helper computes all 35; adapter skips only unused CLAHE columns.
    features35=model.lib.all_features(frames)
    np.testing.assert_array_equal(model.features(frames),features35[:,:31])
    canonical=model.lib.member_probs(model.member,features35,dev="cpu")
    np.testing.assert_array_equal(canonical,expected)
    result=model.predict(frames)
    np.testing.assert_array_equal(result.probabilities,expected)
    manifest=json.loads((fixture/"tinycut_reference.json").read_text())
    assert [[c.a,c.b] for c in result.cuts]==manifest["boundaries"]
    assert model.lib.n_weights(model.member)==18433
    assert model.member["cols"]==list(range(31))


def test_wrong_checkpoint_before_unpickle(monkeypatch,tmp_path):
    path=tmp_path/"wrong.pkl";path.write_bytes(b"not a trusted pickle")
    monkeypatch.setitem(MODEL_PATHS,"tinycut",path)
    with pytest.raises(SBDError,match="SHA-256"):TinyCutDetector()


def test_wrong_decoder(monkeypatch,tmp_path):
    path=tmp_path/"decoder.json";path.write_text('{}')
    monkeypatch.setitem(MODEL_PATHS,"decoder",path)
    with pytest.raises(SBDError,match="SHA-256"):TinyCutDetector()


def test_wrong_frame_shape():
    with pytest.raises(ModelError):TinyCutDetector().features(np.zeros((3,27,47,3),np.uint8))
