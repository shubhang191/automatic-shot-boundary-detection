from pathlib import Path
import json
import numpy as np
import pytest
from sbd_app.config import ROOT
from sbd_app.detectors.base import Detection
from sbd_app.errors import ModelError
from legacy_python.integration_adapter import LegacyVerifier, load_bundle, modules, scenes_to_shots


def test_original_fitted_models_on_historical_feature_tables():
    s3,ov=modules()
    fixture=np.load(ROOT/'tests/fixtures/legacy_historical_features.npz')
    a=load_bundle('autoshot'); o=load_bundle('omnishotcut')
    assert a['tau']==.4 and a['base_thr']==.296 and a['low_thr']==.1
    assert o['tau_o']==.35 and o['tau_r']==.5 and o['rescue'] and o['snap_w']==3
    actual=s3.predict_parts(a['parts'],fixture['autoshot_tab'],fixture['autoshot_windows'])
    np.testing.assert_allclose(actual,fixture['autoshot_keep'],rtol=0,atol=1e-12)
    actual=np.mean([m.predict_proba(fixture['omni_tab'])[:,1] for m in o['models']],axis=0)
    np.testing.assert_allclose(actual,fixture['omni_keep'],rtol=0,atol=1e-12)
    assert len(a['parts'])==len(o['models'])==1
    assert len(o['feats'])==40 and len(a['parts'][0]['cols'])==43
    assert not any(f.startswith('as_') for f in o['feats'])


def test_rescue_and_snap_and_gradual_interval():
    s3,ov=modules()
    from sbd_refine import snap
    frames=np.zeros((32,27,48,3),np.uint8); frames[12:]=255
    sig=s3.frame_signals(frames.astype(np.float32)/255,np.zeros(32))
    assert [(r['a'],r['b']) for r in ov.rescue_candidates(sig,[])]==[(11,12)]
    scenes,moved=snap([[0,9],[10,31]],frames,w=3)
    assert scenes.tolist()==[[0,11],[12,31]] and moved==1
    scenes,moved=snap([[0,8],[10,31]],frames,w=3)
    assert scenes.tolist()==[[0,8],[10,31]] and moved==0
    rows=scenes_to_shots(scenes,25)
    assert rows[0]['end_frame']==8 and rows[1]['start_frame']==10
    assert rows[1]['boundary_type']=='gradual_candidate'


@pytest.mark.parametrize('detector',['autoshot','omnishotcut','omnishotcut_legacy_v15'])
def test_no_transition_single_frame(detector):
    frames=np.zeros((1,27,48,3),np.uint8)
    pred=Detection([],np.zeros(1),[[0,1]],dict(intra_labels=['General'],inter_labels=['New_Start']),{})
    cuts,decisions,detail,scenes=LegacyVerifier().verify(detector,pred,frames)
    assert not cuts and not decisions and scenes==[[0,0]]
    assert detail['bundle_sha256']


def test_unsupported_and_missing_bundle_fail_without_fallback(sample,pipeline,monkeypatch,tmp_path):
    with pytest.raises(ModelError,match='TinyCut'):
        pipeline.analyze(sample,'tinycut',verification='legacy_python')
    with pytest.raises(ValueError,match='Unknown verification'):
        pipeline.analyze(sample,'autoshot',verification='made_up_mode')
    import legacy_python.integration_adapter as adapter
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    with pytest.raises(ModelError,match='Missing or changed'):
        adapter.load_bundle('autoshot')


@pytest.mark.parametrize('detector',['autoshot','omnishotcut','omnishotcut_legacy_v15'])
def test_legacy_pipeline_exports_and_detector_cache(sample,pipeline,detector):
    import zipfile,csv
    raw=pipeline.analyze(sample,detector,verification='raw')
    count=pipeline.calls[detector]
    selected=pipeline.analyze(sample,detector,verification='legacy_python')
    assert pipeline.calls[detector]==count and selected['cache']['detector_hit']
    assert selected['raw_candidates']==raw['raw_candidates']
    assert selected['raw_shots']==raw['raw_shots']
    assert selected['mode']=='legacy_python' and selected['output_dir']!=raw['output_dir']
    with zipfile.ZipFile(selected['export_zip']) as z:
        assert z.testzip() is None
        assert json.loads(z.read('shots.json'))['shots']==selected['shots']
        assert json.loads(z.read('provenance.json'))['bundle_sha256']==selected['verifier']['bundle_sha256']
        assert json.loads(z.read('cuts.json'))==selected['decisions']
        rows=list(csv.DictReader(z.read('shots.csv').decode('utf-8-sig').splitlines()))
        assert len(rows)==len(selected['shots']) and rows[0]['verification']=='legacy_python'
        assert json.loads(rows[0]['thresholds_json'])==selected['verifier']['settings']
        assert len([n for n in z.namelist() if n.startswith('thumbnails/')])==len(rows)
    if detector!='autoshot':
        assert selected['verifier']['rescue_enabled'] and selected['verifier']['snap_enabled']
        assert 'intra_labels' in selected['detector_details']


def test_detector_specific_ui_choices():
    import importlib.util
    spec=importlib.util.spec_from_file_location('legacy_app_under_test',ROOT/'app.py')
    app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
    assert len(app.mode_choices('tinycut'))==2
    assert 'Raw + Legacy Python Verifier' in app.mode_choices('autoshot')
    assert 'Raw + Legacy Python (Rescue + Snap)' in app.mode_choices('omnishotcut')
    assert app.create_app()
