import pytest


@pytest.mark.parametrize('detector',['tinycut','autoshot','omnishotcut'])
def test_real_raw_verified(sample,pipeline,detector):
    raw=pipeline.analyze(sample,detector,False)
    n=raw['metadata']['decoded_frame_count']
    verified=pipeline.analyze(sample,detector,True)
    assert verified['cache']['detector_hit']
    assert verified['raw_candidates']==raw['raw_candidates']
    assert verified['metadata']['frames48_sha256']==raw['metadata']['frames48_sha256']
    assert len(verified['shots'])<=len(raw['shots'])
    assert verified['shots'][0]['start_frame']==0
    assert verified['shots'][-1]['end_frame']==n-1
    assert n>0 and len(raw['shots'])>0
