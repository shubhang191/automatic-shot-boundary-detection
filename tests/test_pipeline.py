import pytest
from sbd_app.config import MODEL_PATHS
from sbd_app.errors import SBDError
from sbd_app.cache import DiskCache


def test_cache_toggle_and_input_isolation(sample,pipeline,tmp_path):
    first=pipeline.analyze(sample,'tinycut',False)
    calls=pipeline.calls['tinycut']
    second=pipeline.analyze(sample,'tinycut',True)
    assert pipeline.calls['tinycut']==calls
    assert second['cache']['detector_hit']
    assert first['raw_candidates']==second['raw_candidates']


def test_missing_model_isolated(sample,pipeline,monkeypatch,tmp_path):
    monkeypatch.setitem(MODEL_PATHS,'omnishotcut',tmp_path/'missing.pth')
    with pytest.raises(SBDError,match='Missing'):pipeline.analyze(sample,'omnishotcut',False)
    assert pipeline.analyze(sample,'tinycut',False)['shots']


def test_cache_eviction(tmp_path):
    cache=DiskCache(tmp_path/'cache')
    entries=[cache.directory(str(i)*64) for i in range(6)]
    outside=tmp_path/'keep.txt';outside.write_text('preserve')
    cache.prune(entries[-1])
    assert len(list(cache.root.iterdir()))==4 and outside.read_text()=='preserve'
    with pytest.raises(ValueError):cache.directory('../outside')


def test_corrupt_requests_still_evict(tmp_path):
    from sbd_app.pipeline import Pipeline
    instance=Pipeline(tmp_path/'failed cache')
    for i in range(6):
        video=tmp_path/f'bad_{i}.mp4';video.write_bytes(f'invalid movie {i}'.encode())
        with pytest.raises(SBDError):instance.analyze(video,'tinycut',False)
    assert len(list(instance.cache.root.iterdir()))==4
