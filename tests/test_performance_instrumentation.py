from pathlib import Path
import hashlib
import json
import zipfile
import pytest
from sbd_app.errors import VideoError


def test_normalized_video_keeps_original_validation(sample, tmp_path, monkeypatch):
    import sbd_app.video_io as module
    original = module.probe
    def probe(path, include_pts=False):
        if not include_pts:
            raise VideoError('normalized metadata validation')
        return original(path, include_pts=True)
    monkeypatch.setattr(module, 'probe', probe)
    with pytest.raises(VideoError, match='normalized metadata validation'):
        module.normalize(sample, tmp_path)


def test_timing_sidecar_and_keyframe_reuse(sample, pipeline):
    first = pipeline.analyze(sample, 'tinycut', False)
    hashes = {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f, _ in first['gallery']}
    second = pipeline.analyze(sample, 'tinycut', False)
    assert second['cache']['detector_hit']
    assert second['timings']['stages'].get('thumbnail_decode', {}).get('calls', 0) == 0
    assert second['timings']['stages'].get('neural_forward', {}).get('calls', 0) == 0
    assert hashes == {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f, _ in second['gallery']}
    profile = json.loads(Path(second['timings_file']).read_text(encoding='utf-8'))
    assert profile['total_wall_s'] >= profile['stages']['pipeline_request']['wall_s'] > 0
    assert profile['annotated_video']['status'] == 'NOT_GENERATED'
    with zipfile.ZipFile(second['export_zip']) as z:
        assert 'timings.json' not in z.namelist()  # no stale or self-referential ZIP duration


def test_tool_and_code_cache_invalidation(tmp_path, monkeypatch):
    import sbd_app.pipeline as module
    from sbd_app.pipeline import Pipeline
    instance = Pipeline(tmp_path / 'cache')
    calls=[]
    original=module.run
    def run(args, **kwargs):
        calls.append(args)
        return original(args, **kwargs)
    monkeypatch.setattr(module, 'run', run)
    assert instance._tool_version() == instance._tool_version()
    assert len(calls) == 1
    code = tmp_path / 'code/src/sbd_app'
    code.mkdir(parents=True)
    source = code / 'demo.py'
    source.write_text('a=1')
    monkeypatch.setattr(module, 'ROOT', tmp_path / 'code')
    old = instance._model_identity('tinycut')
    source.write_text('a=12')
    assert instance._model_identity('tinycut') != old
