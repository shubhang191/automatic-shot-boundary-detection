"""Adversarial cache/failure checks required for the performance release audit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
import hashlib
import json
import os
import shutil
import subprocess
import cv2
import pytest
from sbd_app.config import MODEL_PATHS, executable
from sbd_app.errors import SBDError, ModelError
from sbd_app.profiling import ACTIVE
from sbd_app.video_io import run


def test_instrumentation_does_not_replace_shared_modules(pipeline):
    from sbd_app import video_io, visualizations
    assert video_io.subprocess is not subprocess
    assert visualizations.cv2 is not cv2
    assert subprocess.run.__module__ == 'subprocess'
    assert isinstance(cv2.VideoCapture, type)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([executable('ffmpeg'), '-invalid_audit_option'], capture_output=True, check=True)


def test_tool_cache_reprobes_after_binary_changes(tmp_path, monkeypatch):
    import sbd_app.pipeline as module
    fake = tmp_path / 'ffmpeg.exe'
    fake.write_bytes(b'first')
    calls = []
    monkeypatch.setattr(module, 'executable', lambda name: str(fake))
    def version(args):
        calls.append(args)
        return fake.read_bytes() + b'\n'
    monkeypatch.setattr(module, 'run', version)
    instance = module.Pipeline(tmp_path / 'cache')
    assert instance._tool_version() == instance._tool_version() == 'first'
    stamp = fake.stat()
    fake.write_bytes(b'other')
    os.utime(fake, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert instance._tool_version() == 'other'
    assert len(calls) == 2


def test_code_invalidation_when_size_and_modified_time_are_preserved(tmp_path, monkeypatch):
    import sbd_app.pipeline as module
    root = tmp_path / 'project'
    code = root / 'src/sbd_app/demo.py'
    code.parent.mkdir(parents=True)
    code.write_text('a=1', encoding='utf-8')
    monkeypatch.setattr(module, 'ROOT', root)
    instance = module.Pipeline(tmp_path / 'cache')
    original = instance._model_identity('tinycut')
    stamp = code.stat()
    code.write_text('a=2', encoding='utf-8')
    os.utime(code, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert instance._model_identity('tinycut') != original


@pytest.mark.parametrize('asset,detector,verified', [
    ('tinycut','tinycut',False), ('decoder','tinycut',False),
    ('autoshot','autoshot',False), ('omnishotcut','omnishotcut',False), ('verifier','tinycut',True)])
def test_cached_requests_still_reject_tampered_assets(sample, pipeline, tmp_path, monkeypatch, asset, detector, verified):
    original = pipeline.analyze(sample, detector, verified)
    calls = pipeline.calls[detector]
    copy = tmp_path / MODEL_PATHS[asset].name
    shutil.copy2(MODEL_PATHS[asset], copy)
    with copy.open('r+b') as stream:
        first = stream.read(1)
        stream.seek(0); stream.write(bytes([first[0] ^ 1]))
    with monkeypatch.context() as m:
        m.setitem(MODEL_PATHS, asset, copy)
        with pytest.raises(SBDError, match='SHA-256 mismatch'):
            pipeline.analyze(sample, detector, verified)
    recovered = pipeline.analyze(sample, detector, verified)
    assert recovered['cache']['detector_hit'] and pipeline.calls[detector] == calls
    assert recovered['raw_candidates'] == original['raw_candidates']


def test_same_basename_different_content_and_repeated_upload(sample, pipeline, tmp_path):
    a = tmp_path / 'A/shared.mp4'; a.parent.mkdir()
    b = tmp_path / 'B/shared.mp4'; b.parent.mkdir()
    shutil.copy2(sample, a)
    run([executable('ffmpeg'), '-v', 'error', '-y', '-i', sample, '-t', '1.2', '-an', '-c:v', 'copy', b])
    first = pipeline.analyze(a, 'tinycut', False)
    second = pipeline.analyze(b, 'tinycut', False)
    repeat = pipeline.analyze(a, 'tinycut', True)
    assert first['metadata']['source_video_sha256'] != second['metadata']['source_video_sha256']
    assert Path(first['output_dir']).parent != Path(second['output_dir']).parent
    assert first['metadata']['decoded_frame_count'] != second['metadata']['decoded_frame_count']
    assert repeat['cache']['video_hit'] and repeat['cache']['detector_hit']
    assert repeat['raw_candidates'] == first['raw_candidates']
    assert repeat['metadata']['frames48_sha256'] == first['metadata']['frames48_sha256']
    assert json.loads((Path(second['output_dir']) / 'shots.json').read_text())['metadata']['source_video_sha256'] == hashlib.sha256(b.read_bytes()).hexdigest()


def test_failed_decode_then_same_upload_path_recovers(sample, pipeline, tmp_path):
    path = tmp_path / 'replaced.mp4'
    path.write_bytes(b'broken upload')
    with pytest.raises(SBDError): pipeline.analyze(path, 'tinycut', False)
    assert ACTIVE.get() is None
    shutil.copy2(sample, path)
    result = pipeline.analyze(path, 'tinycut', False)
    assert result['metadata']['source_video_sha256'] == hashlib.sha256(sample.read_bytes()).hexdigest()


@pytest.mark.parametrize('operation', ['render', 'write_export', 'timing_sidecar'])
def test_failed_output_request_recovers_without_reinference(sample, pipeline, monkeypatch, operation):
    import sbd_app.pipeline as module
    original = pipeline.analyze(sample, 'tinycut', False)
    calls = pipeline.calls['tinycut']
    def fail(*a, **kw): raise OSError('injected output failure')
    with monkeypatch.context() as m:
        if operation == 'timing_sidecar':
            real = module.save_json
            def save(path, data):
                if Path(path).name == 'timings.json': fail()
                return real(path, data)
            m.setattr(module, 'save_json', save)
        else: m.setattr(module, operation, fail)
        with pytest.raises(ModelError, match='injected output failure'):
            pipeline.analyze(sample, 'tinycut', False)
    assert ACTIVE.get() is None
    recovered = pipeline.analyze(sample, 'tinycut', False)
    assert pipeline.calls['tinycut'] == calls
    assert recovered['shots'] == original['shots']


def test_verifier_is_recomputed_for_each_verified_request(sample, pipeline, monkeypatch):
    raw = pipeline.analyze(sample, 'tinycut', False)
    pipeline.analyze(sample, 'tinycut', True)
    original = pipeline.verifier.verify
    calls = []
    def verify(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(pipeline.verifier, 'verify', verify)
    selected = [pipeline.analyze(sample, 'tinycut', True) for _ in range(2)]
    assert len(calls) == 2
    assert all(r['cache']['detector_hit'] and r['raw_candidates'] == raw['raw_candidates'] for r in selected)
    assert selected[0]['decisions'] == selected[1]['decisions']
    assert Path(raw['output_dir']) != Path(selected[0]['output_dir'])


def test_sidecar_write_preserves_original_request_serialization(sample, pipeline, monkeypatch):
    import sbd_app.pipeline as module
    entered, finish = Event(), Event()
    real = module.save_json
    def save(path, data):
        if Path(path).name == 'timings.json':
            entered.set()
            assert finish.wait(20), 'Test did not release sidecar writer'
        return real(path, data)
    monkeypatch.setattr(module, 'save_json', save)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(pipeline.analyze, sample, 'tinycut', False)
        try:
            assert entered.wait(20)
            acquired = pipeline.lock.acquire(blocking=False)
            if acquired: pipeline.lock.release()
            assert not acquired, 'Request lock released before timing sidecar was complete'
        finally: finish.set()
        assert task.result(timeout=20)['shots']
