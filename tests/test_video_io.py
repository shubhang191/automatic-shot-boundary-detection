from pathlib import Path
import shutil
import numpy as np
import pytest
from sbd_app.config import executable, MAX_BYTES
from sbd_app.video_io import run, normalize, decode_rgb, probe, validate_upload
from sbd_app.errors import VideoError, SBDError


def test_spaces_and_unicode(sample,tmp_path):
    dest=tmp_path/'video spaces नमस्ते.mp4';shutil.copy2(sample,dest)
    path,frames,meta=normalize(dest,tmp_path)
    assert frames.dtype==np.uint8 and frames.shape[1:]==(27,48,3)
    assert len(frames)==meta['decoded_frame_count']
    assert len(meta['analysis_to_source_frame'])==len(frames)
    assert meta['cfr_transcoded'] and meta['fps_rational']=='25'
    assert meta['analysis_to_source_frame']==list(range(len(frames)))


def test_vfr_mapping(tmp_path):
    frames=np.stack([np.full((54,96,3),20+i*15,np.uint8) for i in range(10)])
    src=tmp_path/'variable.mov'
    import subprocess
    p=subprocess.run([executable('ffmpeg'),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','96x54','-r','10','-i','-',
                      '-vf','setpts=(N+floor(N/2))/(10*TB)','-fps_mode','vfr','-c:v','libx264','-crf','0',str(src)],input=frames.tobytes(),capture_output=True)
    assert p.returncode==0,p.stderr
    output,decoded,meta=normalize(src,tmp_path)
    assert len(set(round(b-a,3) for a,b in zip(meta['source_pts_seconds'],meta['source_pts_seconds'][1:])))>1
    source=decode_rgb(src)
    # Marked source frames prove the exported correspondence matches actual CFR selections.
    for f,idx in zip(decoded,meta['analysis_to_source_frame']):
        assert abs(float(f.mean())-float(source[idx].mean()))<4


@pytest.mark.parametrize('payload',[b'',b'not a video'])
def test_corrupt(tmp_path,payload):
    path=tmp_path/'bad.mp4';path.write_bytes(payload)
    with pytest.raises(VideoError):normalize(path,tmp_path)


def test_missing_ffmpeg(monkeypatch):
    monkeypatch.setenv('SBD_FFMPEG','nonexistent_ffmpeg.exe')
    with pytest.raises(SBDError,match='missing'):executable('ffmpeg')


@pytest.mark.parametrize('extension',['mov','avi'])
def test_supported_containers(sample,tmp_path,extension):
    src=tmp_path/('container.'+extension)
    run([executable('ffmpeg'),'-v','error','-y','-i',sample,'-an','-c:v','mpeg4','-q:v','3',src])
    _,frames,meta=normalize(src,tmp_path)
    assert len(frames)==150 and meta['fps_rational']=='25'
