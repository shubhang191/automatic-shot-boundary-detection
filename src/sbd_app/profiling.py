"""Wall and process-CPU spans; native math and subprocess arguments are unchanged."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import time

ACTIVE = ContextVar('sbd_profile', default=None)


class LocalModule:
    """Override instrumentation dependencies locally, never the shared library module."""
    def __init__(self, original, **overrides):
        self._original = original
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._original, name)


class Profile:
    def __init__(self):
        self.roots = []
        self.stack = []

    @contextmanager
    def span(self, name):
        node = dict(name=name, children=[], wall_s=0., cpu_s=0., child_process_cpu_s=0.)
        (self.stack[-1]['children'] if self.stack else self.roots).append(node)
        self.stack.append(node)
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield node
        finally:
            node['wall_s'] = time.perf_counter() - wall
            node['cpu_s'] = time.process_time() - cpu
            self.stack.pop()

    def summary(self):
        totals = {}
        def walk(node):
            row = totals.setdefault(node['name'], dict(calls=0, wall_s=0., cpu_s=0., exclusive_wall_s=0., exclusive_cpu_s=0., child_process_cpu_s=0.))
            row['calls'] += 1
            for key in ['wall_s', 'cpu_s', 'child_process_cpu_s']: row[key] += node[key]
            row['exclusive_wall_s'] += max(0., node['wall_s'] - sum(c['wall_s'] for c in node['children']))
            row['exclusive_cpu_s'] += max(0., node['cpu_s'] - sum(c['cpu_s'] for c in node['children']))
            for child in node['children']: walk(child)
        for node in self.roots: walk(node)
        return totals


@contextmanager
def span(name):
    profile = ACTIVE.get()
    if profile is None: yield None
    else:
        with profile.span(name) as node: yield node


def timed(name, fn):
    if getattr(fn, '_sbd_profiled', False): return fn
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with span(name): return fn(*args, **kwargs)
    wrapped._sbd_profiled = True
    return wrapped


def windows_process_cpu(proc):
    """Read retained child-process HANDLE after communicate (Windows process already exited)."""
    import os
    if os.name != 'nt': return 0.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_times = kernel.GetProcessTimes
    get_times.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    get_times.restype = wintypes.BOOL
    times = [wintypes.FILETIME() for _ in range(4)]
    if not get_times(int(proc._handle), *(ctypes.byref(x) for x in times)): return 0.
    return sum((t.dwHighDateTime << 32) + t.dwLowDateTime for t in times[2:]) / 1e7


def install():
    """One-time, transparent timing hooks around existing application operations."""
    from . import pipeline as p, config, video_io as v, exports as e, visualizations as vis, cache
    if getattr(p.Pipeline, '_sbd_profiled', False): return
    p.Pipeline._sbd_profiled = True
    config.verified_path = timed('checkpoint_integrity', config.verified_path)
    p.verified_path = config.verified_path
    original_hash = p.sha256
    def hashed(path):
        with span('code_hash' if Path(path).suffix == '.py' else 'input_hash'):
            return original_hash(path)
    p.sha256 = hashed
    p.importlib.metadata.version = timed('package_version_lookup', p.importlib.metadata.version)
    p.identity = timed('identity_json', p.identity)
    cache.DiskCache.prune = timed('cache_eviction', cache.DiskCache.prune)
    p.normalize = timed('normalize_total', p.normalize)
    original_probe = v.probe
    @wraps(original_probe)
    def probe(path, include_pts=False):
        with span('source_pts_probe' if include_pts else 'analysis_header_probe'):
            return original_probe(path, include_pts=include_pts)
    v.probe = probe
    v.decode_rgb = timed('decode_resize', v.decode_rgb)
    # A retained Popen handle lets us measure actual FFmpeg/FFprobe CPU, not just parent wait.
    original_subprocess_run = v.subprocess.run
    def subprocess_run(args, **kwargs):
        if ACTIVE.get() is None or not args or Path(str(args[0])).stem.lower() not in {'ffmpeg','ffprobe'}:
            return original_subprocess_run(args, **kwargs)
        import subprocess
        timeout = kwargs.pop('timeout', None)
        capture = kwargs.pop('capture_output', False)
        if capture: kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with span('ffprobe_process' if Path(str(args[0])).stem.lower() == 'ffprobe' else 'ffmpeg_process') as node:
            proc = subprocess.Popen(args, **kwargs)
            try: stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.communicate(); raise
            if node is not None: node['child_process_cpu_s'] = windows_process_cpu(proc)
            return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
    v.subprocess = LocalModule(v.subprocess, run=subprocess_run)
    original_detector = p.Pipeline.detector
    def detector(self, name):
        with span('model_load_initialize'):
            model = original_detector(self, name)
        if not getattr(model, '_sbd_profiled', False):
            model._sbd_profiled = True
            model.predict = timed('detector_total', model.predict)
            if name == 'tinycut':
                model.features = timed('feature_extraction', model.features)
                lib = model.lib
                original_net = lib._net_of
                def net_of(*a, **kw):
                    with span('tiny_network_initialize'):
                        net = original_net(*a, **kw)
                    net.forward = timed('neural_forward', net.forward)
                    return net
                lib._net_of = net_of
                lib.member_probs = timed('detector_standardize_forward', lib.member_probs)
                lib.decode = timed('boundary_decoder', lib.decode)
            else:
                model.model.forward = timed('neural_forward', model.model.forward)
                if name == 'omnishotcut':
                    import omnishotcut.engine as engine
                    engine.video_transform = timed('feature_extraction', engine.video_transform)
        return model
    p.Pipeline.detector = detector
    from .verifier.sbdv2_adapter import SBDV2Verifier
    original_init = SBDV2Verifier.__init__
    def verifier_init(self):
        with span('verifier_load_initialize'):
            original_init(self)
            self.model._call = timed('verifier_prepare_transfer', self.model._call)
            self.model.L.sbdv_verify = timed('verifier_native_call', self.model.L.sbdv_verify)
    SBDV2Verifier.__init__ = verifier_init
    SBDV2Verifier.verify = timed('verifier_total', SBDV2Verifier.verify)
    original_capture = vis.cv2.VideoCapture
    class Capture:
        def __init__(self, *args, **kwargs): self.cap = original_capture(*args, **kwargs)
        def __getattr__(self, name): return getattr(self.cap, name)
        def set(self, *a, **kw):
            with span('thumbnail_seek'): return self.cap.set(*a, **kw)
        def read(self, *a, **kw):
            with span('thumbnail_decode'): return self.cap.read(*a, **kw)
    vis.cv2 = LocalModule(vis.cv2, VideoCapture=Capture)
    p.render = timed('render_total', p.render)
    original_save = vis.Image.Image.save
    def image_save(self, fp, *args, **kwargs):
        with span('timeline_png' if str(fp).endswith('.png') else 'thumbnail_jpeg'):
            return original_save(self, fp, *args, **kwargs)
    vis.Image.Image.save = image_save
    p.write_export = timed('export_total', p.write_export)
    e.csv_write = timed('export_csv', e.csv_write)
    e.save_json = timed('export_json', e.save_json)
    p.save_json = timed('cache_json_write', p.save_json)
    e.zipfile.ZipFile.write = timed('zip_compression', e.zipfile.ZipFile.write)
    e.zipfile.ZipFile.testzip = timed('zip_crc', e.zipfile.ZipFile.testzip)
