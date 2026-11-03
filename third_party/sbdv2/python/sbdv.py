"""
sbdv.py - Python wrapper of the SBD-Verifier (a thin layer over libsbdv.so and the sbdv command line tool; the real work happens in C++).

    import sbdv
    m = sbdv.Model("models/generic.sbdm")

    # (1) you already have the 48x27 frames in memory (e.g. next to your detector)
    frames = sbdv.decode_frames("clip.mp4")                  # (T, 27, 48, 3) uint8
    r = m.verify_scenes(frames, shots)                       # the detector's shot list [[start, end], ...] (inclusive, 0-based)
    r = m.verify(frames, cuts=[(120, 121), (400, 403)])      # or a list of cuts (a = last frame before, b = first frame after)
    r = m.verify(frames, probs=detector_probs)               # or per-frame probabilities (AutoShot-style models)
    r["scenes"]   # final shots, inclusive [start, end]
    r["cands"]    # one row per candidate: a, b, src, p_keep, kept   (p_keep = -1: skipped, see lenient)

    # (2) you only have the video file: the tool decodes it itself and needs constant memory, however long the video is
    r = sbdv.verify_video("models/generic.sbdm", "clip.mp4", scenes=shots)
"""
import ctypes, json, os, shutil, subprocess, sys, tempfile
import numpy as np

H, W = 27, 48
SRC = ["base", "low", "detector", "rescue", "user"]
_HERE = os.path.dirname(os.path.abspath(__file__))


def _search_dirs():
    return (_HERE, os.path.join(_HERE, "..", "bin"), os.path.join(_HERE, "bin"), os.getcwd(), os.path.join(os.getcwd(), "bin"))


def find_lib(path=None):
    cands = [path] if path else ([os.environ["SBDV_LIB"]] if os.environ.get("SBDV_LIB") else [])
    for d in _search_dirs():
        for n in ("libsbdv.so", "libsbdv.dylib", "sbdv.dll"): cands.append(os.path.join(d, n))
    for c in cands:
        if c and os.path.exists(c): return os.path.abspath(c)
    raise FileNotFoundError("libsbdv.so not found - build it first (see README: ./build.sh)")


def find_binary(path=None):
    cands = [path] if path else ([os.environ["SBDV_BIN"]] if os.environ.get("SBDV_BIN") else [])
    for d in _search_dirs(): cands.append(os.path.join(d, "sbdv.exe" if os.name == "nt" else "sbdv"))
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK): return os.path.abspath(c)
    w = shutil.which("sbdv")
    if w: return w
    raise FileNotFoundError("the sbdv program was not found - build it first (see README: ./build.sh)")


_LIB = None
def _lib(path=None):
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(find_lib(path)); P = ctypes.c_void_p
        L.sbdv_version.restype = ctypes.c_char_p
        L.sbdv_load.restype = P; L.sbdv_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        L.sbdv_free.argtypes = [P]; L.sbdv_input_kind.argtypes = [P]; L.sbdv_get.restype = ctypes.c_double; L.sbdv_get.argtypes = [P, ctypes.c_char_p]
        L.sbdv_set_tau.argtypes = [P, ctypes.c_double, ctypes.c_double]; L.sbdv_flops.restype = ctypes.c_double; L.sbdv_flops.argtypes = [P, ctypes.c_int, ctypes.c_int]
        L.sbdv_set_option.restype = ctypes.c_int; L.sbdv_set_option.argtypes = [P, ctypes.c_char_p, ctypes.c_double]
        L.sbdv_feature_name.restype = ctypes.c_char_p; L.sbdv_n_features.restype = ctypes.c_int
        L.sbdv_verify.restype = ctypes.c_int; L.sbdv_features.restype = ctypes.c_int
        # Integration ABI fix: untyped Python integer handles truncate on 64-bit ctypes.
        I = ctypes.c_int; F = ctypes.POINTER(ctypes.c_float); A = ctypes.POINTER(ctypes.c_int32)
        U = ctypes.POINTER(ctypes.c_uint8); D = ctypes.POINTER(ctypes.c_double); S = ctypes.c_char_p
        L.sbdv_verify.argtypes = [P,U,I,F,I,A,F,I,A,I,S,A,D,U,A,I,A,I,ctypes.POINTER(I),D,S,I]
        L.sbdv_features.argtypes = [P,U,I,F,I,A,F,I,A,I,S,A,A,F,I,S,I]
        _LIB = L
    return _LIB


def version():
    return _lib().sbdv_version().decode()


_FPS_FLAG = None
def _ffmpeg_cmd(path, max_frames=None, cfr=None):
    """the same ffmpeg command the sbdv program uses: frame i = i-th decoded frame (no duplicated / dropped frames), 48x27 RGB"""
    global _FPS_FLAG
    flag = _FPS_FLAG or ["-vsync", "0"]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path] + (["-vf", f"fps={cfr}"] if cfr else []) + ["-an", "-sn", "-dn"] + ([] if cfr else flag)
    return cmd + (["-frames:v", str(int(max_frames))] if max_frames else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-"]


def decode_frames(path, max_frames=None, cfr=None):
    """any video -> (T, 27, 48, 3) uint8 with ffmpeg (the default shrink, like the frames AutoShot reads).  cfr=FPS decodes at a constant frame rate."""
    global _FPS_FLAG
    r = subprocess.run(_ffmpeg_cmd(path, max_frames, cfr), capture_output=True)
    if (r.returncode != 0 or not r.stdout) and not cfr and _FPS_FLAG is None:      # very new ffmpeg may have dropped -vsync: try -fps_mode
        _FPS_FLAG = ["-fps_mode", "passthrough"]; r2 = subprocess.run(_ffmpeg_cmd(path, max_frames, cfr), capture_output=True)
        if r2.returncode == 0 and r2.stdout: r = r2
        else: _FPS_FLAG = None
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("ffmpeg could not decode " + path + ": " + r.stderr.decode(errors="replace")[-300:])
    n = len(r.stdout) // (H * W * 3)
    return np.frombuffer(r.stdout, np.uint8, n * H * W * 3).reshape(n, H, W, 3)


def scenes_to_cuts(scenes):
    """a detector's shot list [[start, end], ...] (inclusive) -> cuts [[last frame of shot i, first frame of shot i+1], ...]"""
    s = np.asarray(scenes, np.int64).reshape(-1, 2)
    return np.stack([s[:-1, 1], s[1:, 0]], 1) if len(s) > 1 else np.zeros((0, 2), np.int64)


class Model:
    def __init__(self, path, lib=None):
        self.L = _lib(lib); err = ctypes.create_string_buffer(512)
        self.h = self.L.sbdv_load(os.fspath(path).encode(), err, 512)
        if not self.h: raise RuntimeError(err.value.decode())
        self.path = os.fspath(path); self.input_kind = self.L.sbdv_input_kind(self.h)

    def __del__(self):
        try:
            if getattr(self, "h", None): self.L.sbdv_free(self.h)
        except Exception: pass

    def info(self):
        g = lambda k: self.L.sbdv_get(self.h, k.encode())
        return dict(input="probs" if self.input_kind == 0 else "cuts", tau=g("tau"), tau_rescue=g("tau_rescue"), base_thr=g("base_thr"), low_thr=g("low_thr"), params=int(g("params")),
                    trees=int(g("trees")), nodes=int(g("nodes")), nets=int(g("nets")), bytes=int(g("bytes")), rescue=bool(g("rescue")), dedupe=int(g("dedupe")), snap=int(g("snap")),
                    streamable=bool(g("streamable")))

    def set_tau(self, tau, tau_rescue=None):
        self.L.sbdv_set_tau(self.h, float(tau), float(tau if tau_rescue is None else tau_rescue))

    def flops(self, T, ncand):
        """arithmetic operations of one run, ESTIMATED by counting the loops of the code (not measured by a profiler)"""
        return self.L.sbdv_flops(self.h, int(T), int(ncand))

    def _call(self, frames, probs, cuts, scores, ranges, intra, inter):
        frames = np.ascontiguousarray(frames, np.uint8)
        if frames.ndim != 4 or frames.shape[1:] != (H, W, 3): raise ValueError("frames must be uint8 [T, 27, 48, 3]")
        T = len(frames); p = None if probs is None else np.ascontiguousarray(probs, np.float32)
        c = None if cuts is None else np.ascontiguousarray(np.asarray(cuts).reshape(-1, 2), np.int32)
        sc = None if scores is None else np.ascontiguousarray(scores, np.float32)
        if sc is not None and c is not None and len(sc) != len(c): raise ValueError("scores must have one number per cut")
        r = None if ranges is None else np.ascontiguousarray(np.asarray(ranges).reshape(-1, 2), np.int32)
        labels = None
        if r is not None:
            ia = intra if intra is not None else ["General"] * len(r); ib = inter if inter is not None else [""] * len(r)
            labels = ("\n".join(f"{x}\t{y}" for x, y in zip(ia, ib)) + "\n").encode()
        ptr = lambda a, t: None if a is None else a.ctypes.data_as(ctypes.POINTER(t))
        return frames, T, p, c, sc, r, labels, ptr

    def verify(self, frames, probs=None, cuts=None, scores=None, ranges=None, intra=None, inter=None, lenient=False):
        """lenient=False: a cut that points outside the video raises ValueError (it means the detector output and the video do not match)"""
        self.L.sbdv_set_option(self.h, b"lenient", 1.0 if lenient else 0.0)
        frames, T, p, c, sc, r, labels, ptr = self._call(frames, probs, cuts, scores, ranges, intra, inter)
        cap = 4096
        while True:
            ab = np.zeros((cap, 2), np.int32); pk = np.zeros(cap); kept = np.zeros(cap, np.uint8); src = np.zeros(cap, np.int32)
            sce = np.zeros((cap + 2, 2), np.int32); ns = ctypes.c_int(0); tm = (ctypes.c_double * 6)(); err = ctypes.create_string_buffer(512)
            n = self.L.sbdv_verify(self.h, ptr(frames, ctypes.c_uint8), T, ptr(p, ctypes.c_float), 0 if p is None else len(p), ptr(c, ctypes.c_int32), ptr(sc, ctypes.c_float),
                                   0 if c is None else len(c), ptr(r, ctypes.c_int32), 0 if r is None else len(r), labels, ptr(ab, ctypes.c_int32), ptr(pk, ctypes.c_double),
                                   ptr(kept, ctypes.c_uint8), ptr(src, ctypes.c_int32), cap, ptr(sce, ctypes.c_int32), cap + 2, ctypes.byref(ns), tm, err, 512)
            if n < 0: raise ValueError(err.value.decode())
            if n <= cap: break
            cap = n + 8
        cands = dict(a=ab[:n, 0].copy(), b=ab[:n, 1].copy(), src=[SRC[i] for i in src[:n]], p_keep=pk[:n].copy(), kept=kept[:n].astype(bool))
        return dict(scenes=sce[:ns.value].copy(), cands=cands, n_cands=n, n_invalid=int(self.L.sbdv_get(self.h, b"last_invalid")), n_bad=int(self.L.sbdv_get(self.h, b"last_bad")),
                    timing_us=dict(zip(["candidates", "signals", "features", "models", "decisions", "total"], tm)))

    def verify_scenes(self, frames, scenes, scores=None, lenient=False):
        return self.verify(frames, cuts=scenes_to_cuts(scenes), scores=scores, lenient=lenient)

    def features(self, frames, probs=None, cuts=None, scores=None, ranges=None, intra=None, inter=None, lenient=False):
        """candidates + every feature (for training your own verifier).  Returns (ab [n,2], src [n], X [n, n_features], feature names)."""
        self.L.sbdv_set_option(self.h, b"lenient", 1.0 if lenient else 0.0)
        frames, T, p, c, sc, r, labels, ptr = self._call(frames, probs, cuts, scores, ranges, intra, inter)
        nf = self.L.sbdv_n_features(); names = [self.L.sbdv_feature_name(i).decode() for i in range(nf)]; cap = 4096
        while True:
            ab = np.zeros((cap, 2), np.int32); src = np.zeros(cap, np.int32); X = np.zeros((cap, nf), np.float32); err = ctypes.create_string_buffer(512)
            n = self.L.sbdv_features(self.h, ptr(frames, ctypes.c_uint8), T, ptr(p, ctypes.c_float), 0 if p is None else len(p), ptr(c, ctypes.c_int32), ptr(sc, ctypes.c_float),
                                     0 if c is None else len(c), ptr(r, ctypes.c_int32), 0 if r is None else len(r), labels, ptr(ab, ctypes.c_int32), ptr(src, ctypes.c_int32),
                                     ptr(X, ctypes.c_float), cap, err, 512)
            if n < 0: raise ValueError(err.value.decode())
            if n <= cap: break
            cap = n + 8
        return ab[:n].copy(), src[:n].copy(), X[:n].copy(), names


def verify_video(model, video, scenes=None, cuts=None, times=None, probs=None, ranges=None, binary=None, extra=(), timeout=None):
    """Run the command line tool on a video FILE (it decodes with ffmpeg and needs constant memory).  Give exactly one of
    scenes ([[start, end], ...] inclusive 0-based), cuts ([[a, b], ...]), times (seconds), probs (per-frame array) or ranges (OmniShotCut text).
    `extra` = more command line options, e.g. ("--lenient",).  Returns the tool's JSON as a dict (scenes as an int array)."""
    given = [k for k, v in dict(scenes=scenes, cuts=cuts, times=times, probs=probs, ranges=ranges).items() if v is not None]
    if len(given) != 1: raise ValueError("give exactly one of scenes, cuts, times, probs, ranges")
    exe = find_binary(binary)
    with tempfile.TemporaryDirectory() as td:
        args = [exe, "verify", "--model", os.fspath(model), "--video", os.fspath(video)]
        if probs is not None:
            fp = os.path.join(td, "p.npy"); np.save(fp, np.asarray(probs, np.float32)); args += ["--probs", fp]
        else:
            key = given[0]; fp = os.path.join(td, key + ".txt")
            with open(fp, "w") as f:
                if key == "ranges": f.write(ranges if isinstance(ranges, str) else "\n".join(" ".join(map(str, r)) for r in ranges))
                elif key == "times": f.write("\n".join(str(float(t)) for t in times))
                else: f.write("\n".join(f"{int(a)} {int(b)}" for a, b in np.asarray(scenes if key == "scenes" else cuts).reshape(-1, 2)))
            args += ["--" + key, fp]
        r = subprocess.run(args + list(extra), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0: raise (ValueError if r.returncode == 3 else RuntimeError)(r.stderr.strip() or f"sbdv exited with {r.returncode}")
        j = json.loads(r.stdout)
    j["scenes"] = np.asarray(j["scenes"], np.int32).reshape(-1, 2)
    return j
