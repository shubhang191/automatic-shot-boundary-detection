"""TriCut v2 shared code. Written to Drive by notebook 16 (do not edit by hand); notebook 17 imports it.

Frames are uint8 arrays (T, 27, 48, 3): every video is shrunk to 48 x 27 pixels first (the same size AutoShot uses).
A boundary is (a, b): a = last frame of the old shot, b = first frame of the new shot.  A hard cut has b = a + 1, a dissolve/fade has the frames a+1 .. b-1 in between.
"""
import subprocess, numpy as np, cv2
from scipy.signal import lfilter
import torch, torch.nn as nn

# ======================================================================================================= video input / output (ffmpeg through subprocess)
def get_frames48(path, w=48, h=27, max_frames=None):
    """decode a video and shrink every frame to w x h (same ffmpeg call AutoShot's get_frames uses) -> (T, h, w, 3) uint8. max_frames: stop after that many frames (long videos)"""
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-frames:v", str(int(max_frames))] if max_frames else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0: raise RuntimeError("ffmpeg failed on " + path + ": " + r.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w, 3)

def write_video(frames, path, fps=25, crf=18):
    """frames (T,H,W,3) uint8 -> .mp4 (H.264)"""
    T, H, W = frames.shape[:3]
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "veryfast", path],
                       input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if r.returncode != 0: raise RuntimeError("ffmpeg could not write " + path + ": " + r.stderr.decode(errors="replace")[-300:])

# ======================================================================================================= the 35 numbers per frame
BASE_NAMES = ["l1", "l1_blur", "hist", "bright", "edge", "frac", "pers3", "pers8", "phist3", "phist8", "lum", "l1_ema", "hist_ema"]
STRUCT_NAMES = ["ncc", "ncc_p3", "ncc_p8", "zdiff"]
CONS_NAMES = ["dz_min", "dz_q25", "dz_med", "dz_q75", "dz_max", "dc_min", "dc_q25", "dc_med", "dc_q75", "dc_max", "pz3_min", "pz3_med", "pz8_min", "pz8_med"]
ALL_NAMES = BASE_NAMES + ["raw_" + n for n in STRUCT_NAMES] + CONS_NAMES + ["clahe_" + n for n in STRUCT_NAMES]        # 13 + 4 + 14 + 4 = 35
GROUP_COLS = {"base": list(range(0, 13)), "raw": list(range(13, 17)), "cons": list(range(17, 31)), "clahe": list(range(31, 35))}
def columns(groups): return sum((GROUP_COLS[g] for g in groups), [])
NUM_LOOKAHEAD = 7                                    # every number looks at most 7 frames AHEAD of the frame it describes
NB_R, NB_C = 3, 4                                    # the 48 x 27 frame is cut into 3 x 4 = 12 blocks of 9 x 12 pixels

def _hist16(F):                                      # (T,27,48,3) uint8 -> (T,48): 16-bin colour histograms
    T = len(F); q = (F >> 4).astype(np.int32); H = np.zeros((T, 3, 16), np.float32); n = q.shape[1] * q.shape[2]
    offs = (np.arange(T, dtype=np.int32) * 16)[:, None]
    for c in range(3): H[:, c, :] = np.bincount((q[..., c].reshape(T, -1) + offs).ravel(), minlength=T * 16).reshape(T, 16) / n
    return H.reshape(T, 48)

def _win_means(A, k):                                # A (T,D): mean of the k frames before t and of the k frames from t on
    T = len(A); cs = np.concatenate([np.zeros((1, A.shape[1]), np.float64), np.cumsum(A, 0, dtype=np.float64)], 0)
    t = np.arange(T); b0 = np.clip(t - k, 0, T); a1 = np.clip(t + k, 0, T)
    return (cs[t] - cs[b0]) / np.maximum(t - b0, 1)[:, None], (cs[a1] - cs[t]) / np.maximum(a1 - t, 1)[:, None]

def ema_rel(x, prior, alpha=0.02):
    """x divided by the running average of its OWN past (no future needed): 'how big is this compared with this video's normal'"""
    e, _ = lfilter([alpha], [1.0, -(1 - alpha)], x.astype(np.float64), zi=[(1 - alpha) * prior]); prev = np.r_[prior, e[:-1]]
    return np.clip(x / (prev + 1e-3), 0, 20).astype(np.float32)

def base_numbers(F):
    """(T,27,48,3) -> (T,13): picture change, blurred change, colour-mix change, brightness change, edge change, fraction of changed pixels,
    persistence (3 and 8 frames before vs after) for picture and colour mix, brightness level, and two 'relative to this video's normal' numbers"""
    T = len(F); out = np.zeros((T, 13), np.float32)
    if T < 3: return out
    X = F.astype(np.float32) / 255.0; G = X.mean(-1); lum = G.mean((1, 2)); d = np.abs(X[1:] - X[:-1])
    l1 = np.r_[0.0, d.mean((1, 2, 3))]; frac = np.r_[0.0, (d.max(-1) > 0.1).mean((1, 2))]
    Gb = G.reshape(T, 9, 3, 16, 3).mean((2, 4)); l1b = np.r_[0.0, np.abs(Gb[1:] - Gb[:-1]).mean((1, 2))]
    gx = G[:, :-1, 1:] - G[:, :-1, :-1]; gy = G[:, 1:, :-1] - G[:, :-1, :-1]; E = np.sqrt(gx ** 2 + gy ** 2); edge = np.r_[0.0, np.abs(E[1:] - E[:-1]).mean((1, 2))]
    H = _hist16(F); hd = np.r_[0.0, 0.5 * np.abs(H[1:] - H[:-1]).sum(1) / 3.0]; bright = np.r_[0.0, lum[1:] - lum[:-1]]
    Xp = X.reshape(T, 9, 3, 16, 3, 3).mean((2, 4)).reshape(T, -1); pers = {}
    for k in (3, 8):
        b, a = _win_means(Xp, k); pk = np.abs(a - b).mean(1); pk[0] = 0.0
        hb, ha = _win_means(H, k); ph = 0.5 * np.abs(ha - hb).sum(1) / 3.0; ph[0] = 0.0; pers[k] = (pk, ph)
    out[:] = np.stack([l1, l1b, hd, bright, edge, frac, pers[3][0], pers[8][0], pers[3][1], pers[8][1], lum, ema_rel(l1, 0.03), ema_rel(hd, 0.08)], 1)
    return out

def grey(F): return np.clip(F[..., 0] * 0.299 + F[..., 1] * 0.587 + F[..., 2] * 0.114, 0, 255).astype(np.uint8)

def clahe_grey(F, clip=2.0, grid=(3, 3)):
    """CLAHE = contrast limited adaptive histogram equalisation: evens out local contrast; returns floats in [0,1]"""
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return np.stack([cl.apply(np.ascontiguousarray(g)) for g in grey(F)]).astype(np.float32) / 255.0

def struct_features(G):
    """G (T,h,w) floats. Every frame is made zero-mean and unit-length first, so a change of brightness or contrast alone changes nothing.
    -> (T,4): 1 - correlation with the previous frame, 1 - correlation of the mean of the 3 (8) frames before vs after, mean absolute difference"""
    T = len(G); out = np.zeros((T, 4), np.float32)
    if T < 3: return out
    V = G.reshape(T, -1).astype(np.float32); V = V - V.mean(1, keepdims=True); Z = V / (np.sqrt((V ** 2).sum(1)) + 1e-6)[:, None]
    out[1:, 0] = 1 - (Z[1:] * Z[:-1]).sum(1); out[1:, 3] = np.abs(Z[1:] - Z[:-1]).mean(1) * np.sqrt(Z.shape[1])
    for j, k in enumerate((3, 8)):
        b, a = _win_means(Z, k); v = 1 - (a * b).sum(1) / ((np.sqrt((b ** 2).sum(1)) + 1e-6) * (np.sqrt((a ** 2).sum(1)) + 1e-6)); v[0] = 0.0; out[:, 1 + j] = v
    return out

def _stats5(d):                                      # d (T,12): min, lower quartile, median, upper quartile, max across the 12 blocks
    S = np.sort(d, axis=1); return np.stack([S[:, 0], S[:, 3], 0.5 * (S[:, 5] + S[:, 6]), S[:, 8], S[:, 11]], 1)

def consensus(F):
    """Spatial consensus (T,14): do ALL parts of the picture change together?  dz = structure change of each block (block mean removed), dc = colour change of each block.
    A cut moves almost every block; a caption box, a glare spot or one moving person moves only some. min / quartiles / median across blocks tell the two apart."""
    T = len(F); out = np.zeros((T, 14), np.float32)
    if T < 3: return out
    X = F.astype(np.float32) / 255.0; Xb = X.reshape(T, NB_R, 9, NB_C, 12, 3)
    Gb = Xb.mean(-1).transpose(0, 1, 3, 2, 4).reshape(T, NB_R * NB_C, 108); Cb = Xb.mean((2, 4)).reshape(T, NB_R * NB_C, 3); Zb = Gb - Gb.mean(2, keepdims=True)
    dz = np.zeros((T, 12), np.float32); dz[1:] = np.abs(Zb[1:] - Zb[:-1]).mean(2); dc = np.zeros((T, 12), np.float32); dc[1:] = np.abs(Cb[1:] - Cb[:-1]).mean(2)
    out[:, 0:5] = _stats5(dz); out[:, 5:10] = _stats5(dc); Z2 = Zb.reshape(T, -1)
    for j, k in enumerate((3, 8)):
        b, a = _win_means(Z2, k); p = np.abs(a - b).reshape(T, 12, 108).mean(2); p[0] = 0.0; S = np.sort(p, axis=1)
        out[:, 10 + 2 * j] = S[:, 0]; out[:, 11 + 2 * j] = 0.5 * (S[:, 5] + S[:, 6])
    return out

def all_features(F):
    """(T,27,48,3) uint8 -> (T,35): 13 base numbers, 4 exposure-proof structure numbers, 14 spatial-consensus numbers, 4 CLAHE structure numbers"""
    return np.concatenate([base_numbers(F), struct_features(grey(F).astype(np.float32) / 255.0), consensus(F), struct_features(clahe_grey(F))], 1)

def features_long(F, chunk=8000, margin=250):         # long videos (BBC): the same numbers computed in pieces so memory stays small
    T = len(F); out = np.zeros((T, 35), np.float32)
    for s0 in range(0, T, chunk):
        e0 = min(s0 + chunk, T); a0, b0 = max(s0 - margin, 0), min(e0 + margin, T); out[s0:e0] = all_features(F[a0:b0])[s0 - a0:s0 - a0 + (e0 - s0)]
    return out

# ======================================================================================================= labels and scoring
def targets(bounds, T):
    """1 on the frames a+1 .. b of every boundary (a hard cut: only the first frame of the new shot)"""
    y = np.zeros(T, np.uint8)
    for a, b in bounds: y[a + 1:b + 1] = 1
    return y

def scenes_from_bounds(bounds, T):
    """[(a,b),...] -> scenes [[0,a1],[b1,a2],...,[bn,T-1]]  (the layout AutoShot's scorer wants)"""
    return np.array(list(zip([0] + [b for a, b in bounds], [a for a, b in bounds] + [T - 1])), dtype=int)

def bounds_from_scenes(s):
    s = np.asarray(s); return [(int(s[i][1]), int(s[i + 1][0])) for i in range(len(s) - 1)]

def f1_of(c):
    tp, fp, fn = [float(x) for x in c]; P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); return 2 * P * R / max(P + R, 1e-9)

def count_bounds(gt_scenes, bounds, T):
    """TP/FP/FN of predicted boundaries against ground-truth scenes with AutoShot's own scorer (a cut counts if it is within 2 frames)"""
    from autoshot_utils import evaluate_scenes
    _, _, _, (a, b, c) = evaluate_scenes(np.asarray(gt_scenes), scenes_from_bounds(bounds, T), n_frames_miss_tolerance=2)
    return np.array([a, b, c])

def count_scenes(gt_scenes, pred_scenes):
    from autoshot_utils import evaluate_scenes
    _, _, _, (a, b, c) = evaluate_scenes(np.asarray(gt_scenes), np.asarray(pred_scenes), n_frames_miss_tolerance=2)
    return np.array([a, b, c])

def boot_diff(per_a, per_b, ids, n=3000, seed=1):
    """paired bootstrap over videos: F1(a) - F1(b), its 95% range, and a plain-words verdict. per_*: {video: np.array([TP,FP,FN])}"""
    A = np.array([per_a[v] for v in ids], float); B = np.array([per_b[v] for v in ids], float); idx = np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))
    d = np.array([f1_of(A[i].sum(0)) - f1_of(B[i].sum(0)) for i in idx]); lo, hi = np.percentile(d, [2.5, 97.5]); diff = f1_of(A.sum(0)) - f1_of(B.sum(0))
    return diff, lo, hi, ("BETTER" if lo > 0 else ("WORSE" if hi < 0 else "cannot be told apart"))

# ======================================================================================================= decoders: probabilities per frame -> boundaries
def pick_cuts(p, thr):
    """the OLD decoder (notebooks 13-16): a cut at every frame that is >= thr and the highest within 2 frames (at most one per 3 frames)"""
    p = np.asarray(p, np.float32); pm = p.copy()
    for s in (1, 2): pm[s:] = np.maximum(pm[s:], p[:-s]); pm[:-s] = np.maximum(pm[:-s], p[s:])
    out = []
    for t in np.where((p >= thr) & (p >= pm))[0]:
        if t >= 1 and (not out or t - out[-1] > 2): out.append(int(t))
    return out

def decode_peaks(p, thr=0.5): return [(t - 1, t) for t in pick_cuts(p, thr)]

def decode_runs(p, thr=0.5, close=2, max_hard=2):
    """the NEW decoder. Frames with p >= thr form runs; runs separated by at most `close` quiet frames are joined.
    A run of <= max_hard frames is ONE hard cut (at its highest frame); a longer run is ONE gradual transition (s-1, e).
    The old decoder put a cut every 3 frames inside a dissolve, which the scorer counts as false cuts."""
    p = np.asarray(p, np.float32); idx = np.flatnonzero(p >= thr); idx = idx[idx >= 1]
    if len(idx) == 0: return []
    br = np.flatnonzero(np.diff(idx) > close + 1); starts = np.r_[idx[0], idx[br + 1]]; ends = np.r_[idx[br], idx[-1]]; out = []
    for s, e in zip(starts, ends):
        if e - s + 1 <= max_hard: t = int(s + np.argmax(p[s:e + 1])); out.append((t - 1, t))
        else: out.append((int(s) - 1, int(e)))
    return out

def decode(p, prm):
    """prm = dict(mode='runs'|'peaks', thr=..., close=..., max_hard=...)"""
    return decode_peaks(p, prm["thr"]) if prm["mode"] == "peaks" else decode_runs(p, prm["thr"], prm["close"], prm["max_hard"])

THR_GRID = [round(float(x), 2) for x in np.arange(0.30, 0.96, 0.05)]
def decoder_grid(modes=("peaks", "runs"), close=(0, 1, 2, 3), max_hard=(2, 3)):
    g = []
    for thr in THR_GRID:
        if "peaks" in modes: g.append(dict(mode="peaks", thr=thr, close=0, max_hard=2))
        if "runs" in modes: g += [dict(mode="runs", thr=thr, close=c, max_hard=h) for c in close for h in max_hard]
    return g

def tune_decoder(P, GT, T, vids, grid=None):
    """choose the decoder settings with the best F1 on the videos `vids` (use TRAINING-side videos only, never the test set).
    P: {video: probabilities}, GT: {video: ground-truth scenes}, T: {video: number of frames}. Returns the best settings, its counts, and a table of the best per mode."""
    grid = grid or decoder_grid(); best, bymode = None, {}
    for prm in grid:
        c = np.zeros(3, int)
        for v in vids: c += count_bounds(GT[v], decode(P[v], prm), T[v])
        f = f1_of(c)
        if best is None or f > best[0]: best = (f, prm, c)
        if prm["mode"] not in bymode or f > bymode[prm["mode"]][0]: bymode[prm["mode"]] = (f, prm, c)
    return best[1], best[2], bymode

# ======================================================================================================= training material: splice real shots together
def shots_from_video(frames, bounds, min_len=20, trim=2):
    """cut-free pieces of a labelled video (edges trimmed so no transition frame is left in)"""
    T = len(frames); st = [0] + [b for a, b in bounds]; en = [a for a, b in bounds] + [T - 1]; out = []
    for s, e in zip(st, en):
        s2, e2 = s + trim, e - trim
        if e2 - s2 + 1 >= min_len: out.append(frames[s2:e2 + 1])
    return out

def _f(x): return x.astype(np.float32)
def _u8(x): return np.clip(x, 0, 255).astype(np.uint8)

def _transition(kind, prev, nxt, rng):
    """returns (new_prev, middle_frames, new_nxt, gap) or None when the pieces are too short"""
    H, W = prev.shape[1:3]
    if kind == "dissolve":
        g = int(rng.integers(4, 21))
        if len(prev) < g + 8 or len(nxt) < g + 8: return None
        al = np.linspace(0, 1, g + 2)[1:-1][:, None, None, None]
        return prev[:-g], _u8((1 - al) * _f(prev[-g:]) + al * _f(nxt[:g])), nxt[g:], g
    if kind == "fade":
        g1, g2 = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        if len(prev) < g1 + 8 or len(nxt) < g2 + 8: return None
        c = 0.0 if rng.random() < 0.7 else 255.0
        a1 = np.linspace(0, 1, g1 + 2)[1:-1][:, None, None, None]; a2 = np.linspace(1, 0, g2 + 2)[1:-1][:, None, None, None]
        return prev[:-g1], _u8(np.concatenate([(1 - a1) * _f(prev[-g1:]) + a1 * c, (1 - a2) * _f(nxt[:g2]) + a2 * c], 0)), nxt[g2:], g1 + g2
    if kind == "wipe":
        g = int(rng.integers(5, 17))
        if len(prev) < g + 8 or len(nxt) < g + 8: return None
        xs = (np.arange(W) + 0.5)[None, :, None]; mid = []
        for i in range(g):
            m = (xs < W * (i + 1) / (g + 1)); mid.append(np.where(m, nxt[i], prev[-g + i]))
        return prev[:-g], np.stack(mid).astype(np.uint8), nxt[g:], g
    return None

def splice(segs, rng, p=(0.72, 0.14, 0.07, 0.07)):
    """join pieces with random transitions (p = chance of hard cut, dissolve, fade, wipe). Returns frames and the exact boundaries [(a, b), ...]."""
    out = [np.asarray(segs[0], np.uint8)]; bounds = []; T = len(segs[0])
    for nxt in segs[1:]:
        nxt = np.asarray(nxt, np.uint8); kind = rng.choice(["hard", "dissolve", "fade", "wipe"], p=np.asarray(p) / np.sum(p)); res = None if kind == "hard" else _transition(kind, out[-1], nxt, rng)
        if res is None: bounds.append((T - 1, T)); out.append(nxt); T += len(nxt); continue
        new_prev, mid, new_nxt, gap = res; T -= len(out[-1]) - len(new_prev); out[-1] = new_prev
        a = T - 1; out.append(mid); out.append(new_nxt); T += len(mid) + len(new_nxt); bounds.append((a, a + gap + 1))
    return np.concatenate(out, 0), bounds

NEG_KINDS = ("flash", "exposure", "caption", "glare", "shake")
def add_negatives(frames, bounds, rng, kinds, n_each=(1, 3), margin=5):
    """disturbances that are NOT cuts: flashes, auto-exposure ramps, caption boxes, glare blobs, camera shake. Returns uint8 frames and a log of (kind, start, length)."""
    T, H, W = frames.shape[:3]; X = np.array(frames, np.uint8, copy=True); log = []; forbid = np.zeros(T, bool)
    for a, b in bounds: forbid[max(a - margin, 0):min(b + margin + 1, T)] = True
    def place(L):
        if T < L + 2: return None
        for _ in range(30):
            s = int(rng.integers(0, T - L + 1))
            if not forbid[s:s + L].any(): return s
        return None
    yy, xx = np.mgrid[0:H, 0:W]; yy = yy.astype(np.float32); xx = xx.astype(np.float32)
    for kind in kinds:
        for _ in range(int(rng.integers(n_each[0], n_each[1] + 1))):
            if kind == "flash":
                L = int(rng.integers(1, 4)); s = place(L)
                if s is None: continue
                seg = _f(X[s:s + L])
                if rng.random() < 0.5: seg = seg + rng.uniform(70, 160)
                else: al = rng.uniform(0.4, 0.9); seg = (1 - al) * seg + al * 255.0
                X[s:s + L] = _u8(seg)
            elif kind == "exposure":
                L = int(rng.integers(12, 41)); s = place(L)
                if s is None: continue
                d = rng.choice([-1, 1]) * rng.uniform(0.35, 0.8); g = 1 + d * np.sin(np.pi * np.arange(L) / (L - 1)); X[s:s + L] = _u8(_f(X[s:s + L]) * g[:, None, None, None])
            elif kind == "caption":
                L = int(rng.integers(20, 91)); s = place(L)
                if s is None: continue
                r0, r1, c0, c1 = int(0.72 * H), int(0.92 * H), int(0.1 * W), int(0.9 * W); box = np.full((r1 - r0, c1 - c0, 3), rng.choice([20.0, 235.0]), np.float32)
                step = max((r1 - r0) // 3, 1); box[::step] = 255.0 - box[0, 0, 0]; X[s:s + L, r0:r1, c0:c1] = _u8(box)
            elif kind == "glare":
                L = int(rng.integers(10, 41)); s = place(L)
                if s is None: continue
                cy, cx = rng.uniform(0.2, 0.8) * H, rng.uniform(0.2, 0.8) * W; rad = rng.uniform(0.15, 0.35) * H; amp = rng.uniform(60, 150)
                blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))[None, :, :, None]; env = np.sin(np.pi * np.arange(L) / (L - 1))[:, None, None, None]
                X[s:s + L] = _u8(_f(X[s:s + L]) + amp * env * blob)
            elif kind == "shake":
                L = int(rng.integers(6, 26)); s = place(L)
                if s is None: continue
                for i in range(L):
                    dy, dx = int(rng.integers(-2, 3)), int(rng.integers(-max(int(0.05 * W), 1), max(int(0.05 * W), 1) + 1)); X[s + i] = np.roll(X[s + i], (dy, dx), (0, 1))
            else: continue
            log.append((kind, s, L))
    return X, log

def random_splice(pool, rng, n_shots=(3, 10), seg_len=(24, 80), neg_p=0.35, p=(0.72, 0.14, 0.07, 0.07)):
    """pool: list of cut-free shots (L,27,48,3). Returns frames, exact boundaries, log of the not-a-cut disturbances."""
    n = int(rng.integers(n_shots[0], n_shots[1] + 1)); idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False); segs = []
    for i in idx:
        sh = pool[i]; L = int(min(len(sh), rng.integers(seg_len[0], seg_len[1] + 1))); s = int(rng.integers(0, len(sh) - L + 1)); segs.append(sh[s:s + L])
    frames, bounds = splice(segs, rng, p); kinds = [k for k in NEG_KINDS if rng.random() < neg_p]; frames, log = add_negatives(frames, bounds, rng, kinds)
    return frames, bounds, log

def make_synthetic(shot_pool, n, seed):
    """n synthetic videos with exact labels -> list of dict(f=(T,35) numbers, y=(T,) targets, T, bounds). Also returns how many of each disturbance were inserted."""
    rng = np.random.default_rng(seed); out = []; kinds = {}
    for _ in range(n):
        fr, b, log = random_splice(shot_pool, rng); out.append(dict(f=all_features(fr), y=targets(b, len(fr)), T=len(fr), bounds=b))
        for k, _, _ in log: kinds[k] = kinds.get(k, 0) + 1
    return out, kinds

# ======================================================================================================= the tiny networks and their training
class TinyCut(nn.Module):
    """symmetric 1D convolution network: sees 22 frames to each side of a frame (about 18 thousand weights)"""
    def __init__(self, c_in=31, h=32, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c_in, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.ReLU(), nn.Dropout(p),
                                 nn.Conv1d(h, h, 5, padding=8, dilation=4), nn.ReLU(), nn.Dropout(p), nn.Conv1d(h, h, 3, padding=8, dilation=8), nn.ReLU(), nn.Conv1d(h, 1, 1))
    def forward(self, x): return self.net(x).squeeze(1)

class CausalTinyCut(nn.Module):
    """same size, but every convolution only looks at the PAST. Its output at time u is read as the decision for frame u - LOOK."""
    def __init__(self, c_in=31, h=32, p=0.1):
        super().__init__()
        def blk(ci, co, k, d): return [nn.ConstantPad1d(((k - 1) * d, 0), 0.0), nn.Conv1d(ci, co, k, dilation=d), nn.ReLU()]
        self.net = nn.Sequential(*blk(c_in, h, 5, 1), *blk(h, h, 5, 2), nn.Dropout(p), *blk(h, h, 5, 4), nn.Dropout(p), *blk(h, h, 3, 8), nn.Conv1d(h, 1, 1))
    def forward(self, x): return self.net(x).squeeze(1)

RF_HALF = 2 + 4 + 8 + 8                              # frames the symmetric network looks ahead of a frame
DEFAULT_CFG = dict(L=512, B=32, STEPS=60, EPOCHS=30, PATIENCE=6, POSW=8.0, LOOK=4, LR=2e-3, DEV=None)

def _standardise(cols, real_train, syn_train):
    allf = np.concatenate([r["f"][:, cols] for r in real_train] + [r["f"][:, cols] for r in syn_train[:300]], 0); return allf.mean(0).astype(np.float32), (allf.std(0) + 1e-6).astype(np.float32)

def _make_batch(pools, probs, cols, mu, sd, rng, B, Lw, dev):
    X = np.zeros((B, len(cols), Lw), np.float32); Y = np.zeros((B, Lw), np.float32); M = np.zeros((B, Lw), np.float32)
    for i in range(B):
        pool = pools[int(rng.choice(len(pools), p=probs))]; r = pool[int(rng.integers(len(pool)))]; f, y = r["f"], r["y"]; T = len(y)
        a = int(rng.integers(0, max(T - Lw, 0) + 1)); e = min(a + Lw, T); n = e - a; X[i, :, :n] = ((f[a:e][:, cols] - mu) / sd).T; Y[i, :n] = y[a:e]; M[i, :n] = 1
    return torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev), torch.from_numpy(M).to(dev)

def _net_probs(net, f, cols, mu, sd, causal, look, dev):
    net.eval(); x = ((f[:, cols] - mu) / sd).T.astype(np.float32)
    if causal: x = np.concatenate([x, np.zeros((x.shape[0], look), np.float32)], 1)                 # the stream ends: feed 'average' numbers for the last frames
    with torch.no_grad(): out = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(x[None])).to(dev)))[0].cpu().numpy()
    return out[look:] if causal else out

def train_member(cols, causal, real_train, syn_train, real_stop, seed, cfg=None):
    """train ONE network. real_train / syn_train / real_stop: lists of dict(f=(T,35) numbers, y=(T,) targets).
    Training batches are half real, half synthetic; training stops early when the average precision on `real_stop` (real videos only) stops improving."""
    from sklearn.metrics import average_precision_score
    c = dict(DEFAULT_CFG); c.update(cfg or {}); dev = c["DEV"] or ("cuda" if torch.cuda.is_available() else "cpu"); look = c["LOOK"]; Lw = c["L"]
    torch.manual_seed(seed); rng = np.random.default_rng(seed); mu, sd = _standardise(cols, real_train, syn_train)
    net = (CausalTinyCut if causal else TinyCut)(c_in=len(cols)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=c["LR"], weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([c["POSW"]], device=dev)); pools, probs = ([real_train, syn_train], [0.5, 0.5]) if len(syn_train) else ([real_train], [1.0])
    y_stop = np.concatenate([r["y"] for r in real_stop]); best = (-1.0, None, 0); bad = 0
    for ep in range(1, c["EPOCHS"] + 1):
        net.train()
        for _ in range(c["STEPS"]):
            X, Y, M = _make_batch(pools, probs, cols, mu, sd, rng, c["B"], Lw, dev); opt.zero_grad(); out = net(X)
            if causal: out, Y, M = out[:, look:], Y[:, :Lw - look], M[:, :Lw - look]                  # output at time u is trained to say what happened at frame u - LOOK
            loss = (lossf(out, Y) * M).sum() / M.sum().clamp(min=1.0); loss.backward(); opt.step()
        ap = average_precision_score(y_stop, np.concatenate([_net_probs(net, r["f"], cols, mu, sd, causal, look, dev) for r in real_stop]))
        if ap > best[0]: best = (ap, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep); bad = 0
        else: bad += 1
        if bad >= c["PATIENCE"]: break
    return dict(state=best[1], cols=list(cols), mu=mu, sd=sd, causal=bool(causal), ap=float(best[0]), epoch=int(best[2]), look=look)

def _net_of(m, dev="cpu"):
    key = "_net_" + dev
    if key not in m:
        net = (CausalTinyCut if m["causal"] else TinyCut)(c_in=len(m["cols"])); net.load_state_dict(m["state"]); m[key] = net.to(dev).eval()
    return m[key]

def member_probs(m, f, dev="cpu"):
    """probability that a cut happens at each frame, from ONE trained member. f: (T,35) numbers of the video."""
    return _net_probs(_net_of(m, dev), f, m["cols"], m["mu"], m["sd"], m["causal"], m["look"], dev)

def ensemble_probs(members, f, dev="cpu"): return np.mean([member_probs(m, f, dev) for m in members], 0)

def n_weights(m): return int(sum(v.numel() for v in m["state"].values()))
def conv_macs(m):
    """multiply-adds per frame of one member (convolutions only)"""
    s = m["state"]; return int(sum(v.shape[0] * v.shape[1] * v.shape[2] for k, v in s.items() if k.endswith("weight") and v.dim() == 3))
def decision_delay(m, prm):
    """frames between a cut and the moment the detector can report it: numbers look 7 frames ahead + the network's own lookahead + frames to see that a run of high scores has ended"""
    return NUM_LOOKAHEAD + (m["look"] if m["causal"] else RF_HALF) + (prm.get("close", 0) + 1 if prm["mode"] == "runs" else 3)

def strip_runtime(m): return {k: v for k, v in m.items() if not k.startswith("_net")}

def save_obj(path, obj):
    """pickle a dict that may hold trained members (the cached torch modules are left out)"""
    import pickle
    def clean(o):
        if isinstance(o, dict) and "cols" in o and "mu" in o: return strip_runtime(o)
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return type(o)(clean(v) for v in o)
        return o
    with open(path + ".tmp", "wb") as fh: pickle.dump(clean(obj), fh)
    import os; os.replace(path + ".tmp", path)

def load_obj(path):
    import pickle
    with open(path, "rb") as fh: return pickle.load(fh)

def missed_kinds(gt_scenes, pred_scenes):
    """how many ground-truth boundaries were missed: (hard cuts, gradual transitions)"""
    from autoshot_utils import evaluate_scenes
    r = evaluate_scenes(np.asarray(gt_scenes), np.asarray(pred_scenes), return_mistakes=True, n_frames_miss_tolerance=2); fn = r[5]
    hard = sum(1 for g in fn if (g[1] - g[0]) < 2.5); return hard, len(fn) - hard
