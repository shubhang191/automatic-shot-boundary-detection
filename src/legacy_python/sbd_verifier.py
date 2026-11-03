"""
sbd_verifier.py - the LEARNED CUT VERIFIER for the shot-boundary project (inference side).
Put this file next to sbd_refine.py in MyDrive/SBD_Prototype/.  No GPU needed (AutoShot's probabilities come from the app).

What it does, in plain words:
  1. AutoShot gives a probability per frame.  Peaks above 0.296 are its normal cuts ('base'); peaks between 0.10 and 0.296 are
     borderline candidates ('low').
  2. For every candidate we measure AutoShot's confidence, how much the picture really changes across the cut, and the spacing
     to neighbouring cuts (the same measurements as in notebook 6, copied unchanged).
  3. A small model trained on SHOT (saved by notebook 6 as verifier_v15_autoshot_features.joblib) says KEEP or DROP for each candidate.
Returns the final shots plus, for every candidate, the model's confidence - so the app can show which cuts are doubtful.
"""
import numpy as np
from sbd_refine import look_diff, look_frac


def run_pairs(p, thr):
    b = p > thr; n = len(b); out = []; i = 0
    while i < n:
        if b[i]:
            j = i
            while j + 1 < n and b[j + 1]: j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def bounds_of(runs, n):                  # (end of previous shot, start of next shot) - the same convention as AutoShot's scenes
    return [(s, e + 1) for s, e in runs if s > 0 and e < n - 1]


def build_scenes(cands, n):              # turn a list of (a, b) cuts into scenes; skips cuts that overlap an earlier one
    out, start, used = [], 0, []
    for k, (a, b) in enumerate(cands):
        if a < start or b <= a or b > n - 1: continue
        out.append([start, a]); start = b; used.append(k)
    out.append([start, n - 1])
    return np.array(out), used


def hist3(x):
    h = np.stack([np.histogram(x[..., c], bins=8, range=(0, 1))[0] for c in range(3)]).astype(np.float32)
    return h / max(float(h.sum()), 1.0)


def pix_feats(F, a, b):                  # simple picture measurements on the tiny 48x27 frames (F = frames / 255, float32)
    T = len(F); a = int(np.clip(a, 0, T - 1)); b = int(np.clip(b, 0, T - 1))
    fa, fb = F[a], F[b]
    d_adj = float(np.abs(fb - fa).mean())
    pre = F[max(0, a - 3):a + 1].mean(0); post = F[b:min(T, b + 4)].mean(0)
    d_ctx = float(np.abs(post - pre).mean())
    lo, hi = max(0, a - 15), min(T - 1, b + 15)
    seg = np.abs(np.diff(F[lo:hi + 1], axis=0)).mean(axis=(1, 2, 3)) if hi > lo else np.zeros(1, np.float32)
    spike = float(np.clip(d_adj / (float(np.median(seg)) + 1e-3), 0, 50))
    far_a, far_b = F[max(0, a - 2)], F[min(T - 1, b + 2)]
    persist = float(np.clip(np.abs(far_b - far_a).mean() / (d_adj + 1e-3), 0, 20))
    hist_d = float(0.5 * np.abs(hist3(fa) - hist3(fb)).sum())
    bright = abs(float(fb.mean()) - float(fa.mean()))
    return dict(d_adj=d_adj, d_ctx=d_ctx, spike=spike, persist=persist, hist_d=hist_d, bright=bright)


def safe(fn, fr, a, b):
    try: return float(fn(fr, a, b))
    except Exception: return np.nan


def candidates(probs, frames, base_thr=0.296, low_thr=0.10):
    """All candidate cuts of ONE video with the 'AutoShot features' of notebook 6. Returns a list of dicts sorted by position."""
    n = len(frames); F = frames.astype(np.float32) / 255.0; p = np.asarray(probs, np.float32)
    if len(p) != n: p = np.pad(p[:n], (0, max(0, n - len(p))))
    base = bounds_of(run_pairs(p, base_thr), n)
    low = [(s, e + 1) for s, e in run_pairs(p, low_thr) if s > 0 and e < n - 1 and p[s:e + 1].max() <= base_thr]
    cand = sorted([dict(a=a, b=b, src="base") for a, b in base] + [dict(a=a, b=b, src="low") for a, b in low],
                  key=lambda r: (r["a"], r["b"]))
    cs = np.array(sorted((r["a"] + r["b"]) / 2 for r in cand)); dens = len(cs) / max(n, 1) * 1000
    for r in cand:
        c = (r["a"] + r["b"]) / 2
        prev = cs[cs < c - 1e-9]; nxt = cs[cs > c + 1e-9]
        r["gap_prev"] = float(np.log1p(c - (prev[-1] if len(prev) else 0)))
        r["gap_next"] = float(np.log1p((nxt[0] if len(nxt) else n) - c))
        r["density"] = dens
        r.update(pix_feats(F, r["a"], r["b"]))
        r["ld"] = safe(look_diff, frames, r["a"], r["b"]); r["lf"] = safe(look_frac, frames, r["a"], r["b"])
        aa, bb = int(np.clip(r["a"], 0, n - 1)), int(np.clip(r["b"], 0, n - 1))
        r["p_at"] = float(p[aa:max(bb, aa + 1)].max()); r["p_near"] = float(p[max(0, aa - 2):bb + 3].max())
        r["gap"] = int(r["b"] - r["a"] - 1); r["is_low"] = int(r["src"] == "low")
    return cand


def verify(probs, frames, bundle):
    """bundle = joblib.load('verifier_v15_autoshot_features.joblib').
    Returns (scenes as int array [[start, end], ...], info) where info has one dict per candidate: a, b, src, p_keep, kept."""
    n = len(frames)
    cand = candidates(probs, frames, bundle.get("base_thr", 0.296), bundle.get("low_thr", 0.10))
    if not cand:
        return np.array([[0, n - 1]]), []
    X = np.nan_to_num(np.array([[r[f] for f in bundle["features"]] for r in cand], dtype=np.float32))
    pk = bundle["model"].predict_proba(X)[:, 1]
    tau = float(bundle["tau"])
    for r, q in zip(cand, pk):
        r["p_keep"] = float(q); r["kept"] = bool(q >= tau)
    kept = [r for r in cand if r["kept"]]
    scenes, used = build_scenes([(r["a"], r["b"]) for r in kept], n)
    used_set = set(used)
    for k, r in enumerate(kept):
        if k not in used_set: r["kept"] = False          # overlapped an earlier cut
    info = [dict(a=r["a"], b=r["b"], src=r["src"], p_keep=r["p_keep"], kept=r["kept"]) for r in cand]
    return scenes, info
