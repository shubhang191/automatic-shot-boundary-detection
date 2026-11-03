"""
sbd_verifier2.py - EXTRA evidence for the learned cut verifier (used by notebook 6c and, if it is adopted, by the app).
Put this file next to sbd_refine.py and sbd_verifier.py in MyDrive/SBD_Prototype/.

Notebook 6 judged every candidate cut with 15 numbers. This module adds 20 more, aimed at the mistakes seen on real videos:
  * camera motion vs a real cut         -> mc_d, mc_ratio  (best match after sliding the picture a little)
  * structure (edges) really changing   -> edge_d, edge_rel
  * how much of the picture changed     -> frac05, frac10, frac20, ctr_d, brd_d
  * does the new look last, is it calm  -> d_wide, wide_ratio, pre_stab, post_stab, cut_vs_stab
  * crowding: other cuts / a doubled cut -> n15, dup_dist, dup_p, nb_p_prev, nb_p_next, p_rank
It also has a 'near-duplicate guard': if two kept cuts sit within a few frames of each other, keep only the more certain one.
The training notebook and the app call the SAME functions, so the numbers are identical in both places.
"""
import numpy as np
from sbd_verifier import candidates, build_scenes

NEW_FEATS = ["mc_d", "mc_ratio", "edge_d", "edge_rel", "frac05", "frac10", "frac20", "ctr_d", "brd_d",
             "d_wide", "wide_ratio", "pre_stab", "post_stab", "cut_vs_stab",
             "n15", "dup_dist", "dup_p", "nb_p_prev", "nb_p_next", "p_rank"]


def _shift_min_diff(ga, gb, s=2):
    """Smallest mean |difference| between two grey pictures after sliding one by up to s pixels in each direction."""
    H, W = ga.shape
    core = ga[s:H - s, s:W - s]
    best = 1e9
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            d = float(np.abs(core - gb[s + dy:H - s + dy, s + dx:W - s + dx]).mean())
            if d < best: best = d
    return best


def _edges(g):
    gx = g[:26, 1:48] - g[:26, 0:47]
    gy = g[1:27, :47] - g[0:26, :47]
    return np.hypot(gx, gy)


def _stab(seg):
    return float(np.abs(np.diff(seg, axis=0)).mean()) if len(seg) > 1 else 0.0


def pic_feats2(F, a, b):
    """F = frames / 255 as float32, shape (T, 27, 48, 3). a = last frame before the cut, b = first frame after it."""
    T = len(F); a = int(np.clip(a, 0, T - 1)); b = int(np.clip(b, 0, T - 1))
    fa, fb = F[a], F[b]
    diff = np.abs(fb - fa).mean(-1)                                   # (27, 48) change per pixel
    d_adj = float(diff.mean())
    ga, gb = fa.mean(-1), fb.mean(-1)
    mc = _shift_min_diff(ga, gb)
    mc_ratio = float(np.clip(mc / (float(np.abs(gb - ga).mean()) + 1e-3), 0, 2))
    ma, mb = _edges(ga), _edges(gb)
    edge_d = float(np.abs(mb - ma).mean())
    edge_rel = float(np.clip(edge_d / (float(ma.mean() + mb.mean()) + 1e-3), 0, 5))
    pre_w = F[max(0, a - 5):max(1, a - 1)].mean(0)
    hi = min(T, b + 7); lo = min(T - 1, b + 3)
    post_w = F[lo:max(hi, lo + 1)].mean(0)
    d_wide = float(np.abs(post_w - pre_w).mean())
    pre_stab = _stab(F[max(0, a - 8):a + 1].mean(-1)); post_stab = _stab(F[b:min(T, b + 9)].mean(-1))
    return dict(mc_d=mc, mc_ratio=mc_ratio, edge_d=edge_d, edge_rel=edge_rel,
                frac05=float((diff > 0.05).mean()), frac10=float((diff > 0.10).mean()), frac20=float((diff > 0.20).mean()),
                ctr_d=float(diff[7:20, 12:36].mean()), brd_d=float((diff.sum() - diff[7:20, 12:36].sum()) / (diff.size - diff[7:20, 12:36].size)),
                d_wide=d_wide, wide_ratio=float(np.clip(d_wide / (d_adj + 1e-3), 0, 20)),
                pre_stab=pre_stab, post_stab=post_stab,
                cut_vs_stab=float(np.clip(d_adj / (pre_stab + post_stab + 1e-3), 0, 50)))


def extra_feats(F, p, cands, n):
    """cands = the AutoShot candidates of ONE video (base + low), sorted by position, each a dict with a, b and p_at.
    Returns one dict of the 20 new features per candidate."""
    cen = np.array([(r["a"] + r["b"]) / 2 for r in cands], np.float64)
    pat = np.array([r["p_at"] for r in cands], np.float64)
    out = []
    for k, r in enumerate(cands):
        e = pic_feats2(F, r["a"], r["b"])
        d = np.abs(cen - cen[k]); d[k] = 1e9
        near = d <= 15; n15 = int(near.sum())
        j = int(d.argmin()) if len(d) > 1 else -1
        e["n15"] = n15
        e["dup_dist"] = float(min(d[j], 30.0)) if j >= 0 else 30.0
        e["dup_p"] = float(pat[j]) if j >= 0 and d[j] <= 30 else 0.0
        e["nb_p_prev"] = float(pat[k - 1]) if k > 0 else 0.0
        e["nb_p_next"] = float(pat[k + 1]) if k + 1 < len(cands) else 0.0
        win = (d <= 10); m = max([pat[k]] + list(pat[win]))
        e["p_rank"] = float(pat[k] / (m + 1e-6))
        out.append(e)
    return out


def dedupe_mask(centers, p_keep, kept, dist):
    """Near-duplicate guard. kept = boolean array. If two kept cuts have centres within `dist` frames, drop the less certain one."""
    kept = np.array(kept, bool).copy()
    if dist <= 0: return kept
    idx = [i for i in np.argsort(centers, kind="stable") if kept[i]]
    last = None
    for i in idx:
        if last is not None and centers[i] - centers[last] <= dist:
            if p_keep[i] > p_keep[last]:
                kept[last] = False; last = i
            else:
                kept[i] = False
        else:
            last = i
    return kept


def predict_keep(bundle, X):
    """Average of the models stored in the bundle."""
    return np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]], axis=0)


def verify2(probs, frames, bundle):
    """bundle = joblib.load('verifier_v15_reinforced.joblib').  Same output as sbd_verifier.verify:
    (scenes as an int array [[start, end], ...], info with one dict per candidate: a, b, src, p_keep, kept)."""
    n = len(frames)
    cand = candidates(probs, frames, bundle.get("base_thr", 0.296), bundle.get("low_thr", 0.10))
    if not cand:
        return np.array([[0, n - 1]]), []
    F = frames.astype(np.float32) / 255.0
    for r, e in zip(cand, extra_feats(F, None, cand, n)): r.update(e)
    X = np.nan_to_num(np.array([[r[f] for f in bundle["features"]] for r in cand], dtype=np.float32))
    pk = predict_keep(bundle, X)
    keep = pk >= float(bundle["tau"])
    cen = np.array([(r["a"] + r["b"]) / 2 for r in cand])
    keep = dedupe_mask(cen, pk, keep, int(bundle.get("dedupe", 0)))
    kept = [r for r, k in zip(cand, keep) if k]
    scenes, used = build_scenes([(r["a"], r["b"]) for r in kept], n)
    used_set = set(used)
    final = {id(r): (i in used_set) for i, r in enumerate(kept)}
    info = [dict(a=r["a"], b=r["b"], src=r["src"], p_keep=float(q), kept=bool(final.get(id(r), False))) for r, q in zip(cand, pk)]
    return scenes, info
