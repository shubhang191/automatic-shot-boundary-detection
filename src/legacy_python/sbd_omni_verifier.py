"""
sbd_omni_verifier.py - the LEARNED VERIFIER FOR OMNISHOTCUT (used by notebook 7 and by the app's OmniShotCut mode).
Put this file next to sbd_refine.py, sbd_verifier.py, sbd_verifier2.py and sbd_verifier3.py in MyDrive/SBD_Prototype/ (notebook 7 writes it for you).

The idea, in plain words
  OmniShotCut gives a list of shots. Three things are wrong with that list on SHOT:
    1. Too many false cuts: it marks stylised transitions, caption changes and jumps inside one scene as cuts.
    2. It misses some real cuts.
    3. Some cuts are a few frames off.
  This module fixes them with the SAME tools as the AutoShot verifier, but built around OmniShotCut's own output:
    * every boundary OmniShotCut reports becomes a CANDIDATE. The verifier looks at the pictures around it (colour change, edges,
      how much of the picture changed, whether the new look lasts, how calm the video is), at how OmniShotCut labelled the boundary
      (Hard_Cut, Sudden_Jump, Transition ...) and at how this video normally behaves, and says KEEP or DROP.
    * optional RESCUE candidates: big picture jumps that OmniShotCut did not report. The verifier judges them in the same way and a
      stricter threshold decides whether to add them.
    * optional SNAP: move a kept hard cut to the exact frame of the biggest picture change (fixes cuts that are a few frames early/late).
  It does NOT use AutoShot's output (the 'hybrid' variant in the notebook is only a diagnostic and is never used in the app).
"""
import numpy as np
from sbd_refine import look_diff, look_frac, snap, to_gt_format
from sbd_verifier import pix_feats, safe, build_scenes
from sbd_verifier2 import NEW_FEATS, extra_feats
from sbd_verifier3 import frame_signals, video_feats

PIC  = ["d_adj", "d_ctx", "spike", "persist", "hist_d", "bright", "ld", "lf", "gap", "gap_prev", "gap_next", "density"]
PIC2 = [f for f in NEW_FEATS if f not in ("dup_p", "nb_p_prev", "nb_p_next", "p_rank")]      # the 4 removed ones need AutoShot's probabilities
VID  = ["vid_motion", "vid_motion_p90", "vid_bright", "vid_contrast", "vid_len", "d_adj_rel"]
LAB  = ["lab_hard_cut", "lab_sudden_jump", "lab_transition", "lab_new_start", "n_dropped", "is_rescue"]
OMNI_FEATS = PIC + PIC2 + VID + LAB
AS_FEATS   = ["as_p_at", "as_p_near", "as_agrees"]                                           # hybrid diagnostic only
HYB_FEATS  = OMNI_FEATS + AS_FEATS


# ---------------------------------------------------------------- candidates
def _lab_flags(lab):
    s = str(lab).lower()
    return dict(lab_hard_cut=int("hard" in s), lab_sudden_jump=int("sudden" in s),
                lab_transition=int("transition" in s), lab_new_start=int("new" in s))


def omni_boundaries(ranges, intra, inter):
    """Boundaries of OmniShotCut's 'fair' (SHOT-style) shot list: only the 'General' ranges count as shots, ends made inclusive,
    a dropped transition range becomes the gap between two shots. One dict per boundary: a = last frame before, b = first frame after."""
    keep = [k for k in range(len(ranges)) if intra[k] == "General"]
    sc, labs = to_gt_format(ranges, intra, inter)
    out = []
    for j in range(1, len(keep)):
        d = dict(a=int(sc[j - 1][1]), b=int(sc[j][0]), src="omni", n_dropped=int(keep[j] - keep[j - 1] - 1), is_rescue=0)
        d.update(_lab_flags(labs[j] if labs is not None else ""))
        out.append(d)
    return out, sc


def rescue_candidates(sig, existing, ratio=4.0, min_d=0.05, pad=3, cap=40):
    """Frames where the picture jumps (frame-to-frame change is large and much larger than the video's local normal) and that OmniShotCut
    did not report. They are only CANDIDATES: the verifier decides."""
    d1 = sig["d1"]; T = len(d1)
    if T < 8: return []
    loc = np.zeros(T, np.float32)
    for t in range(T):
        loc[t] = np.median(d1[max(1, t - 15):min(T, t + 16)])
    peaks = []
    for t in range(1, T):
        if d1[t] < min_d or d1[t] < ratio * (loc[t] + 1e-3): continue
        if d1[t] < d1[max(1, t - 2):min(T, t + 3)].max(): continue
        if any(t >= e["a"] - pad and t - 1 <= e["b"] + pad for e in existing): continue
        peaks.append((float(d1[t] / (loc[t] + 1e-3)), t))
    peaks = sorted(peaks, reverse=True)[:cap]
    out = []
    for _, t in sorted(peaks, key=lambda x: x[1]):
        out.append(dict(a=t - 1, b=t, src="rescue", n_dropped=0, is_rescue=1, lab_hard_cut=0, lab_sudden_jump=0, lab_transition=0, lab_new_start=0))
    return out


def build_candidates(ranges, intra, inter, frames, probs=None, rescue=True, r_ratio=4.0, r_min=0.05):
    """All candidates of ONE video with every number the models can need. frames = (n, 27, 48, 3) uint8.
    probs (AutoShot's probabilities) is only used for the hybrid diagnostic columns; the app never passes it."""
    n = len(frames); F = frames.astype(np.float32) / 255.0
    sig = frame_signals(F, np.zeros(n, np.float32))
    cand, _ = omni_boundaries(ranges, intra, inter)
    if rescue: cand = cand + rescue_candidates(sig, cand, r_ratio, r_min)
    cand = sorted(cand, key=lambda r: (r["a"], r["b"]))
    if not cand: return cand
    cs = np.array(sorted((r["a"] + r["b"]) / 2 for r in cand)); dens = len(cs) / max(n, 1) * 1000
    p = None
    if probs is not None:
        p = np.asarray(probs, np.float32)
        if len(p) != n: p = np.pad(p[:n], (0, max(0, n - len(p))))
    for r in cand:
        c = (r["a"] + r["b"]) / 2
        prev = cs[cs < c - 1e-9]; nxt = cs[cs > c + 1e-9]
        r["gap_prev"] = float(np.log1p(c - (prev[-1] if len(prev) else 0)))
        r["gap_next"] = float(np.log1p((nxt[0] if len(nxt) else n) - c))
        r["density"] = dens; r["gap"] = int(r["b"] - r["a"] - 1)
        r.update(pix_feats(F, r["a"], r["b"]))
        r["ld"] = safe(look_diff, frames, r["a"], r["b"]); r["lf"] = safe(look_frac, frames, r["a"], r["b"])
        r["p_at"] = 1.0; r["is_low"] = 0                                                       # neutral stand-ins for the helper functions
        aa, bb = int(np.clip(r["a"], 0, n - 1)), int(np.clip(r["b"], 0, n - 1))
        if p is not None:
            r["as_p_at"] = float(p[aa:max(bb, aa + 1)].max()); r["as_p_near"] = float(p[max(0, aa - 2):bb + 3].max())
            r["as_agrees"] = int(r["as_p_near"] >= 0.296)
        else:
            r["as_p_at"] = r["as_p_near"] = 0.0; r["as_agrees"] = 0
    for r, e in zip(cand, extra_feats(F, None, cand, n)): r.update(e)
    for r, e in zip(cand, video_feats(F, sig, cand)): r.update(e)
    return cand


def table_of(cand, feats):
    return np.nan_to_num(np.array([[r[f] for f in feats] for r in cand], dtype=np.float32))


# ---------------------------------------------------------------- what the app calls
def verify_omni(ranges, intra, inter, frames, bundle):
    """bundle = joblib.load('omni_verifier_v15.joblib'). Returns (scenes as int array [[start, end], ...], info)
    where info has one dict per candidate: a, b, src ('omni' or 'rescue'), p_keep, kept."""
    n = len(frames)
    cand = build_candidates(ranges, intra, inter, frames, None, bundle["rescue"], bundle.get("r_ratio", 4.0), bundle.get("r_min", 0.05))
    if not cand:
        return np.array([[0, n - 1]]), []
    X = table_of(cand, bundle["feats"])
    pk = np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]], axis=0)
    thr = np.array([bundle["tau_r"] if r["is_rescue"] else bundle["tau_o"] for r in cand])
    kept = [r for r, q, t in zip(cand, pk, thr) if q >= t]
    scenes, used = build_scenes([(r["a"], r["b"]) for r in kept], n)
    if int(bundle.get("snap_w", 0)) > 0 and len(scenes) > 1:
        scenes = snap(scenes, frames, w=int(bundle["snap_w"]))[0]
    used_set = set(used)
    final = {id(r): (i in used_set) for i, r in enumerate(kept)}
    info = [dict(a=r["a"], b=r["b"], src=r["src"], p_keep=float(q), kept=bool(final.get(id(r), False))) for r, q in zip(cand, pk)]
    return np.asarray(scenes, dtype=int).reshape(-1, 2), info
