"""
sbd_refine.py  -  shared logic for the shot-boundary 'false cut' project.
Put this file in ONE Drive folder (e.g. MyDrive/SBD_Prototype/) and import it from
(1) the analysis notebook and (2) the Gradio app notebook.  No GPU needed.

Conventions (same as the official AutoShot evaluator):
  scenes = array of [start_frame, end_frame] per shot.
  A boundary j sits between shot j and shot j+1.
"""
import numpy as np


# ---------------------------------------------------------------- scoring helpers
def _transitions(scenes, tol=2):
    """Same transformation as AutoShot's evaluate_scenes()."""
    sh = tol / 2
    s = np.asarray(scenes, np.float32) + np.array([[-0.5 + sh, 0.5 - sh]])
    return np.stack([s[:-1, 1], s[1:, 0]], 1)


def match_boundaries(gt_scenes, pred_scenes, tol=2):
    """Per-boundary version of the official matching loop.
    Returns (pred_is_tp, gt_is_hit): bool arrays, one entry per predicted / GT boundary.
    pred_is_tp[j] == False  ->  predicted boundary j is a FALSE POSITIVE."""
    gt_tr, pr_tr = _transitions(gt_scenes, tol), _transitions(pred_scenes, tol)
    pred_is_tp = np.zeros(len(pr_tr), bool)
    gt_is_hit = np.zeros(len(gt_tr), bool)
    i = j = 0
    while i < len(gt_tr) and j < len(pr_tr):
        if pr_tr[j, 1] < gt_tr[i, 0]:
            j += 1
        elif pr_tr[j, 0] > gt_tr[i, 1]:
            i += 1
        else:
            pred_is_tp[j] = True; gt_is_hit[i] = True; i += 1; j += 1
    return pred_is_tp, gt_is_hit


def boundary_frame(pred_scenes, j):
    """Frame index where predicted boundary j happens (first frame of the next shot)."""
    return int(pred_scenes[j][1]), int(pred_scenes[j + 1][0])   # (last frame before, first frame after)



# ---------------------------------------------------------------- format conversion (OmniShotCut -> SHOT / AutoShot)
def to_gt_format(ranges, intra_labels=None, inter_labels=None, drop_transitions=True):
    """OmniShotCut returns HALF-OPEN ranges [start, end) that partition [0, n_frames], and in mode="default" it writes every
    dissolve / fade / wipe as a range of its own. SHOT ground truth and AutoShot use INCLUSIVE ranges [first, last] and
    represent a gradual transition as the GAP between two shots.
    This keeps only the 'General' ranges (what mode="clean_shot" does) and makes the ends inclusive (end - 1).
    Returns (new_ranges as int array, labels of the kept ranges or None)."""
    keep = [k for k in range(len(ranges))
            if (not drop_transitions) or intra_labels is None or intra_labels[k] == "General"]
    out = np.array([[int(ranges[k][0]), int(ranges[k][1]) - 1] for k in keep], dtype=int).reshape(-1, 2)
    labs = [inter_labels[k] for k in keep] if inter_labels is not None else None
    return out, labs


def look_frac(frames, a, b, k=3, thr=0.12):
    """Share of pixels (0..1) that change clearly between the video just before a and just after b.
    A caption, button or popup changing on an otherwise unchanged picture gives a SMALL share; a real cut gives a large one."""
    b = max(b, a + 1)
    pre = frames[max(0, a - k + 1): a + 1]
    post = frames[b: b + k]
    if len(pre) == 0 or len(post) == 0:
        return 1.0
    d = np.abs(pre.astype(np.float32).mean(0) - post.astype(np.float32).mean(0)).max(-1) / 255.0
    return float((d > thr).mean())


def snap(scenes, frames, w=8, min_ratio=1.5):
    """Move each hard cut to the frame with the largest frame-to-frame change within +-w frames
    (fixes cuts that are a few frames early/late). Returns (scenes, number_of_cuts_moved)."""
    S = [list(map(int, r)) for r in scenes]
    n = len(frames)
    if w <= 0 or n < 2 or len(S) < 2:
        return np.array(S, dtype=int).reshape(-1, 2), 0
    d = np.zeros(n, np.float32)
    d[1:] = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).mean(axis=(1, 2, 3)) / 255.0
    moved = 0
    for i in range(1, len(S)):
        if S[i][0] - S[i - 1][1] != 1:           # only hard cuts (no gap between the shots)
            continue
        b = S[i][0]
        lo, hi = max(S[i - 1][0] + 1, b - w, 1), min(S[i][1], b + w, n - 1)
        if hi < lo:
            continue
        cand = np.arange(lo, hi + 1)
        best = int(cand[np.argmax(d[cand])])
        if best != b and d[best] > min_ratio * max(float(d[b]), 1e-6):
            S[i][0] = best; S[i - 1][1] = best - 1; moved += 1
    return np.array(S, dtype=int).reshape(-1, 2), moved

# ---------------------------------------------------------------- the refinement stage
def look_diff(frames, a, b, k=3):
    """How different is the video just BEFORE frame a (inclusive) from just AFTER frame b (inclusive)?
    frames: (n,27,48,3) uint8.  Returns 0..1 (mean absolute pixel difference of k-frame averages).
    Returns 1.0 (= 'very different', keep the cut) if there are not enough frames to compare."""
    b = max(b, a + 1)                      # OmniShotCut ranges share the end/start frame
    pre = frames[max(0, a - k + 1): a + 1]
    post = frames[b: b + k]
    if len(pre) == 0 or len(post) == 0:
        return 1.0
    return float(np.abs(pre.astype(np.float32).mean(0) - post.astype(np.float32).mean(0)).mean() / 255.0)


def refine(scenes, frames, lmax=8, tau_a=0.10, tau_b=0.0, k=3, labels=None, drop_labels=(), tau_f=0.0, snap_w=0):
    """Clean up a list of shot ranges (INCLUSIVE ends, i.e. SHOT / AutoShot format). Training-free; uses only 48x27 frames.
    Order of the rules:
      S   snap_w>0   : move hard cuts to the nearest peak of frame-to-frame change (+-snap_w frames)
      C   drop_labels: drop boundaries whose OmniShotCut inter-label is in the set (needs `labels`, one per shot)
      A   lmax>0     : a shot of at most lmax frames whose neighbours look alike (look_diff < tau_a) is a flash: merge it away
      B   tau_b>0    : drop a boundary where before/after look alike (look_diff < tau_b)
      B'  tau_f>0    : drop a boundary where fewer than tau_f of the pixels change (caption / overlay / UI change)
    Returns (new_scenes as int array, number_of_boundaries_removed)."""
    R = [list(map(int, r)) for r in scenes]
    n0 = len(R)
    if snap_w > 0 and n0 > 1:
        R = [list(map(int, r)) for r in snap(R, frames, w=snap_w)[0]]
    L = list(labels) if labels is not None else [None] * len(R)
    i = 1
    while i < len(R):                                                     # Rule C
        if L[i] in drop_labels:
            R[i - 1][1] = R[i][1]; del R[i]; del L[i]
        else:
            i += 1
    i = 1
    while i < len(R) - 1:                                                 # Rule A
        if (R[i][1] - R[i][0] + 1) <= lmax and look_diff(frames, R[i - 1][1], R[i + 1][0], k) < tau_a:
            R[i - 1][1] = R[i + 1][1]; del R[i:i + 2]; del L[i:i + 2]
        else:
            i += 1
    for rule_on, fn, tau in ((tau_b > 0, look_diff, tau_b), (tau_f > 0, look_frac, tau_f)):   # Rules B, B'
        if not rule_on:
            continue
        i = 1
        while i < len(R):
            if fn(frames, R[i - 1][1], R[i][0], k) < tau:
                R[i - 1][1] = R[i][1]; del R[i]; del L[i]
            else:
                i += 1
    return np.array(R, dtype=int).reshape(-1, 2), n0 - len(R)
