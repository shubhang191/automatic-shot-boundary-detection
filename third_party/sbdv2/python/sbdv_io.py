"""
sbdv_io.py - bring the output of ANY detector into the form the verifier wants, and check that the frame numbers fit the video.

Why this file exists: detectors disagree on how they number frames (start at 0 or at 1?) and on what the END of a shot means (the last frame of the
shot, or the first frame of the next one?).  If those are read wrongly, every cut lands one frame off.  We do not guess a tool's convention from memory:
  * you can SAY the convention (base=0|1, end="inclusive"|"exclusive"), or
  * detect_convention() LOOKS AT THE PICTURES: at a real cut the picture changes the most, so the reading under which the cuts sit on the biggest
    changes is the right one.  It reports how sure it is, and refuses to answer when it is not clear.

    import sbdv, sbdv_io
    frames = sbdv.decode_frames("video.mp4")
    scenes = sbdv_io.read_scenes_csv("scenes.csv")                    # any CSV with start / end columns, or a plain "start end" text file
    conv   = sbdv_io.detect_convention(frames, scenes)                # {'base': 0, 'end': 'inclusive', 'sure': True, ...}
    cuts   = sbdv_io.scenes_to_cuts(scenes, conv["base"], conv["end"])
    out    = model.verify(frames, cuts=cuts)
"""
import csv, re
import numpy as np


def scenes_to_cuts(scenes, base=0, end="inclusive"):
    """shots [[start, end], ...] -> cuts [[a, b], ...] in 0-based frame numbers (a = last frame of the old shot, b = first frame of the new shot).
    base: are the numbers counted from 0 or from 1.   end: is `end` the last frame of the shot ("inclusive") or the first frame of the next shot ("exclusive")."""
    if base not in (0, 1) or end not in ("inclusive", "exclusive"): raise ValueError("base must be 0 or 1 and end must be 'inclusive' or 'exclusive'")
    s = np.asarray(scenes, np.float64).reshape(-1, 2) - base
    if len(s) < 2: return np.zeros((0, 2), np.int64)
    e = s[:, 1] - (1 if end == "exclusive" else 0)
    return np.stack([e[:-1], s[1:, 0]], 1).round().astype(np.int64)


def times_to_cuts(times, fps):
    """cut times in seconds (the time at which the NEW shot starts) -> cuts [[a, b]] with b = round(time * fps)"""
    b = np.round(np.asarray(times, np.float64) * float(fps)).astype(np.int64); return np.stack([b - 1, b], 1)


def read_scenes_csv(path, start=None, end=None):
    """Read shots from a CSV or text file.  With a header row the columns are found by name: the first column whose name contains 'start' and 'frame'
    (override with start="name"), and the first containing 'end' and 'frame' (override with end="name").  Without a header, the first two numbers of
    every line are used.  Lines that are not numbers (for example a title line above the header) are skipped.  Returns an array [n, 2] of the numbers
    as written in the file (you still need base / end, see detect_convention)."""
    with open(path, newline="") as f: text = f.read()
    rows = list(csv.reader(text.replace(";", ",").splitlines()))
    rows = [r for r in rows if any(c.strip() for c in r)]
    si = ei = None
    def norm(c): return re.sub(r"[^a-z0-9]+", " ", c.lower()).strip()
    for r in rows:
        names = [norm(c) for c in r]
        if start is not None and norm(start) in names: si = names.index(norm(start))
        elif start is None: si = next((i for i, c in enumerate(names) if "start" in c and "frame" in c), None)
        if end is not None and norm(end) in names: ei = names.index(norm(end))
        elif end is None: ei = next((i for i, c in enumerate(names) if "end" in c and "frame" in c), None)
        if si is not None and ei is not None: break
        si = ei = None
    out = []
    for r in rows:
        try:
            if si is not None: out.append((float(r[si]), float(r[ei])))
            else:
                v = [float(c) for c in " ".join(r).replace(",", " ").split()[:2]]
                if len(v) == 2: out.append(tuple(v))
        except (ValueError, IndexError):
            continue
    if not out: raise ValueError(f"{path}: found no 'start end' rows (use start='column name', end='column name' if your columns are named differently)")
    return np.array(out)


def _step(frames):
    """mean absolute change between neighbouring frames, step[i] = |frame i+1 - frame i|"""
    f = np.asarray(frames); d = np.zeros(max(len(f) - 1, 0), np.float32)
    for t0 in range(0, len(d), 512):
        t1 = min(len(d), t0 + 512); d[t0:t1] = np.abs(f[t0 + 1:t1 + 1].astype(np.int16) - f[t0:t1].astype(np.int16)).mean(axis=(1, 2, 3))
    return d


def cut_sharpness(frames, cuts, step=None):
    """Only plain cuts (b = a + 1) are used: how much bigger is the picture change from frame a to frame b than in the frames right around it?
    Returns (median ratio, number of cuts used).  A ratio well above 1 means the cuts sit on picture jumps.  A reading that is off by one frame puts
    a and b inside one shot (ratio about 1) or makes b != a + 1 (not used), so the right reading stands out."""
    d = _step(frames) if step is None else step; T = len(d) + 1; r = []
    for a, b in np.asarray(cuts, np.int64).reshape(-1, 2):
        if b != a + 1 or a < 2 or b > T - 3: continue
        near = np.mean([d[a - 2], d[a - 1], d[b], d[b + 1]]); r.append(d[a] / max(near, 0.5))
    return (float(np.median(r)) if r else float("nan")), len(r)


def detect_convention(frames, scenes, need_ratio=1.5):
    """Which reading of the numbers puts the cuts on the picture jumps?  Tries base 0/1 x end inclusive/exclusive and compares.
    Returns dict(base, end, sure, table).  sure=False means "cannot tell" (very few cuts, or dissolves everywhere): then say the convention yourself."""
    step = _step(frames); table = []
    for base in (0, 1):
        for end in ("inclusive", "exclusive"):
            c = scenes_to_cuts(scenes, base, end); sh, used = cut_sharpness(frames, c, step); table.append(dict(base=base, end=end, sharpness=sh, cuts=len(c), used=used))
    ok = [t for t in table if t["sharpness"] == t["sharpness"]]
    if not ok: return dict(base=0, end="inclusive", sure=False, table=table, note="no plain cut (b = a + 1) away from the video edges to decide with")
    ok.sort(key=lambda t: -t["sharpness"]); best, second = ok[0], (ok[1] if len(ok) > 1 else None)
    # readings that give the SAME cuts (for example base 1 + exclusive vs base 0 + inclusive when ends and starts are consecutive) are equivalent, not rivals
    def same(t1, t2): return np.array_equal(scenes_to_cuts(scenes, t1["base"], t1["end"]), scenes_to_cuts(scenes, t2["base"], t2["end"]))
    rival = next((t for t in ok[1:] if not same(best, t)), None)
    sure = best["used"] >= 3 and best["sharpness"] >= need_ratio and (rival is None or best["sharpness"] >= 1.3 * rival["sharpness"])
    return dict(base=best["base"], end=best["end"], sure=bool(sure), table=table, note="" if sure else "the readings are too close to tell; say base= and end= yourself")


if __name__ == "__main__":
    import sys, sbdv
    if len(sys.argv) != 3: print("usage: python sbdv_io.py scenes.csv video.mp4   -> tells which frame-number convention the file uses"); sys.exit(2)
    sc = read_scenes_csv(sys.argv[1]); fr = sbdv.decode_frames(sys.argv[2]); r = detect_convention(fr, sc)
    for t in r["table"]: print(f"  base {t['base']}  end {t['end']:9s}  cuts {t['cuts']:4d}  used {t['used']:4d}  sharpness {t['sharpness']:.2f}")
    print(f"-> base {r['base']}, end {r['end']}" + ("" if r["sure"] else "   (NOT SURE: " + r["note"] + ")"))
