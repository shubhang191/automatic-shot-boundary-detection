"""
synth.py - synthetic 48x27 videos with KNOWN cuts and typical false-cut causes, for testing the tool without the SHOT data.

Each video is a list of shots.  A shot is a smooth random picture that drifts slowly and has a little noise.  Between shots there is a hard
cut, a dissolve or a fade.  Inside shots there are "disturbances" a detector can mistake for cuts: flashes (brightness jumps up and comes
back), captions (a bright block appears for a while), camera shake (a few frames shifted).
fake_detector() turns the known events into the cut list a detector might produce (finds most real cuts, some disturbances, misses a few).
These videos test that the code works; they say NOTHING about accuracy on real videos.
"""
import numpy as np

H, W = 27, 48


def _smooth_picture(rng):
    base = rng.random((5, 8, 3)).astype(np.float32)
    big = np.kron(base, np.ones((6, 8, 1), np.float32))                            # (30, 64, 3)            # blocky, then blur
    for _ in range(3):
        big = (big + np.roll(big, 1, 0) + np.roll(big, -1, 0) + np.roll(big, 1, 1) + np.roll(big, -1, 1)) / 5.0
    return big


def make_video(rng, n_shots=8, shot_len=(25, 90), p_flash=0.5, p_caption=0.4, p_shake=0.25, p_dissolve=0.25, noise=2.0):
    pics = [_smooth_picture(rng) for _ in range(n_shots)]
    shots, shot_events = [], []
    for s in range(n_shots):
        L = int(rng.integers(shot_len[0], shot_len[1] + 1)); pic = pics[s]
        dx, dy = rng.normal(0, 0.05, 2)
        seg = np.stack([pic[int(round(2 + dy * t)) % 4:int(round(2 + dy * t)) % 4 + H, int(round(2 + dx * t)) % 4:int(round(2 + dx * t)) % 4 + W].copy() for t in range(L)])
        ev = []                                                                   # (kind, offset inside the shot)
        if L > 40 and rng.random() < p_flash:
            t0 = int(rng.integers(12, L - 12)); k = int(rng.integers(1, 3))
            seg[t0:t0 + k] = np.clip(seg[t0:t0 + k] + rng.uniform(0.25, 0.5), 0, 1); ev.append(("flash", t0))
        if L > 40 and rng.random() < p_caption:
            t0 = int(rng.integers(10, L - 25)); t1 = t0 + int(rng.integers(10, 15)); y0, x0 = int(rng.integers(3, 15)), int(rng.integers(3, 25))
            seg[t0:t1, y0:y0 + 8, x0:x0 + 20] = np.clip(seg[t0:t1, y0:y0 + 8, x0:x0 + 20] * 0.3 + 0.65, 0, 1); ev.append(("caption", t0))
        if L > 40 and rng.random() < p_shake:
            t0 = int(rng.integers(12, L - 12)); k = int(rng.integers(2, 4))
            for j in range(k): seg[t0 + j] = np.roll(seg[t0 + j], int(rng.choice([-3, 3])), axis=1)
            ev.append(("shake", t0))
        shots.append(seg); shot_events.append(ev)
    # join: a hard cut, or a dissolve of n extra frames between the last frame of the old shot and the first frame of the new one
    parts, cuts, events, pos = [], [], [], 0
    for s in range(n_shots):
        if s > 0:
            if rng.random() < p_dissolve:
                n = int(rng.integers(5, 10)); w = np.linspace(0, 1, n + 2)[1:-1][:, None, None, None].astype(np.float32)
                parts.append(shots[s - 1][-1:] * (1 - w) + shots[s][:1] * w)
                cuts.append(dict(kind="dissolve", a=pos - 1, b=pos + n)); pos += n
            else:
                cuts.append(dict(kind="hard", a=pos - 1, b=pos))
        parts.append(shots[s])
        for kind, t0 in shot_events[s]: events.append(dict(kind=kind, a=pos + t0 - 1, b=pos + t0))
        pos += len(shots[s])
    V = np.concatenate(parts); T = len(V)
    V = np.clip(V * 255.0 + rng.normal(0, noise, V.shape), 0, 255).astype(np.uint8)
    gt_scenes, start = [], 0
    for c in cuts:
        gt_scenes.append([start, c["a"]]); start = c["b"]
    gt_scenes.append([start, T - 1])
    return V, np.array(gt_scenes), cuts, events


def fake_detector(rng, T, cuts, events, recall=0.93, p_dist=0.7, jitter=1):
    """a cut list [(a, b), ...] as a detector might give it: finds most real cuts, many disturbances, a few extra random ones"""
    out = []
    for c in cuts:
        if rng.random() < recall:
            j = int(rng.integers(-jitter, jitter + 1)) if c["kind"] != "hard" else 0
            a = max(c["a"] + j, 1); b = max(c["b"] + j, a + 1)
            out.append((a, b))
    for e in events:
        if rng.random() < p_dist: out.append((e["a"], e["b"]))
    for _ in range(int(rng.integers(0, 2))):
        a = int(rng.integers(20, max(T - 20, 21))); out.append((a, a + 1))
    out = sorted(set(out))
    return np.array(out, np.int64).reshape(-1, 2)


def cuts_to_scenes(cuts, T):
    s, start = [], 0
    for a, b in sorted(map(tuple, cuts)):
        if a < start or b <= a or b > T - 1: continue
        s.append([start, a]); start = b
    s.append([start, T - 1]); return np.array(s)


def make_dataset(n_videos=40, seed=0, **kw):
    rng = np.random.default_rng(seed); vids = {}
    for i in range(n_videos):
        n_shots = int(rng.integers(2, 12))
        V, gt, cuts, events = make_video(rng, n_shots=n_shots, **kw)
        vids[f"v{i:03d}"] = dict(frames=V, gt=gt, cuts=cuts, events=events)
    return vids
