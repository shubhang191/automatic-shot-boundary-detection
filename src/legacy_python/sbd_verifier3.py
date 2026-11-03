"""
sbd_verifier3.py - TWO new ideas for the learned cut verifier (used by notebook 6d and, if one is adopted, by the app).
Put this file next to sbd_refine.py, sbd_verifier.py and sbd_verifier2.py in MyDrive/SBD_Prototype/ (notebook 6d writes it for you).

  Idea 1  A small TEMPORAL neural network (1D convolutions). Instead of 15-35 hand-made numbers per candidate cut, it reads the
          64 frames around the cut as curves: AutoShot's probability, how much the picture changes from frame to frame, over 2 frames,
          how the brightness changes, how big the change is compared with the video's normal motion, and where the candidate sits.
  Idea 2  PER-VIDEO signals. The same picture change means different things in a calm interview and in a sports match, so the verifier
          also sees how much this video normally moves, how bright and contrasty it is, how long it is, and how strong this cut is
          compared with the video's own usual cut strength. This lets the keep/drop decision adapt to the type of video.
The training notebook and the app call the SAME functions, so the numbers are identical in both places.
"""
import numpy as np
from sbd_verifier import candidates, build_scenes
from sbd_verifier2 import NEW_FEATS, extra_feats, dedupe_mask

OLD_FEATS = ["d_adj", "d_ctx", "spike", "persist", "hist_d", "bright", "ld", "lf", "p_at", "p_near", "gap", "is_low",
             "gap_prev", "gap_next", "density"]
VID_FEATS = ["vid_motion", "vid_motion_p90", "vid_bright", "vid_contrast", "vid_len", "vid_low_share", "d_adj_rel", "p_vs_vid"]
ALL_FEATS = OLD_FEATS + NEW_FEATS + VID_FEATS
WIN = 64          # frames around each candidate cut
WIN_CH = 6        # curves fed to the network


# ---------------------------------------------------------------- per-frame signals, per-video signals, windows
def frame_signals(F, p):
    """F = frames / 255 (float32, shape (T, 27, 48, 3)), p = AutoShot's probability per frame. Returns the per-frame curves of ONE video."""
    T = len(F)
    pp = np.zeros(T, np.float32); m = min(T, len(p)); pp[:m] = np.asarray(p, np.float32)[:m]
    d1 = np.zeros(T, np.float32); d2 = np.zeros(T, np.float32); b1 = np.zeros(T, np.float32)
    if T > 1:
        d1[1:] = np.abs(F[1:] - F[:-1]).mean(axis=(1, 2, 3))
        br = F.mean(axis=(1, 2, 3)); b1[1:] = np.abs(br[1:] - br[:-1])
    if T > 2:
        d2[2:] = np.abs(F[2:] - F[:-2]).mean(axis=(1, 2, 3))
    med = float(np.median(d1[1:])) if T > 1 else 0.0
    rel = (np.clip(d1 / (med + 1e-3), 0, 10) / 10).astype(np.float32)
    return dict(p=pp, d1=d1, d2=d2, b1=b1, rel=rel)


def video_feats(F, sig, cands):
    """Idea 2. cands = the candidates of ONE video (dicts with d_adj, p_at, is_low). Returns one dict of the 8 per-video numbers per candidate."""
    d1 = sig["d1"][1:]
    vm = float(np.median(d1)) if len(d1) else 0.0
    vp = float(np.percentile(d1, 90)) if len(d1) else 0.0
    bright, contrast, vlen = float(F.mean()), float(F.std()), float(np.log(max(len(F), 1)))
    pats = np.array([r["p_at"] for r in cands], np.float64)
    med_p = float(np.median(pats)) if len(pats) else 0.0
    low = float(np.mean([r["is_low"] for r in cands])) if len(cands) else 0.0
    return [dict(vid_motion=vm, vid_motion_p90=vp, vid_bright=bright, vid_contrast=contrast, vid_len=vlen, vid_low_share=low,
                 d_adj_rel=float(np.clip(r["d_adj"] / (vm + 1e-3), 0, 50)),
                 p_vs_vid=float(np.clip(r["p_at"] / (med_p + 1e-3), 0, 20))) for r in cands]


def window(sig, a, b, W=WIN):
    """Idea 1. The 6 curves around one candidate cut, shape (6, W), centred on the cut."""
    T = len(sig["p"]); c = (int(a) + int(b)) // 2; lo = c - W // 2; hi = lo + W
    curves = [sig["p"], np.clip(sig["d1"] * 5, 0, 3), np.clip(sig["d2"] * 5, 0, 3), np.clip(sig["b1"] * 10, 0, 3), sig["rel"]]
    out = np.zeros((WIN_CH, W), np.float32)
    s, e = max(lo, 0), min(hi, T)
    if e > s:
        for i, ch in enumerate(curves): out[i, s - lo:e - lo] = ch[s:e]
    m0, m1 = max(int(a), lo), min(int(b), hi - 1)
    if m1 >= m0: out[5, m0 - lo:m1 - lo + 1] = 1.0
    return out


# ---------------------------------------------------------------- models (all share fit(tab, win, y) / predict(tab, win) -> 1-D probabilities)
class TabModel:
    """A scikit-learn model on a chosen subset of the numbers."""
    def __init__(self, cols, make):
        self.cols = list(cols); self.make = make; self.ix = [ALL_FEATS.index(c) for c in self.cols]
    def fit(self, tab, win, y):
        self.m = self.make().fit(tab[:, self.ix], y); return self
    def predict(self, tab, win):
        return self.m.predict_proba(tab[:, self.ix])[:, 1]
    def export(self):
        return dict(type="tab", cols=self.cols, model=self.m)


def _build_net(n_ch, n_tab, W=WIN):
    import torch, torch.nn as nn
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Sequential(nn.Conv1d(n_ch, 24, 5, padding=2), nn.ReLU(),
                                   nn.Conv1d(24, 24, 5, padding=4, dilation=2), nn.ReLU(),
                                   nn.Conv1d(24, 32, 3, padding=2, dilation=2), nn.ReLU())
            s.tab = nn.Sequential(nn.Linear(n_tab, 32), nn.ReLU()) if n_tab else None
            s.head = nn.Sequential(nn.Linear(64 + (32 if n_tab else 0), 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        def forward(s, w, t=None):
            h = s.conv(w); c = W // 2
            z = torch.cat([h.max(dim=2).values, h[:, :, c - 4:c + 4].mean(dim=2)], dim=1)   # strongest response anywhere + response at the cut itself
            if s.tab is not None: z = torch.cat([z, s.tab(t)], dim=1)
            return s.head(z).squeeze(1)
    return Net()


class TCNModel:
    """Idea 1: 1D convolutions over the 64-frame window. cols = optional picture/per-video numbers appended before the final decision."""
    def __init__(self, cols=(), epochs=40, seed=0, lr=3e-3, batch=128):
        self.cols = list(cols); self.epochs = epochs; self.seed = seed; self.lr = lr; self.batch = batch
        self.ix = [ALL_FEATS.index(c) for c in self.cols]
    def _tab(self, tab):
        if not self.ix: return None
        return np.clip((np.nan_to_num(tab[:, self.ix]).astype(np.float32) - self.mu) / self.sd, -8, 8)
    def fit(self, tab, win, y):
        import torch, torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"; self.dev = dev
        if self.ix:
            X = np.nan_to_num(tab[:, self.ix]).astype(np.float32); self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.net = _build_net(win.shape[1], len(self.ix)).to(dev)
        W_ = torch.tensor(win, dtype=torch.float32, device=dev)
        T_ = torch.tensor(self._tab(tab), dtype=torch.float32, device=dev) if self.ix else None
        Y_ = torch.tensor(np.asarray(y), dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss(); n = len(Y_)
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                if len(idx) < 4: continue
                w = W_[idx] + 0.02 * torch.randn_like(W_[idx])                          # a little noise so the network does not memorise
                out = self.net(w, T_[idx] if T_ is not None else None)
                loss = lossf(out, Y_[idx]); opt.zero_grad(); loss.backward(); opt.step()
        return self
    def predict(self, tab, win):
        import torch
        self.net.eval(); dev = next(self.net.parameters()).device
        T_all = self._tab(tab); res = []
        with torch.no_grad():
            for i in range(0, len(win), 1024):
                w = torch.tensor(win[i:i + 1024], dtype=torch.float32, device=dev)
                t = torch.tensor(T_all[i:i + 1024], dtype=torch.float32, device=dev) if T_all is not None else None
                res.append(torch.sigmoid(self.net(w, t)).cpu().numpy())
        return np.concatenate(res) if res else np.zeros(0)
    def export(self):
        d = dict(type="tcn", cols=self.cols, n_ch=int(self.net.conv[0].in_channels))
        d["state"] = {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        if self.ix: d["mu"] = self.mu; d["sd"] = self.sd
        return d
    @classmethod
    def from_export(cls, d):
        import torch
        m = cls(cols=d["cols"]); m.mu = d.get("mu"); m.sd = d.get("sd")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m.net = _build_net(d["n_ch"], len(m.ix)).to(dev)
        m.net.load_state_dict({k: torch.tensor(v) for k, v in d["state"].items()}); m.net.eval()
        return m


def predict_parts(parts, tab, win):
    """parts = the list stored in the bundle (exports of the models). The final probability is the average of the parts."""
    ps = []
    for d in parts:
        if d["type"] == "tab":
            ix = [ALL_FEATS.index(c) for c in d["cols"]]; ps.append(d["model"].predict_proba(tab[:, ix])[:, 1])
        else:
            ps.append(TCNModel.from_export(d).predict(tab, win))
    return np.mean(ps, axis=0)


# ---------------------------------------------------------------- what the app calls
def build_inputs(probs, frames, base_thr=0.296, low_thr=0.10):
    """All candidates of ONE video with every number the models can need: returns (candidates, tab (N, len(ALL_FEATS)), win (N, 6, 64))."""
    n = len(frames)
    cand = candidates(probs, frames, base_thr, low_thr)
    if not cand: return cand, None, None
    F = frames.astype(np.float32) / 255.0
    p = np.asarray(probs, np.float32)
    if len(p) != n: p = np.pad(p[:n], (0, max(0, n - len(p))))
    for r, e in zip(cand, extra_feats(F, None, cand, n)): r.update(e)
    sig = frame_signals(F, p)
    for r, e in zip(cand, video_feats(F, sig, cand)): r.update(e)
    tab = np.nan_to_num(np.array([[r[f] for f in ALL_FEATS] for r in cand], dtype=np.float32))
    win = np.stack([window(sig, r["a"], r["b"]) for r in cand]).astype(np.float32)
    return cand, tab, win


def verify3(probs, frames, bundle):
    """bundle = joblib.load('verifier_v15_idea12.joblib'). Same output as verify / verify2: (scenes, info)."""
    n = len(frames)
    cand, tab, win = build_inputs(probs, frames, bundle.get("base_thr", 0.296), bundle.get("low_thr", 0.10))
    if not cand: return np.array([[0, n - 1]]), []
    pk = predict_parts(bundle["parts"], tab, win)
    keep = pk >= float(bundle["tau"])
    kept = [r for r, k in zip(cand, keep) if k]
    scenes, used = build_scenes([(r["a"], r["b"]) for r in kept], n)
    used_set = set(used)
    final = {id(r): (i in used_set) for i, r in enumerate(kept)}
    info = [dict(a=r["a"], b=r["b"], src=r["src"], p_keep=float(q), kept=bool(final.get(id(r), False))) for r, q in zip(cand, pk)]
    return scenes, info
