"""
sbdv_train.py - train, test and export a cut verifier for YOUR detector.  Python + scikit-learn are only needed for this (training); using the
exported .sbdm model needs neither.

The recipe (README section 4):
    items = [dict(vid="a", frames=frames_uint8, cuts=[(a, b), ...], gt=ground_truth_scenes, det="mydetector"), ...]   # one entry per (video, detector)
    df, X, names = build_table(engine, items)                    # every candidate cut + its picture numbers + label (1 = a real cut)
    res = cross_validate(df, X, names, FEATS_LOCAL, items, ["mydetector"], ["mydetector"])   # honest test, split BY VIDEO
    export_generic(df, X, names, FEATS_LOCAL, res["tau"], "mine.sbdm")                         # the file the C++ tool reads

What "honest" means here (the mistakes this version fixes are marked FIX):
  * the data are split BY VIDEO into folds; the verifier that judges a video never saw that video (and neither did the threshold choice);
  * FIX: videos for which the detector proposed NO cut are counted too (their missed cuts are false negatives).  The first version of this file
    dropped them, which made the false-negative count too small;
  * FIX: every row that reports "detector alone" is checked to add up: TP + FN must equal the number of real cuts (check_totals).
"""
import numpy as np, pandas as pd

# ------------------------------------------------------------------------------------------------ the picture numbers a model may use
# LOCAL numbers only look at the frames around the candidate: such a model runs in constant memory, frame by frame, on videos of any length.
FEATS_LOCAL = ("d_adj d_ctx spike persist hist_d bright ld lf gap gprev gnext n60 mc_d mc_ratio edge_d edge_rel frac05 frac10 frac20 ctr_d brd_d "
               "d_wide wide_ratio pre_stab post_stab cut_vs_stab n15 dup_dist loc_motion d_adj_loc").split()
# the numbers of the first generic model also used whole-video numbers (and the video length): they need all frames in memory
FEATS_GLOBAL_EXTRA = "density gap_next vid_motion vid_motion_p90 vid_bright vid_contrast vid_len d_adj_rel".split()
FEATS_FULL = FEATS_LOCAL + FEATS_GLOBAL_EXTRA
FEATS_TINY = "d_adj persist hist_d bright".split()          # four numbers: a deliberately simple model to compare against
FEATS = FEATS_LOCAL                                          # default


# ------------------------------------------------------------------------------------------------ scoring (the official SHOT rule)
def count_scenes(gt, pred, tol=2):
    """AutoShot's official counting (evaluate_scenes): returns [TP, FP, FN].  Same rule as tricut2_lib.count_scenes."""
    def tr(s):
        sh = tol / 2; s = np.asarray(s, np.float32).reshape(-1, 2) + np.array([[-0.5 + sh, 0.5 - sh]]); return np.stack([s[:-1, 1], s[1:, 0]], 1)
    g, p = tr(gt), tr(pred); i = j = tp = 0
    while i < len(g) and j < len(p):
        if p[j, 1] < g[i, 0]: j += 1
        elif p[j, 0] > g[i, 1]: i += 1
        else: tp += 1; i += 1; j += 1
    return np.array([tp, len(p) - tp, len(g) - tp])


def f1(c):
    return 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)


def prf(c):
    tp, fp, fn = [float(x) for x in c]
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); return p, r, 2 * tp / max(2 * tp + fp + fn, 1)


def scenes_from_cuts(ab, T):
    """final shots from kept cuts (a,b): cuts that overlap an earlier one are skipped, exactly like build_scenes() in the C++ code"""
    out, start = [], 0
    for a, b in sorted(map(tuple, ab)):
        if a < start or b <= a or b > T - 1: continue
        out.append([start, a]); start = b
    out.append([start, T - 1]); return np.array(out)


def clip_cuts(cuts, T):
    """cuts that point outside a video of T frames (or make no sense) are removed; returns (kept cuts as int array [n,2], number removed)"""
    c = np.asarray(cuts, np.int64).reshape(-1, 2)
    ok = (c[:, 0] >= 0) & (c[:, 1] > c[:, 0]) & (c[:, 1] <= T - 1)
    return c[ok], int((~ok).sum())


def _label(a, b, gt):
    """1 when the candidate (a, b) overlaps a real cut of the ground truth, with the same tolerance as the official scorer"""
    for i in range(len(gt) - 1):
        ga, gb = int(gt[i][1]), int(gt[i + 1][0])
        if a - 0.5 <= gb + 0.5 and b + 0.5 >= ga - 0.5: return 1
    return 0


# ------------------------------------------------------------------------------------------------ a simple detector, to test "any detector"
def classic_cuts(frames, thr=0.12, min_gap=6):
    """A plain frame-difference detector (our own, written for this test; in the spirit of classic threshold detectors, NOT PySceneDetect itself):
    a cut where the mean absolute change between two neighbouring 48x27 frames (0..1) is above `thr` and is the biggest within `min_gap` frames."""
    f = np.asarray(frames)
    if len(f) < 2: return np.zeros((0, 2), np.int64)
    d = np.zeros(len(f), np.float32)
    for t0 in range(1, len(f), 512):                                         # in blocks, to keep the memory small
        t1 = min(len(f), t0 + 512); d[t0:t1] = np.abs(f[t0:t1].astype(np.int16) - f[t0 - 1:t1 - 1].astype(np.int16)).mean(axis=(1, 2, 3)) / 255.0
    idx = np.where(d > thr)[0]; out = []
    for t in idx:
        if out and t - out[-1] < min_gap:
            if d[t] > d[out[-1]]: out[-1] = t
        else: out.append(int(t))
    return np.array([(t - 1, t) for t in out], np.int64).reshape(-1, 2)


# ------------------------------------------------------------------------------------------------ candidate table
def build_table(engine, items, log=print):
    """engine: an sbdv.Model (any .sbdm; only its candidate rules are used).  items: list of dict(vid, frames, cuts, gt, det[, scores]).
    Returns (df, X, names).  df has one row per candidate cut: item (index in items), vid, det, a, b, src, label."""
    rows, Xs, names, skipped = [], [], None, 0
    for k, it in enumerate(items):
        T = len(it["frames"]); cuts, bad = clip_cuts(it["cuts"], T); skipped += bad
        ab, src, X, names = engine.features(it["frames"], cuts=cuts, scores=it.get("scores"), lenient=True)
        for (a, b), s in zip(ab, src): rows.append((k, it["vid"], it["det"], int(a), int(b), int(s), _label(int(a), int(b), it["gt"])))
        Xs.append(X)
    df = pd.DataFrame(rows, columns=["item", "vid", "det", "a", "b", "src", "label"]); X = np.concatenate(Xs) if Xs else np.zeros((0, 1), np.float32)
    if log: log(f"{len(df)} candidates from {len(items)} runs; {df.label.mean():.3f} of them are real cuts" + (f"; {skipped} detector cut(s) pointed outside their video and were left out" if skipped else ""))
    return df, X, names


def counts_by_item(df, keep, items, dets):
    """{item index: [TP, FP, FN]} for EVERY item whose detector is in `dets`: a video without any kept cut is one shot, and its real cuts are all misses.
    (FIX: the first version only looked at videos that had candidate rows.)"""
    dets = [dets] if isinstance(dets, str) else list(dets)
    rows_of = {}
    for r, k in enumerate(df.item.values): rows_of.setdefault(int(k), []).append(r)
    a_all, b_all = df.a.values, df.b.values; out = {}
    for k, it in enumerate(items):
        if it["det"] not in dets: continue
        idx = np.array(rows_of.get(k, []), dtype=int); idx = idx[keep[idx]] if len(idx) else idx
        out[k] = count_scenes(it["gt"], scenes_from_cuts(zip(a_all[idx], b_all[idx]), len(it["frames"])))
    return out


def total_real_cuts(items, dets):
    dets = [dets] if isinstance(dets, str) else list(dets)
    return int(sum(len(it["gt"]) - 1 for it in items if it["det"] in dets))


def check_totals(per, items, dets):
    """TP + FN must equal the number of real cuts, for every row of every table.  Raises when a video was lost on the way."""
    want = total_real_cuts(items, dets); got = int(sum(c[0] + c[2] for c in per.values()))
    if got != want: raise AssertionError(f"TP+FN = {got} but the ground truth has {want} real cuts: some videos were lost")
    return want


# ------------------------------------------------------------------------------------------------ models
def make_model(kind="hgb", n_iter=60, depth=4, seed=0):
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier as C
        return C(max_iter=n_iter, max_depth=depth, learning_rate=0.1, early_stopping=False, random_state=seed)
    if kind == "gbc":
        from sklearn.ensemble import GradientBoostingClassifier as C
        return C(n_estimators=n_iter, max_depth=depth, learning_rate=0.1, subsample=1.0, random_state=seed)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    raise ValueError("model kind must be hgb, gbc or logreg")


def _fit(X, y, kind, n_iter, depth, seed):
    return make_model(kind, n_iter, depth, seed).fit(X, y)


# ------------------------------------------------------------------------------------------------ honest test, split by video
def cross_validate(df, X, names, feats, items, train_dets, test_dets, kind="hgb", n_iter=60, depth=4, folds=5, seed=0, grid=np.arange(0.2, 0.81, 0.05), log=print):
    """Split BY VIDEO.  For each fold the keep-threshold is chosen on the training videos only (inner 3-fold).  Returns per-video counts with and without
    the verifier.  train_dets / test_dets: which detectors' candidates are used for training / judged (they may differ: test on a detector never seen)."""
    train_dets, test_dets = ([train_dets] if isinstance(train_dets, str) else list(train_dets)), ([test_dets] if isinstance(test_dets, str) else list(test_dets))
    cols = [names.index(f) for f in feats]; Xf = X[:, cols]; y = df.label.values
    vids = np.array(sorted(set(it["vid"] for it in items))); rng = np.random.default_rng(seed); fold_of = dict(zip(vids, rng.permutation(len(vids)) % folds))
    fv = df.vid.map(fold_of).values; tr_det = df.det.isin(train_dets).values; te_det = df.det.isin(test_dets).values
    prob = np.full(len(df), np.nan); tau_of_row = np.full(len(df), np.nan); taus = []
    for f in range(folds):
        tr = (fv != f) & tr_det; te = (fv == f) & te_det
        tr_vids = sorted(set(df.vid[tr])); inner = {v: i % 3 for i, v in enumerate(tr_vids)}; iv = df.vid.map(inner).values; oof = np.full(len(df), np.nan)
        for g in range(3):
            a = tr & (iv != g); b = tr & (iv == g)
            if a.sum() and b.sum() and len(np.unique(y[a])) > 1: oof[b] = _fit(Xf[a], y[a], kind, n_iter, depth, seed).predict_proba(Xf[b])[:, 1]
        best, tau = -1, 0.5
        for t in grid:
            keep = np.where(np.isnan(oof), True, oof >= t)                       # (rows without an inner prediction are only in tiny test setups)
            c = sum(counts_by_item(df, keep & tr, items, train_dets).values()); s = f1(c)
            if s > best: best, tau = s, float(t)
        taus.append(tau)
        if te.sum(): prob[te] = _fit(Xf[tr], y[tr], kind, n_iter, depth, seed).predict_proba(Xf[te])[:, 1]
        tau_of_row[te] = tau
    keep = np.where(np.isnan(prob), False, prob >= tau_of_row)
    res = dict(tau=float(np.median(taus)), taus=taus, prob=prob, keep=keep, per={}, fold_of=fold_of, feats=list(feats), kind=kind)
    for d in test_dets:
        # rows of OTHER detectors must not be kept: only this detector's rows count for its items
        ver = counts_by_item(df, keep & (df.det == d).values, items, d); pla = counts_by_item(df, np.ones(len(df), bool), items, d)
        check_totals(ver, items, d); check_totals(pla, items, d)
        res["per"][d] = dict(verified=ver, plain=pla)
        cv_, cp_ = sum(ver.values()), sum(pla.values())
        if log: log(f"{d:14s} detector alone: TP/FP/FN {cp_.tolist()} F1 {f1(cp_):.4f}  ->  with verifier: {cv_.tolist()} F1 {f1(cv_):.4f}")
    return res


def per_video(per_item, items, det):
    """{vid: counts} of one detector, from the {item index: counts} dict of cross_validate"""
    return {items[k]["vid"]: c for k, c in per_item.items() if items[k]["det"] == det}


def boot_diff(per_a, per_b, vids=None, n=2000, seed=0):
    """paired bootstrap over VIDEOS: how much better is A than B in F1?  per_a / per_b = {vid: [TP,FP,FN]}.  Returns (mean gain, low, high, verdict);
    the verdict is BETTER / WORSE only when the 95% range excludes 0, otherwise 'cannot be told apart'."""
    vids = sorted(per_a) if vids is None else [v for v in vids if v in per_a]
    A = np.array([per_a[v] for v in vids], float); B = np.array([per_b[v] for v in vids], float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(n, len(vids)))
    def F(M):
        s = M[idx].sum(1); return 2 * s[:, 0] / np.maximum(2 * s[:, 0] + s[:, 1] + s[:, 2], 1)
    d = F(A) - F(B); lo, hi = np.percentile(d, [2.5, 97.5])
    full = f1(A.sum(0)) - f1(B.sum(0))
    return float(full), float(lo), float(hi), ("BETTER" if lo > 0 else "WORSE" if hi < 0 else "cannot be told apart")


def summary_table(res, items, test_dets, label=""):
    """one row per tested detector: detector alone vs with verifier, with the bootstrap range"""
    rows = []
    for d in ([test_dets] if isinstance(test_dets, str) else test_dets):
        pl, ve = sum(res["per"][d]["plain"].values()), sum(res["per"][d]["verified"].values())
        g, lo, hi, vd = boot_diff(per_video(res["per"][d]["verified"], items, d), per_video(res["per"][d]["plain"], items, d))
        p0, r0, f0 = prf(pl); p1, r1, f1_ = prf(ve)
        rows.append(dict(test=label, detector=d, alone_TP=int(pl[0]), alone_FP=int(pl[1]), alone_FN=int(pl[2]), alone_P=round(p0, 4), alone_R=round(r0, 4), alone_F1=round(f0, 4),
                         verified_TP=int(ve[0]), verified_FP=int(ve[1]), verified_FN=int(ve[2]), verified_P=round(p1, 4), verified_R=round(r1, 4), verified_F1=round(f1_, 4),
                         F1_gain=round(g, 4), range95=f"{lo:+.4f} to {hi:+.4f}", verdict=vd))
    return pd.DataFrame(rows)


def leave_one_family_out(df, X, names, feats, items, families, **kw):
    """train on every detector family except one, test on the family left out (never seen in training).  families = {name: [det, ...]}"""
    out = []
    for fam, dets in families.items():
        train = [d for f2, ds in families.items() if f2 != fam for d in ds]
        res = cross_validate(df, X, names, feats, items, train, dets, log=None, **kw)
        out.append(summary_table(res, items, dets, label=f"train without {fam}"))
    return pd.concat(out, ignore_index=True)


def model_size(est, feats, tau, tmp_path="/tmp/_sbdv_size.sbdm"):
    """parameters and file size of the exported model"""
    import sbdv_export as EX
    info = EX.export_estimator(est, feats, tmp_path, tau, name="size", verbose=False)
    n_par = 0
    for m in info["members"]:
        n_par += (len(m["val"]) + 1) if "val" in m else (len(m["w"]) + 1)
    return n_par, info["bytes"]


def ablation(df, X, names, items, configs, dets, folds=5, seed=0, log=print):
    """size vs accuracy: configs = [(label, feats, kind, n_iter, depth), ...].  Pooled over all detectors in `dets`; returns a table"""
    rows = []
    for label, feats, kind, n_iter, depth in configs:
        res = cross_validate(df, X, names, feats, items, dets, dets, kind=kind, n_iter=n_iter, depth=depth, folds=folds, seed=seed, log=None)
        ver = sum(sum(res["per"][d]["verified"].values()) for d in dets); pla = sum(sum(res["per"][d]["plain"].values()) for d in dets)
        est = _fit(X[:, [names.index(f) for f in feats]], df.label.values, kind, n_iter, depth, seed)
        n_par, nbytes = model_size(est, feats, res["tau"])
        rows.append(dict(model=label, features=len(feats), trees=n_iter if kind != "logreg" else 0, depth=depth if kind != "logreg" else 0, parameters=n_par, size_KB=round(nbytes / 1024, 1),
                         pooled_F1_alone=round(f1(pla), 4), pooled_F1_verified=round(f1(ver), 4), gain=round(f1(ver) - f1(pla), 4)))
        if log: log(rows[-1])
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ export
def export_generic(df, X, names, feats, tau, path, kind="hgb", n_iter=60, depth=4, seed=0, train_dets=None, name="generic-picture-verifier"):
    """train on all rows (of train_dets) and write the .sbdm file for the C++ tool (works on the cut list of any detector).
    The export is checked against scikit-learn before the file is written.  Returns the export info plus the fitted estimator."""
    import sbdv_export as EX
    cols = [names.index(f) for f in feats]; m = np.ones(len(df), bool) if train_dets is None else df.det.isin(train_dets).values
    clf = _fit(X[m][:, cols], df.label.values[m], kind, n_iter, depth, seed)
    info = EX.export_estimator(clf, feats, path, tau, name=name)
    info["estimator"] = clf; info["streamable"] = EX.is_streamable(feats); return info
