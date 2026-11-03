"""
sbdv_export.py - turn a trained verifier (a joblib bundle from the Python project, or models you train yourself) into a
portable .sbdm file that the C++ tool reads.  Needs numpy; scikit-learn is only needed to READ the bundle (it is already in Colab).

    python sbdv_export.py verifier_v15_reinforced.joblib  autoshot_v2.sbdm
    python sbdv_export.py omni_verifier_v15.joblib        omni.sbdm

The bundle type is detected from its keys:
    "model"  + "features"                 -> sbd_verifier.py   (v1)  one model
    "models" + "features"                 -> sbd_verifier2.py  (v2)  average of models, near-duplicate guard
    "parts"                               -> sbd_verifier3.py  (v3)  average of tree models and small neural nets (TCN)
    "models" + "feats" + "tau_o"          -> sbd_omni_verifier.py    OmniShotCut verifier (works on any list of cuts)

Every exported model is checked against the original (sklearn predict_proba) on test rows before the file is written.
A model whose features are all "local" (none of GLOBAL_FEATURES) runs in constant-memory stream mode in the command line tool.
Supported scikit-learn models: HistGradientBoosting, GradientBoosting, RandomForest, ExtraTrees, DecisionTree, LogisticRegression,
and Pipeline(StandardScaler, one of those).  Anything else stops with a clear message instead of exporting something wrong.
"""
import struct, sys, math
import numpy as np

FEATURES = ("d_adj d_ctx spike persist hist_d bright ld lf p_at p_near gap is_low gap_prev gap_next density "
            "mc_d mc_ratio edge_d edge_rel frac05 frac10 frac20 ctr_d brd_d d_wide wide_ratio pre_stab post_stab cut_vs_stab n15 dup_dist dup_p nb_p_prev nb_p_next p_rank "
            "vid_motion vid_motion_p90 vid_bright vid_contrast vid_len vid_low_share d_adj_rel p_vs_vid "
            "lab_hard_cut lab_sudden_jump lab_transition lab_new_start n_dropped is_rescue as_p_at as_p_near as_agrees "
            "n60 loc_motion d_adj_loc gprev gnext").split()
# numbers that need the WHOLE video (or its length): a model that uses any of them cannot run in constant-memory stream mode
GLOBAL_FEATURES = ("density gap_next vid_motion vid_motion_p90 vid_bright vid_contrast vid_len vid_low_share d_adj_rel p_vs_vid").split()
M_LOGREG, M_BOOST, M_BAG, M_TCN = 1, 2, 3, 4


class ExportError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- members (plain dicts)
def _forest_member(kind, cols, bias, feat, thr, left, right, val, roots):
    return dict(type=kind, cols=list(cols), bias=float(bias), feat=np.asarray(feat, np.int32), thr=np.asarray(thr, np.float64),
                left=np.asarray(left, np.int32), right=np.asarray(right, np.int32), val=np.asarray(val, np.float64), roots=np.asarray(roots, np.int32))


def _unscale_thr(thr, feat, mean, scale):
    """a tree trained on (x-mean)/scale: x_scaled <= t  <=>  x <= t*scale+mean"""
    out = np.array(thr, np.float64, copy=True)
    for i, f in enumerate(feat):
        if f >= 0:
            out[i] = out[i] * scale[f] + mean[f]
    return out


def _sk_tree_arrays(tree, base, value_fn):
    """arrays of one sklearn tree_ in the common node format; children are made absolute (offset base)"""
    n = tree.node_count
    feat = np.where(tree.children_left == -1, -1, tree.feature).astype(np.int32)
    left = np.where(tree.children_left == -1, -1, tree.children_left + base).astype(np.int32)
    right = np.where(tree.children_right == -1, -1, tree.children_right + base).astype(np.int32)
    thr = np.where(tree.children_left == -1, 0.0, tree.threshold).astype(np.float64)
    val = value_fn(tree).astype(np.float64)
    return feat, thr, left, right, val


def sklearn_member(est, cols):
    """one fitted scikit-learn estimator (or a Pipeline of StandardScaler + estimator) -> member dict"""
    from sklearn.pipeline import Pipeline
    mean = scale = None
    if isinstance(est, Pipeline):
        steps = [s for _, s in est.steps]
        if len(steps) == 2 and type(steps[0]).__name__ == "StandardScaler":
            sc = steps[0]; mean = np.asarray(sc.mean_ if sc.with_mean else np.zeros(len(sc.scale_)), np.float64)
            scale = np.asarray(sc.scale_ if sc.with_std else np.ones(len(mean)), np.float64); est = steps[1]
        elif len(steps) == 1:
            est = steps[0]
        else:
            raise ExportError("unsupported Pipeline: " + str([type(s).__name__ for s in steps]))
    name = type(est).__name__
    if name == "LogisticRegression":
        if est.coef_.shape[0] != 1: raise ExportError("LogisticRegression must be binary")
        d = dict(type=M_LOGREG, cols=list(cols), w=np.asarray(est.coef_[0], np.float64), b=float(est.intercept_[0]))
        if mean is not None: d.update(mean=mean, scale=scale)
        return d
    feats, thrs, lefts, rights, vals, roots = [], [], [], [], [], []
    def add(tree, value_fn):
        base = sum(len(f) for f in feats); roots.append(base)
        f, t, l, r, v = _sk_tree_arrays(tree, base, value_fn)
        feats.append(f); thrs.append(t); lefts.append(l); rights.append(r); vals.append(v)
    if name == "HistGradientBoostingClassifier":
        if getattr(est, "n_trees_per_iteration_", 1) != 1: raise ExportError("HistGradientBoosting must be binary")
        feat, thr, left, right, val, rts = [], [], [], [], [], []
        for it in est._predictors:
            nodes = it[0].nodes; base = len(feat); rts.append(base)
            if np.any(nodes["is_categorical"]): raise ExportError("categorical splits are not supported")
            for nd in nodes:
                if nd["is_leaf"]:
                    feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(float(nd["value"]))
                else:
                    feat.append(int(nd["feature_idx"])); thr.append(float(nd["num_threshold"])); left.append(base + int(nd["left"])); right.append(base + int(nd["right"])); val.append(0.0)
        bias = float(np.ravel(est._baseline_prediction)[0])
        d = _forest_member(M_BOOST, cols, bias, feat, thr, left, right, val, rts)
    elif name == "GradientBoostingClassifier":
        if est.n_classes_ != 2: raise ExportError("GradientBoosting must be binary")
        lr = float(est.learning_rate)
        if est.init_ == "zero": raw0 = 0.0
        else:
            p1 = float(est.init_.predict_proba(np.zeros((1, est.n_features_in_)))[0, 1]); raw0 = math.log(p1 / (1.0 - p1))
        for t in est.estimators_[:, 0]: add(t.tree_, lambda tr: tr.value[:, 0, 0] * lr)
        d = _forest_member(M_BOOST, cols, raw0, np.concatenate(feats), np.concatenate(thrs), np.concatenate(lefts), np.concatenate(rights), np.concatenate(vals), roots)
    elif name in ("RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier"):
        if list(est.classes_) != [0, 1]: raise ExportError("classes must be [0, 1]")
        def p1(tr):
            v = tr.value[:, 0, :]; return v[:, 1] / np.maximum(v.sum(1), 1e-300)
        for t in (est.estimators_ if name != "DecisionTreeClassifier" else [est]): add(t.tree_, p1)
        d = _forest_member(M_BAG, cols, 0.0, np.concatenate(feats), np.concatenate(thrs), np.concatenate(lefts), np.concatenate(rights), np.concatenate(vals), roots)
    else:
        raise ExportError(f"model type {name} is not supported by the exporter (supported: HistGradientBoosting, GradientBoosting, RandomForest, ExtraTrees, DecisionTree, LogisticRegression, Pipeline(StandardScaler, ...))")
    if mean is not None: d["thr"] = _unscale_thr(d["thr"], d["feat"], mean, scale)
    return d


def tcn_member(d):
    """TCNModel.export() dict of sbd_verifier3.py -> member dict"""
    st = {k: np.asarray(v, np.float32) for k, v in d["state"].items()}
    cols = list(d["cols"]); nt = len(cols)
    def g(k): return np.ascontiguousarray(st[k], np.float32)
    need = ["conv.0.weight", "conv.0.bias", "conv.2.weight", "conv.2.bias", "conv.4.weight", "conv.4.bias", "head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias"]
    for k in need:
        if k not in st: raise ExportError("TCN state is missing " + k)
    if g("conv.0.weight").shape != (24, 6, 5) or g("conv.2.weight").shape != (24, 24, 5) or g("conv.4.weight").shape != (32, 24, 3):
        raise ExportError("TCN has a different shape than the one in sbd_verifier3.py")
    m = dict(type=M_TCN, cols=cols, n_ch=int(d["n_ch"]), n_tab=nt, c0w=g("conv.0.weight"), c0b=g("conv.0.bias"), c1w=g("conv.2.weight"), c1b=g("conv.2.bias"),
             c2w=g("conv.4.weight"), c2b=g("conv.4.bias"), h0w=g("head.0.weight"), h0b=g("head.0.bias"), h1w=g("head.3.weight"), h1b=g("head.3.bias"))
    if nt:
        m.update(mu=np.asarray(d["mu"], np.float32), sd=np.asarray(d["sd"], np.float32), tabw=g("tab.0.weight"), tabb=g("tab.0.bias"))
    else:
        m.update(mu=np.zeros(0, np.float32), sd=np.zeros(0, np.float32), tabw=np.zeros(0, np.float32), tabb=np.zeros(0, np.float32))
    if m["n_ch"] != 6: raise ExportError("TCN must have 6 input curves")
    return m


# ----------------------------------------------------------------------------------------------- pure python evaluation (self-check of the export)
def member_predict(m, X):
    """probability of one member on a feature matrix X (rows = candidates, columns = the member's cols, in order)"""
    X = np.asarray(X, np.float64)
    if m["type"] == M_LOGREG:
        Z = X if "mean" not in m else (X - m["mean"]) / m["scale"]
        return 1.0 / (1.0 + np.exp(-(Z @ m["w"] + m["b"])))
    if m["type"] in (M_BOOST, M_BAG):
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            tot = 0.0
            for r in m["roots"]:
                n = int(r)
                while m["feat"][n] >= 0: n = int(m["left"][n] if x[m["feat"][n]] <= m["thr"][n] else m["right"][n])
                tot += m["val"][n]
            out[i] = (1.0 / (1.0 + math.exp(-(m["bias"] + tot)))) if m["type"] == M_BOOST else tot / len(m["roots"])
        return out
    raise ExportError("member_predict: TCN needs windows")


def tcn_forward(m, win, tab=None):
    """numpy version of the network of sbd_verifier3._build_net (cross-correlation, zero padding, dilation) - reference for tests"""
    def conv(x, w, b, dil, pad):
        B, Ci, L = x.shape; Co, _, K = w.shape; xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
        out = np.zeros((B, Co, L), np.float32) + b[None, :, None]
        for k in range(K):
            out += np.einsum("bil,oi->bol", xp[:, :, k * dil:k * dil + L], w[:, :, k]).astype(np.float32)
        return np.maximum(out, 0)
    h = conv(win.astype(np.float32), m["c0w"], m["c0b"], 1, 2); h = conv(h, m["c1w"], m["c1b"], 2, 4); h = conv(h, m["c2w"], m["c2b"], 2, 2)
    c = h.shape[2] // 2
    z = np.concatenate([h.max(axis=2), h[:, :, c - 4:c + 4].mean(axis=2)], axis=1)
    if m["n_tab"]:
        t = np.clip((np.nan_to_num(tab).astype(np.float32) - m["mu"]) / m["sd"], -8, 8)
        z = np.concatenate([z, np.maximum(t @ m["tabw"].T + m["tabb"], 0)], axis=1)
    hid = np.maximum(z @ m["h0w"].T + m["h0b"], 0)
    return 1.0 / (1.0 + np.exp(-(hid @ m["h1w"].T + m["h1b"]).squeeze(1)))


# ----------------------------------------------------------------------------------------------- file writer
def _str(s):
    b = s.encode(); return struct.pack("<I", len(b)) + b


def write_model(path, name, members, input_kind, tau, tau_rescue=None, base_thr=0.296, low_thr=0.10, rescue=False, r_ratio=4.0, r_min=0.05, dedupe=0, snap_w=0):
    for m in members:
        for c in m["cols"]:
            if c not in FEATURES: raise ExportError("unknown feature name " + c)
            if c.startswith("as_"): raise ExportError("feature " + c + " (hybrid diagnostic) is not supported")
    out = [b"SBDVMDL\x01", struct.pack("<III", 1, int(input_kind), 1 if rescue else 0),
           struct.pack("<6d", float(tau), float(tau if tau_rescue is None else tau_rescue), float(base_thr), float(low_thr), float(r_ratio), float(r_min)),
           struct.pack("<ii", int(dedupe), int(snap_w)), _str(name), struct.pack("<I", len(members))]
    for m in members:
        out.append(struct.pack("<II", m["type"], len(m["cols"]))); out += [_str(c) for c in m["cols"]]
        if m["type"] == M_LOGREG:
            hs = "mean" in m; out.append(struct.pack("<I", int(hs)))
            if hs: out += [np.asarray(m["mean"], "<f8").tobytes(), np.asarray(m["scale"], "<f8").tobytes()]
            out += [np.asarray(m["w"], "<f8").tobytes(), struct.pack("<d", m["b"])]
        elif m["type"] in (M_BOOST, M_BAG):
            out.append(struct.pack("<dII", m["bias"], len(m["roots"]), len(m["feat"])))
            out += [np.asarray(m["feat"], "<i4").tobytes(), np.asarray(m["thr"], "<f8").tobytes(), np.asarray(m["left"], "<i4").tobytes(),
                    np.asarray(m["right"], "<i4").tobytes(), np.asarray(m["val"], "<f8").tobytes(), np.asarray(m["roots"], "<i4").tobytes()]
        else:
            out.append(struct.pack("<II", m["n_ch"], m["n_tab"]))
            for k in ("mu", "sd", "c0w", "c0b", "c1w", "c1b", "c2w", "c2b", "tabw", "tabb", "h0w", "h0b", "h1w", "h1b"):
                out.append(np.ascontiguousarray(m[k], "<f4").tobytes())
    blob = b"".join(out)
    with open(path, "wb") as f: f.write(blob)
    return len(blob)


# ----------------------------------------------------------------------------------------------- bundles
def _g(b, k, default=None):
    return b[k] if k in b else default


def export_bundle(bundle, out_path, check_X=None, name=None, verbose=True):
    """bundle: a dict (already loaded with joblib) or a path to a .joblib file.  check_X: optional dict {feature name: 1-D array} of REAL
    feature values to test the export on (otherwise random rows around the split thresholds are used)."""
    if isinstance(bundle, str):
        import joblib; bundle = joblib.load(bundle)
    members = []; kw = {}
    if "parts" in bundle:
        kind = "autoshot-v3"; kw.update(input_kind=0, tau=float(bundle["tau"]))
        for d in bundle["parts"]:
            members.append(tcn_member(d) if d["type"] == "tcn" else sklearn_member(d["model"], d["cols"]))
    elif "models" in bundle and "tau_o" in bundle:
        kind = "omni"; kw.update(input_kind=1, tau=float(bundle["tau_o"]), tau_rescue=float(bundle["tau_r"]), rescue=bool(bundle["rescue"]),
                                 r_ratio=float(_g(bundle, "r_ratio", 4.0)), r_min=float(_g(bundle, "r_min", 0.05)), snap_w=int(_g(bundle, "snap_w", 0)))
        for m in bundle["models"]: members.append(sklearn_member(m, bundle["feats"]))
    elif "models" in bundle:
        kind = "autoshot-v2"; kw.update(input_kind=0, tau=float(bundle["tau"]), dedupe=int(_g(bundle, "dedupe", 0)))
        for m in bundle["models"]: members.append(sklearn_member(m, bundle["features"]))
    elif "model" in bundle:
        kind = "autoshot-v1"; kw.update(input_kind=0, tau=float(bundle["tau"]))
        members.append(sklearn_member(bundle["model"], bundle["features"]))
    else:
        raise ExportError("cannot recognise this bundle; its keys are " + str(list(bundle.keys())))
    if kw["input_kind"] == 0: kw.update(base_thr=float(_g(bundle, "base_thr", 0.296)), low_thr=float(_g(bundle, "low_thr", 0.10)))
    # ---- self-check of the tree/linear members against the original estimators
    rng = np.random.default_rng(0); checked = 0; worst = 0.0
    originals = []
    if "parts" in bundle: originals = [None if d["type"] == "tcn" else d["model"] for d in bundle["parts"]]
    elif "models" in bundle: originals = list(bundle["models"])
    else: originals = [bundle["model"]]
    for m, orig in zip(members, originals):
        if m["type"] == M_TCN or orig is None: continue
        nf = len(m["cols"]); rows = []
        if check_X is not None:
            rows.append(np.stack([np.asarray(check_X[c], np.float64) for c in m["cols"]], 1))
        thr_pool = [m["thr"][m["feat"] == j] for j in range(nf)] if "feat" in m else [np.zeros(0)] * nf
        R = np.zeros((2000, nf))
        for j in range(nf):
            base = rng.normal(0, 1, 2000)
            if len(thr_pool[j]):
                pick = rng.choice(thr_pool[j], 2000); jitter = rng.choice([-1e-3, 0.0, 1e-3, 0.3, -0.3], 2000) * (np.abs(pick) + 1)
                base = np.where(rng.random(2000) < 0.8, pick + jitter, base)
            R[:, j] = base
        rows.append(R); X = np.concatenate(rows).astype(np.float32).astype(np.float64)
        ref = orig.predict_proba(X)[:, 1]; mine = member_predict(m, X)
        worst = max(worst, float(np.max(np.abs(ref - mine)))); checked += 1
    if worst > 1e-6: raise ExportError(f"export does not reproduce sklearn (max difference {worst:.2e}); not written")
    nm = name or kind
    nbytes = write_model(out_path, nm, members, **kw)
    if verbose:
        trees = sum(len(m["roots"]) for m in members if "roots" in m); nets = sum(1 for m in members if m["type"] == M_TCN)
        print(f"{nm}: {len(members)} member(s), {trees} trees, {nets} neural net(s) -> {out_path} ({nbytes / 1024:.1f} KB); checked {checked} member(s) vs sklearn, max difference {worst:.1e}")
    return dict(kind=kind, members=members, bytes=nbytes, worst=worst, kw=kw)


# ----------------------------------------------------------------------------------------------- own models (trained by sbdv_train.py)
def is_streamable(feats):
    """True when no feature of the list needs the whole video (so the model can run frame by frame with constant memory)"""
    return not (set(feats) & set(GLOBAL_FEATURES))


def export_estimator(est, feats, out_path, tau, name="verifier", check_X=None, verbose=True):
    """export ONE fitted scikit-learn estimator (HistGradientBoosting, GradientBoosting, RandomForest, LogisticRegression ...) that was trained on the
    columns `feats` (feature names of the C++ tool, in that order) as a cut-list verifier.  The export is checked against sklearn first."""
    bundle = dict(models=[est], feats=list(feats), tau_o=float(tau), tau_r=float(tau), rescue=False, snap_w=0)
    return export_bundle(bundle, out_path, check_X=check_X, name=name, verbose=verbose)


def selftest_export(verbose=True):
    """Can the scikit-learn installed here be exported exactly?  Trains a tiny model on random data and exports it.  Returns the model kind that works
    ("hgb" first, then "gbc"), or raises ExportError when none does."""
    import tempfile, os
    rng = np.random.default_rng(0); X = rng.normal(size=(600, 5)); y = (X[:, 0] + 0.5 * X[:, 1] * X[:, 2] > 0).astype(int)
    feats = ["d_adj", "d_ctx", "spike", "persist", "hist_d"]; last = None
    for kind in ("hgb", "gbc"):
        try:
            if kind == "hgb":
                from sklearn.ensemble import HistGradientBoostingClassifier as C
                est = C(max_iter=15, max_depth=3, early_stopping=False, random_state=0).fit(X, y)
            else:
                from sklearn.ensemble import GradientBoostingClassifier as C
                est = C(n_estimators=15, max_depth=3, random_state=0).fit(X, y)
            fd, p = tempfile.mkstemp(suffix=".sbdm"); os.close(fd)
            try:
                export_estimator(est, feats, p, 0.5, name="selftest", verbose=False)
            finally:
                os.remove(p)
            if verbose: print(f"export self-test: scikit-learn model type '{kind}' can be exported exactly")
            return kind
        except Exception as e:  # noqa
            last = e
            if verbose: print(f"export self-test: '{kind}' cannot be exported here ({type(e).__name__}: {str(e)[:120]})")
    raise ExportError("no scikit-learn model type could be exported: " + str(last))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    export_bundle(sys.argv[1], sys.argv[2])
