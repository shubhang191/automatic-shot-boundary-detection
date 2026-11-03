"""
run_tests.py - checks that the tool works (on synthetic videos, so it needs no data).    python tests/run_tests.py [--quick]
These tests prove the CODE is right (features, export, streaming, errors, training bookkeeping).  They say nothing about accuracy on real videos.
"""
import os, sys, subprocess, tempfile, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "python"), HERE]
import numpy as np
import sbdv, sbdv_export as EX, sbdv_train as TR, synth

QUICK = "--quick" in sys.argv
BIN_DIR = os.environ.get("SBDV_BIN", os.path.join(ROOT, "bin"))
DBG_DIR = os.environ.get("SBDV_BIN_DEBUG", os.path.join(ROOT, "bin_debug"))
EXE = os.path.join(BIN_DIR, "sbdv"); EXE_DBG = os.path.join(DBG_DIR, "sbdv")
passed, failed = [], []


def check(name, cond, detail=""):
    cond = bool(cond); (passed if cond else failed).append(name); print(("  ok    " if cond else "  FAIL  ") + name + (("   " + str(detail)) if (len(str(detail)) and not cond) else ""))


def run(args, exe=EXE, **kw):
    return subprocess.run([exe] + [str(a) for a in args], capture_output=True, text=True, **kw)


def write_cuts(path, cuts, scores=None):
    with open(path, "w") as f:
        for i, (a, b) in enumerate(cuts): f.write(f"{a} {b}" + (f" {scores[i]}" if scores is not None else "") + "\n")


def items_from(ds, det_name="fake", seed=1, recall=0.93, **kw):
    rng = np.random.default_rng(seed); items = []
    for vid, d in ds.items():
        T = len(d["frames"]); cuts = synth.fake_detector(rng, T, d["cuts"], d["events"], recall=recall, **kw)
        items.append(dict(vid=vid, frames=d["frames"], cuts=cuts, gt=d["gt"], det=det_name))
    return items


tmp = tempfile.mkdtemp(prefix="sbdv_test_")
print("sbdv version:", sbdv.version(), "| binary:", EXE)

# ---------------------------------------------------------------------------------------------- 1. export self-test and parity
print("\n[1] export: the .sbdm file must reproduce scikit-learn exactly")
ds = synth.make_dataset(30 if not QUICK else 14, seed=3); items = items_from(ds)
dummy = os.path.join(tmp, "dummy.sbdm")
KIND = os.environ.get("SBDV_FORCE_KIND") or EX.selftest_export(verbose=False); check("export self-test finds a working model type", KIND in ("hgb", "gbc"), KIND)
# the first export needs a model file to compute candidate features: make a tiny one from random data
rng = np.random.default_rng(0)
tiny = TR.make_model("logreg").fit(rng.normal(size=(200, 4)), rng.integers(0, 2, 200))
EX.export_estimator(tiny, TR.FEATS_TINY, dummy, 0.5, name="dummy", verbose=False)
eng = sbdv.Model(dummy)
df, X, names = TR.build_table(eng, items, log=None)
check("feature table has all feature names", len(names) == X.shape[1] and "d_adj" in names and "gprev" in names, names[:5])
check("every row is labelled 0/1 and both classes exist", set(df.label.unique()) == {0, 1}, df.label.value_counts().to_dict())

for kind_, fe in tuple((k, TR.FEATS_LOCAL) for k in dict.fromkeys([KIND, "gbc", "logreg"])) + ((KIND, TR.FEATS_FULL),):
    info = TR.export_generic(df, X, names, fe, 0.5, os.path.join(tmp, f"{kind_}_{len(fe)}.sbdm"), kind=kind_, n_iter=25, depth=3)
    check(f"export {kind_}/{len(fe)} feats matches sklearn (max diff {info['worst']:.1e})", info["worst"] < 1e-6)
    m = sbdv.Model(os.path.join(tmp, f"{kind_}_{len(fe)}.sbdm")); i = m.info()
    check(f"  streamable flag correct for {kind_}/{len(fe)}", i["streamable"] == (fe is TR.FEATS_LOCAL), i)
    # full C++ pipeline vs sklearn on the candidate rows
    cols = [names.index(f) for f in fe]; sk = info["estimator"].predict_proba(X[:, cols])[:, 1]
    it = items[0]; r = m.verify(it["frames"], cuts=it["cuts"], lenient=True)
    sel = (df.item.values == 0)
    cpp = r["cands"]["p_keep"]; ref = sk[sel]
    check(f"  C++ p_keep equals sklearn on a whole video ({kind_}/{len(fe)})", len(cpp) == len(ref) and np.allclose(cpp, ref, atol=2e-5), (len(cpp), len(ref), np.abs(cpp - ref).max() if len(cpp) == len(ref) else ""))

# ---------------------------------------------------------------------------------------------- 2. stream mode == RAM mode
print("\n[2] constant-memory stream mode gives the same answer as RAM mode (ring-checked build)")
loc = os.path.join(tmp, f"{KIND}_{len(TR.FEATS_LOCAL)}.sbdm"); tauv = 0.5
nbad = 0; nvid = 0; ncand = 0
for k, it in enumerate(items):
    fr = it["frames"]; fp = os.path.join(tmp, f"v{k}.npy"); np.save(fp, fr); cf = os.path.join(tmp, f"c{k}.txt"); write_cuts(cf, it["cuts"])
    outs = {}
    for mode in ("--ram", "--stream"):
        sh = os.path.join(tmp, f"s_{k}{mode}.txt"); co = os.path.join(tmp, f"co_{k}{mode}.txt")
        r = run(["verify", "--model", loc, "--npy", fp, "--cuts", cf, mode, "--shots", sh, "--cuts-out", co, "--quiet", "--lenient"], exe=EXE_DBG)
        if r.returncode != 0: print("   exit", r.returncode, r.stderr.strip()[:200]); nbad += 1; break
        outs[mode] = (open(sh).read(), open(co).read())
    else:
        nvid += 1; ncand += outs["--ram"][1].count("\n") - 1
        if outs["--ram"] != outs["--stream"]: nbad += 1; print("   differs on video", k)
check(f"stream == RAM (shots and every p_keep) on {nvid} videos / {ncand} candidates", nbad == 0 and nvid == len(items), f"{nbad} videos differ or failed")

# the same on edge cases: a cut right at the start/end, two cuts one frame apart, a cut list that is empty, a very short video
edge = [("cut near start", [(2, 3), (40, 41)]), ("cut at the very end", [(len(items[0]["frames"]) - 2, len(items[0]["frames"]) - 1)]), ("neighbours", [(30, 31), (31, 32), (33, 34)]), ("empty", [])]
fr0 = items[0]["frames"]; fp0 = os.path.join(tmp, "e.npy"); np.save(fp0, fr0); ok = True
for name, cs in edge:
    cf = os.path.join(tmp, "ec.txt"); write_cuts(cf, cs); res = {}
    for mode in ("--ram", "--stream"):
        co = os.path.join(tmp, f"eo{mode}.txt"); sh = os.path.join(tmp, f"es{mode}.txt")
        r = run(["verify", "--model", loc, "--npy", fp0, "--cuts", cf, mode, "--cuts-out", co, "--shots", sh, "--quiet", "--lenient"], exe=EXE_DBG)
        res[mode] = (r.returncode, open(co).read() if r.returncode == 0 else r.stderr, open(sh).read() if r.returncode == 0 else "")
    same = res["--ram"] == res["--stream"] and res["--ram"][0] == 0; ok &= same
    check(f"  edge case '{name}': stream == RAM", same, res)
short = fr0[:30]; np.save(os.path.join(tmp, "short.npy"), short); cf = os.path.join(tmp, "sc.txt"); write_cuts(cf, [(10, 11)]); res = {}
for mode in ("--ram", "--stream"):
    r = run(["verify", "--model", loc, "--npy", os.path.join(tmp, "short.npy"), "--cuts", cf, mode, "--quiet"], exe=EXE_DBG); res[mode] = (r.returncode, r.stdout)
check("  video shorter than the context window: stream == RAM", res["--ram"] == res["--stream"] and res["--ram"][0] == 0, res)

# whole-video models must refuse stream mode, and still work in RAM mode
r = run(["verify", "--model", os.path.join(tmp, f"{KIND}_{len(TR.FEATS_FULL)}.sbdm"), "--npy", fp0, "--cuts", cf, "--stream", "--quiet"])
check("whole-video model refuses --stream with a clear message", r.returncode == 2 and "whole video" in r.stderr, (r.returncode, r.stderr[:150]))

# probability/Python API == CLI on the same input
m = sbdv.Model(loc)
for k in (0, 1, 2):
    it = items[k]; py = m.verify(it["frames"], cuts=it["cuts"], lenient=True); cf = os.path.join(tmp, "pc.txt"); write_cuts(cf, it["cuts"]); fpn = os.path.join(tmp, f"v{k}.npy")
    r = run(["verify", "--model", loc, "--npy", fpn, "--cuts", cf, "--lenient", "--out", os.path.join(tmp, "o.json")]); j = json.load(open(os.path.join(tmp, "o.json")))
    check(f"Python API == command line (video {k})", np.array_equal(np.array(j["scenes"]), py["scenes"]), (j["scenes"][:3], py["scenes"][:3].tolist()))

# ---------------------------------------------------------------------------------------------- 3. memory: stream mode must not grow with the video
print("\n[3] memory: stream mode uses constant memory, RAM mode grows with the video")
if not QUICK:
    base = np.concatenate([it["frames"] for it in items[:6]]); mem = {}; nb = len(base)
    for reps in (6, 60):
        big = np.concatenate([base] * reps); rawp = os.path.join(tmp, f"big{reps}.u8"); big.tofile(rawp); vp = os.path.join(tmp, f"big{reps}.avi"); T = len(big)
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x27", "-r", "25", "-i", rawp, "-c:v", "ffv1", vp], capture_output=True)
        os.remove(rawp)
        cuts = [(nb * i - 1, nb * i) for i in range(1, reps)]; cf = os.path.join(tmp, f"bc{reps}.txt"); write_cuts(cf, cuts)
        for mode in ("--ram", "--stream"):
            o = os.path.join(tmp, f"bo{reps}{mode}.json"); r = run(["verify", "--model", loc, "--video", vp, "--cuts", cf, mode, "--out", o]); j = json.load(open(o)); mem[(reps, mode)] = (j["peak_rss_mb"], j["decoder_peak_rss_mb"], T)
        os.remove(vp)
    print("   (frames, program MB, ffmpeg MB):", {f"{k[0]}x {k[1]}": (v[2], round(v[0], 1), round(v[1], 1)) for k, v in mem.items()})
    grow_stream = mem[(60, "--stream")][0] - mem[(6, "--stream")][0]; grow_ram = mem[(60, "--ram")][0] - mem[(6, "--ram")][0]
    check(f"stream mode: 10x longer video adds < 5 MB (added {grow_stream:.1f} MB)", grow_stream < 5.0, mem)
    check(f"RAM mode really grows with the video (added {grow_ram:.1f} MB, 3.9 KB per frame)", grow_ram > 0.5 * 3.9e-3 * (mem[(60, '--ram')][2] - mem[(6, '--ram')][2]), mem)
else:
    print("   skipped (--quick)")

# ---------------------------------------------------------------------------------------------- 4. errors
print("\n[4] errors are clear, and a mismatch between detector output and video is NOT silently accepted")
cf = os.path.join(tmp, "bad.txt"); T0 = len(fr0); write_cuts(cf, [(10, 11), (T0 + 50, T0 + 51)])
r = run(["verify", "--model", loc, "--npy", fp0, "--cuts", cf, "--quiet"]); check("cut beyond the end -> exit code 3", r.returncode == 3, (r.returncode, r.stderr[:200]))
r = run(["verify", "--model", loc, "--npy", fp0, "--cuts", cf, "--quiet", "--lenient"]); check("same input with --lenient -> exit 0 and a warning in JSON", r.returncode == 0, (r.returncode, r.stderr[:200]))
r = run(["verify", "--model", loc, "--npy", fp0, "--cuts", cf, "--lenient", "--out", os.path.join(tmp, "w.json")]); j = json.load(open(os.path.join(tmp, "w.json")))
check("  warnings list is not empty", len(j.get("warnings", [])) >= 1, j.get("warnings"))
r = run(["verify", "--model", loc, "--npy", fp0]); check("no detector output -> exit code 2", r.returncode == 2, r.returncode)
r = run(["verify", "--model", os.path.join(tmp, "nonexistent.sbdm"), "--npy", fp0, "--cuts", cf]); check("missing model -> exit code 1", r.returncode == 1, r.returncode)
bad = os.path.join(tmp, "bad.sbdm"); open(bad, "wb").write(open(loc, "rb").read()[:50]); r = run(["verify", "--model", bad, "--npy", fp0, "--cuts", cf]); check("truncated model file -> clean error", r.returncode == 1 and "error" in r.stderr, (r.returncode, r.stderr[:100]))
garbage = os.path.join(tmp, "garbage.txt"); open(garbage, "w").write("10 11\nhello world\n"); r = run(["verify", "--model", loc, "--npy", fp0, "--cuts", garbage, "--quiet"]); check("unreadable line in the cut list -> exit 1 (not silently skipped)", r.returncode == 1, (r.returncode, r.stderr[:150]))
try:
    m.verify(fr0, cuts=[(10, 11), (T0 + 50, T0 + 51)]); check("Python API raises on a cut beyond the end", False)
except ValueError:
    check("Python API raises on a cut beyond the end", True)
r = run(["verify", "--model", loc, "--npy", fp0, "--cuts", write_cuts(os.path.join(tmp, "e2.txt"), []) or os.path.join(tmp, "e2.txt"), "--quiet"]); check("empty cut list works (one shot)", r.returncode == 0 and r.stdout.strip() == f"0 {T0 - 1}", (r.returncode, r.stdout))
# 1-based and exclusive-end readings
open(os.path.join(tmp, "sc1.txt"), "w").write("1 40\n41 %d\n" % T0)
r = run(["verify", "--model", loc, "--npy", fp0, "--scenes", os.path.join(tmp, "sc1.txt"), "--base", "1", "--cuts-out", os.path.join(tmp, "sc1o.txt"), "--quiet", "--lenient"])
check("--scenes with --base 1 is read as cut (39, 40) in 0-based numbers", r.returncode == 0 and "39 40" in open(os.path.join(tmp, "sc1o.txt")).read(), (r.returncode, r.stderr[:100]))
open(os.path.join(tmp, "sc2.txt"), "w").write("0 40\n40 %d\n" % T0)
r = run(["verify", "--model", loc, "--npy", fp0, "--scenes", os.path.join(tmp, "sc2.txt"), "--end", "exclusive", "--cuts-out", os.path.join(tmp, "sc2o.txt"), "--quiet", "--lenient"])
check("--end exclusive: shots 0-40 and 40-end -> cut (39, 40)", r.returncode == 0 and "39 40" in open(os.path.join(tmp, "sc2o.txt")).read(), (r.returncode, r.stderr[:100]))

# ---------------------------------------------------------------------------------------------- 5. real video file through ffmpeg
print("\n[5] real video file: decoded with ffmpeg == the frames we made")
try:
    import cv2
    it = items[0]; fr = it["frames"]; vp = os.path.join(tmp, "t.avi"); big = np.stack([cv2.resize(f, (192, 108), interpolation=cv2.INTER_AREA) for f in fr])
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"FFV1"), 25, (192, 108))
    if vw.isOpened():
        for f in big: vw.write(f[:, :, ::-1])
        vw.release(); dec = sbdv.decode_frames(vp)
        check("ffmpeg decodes the same number of frames", len(dec) == len(fr), (len(dec), len(fr)))
        cf = os.path.join(tmp, "vc.txt"); write_cuts(cf, it["cuts"]); o = os.path.join(tmp, "vo.json")
        r = run(["verify", "--model", loc, "--video", vp, "--cuts", cf, "--out", o, "--lenient"]); j = json.load(open(o))
        py = m.verify(dec, cuts=it["cuts"], lenient=True)
        check("--video (stream, ffmpeg pipe) == Python API on the same decoded frames", r.returncode == 0 and j["mode"] == "stream" and np.array_equal(np.array(j["scenes"]), py["scenes"]), (r.returncode, r.stderr[:150]))
        check("JSON reports peak memory of the program and of ffmpeg separately", "peak_rss_mb" in j and "decoder_peak_rss_mb" in j, list(j))
    else:
        print("   skipped: this OpenCV cannot write FFV1 video")
except Exception as e:
    print("   skipped:", type(e).__name__, str(e)[:120])

# ---------------------------------------------------------------------------------------------- 6. training bookkeeping
print("\n[6] training / evaluation bookkeeping")
want = TR.total_real_cuts(items, "fake")
dets_all = TR.counts_by_item(df, np.ones(len(df), bool), items, "fake"); got = sum(c[0] + c[2] for c in dets_all.values())
check("TP + FN equals the number of real cuts (detector alone)", got == want, (got, want))
empty_items = [dict(it) for it in items]; empty_items.append(dict(vid="no_cut", frames=items[0]["frames"], cuts=np.zeros((0, 2), np.int64), gt=items[0]["gt"], det="fake"))
df2, X2, _ = TR.build_table(eng, empty_items, log=None)
c2 = TR.counts_by_item(df2, np.ones(len(df2), bool), empty_items, "fake"); check("a video with NO candidates is still counted (the old code dropped it)", len(c2) == len(empty_items) and sum(c[0] + c[2] for c in c2.values()) == TR.total_real_cuts(empty_items, "fake"))
res = TR.cross_validate(df, X, names, TR.FEATS_LOCAL, items, ["fake"], ["fake"], kind=KIND, n_iter=30, depth=3, folds=4, log=None)
check("cross-validation totals add up (TP+FN == real cuts, with and without verifier)", all(sum(c[0] + c[2] for c in res["per"]["fake"][k].values()) == want for k in ("verified", "plain")))
pv_v, pv_p = TR.per_video(res["per"]["fake"]["verified"], items, "fake"), TR.per_video(res["per"]["fake"]["plain"], items, "fake")
g, lo, hi, verdict = TR.boot_diff(pv_v, pv_p); print(f"   synthetic gain {g:+.3f} ({lo:+.3f}..{hi:+.3f}) {verdict}")
check("verifier removes false cuts on the synthetic videos (sanity: gain > 0)", g > 0, g)
# out-of-fold must be lower or equal to in-sample (this is the 0.889 vs 0.8855 lesson)
clf = TR._fit(X[:, [names.index(f) for f in TR.FEATS_LOCAL]], df.label.values, KIND, 30, 3, 0)
insample = clf.predict_proba(X[:, [names.index(f) for f in TR.FEATS_LOCAL]])[:, 1] >= res["tau"]
f_in = TR.f1(sum(TR.counts_by_item(df, insample, items, "fake").values())); f_oof = TR.f1(sum(res["per"]["fake"]["verified"].values()))
print(f"   in-sample F1 {f_in:.4f}  vs  out-of-fold F1 {f_oof:.4f}")
check("in-sample F1 >= out-of-fold F1 (so in-sample numbers must never be reported)", f_in >= f_oof - 1e-9)

tab = TR.summary_table(res, items, "fake", "synthetic"); check("summary table has the bootstrap columns", {"range95", "verdict"} <= set(tab.columns))
cl_items = []
for vid, d in ds.items():
    c = TR.classic_cuts(d["frames"]); cl_items.append(dict(vid=vid, frames=d["frames"], cuts=c, gt=d["gt"], det="classic"))
check("classic frame-difference detector produces cuts", sum(len(i["cuts"]) for i in cl_items) > 0)
items2 = items + cl_items; df3, X3, _ = TR.build_table(eng, items2, log=None)
lo_t = TR.leave_one_family_out(df3, X3, names, TR.FEATS_LOCAL, items2, {"fake": ["fake"], "classic": ["classic"]}, kind=KIND, n_iter=20, depth=3, folds=3)
check("leave-one-family-out runs and returns one row per family", len(lo_t) == 2, lo_t.to_string())
ab = TR.ablation(df, X, names, items, [("tiny logreg", TR.FEATS_TINY, "logreg", 0, 0), ("local trees", TR.FEATS_LOCAL, KIND, 20, 3)], ["fake"], folds=3, log=None)
check("size-vs-accuracy table runs and sizes are positive", (ab.size_KB > 0).all(), ab.to_string())


# ---------------------------------------------------------------------------------------------- 7. reading other detectors' output
print("\n[7] sbdv_io: frame-number conventions")
import sbdv_io
good = 0; total = 0; unsure = 0; wrong = 0
rng7 = np.random.default_rng(77)
for k in range(10):
    V7, gt, cuts7, ev7 = synth.make_video(rng7, n_shots=8, p_dissolve=0.0); d = dict(frames=V7, gt=gt)
    for base in (0, 1):
        for end in ("inclusive", "exclusive"):
            sc = (gt + base).copy()
            if end == "exclusive": sc[:-1, 1] = gt[1:, 0] + base            # END = first frame of the next shot
            r = sbdv_io.detect_convention(d["frames"], sc); total += 1
            assert np.array_equal(sbdv_io.scenes_to_cuts(sc, base, end), sbdv_io.scenes_to_cuts(gt, 0, "inclusive")), (base, end)
            if not r["sure"]: unsure += 1
            elif (r["base"], r["end"]) == (base, end): good += 1
            else: wrong += 1
check(f"scenes_to_cuts maps all 4 conventions to the same cuts; detect_convention: {good} right, {unsure} unsure, {wrong} WRONG of {total}", total > 0 and wrong == 0 and good >= 0.7 * total, (good, unsure, wrong, total))
csvp = os.path.join(tmp, "sc.csv"); open(csvp, "w").write("Timecode List:,00:00:01.000\nScene Number,Start Frame,Start Timecode,End Frame,End Timecode\n1,1,00:00:00.000,40,00:00:01.600\n2,41,00:00:01.640,90,00:00:03.600\n")
rs = sbdv_io.read_scenes_csv(csvp); check("read_scenes_csv finds 'Start Frame' / 'End Frame' columns by name and skips title lines", rs.tolist() == [[1, 40], [41, 90]], rs)
txt = os.path.join(tmp, "sc.txt"); open(txt, "w").write("0 39\n40 89\n"); check("read_scenes_csv reads a plain 'start end' text file", sbdv_io.read_scenes_csv(txt).tolist() == [[0, 39], [40, 89]])
check("times_to_cuts", sbdv_io.times_to_cuts([1.0, 2.0], 25).tolist() == [[24, 25], [49, 50]])

# ---------------------------------------------------------------------------------------------- summary
print(f"\n{len(passed)} passed, {len(failed)} failed")
if failed:
    print("FAILED:", *failed, sep="\n  "); sys.exit(1)
print("all tests passed")
