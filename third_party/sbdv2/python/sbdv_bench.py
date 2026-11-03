"""
sbdv_bench.py - measure the verifier the same way every time, and build the efficiency table from ONE base number.

The rules (each one fixes a mistake of the earlier benchmark notebooks):
  1. ONE timing method.  total_ms = wall-clock time of the whole `sbdv verify` process, measured from outside (decoding + verifying).  verify_ms = the
     compute time of the verifier alone, measured inside (median of repeats).  Both are reported, and every other number is derived from total_ms.
  2. ONE base unit.  Everything per-frame, FPS, "seconds per hour of video" comes from  ms_per_frame = total_ms / frames.  FPS = 1000 / ms_per_frame.
  3. Adding a stage can never make a pipeline faster: combine() adds the times, so FPS(detector + verifier) <= FPS(detector) always (asserted).
  4. Memory: the peak of the sbdv process (VmHWM) and the peak of its ffmpeg child are both recorded; they run at the same time, so their sum is the
     worst case.  Python's own memory is NOT part of the tool and is never counted.
  5. Power is not measured unless the computer has an energy counter (Intel RAPL).  On Colab there is none, so the energy column stays empty rather than guessed.
  6. A decision for a cut can only be made LOOKAHEAD_FRAMES frames after the new shot started (the verifier looks at the pictures after the cut).  That
     waiting time is part of the real latency of a live system and is reported separately from compute time.
"""
import os, sys, json, time, subprocess, tempfile, platform, shutil
import numpy as np

LOOKAHEAD_FRAMES = 60        # the picture window after a cut that the features use (the same number as PAD_F in the C++ code)


def environment():
    """where the numbers were measured (print this next to every table)"""
    cpu = "unknown"
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.lower().startswith("model name"): cpu = ln.split(":", 1)[1].strip(); break
    except OSError: pass
    ff = ""
    try: ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    except OSError: pass
    return dict(cpu=cpu, cores=os.cpu_count(), os=platform.platform(), python=platform.python_version(), ffmpeg=ff, rapl=os.path.exists("/sys/class/powercap/intel-rapl"))


def read_rapl_joules():
    """energy counter of the CPU package in joules, or None when this computer has no readable one (Colab does not)"""
    p = "/sys/class/powercap/intel-rapl:0/energy_uj"
    try: return int(open(p).read()) / 1e6
    except (OSError, ValueError): return None


def _write_cuts(path, cuts):
    with open(path, "w") as f:
        for a, b in np.asarray(cuts).reshape(-1, 2): f.write(f"{int(a)} {int(b)}\n")


def run_once(exe, model, video, cuts_file, mode="stream", extra=()):
    """one run of the command line tool; returns (JSON dict, total wall time in ms measured from outside)"""
    with tempfile.TemporaryDirectory() as td:
        o = os.path.join(td, "o.json"); flag = {"stream": "--stream", "ram": "--ram", "auto": None}[mode]
        args = [exe, "verify", "--model", os.fspath(model), "--video", os.fspath(video), "--cuts", cuts_file, "--out", o, "--lenient"] + ([flag] if flag else []) + list(extra)
        t0 = time.perf_counter(); r = subprocess.run(args, capture_output=True, text=True); dt = (time.perf_counter() - t0) * 1000
        if r.returncode != 0: raise RuntimeError(f"sbdv failed ({r.returncode}): {r.stderr.strip()[:300]}")
        return json.load(open(o)), dt


def bench_video(model, video, cuts, mode="stream", repeats=5, exe=None):
    """repeat the same run (one warm-up run first, not counted) and report medians.  mode: "stream" | "ram"."""
    exe = exe or shutil.which("sbdv") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "sbdv")
    with tempfile.TemporaryDirectory() as td:
        cf = os.path.join(td, "c.txt"); _write_cuts(cf, cuts); run_once(exe, model, video, cf, mode)
        runs = [run_once(exe, model, video, cf, mode) for _ in range(repeats)]
    med = lambda xs: float(np.median(xs))
    j0 = runs[0][0]; frames = j0["frames"]; total = [t for _, t in runs]
    row = dict(mode=j0["mode"], frames=frames, candidates=j0["candidates_total"], repeats=repeats,
               total_ms=med(total), total_ms_min=min(total), total_ms_max=max(total),
               verify_ms=med([j["timing_ms"]["verify_median"] for j, _ in runs]),
               peak_rss_mb=med([j["peak_rss_mb"] for j, _ in runs]), decoder_peak_rss_mb=med([j["decoder_peak_rss_mb"] for j, _ in runs]))
    row["peak_total_mb_worst_case"] = row["peak_rss_mb"] + row["decoder_peak_rss_mb"]
    row["ms_per_frame"] = row["total_ms"] / max(frames, 1)                       # the ONE base unit
    row["verify_us_per_frame"] = 1000.0 * row["verify_ms"] / max(frames, 1)
    return row


def combine(detector_ms_per_frame, verifier_ms_per_frame):
    """pipeline detector -> verifier, run one after the other: times add, so the FPS can only go down.  Returns the derived numbers from ONE base."""
    d, v = float(detector_ms_per_frame), float(verifier_ms_per_frame); tot = d + v
    out = dict(detector_ms_per_frame=d, verifier_ms_per_frame=v, total_ms_per_frame=tot, fps_detector=1000.0 / d, fps_with_verifier=1000.0 / tot,
               overhead_percent=100.0 * v / d, slowdown_x=tot / d)
    assert out["fps_with_verifier"] <= out["fps_detector"] + 1e-9, "adding a stage cannot make the pipeline faster"
    return out


def efficiency_table(rows):
    """rows: [dict(name=..., detector_ms_per_frame=..., verifier_ms_per_frame=..., fps_of_video=25 (optional), energy_j=None)].  Returns a pandas DataFrame
    where every speed number comes from the two ms-per-frame inputs; the 'real time' column says whether the pipeline keeps up with the video's own FPS."""
    import pandas as pd
    out = []
    for r in rows:
        c = combine(r["detector_ms_per_frame"], r["verifier_ms_per_frame"]); vf = r.get("fps_of_video", 25.0)
        out.append(dict(name=r["name"], detector_ms_per_frame=round(c["detector_ms_per_frame"], 4), verifier_ms_per_frame=round(c["verifier_ms_per_frame"], 4),
                        fps_detector=round(c["fps_detector"], 1), fps_with_verifier=round(c["fps_with_verifier"], 1), overhead_percent=round(c["overhead_percent"], 2),
                        seconds_per_hour_of_video=round(c["total_ms_per_frame"] * vf * 3600 / 1000.0, 1), real_time_at=f"{vf:g} fps: " + ("yes" if c["fps_with_verifier"] >= vf else "NO"),
                        energy_j=r.get("energy_j")))
    return pd.DataFrame(out)


def decision_delay_seconds(video_fps):
    """the unavoidable wait of a live system: a cut can be judged only after LOOKAHEAD_FRAMES more frames arrived (this is NOT compute time)"""
    return LOOKAHEAD_FRAMES / float(video_fps)


def synthetic_long_video(path, minutes=10.0, fps=25, seed=0, shot_len=(25, 90)):
    """write a lossless 48x27 test video of the given length (made of synthetic shots) and return (cuts [n,2] of a fake detector).  For speed / memory tests only."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    import synth
    rng = np.random.default_rng(seed); need = int(minutes * 60 * fps); frames, cuts, pos = [], [], 0
    while pos < need:
        V, gt, cs, ev = synth.make_video(rng, n_shots=12, shot_len=shot_len); det = synth.fake_detector(rng, len(V), cs, ev)
        cuts += [(int(a) + pos, int(b) + pos) for a, b in det]; frames.append(V); pos += len(V)
    allf = np.concatenate(frames)[:need]; cuts = [(a, b) for a, b in cuts if b < need]
    raw = path + ".u8"; allf.tofile(raw)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x27", "-r", str(fps), "-i", raw, "-c:v", "ffv1", path], capture_output=True)
    os.remove(raw)
    if r.returncode != 0: raise RuntimeError(r.stderr.decode()[-300:])
    return np.array(cuts, np.int64)


if __name__ == "__main__":
    if len(sys.argv) < 3: print("usage: python sbdv_bench.py MODEL.sbdm VIDEO CUTS.txt   (cuts: 'A B' per line)"); sys.exit(2)
    cuts = np.loadtxt(sys.argv[3], ndmin=2)[:, :2] if len(sys.argv) > 3 else np.zeros((0, 2))
    print(environment())
    for mode in ("stream", "ram"):
        try: print(mode, bench_video(sys.argv[1], sys.argv[2], cuts, mode))
        except RuntimeError as e: print(mode, "not possible:", e)
