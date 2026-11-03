# SBD-Verifier 2.0: a small offline tool that cleans up the cuts of ANY shot-boundary detector

A detector (AutoShot, OmniShotCut, TransNetV2, a classic algorithm, your own network ...) proposes cuts. Some are false: a camera flash, a caption popping up,
a quick camera shake. The verifier looks at the tiny 48x27 pictures around every proposed cut and says KEEP or DROP.
Its cost depends on the **number of cuts**, not on the length of the video.

* Core: one C++17 header (`src/sbdv.hpp`), **no libraries**. Builds to a command line tool (`bin/sbdv`) and a shared library (`bin/libsbdv.so`).
* Offline: no network, no GPU, no Python needed to RUN it. Python + scikit-learn are only needed to TRAIN a model (and for the optional wrapper).
* Any video goes through `ffmpeg` (the same 48x27 RGB shrink AutoShot uses), or hand it frames you already have.
* **Stream mode (new in 2.0):** a model made only of "local" picture numbers runs frame by frame with a small, fixed amount of memory (a ring buffer of 123
  frames, about 0.5 MB, plus the length of the longest dissolve a detector reports) on a video of any length. Models that use whole-video numbers (the old v1-v3 / omni bundles) keep all frames in RAM (3.9 KB per frame).

## 1. Build (10 seconds)
    ./build.sh                  # needs g++ or clang (C++17). Gives bin/sbdv and bin/libsbdv.so   (NATIVE=1 ./build.sh = tuned for this CPU only)
    python tests/run_tests.py   # ~1 minute, synthetic videos, checks export / streaming / errors / training bookkeeping

## 2. Use it with YOUR detector: hand over what the detector found
    bin/sbdv verify --model models/generic_local.sbdm --video clip.mp4 --scenes shots.txt --out result.json     # shots: "START END" per line
    bin/sbdv verify --model models/generic_local.sbdm --video clip.mp4 --cuts cuts.txt                          # cuts:  "A B [score]" per line
    bin/sbdv verify --model models/generic_local.sbdm --video clip.mp4 --times times.txt --fps 25               # cut times in seconds
    bin/sbdv verify --model models/autoshot_v2.sbdm   --video clip.mp4 --probs probs.npy                       # per-frame probabilities (AutoShot style)
`--quiet` prints only the final shots (`START END` per line). `--shots FILE` writes them, `--cuts-out FILE` writes every candidate with its keep-probability.
`bin/sbdv info models/x.sbdm` shows what a model contains and whether it can stream. Run `bin/sbdv` without arguments for every option.

**Frame numbers must match the video.** Detectors disagree about two things: do frame numbers start at 0 or 1 (`--base 0|1`), and is the END of a shot the last
frame of the shot or the first frame of the next one (`--end inclusive|exclusive`)? The tool does not guess. If a cut points outside the video it stops with
exit code 3 (use `--lenient` to skip such cuts with a warning). `python python/sbdv_io.py scenes.csv video.mp4` looks at the pictures and tells which
convention a file uses (and says "not sure" when it cannot tell). If your detector worked on a constant-frame-rate copy, pass `--cfr FPS`.

Exit codes: 0 ok, 1 error (bad file, bad model), 2 wrong usage, 3 the detector output does not fit the video.

Python, in the same process as your detector:

    import sys; sys.path.insert(0, "python"); import sbdv
    frames = sbdv.decode_frames("clip.mp4")                  # (T,27,48,3) uint8
    m = sbdv.Model("models/generic_local.sbdm")
    r = m.verify_scenes(frames, my_detector_scenes)          # or m.verify(frames, probs=...) / m.verify(frames, cuts=...)
    r["scenes"]                                              # the cleaned shot list
    sbdv.verify_video(m.path, "clip.mp4", scenes=my_scenes)  # same through the command line tool (stream mode, constant memory)

From C/C++ (video editors): `include/sbdv.h` (`sbdv_load`, `sbdv_verify`, `sbdv_set_option`), frames are `uint8 [T][27][48][3]`.

## 3. Which model for which detector
| model file | needs from the detector | memory | made from |
|---|---|---|---|
| `generic_local.sbdm` | **any** detector's cut list: pictures only, no detector score | stream, constant | `sbdv_train.py` (notebook 25) |
| `generic_full.sbdm` | the same, plus whole-video numbers (a little more information) | RAM, 3.9 KB per frame | `sbdv_train.py` |
| `autoshot_v1/v2/v3.sbdm` | per-frame probabilities **of AutoShot** (thresholds 0.296 / 0.10, score features) | RAM | your joblib bundles |
| `omni.sbdm` | the shot list of OmniShotCut with its labels | RAM | your omni bundle |

A model trained on one detector's candidates is only guaranteed on that detector. Notebook 25 measures how far a model carries: it trains without one detector
and tests on it (leave-one-detector-out), including a plain frame-difference detector. For a new detector, train on its own output (section 4).

## 4. Train a verifier for your own detector (needs python + scikit-learn)
    import sbdv, sbdv_train as TR
    items = [dict(vid="a", frames=frames, cuts=detector_cuts, gt=ground_truth_scenes, det="mydet"), ...]   # your detector's output on labelled videos
    df, X, names = TR.build_table(engine_model, items)                                                    # every candidate + all picture numbers + label
    res = TR.cross_validate(df, X, names, TR.FEATS_LOCAL, items, ["mydet"], ["mydet"])                    # honest test, split BY VIDEO
    print(TR.summary_table(res, items, ["mydet"]))                                                        # with bootstrap range and verdict
    TR.export_generic(df, X, names, TR.FEATS_LOCAL, res["tau"], "models/mydet.sbdm")                      # checked against scikit-learn, then written
The exporter refuses anything it cannot reproduce exactly (supported: HistGradientBoosting, GradientBoosting, RandomForest, ExtraTrees, DecisionTree,
LogisticRegression, Pipeline(StandardScaler, ...), and the small TCN of `sbd_verifier3.py`). Export an old sklearn bundle with
`python python/sbdv_export.py bundle.joblib out.sbdm`.

## 5. What it can and cannot do (read this before quoting numbers)
* A verifier only removes false cuts (the old Omni model can also add big picture jumps, "rescue"). It **cannot find a cut the detector did not propose**; recall is capped by the candidates.
* **Delay of a live system:** a cut can only be judged 60 frames after the new shot starts (the numbers use the pictures after the cut): 2.4 s at 25 fps. That is
  waiting time, not compute time, and it is reported separately (`sbdv_bench.decision_delay_seconds`).
* Accuracy numbers are only honest when they are out-of-fold: each video is judged by a verifier that never saw that video, and the keep-threshold is chosen on the other videos.
  A verifier judged on the videos it was trained on looks better than it is (that was the 0.889 vs 0.8855 discrepancy of the earlier notebooks).
* Everything was fitted and tested on the SHOT dataset. Other kinds of video are not shown until you test them.
* Speed: `sbdv_bench.py` uses one timing method (wall-clock of the whole process from outside) and derives FPS and per-hour times from one number. Power is not measured
  (Colab has no energy counter); the energy column stays empty rather than guessed. FLOPs in the tool's output are an estimate from counting loops, not a profiler.
* Frames are the decoded frames of the file (frame i = i-th decoded frame; no duplication or dropping). If your detector resampled the video, say so with `--cfr`.

## 6. What changed in 2.0 (the audit)
| problem in the earlier notebooks | fix |
|---|---|
| in-sample F1 (0.889) compared with out-of-fold F1 (0.8855) as if equal | all reported numbers are out-of-fold, threshold chosen on training videos only, paired bootstrap over videos, leave-one-detector-out |
| videos with no proposed cut were dropped when counting TP/FP/FN, so false negatives were too few | counted; every table is checked: TP + FN = real cuts |
| RAM measured with a duplicate copy of the frames; ffmpeg's memory not counted | one copy; peak of the tool and of ffmpeg recorded separately |
| FPS got *higher* after adding the verifier; several timing methods mixed | one method, one base unit, `combine()` asserts FPS can only go down |
| decode time left out of "latency" | total wall time of the whole process is the headline number, compute time next to it |
| `-vsync 0` only (removed in new ffmpeg) | `-vsync 0` with automatic fallback to `-fps_mode passthrough` |
| a detector output that does not fit the video was silently clipped | exit code 3 / `ValueError`; `--lenient` skips with a warning; unreadable lines are errors |
| the whole video had to be in RAM | stream mode with a ring buffer for local models (checked bit-identical to RAM mode) |
| model could silently use whole-video numbers | `info` and the exporter say whether a model can stream; `--stream` refuses a model that cannot |

## Files
    src/sbdv.hpp  src/sbdv_cli.cpp  src/sbdv_capi.cpp  include/sbdv.h       the tool
    python/sbdv.py         ctypes wrapper + ffmpeg decode                    python/sbdv_export.py   sklearn / joblib -> .sbdm
    python/sbdv_train.py   honest training and evaluation                    python/sbdv_io.py       other detectors' formats, frame-number check
    python/sbdv_bench.py   timing and memory, efficiency table               tests/                  run_tests.py + synthetic videos
    build.sh Makefile CMakeLists.txt                                         models/*.sbdm           exported models
