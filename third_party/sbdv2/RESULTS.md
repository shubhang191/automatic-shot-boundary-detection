# SBD-Verifier 2.0 - results (generated 2026-10-26 17:06)

Measured on: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 cores. Dataset: SHOT test set, 200 videos, 2516 real cuts.

## 1. Accuracy, out-of-fold (each video judged by a verifier that never saw it; threshold chosen on training videos only)

Model: `generic_local` (pictures around the cut only; constant-memory stream mode).

| test               | detector    |   alone_TP |   alone_FP |   alone_FN |   alone_F1 |   verified_TP |   verified_FP |   verified_FN |   verified_F1 |   F1_gain | range95            | verdict              |
|:-------------------|:------------|-----------:|-----------:|-----------:|-----------:|--------------:|--------------:|--------------:|--------------:|----------:|:-------------------|:---------------------|
| local (30 numbers) | AutoShot    |       2098 |        377 |        418 |     0.8407 |          2046 |           281 |           470 |        0.8449 |    0.0042 | -0.0023 to +0.0098 | cannot be told apart |
| local (30 numbers) | OmniShotCut |       1960 |        592 |        556 |     0.7735 |          1906 |           315 |           610 |        0.8047 |    0.0312 | +0.0237 to +0.0388 | BETTER               |
| local (30 numbers) | Classic     |       1780 |        400 |        736 |     0.7581 |          1734 |           222 |           782 |        0.7755 |    0.0174 | +0.0063 to +0.0288 | BETTER               |

## 2. A detector the verifier never saw (trained on the other two)

| test                                      | detector    |   alone_TP |   alone_FP |   alone_FN |   alone_F1 |   verified_TP |   verified_FP |   verified_FN |   verified_F1 |   F1_gain | range95            | verdict              |
|:------------------------------------------|:------------|-----------:|-----------:|-----------:|-----------:|--------------:|--------------:|--------------:|--------------:|----------:|:-------------------|:---------------------|
| train without AutoShot [local numbers]    | AutoShot    |       2098 |        377 |        418 |     0.8407 |          2047 |           278 |           469 |        0.8457 |    0.005  | -0.0019 to +0.0109 | cannot be told apart |
| train without OmniShotCut [local numbers] | OmniShotCut |       1960 |        592 |        556 |     0.7735 |          1931 |           433 |           585 |        0.7914 |    0.0179 | +0.0124 to +0.0236 | BETTER               |
| train without Classic [local numbers]     | Classic     |       1780 |        400 |        736 |     0.7581 |          1742 |           260 |           774 |        0.7711 |    0.013  | +0.0029 to +0.0235 | BETTER               |

## 3. Size vs accuracy

| model                 |   features |   trees |   depth |   parameters |   size_KB |   pooled_F1_alone |   pooled_F1_verified |   gain |
|:----------------------|-----------:|--------:|--------:|-------------:|----------:|------------------:|---------------------:|-------:|
| 4 numbers, linear     |          4 |       0 |       0 |            5 |       0.2 |            0.7913 |               0.7974 | 0.0061 |
| local, linear         |         30 |       0 |       0 |           31 |       1.1 |            0.7913 |               0.8076 | 0.0162 |
| local, 20 small trees |         30 |      20 |       3 |          291 |       8.4 |            0.7913 |               0.8043 | 0.0129 |
| local, 60 trees       |         30 |      60 |       4 |         1633 |      45.3 |            0.7913 |               0.8093 | 0.018  |
| full, 60 trees        |         38 |      60 |       4 |         1633 |      45.4 |            0.7913 |               0.8112 | 0.0199 |

## 4. Speed and memory

Verifier compute with frames in memory: 1.73 microseconds per frame.

| video                     | mode   |   frames |   candidates |   total_ms |   verify_ms |   ms_per_frame |   peak_rss_mb |   decoder_peak_rss_mb |   peak_total_mb_worst_case |
|:--------------------------|:-------|---------:|-------------:|-----------:|------------:|---------------:|--------------:|----------------------:|---------------------------:|
| real SHOT video (mp4)     | stream |     1229 |           16 |     817.59 |        3.99 |           0.67 |          4.25 |                 62.81 |                      67.06 |
| real SHOT video (mp4)     | ram    |     1229 |           16 |     922.88 |        2.53 |           0.75 |          8.38 |                 62.86 |                      71.23 |
| 30-minute synthetic video | stream |    45000 |         1175 |    9955.42 |      239.86 |           0.22 |          4.75 |                 55.29 |                      60.04 |
| 30-minute synthetic video | ram    |    45000 |         1175 |   10166.8  |      155.93 |           0.23 |        171.62 |                 55.64 |                     227.26 |

A cut can be judged 60 frames after the new shot starts: 2.4 s at 25 fps (waiting time, not compute).

The test videos are tiny (48x27 / 384x216): ffmpeg decode time for real HD video is larger and is not a property of the verifier.

## 5. Overhead on top of the detector

| name                                                                                                                                            |   detector_ms_per_frame |   verifier_ms_per_frame |   fps_detector |   fps_with_verifier |   overhead_percent |   seconds_per_hour_of_video | real_time_at   | energy_j   |
|:------------------------------------------------------------------------------------------------------------------------------------------------|------------------------:|------------------------:|---------------:|--------------------:|-------------------:|----------------------------:|:---------------|:-----------|
| AutoShot (T4 GPU, forward pass only) + verifier (frames already decoded)                                                                        |                  1.5932 |                  0.0017 |          627.7 |               627   |               0.11 |                       143.5 | 25 fps: yes    |            |
| AutoShot (T4 GPU, forward pass only) + verifier run on a video file (incl. ffmpeg decode of a tiny 48x27 test video; real video decodes slower) |                  1.5932 |                  0.22   |          627.7 |               551.5 |              13.81 |                       163.2 | 25 fps: yes    |            |

## What is NOT claimed

* Power / energy was not measured (no counter on Colab).
* Results are for SHOT only; other video types are untested.
* The verifier cannot find cuts the detector missed; recall can only go down or stay.
* The exported model files have seen all of SHOT; only the out-of-fold numbers above are accuracy results.
