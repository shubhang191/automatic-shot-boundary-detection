from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbd_app.pipeline import Pipeline
from sbd_app.cache import save_json


def smoke(video, detectors, output):
    pipeline = Pipeline()
    records = []
    for detector in detectors:
        for verified in [False, True]:
            result = pipeline.analyze(video, detector, verified)
            shots = result["shots"]
            n = result["metadata"]["decoded_frame_count"]
            assert shots[0]["start_frame"] == 0 and shots[-1]["end_frame"] == n - 1
            assert sum(s["end_frame"] - s["start_frame"] + 1 for s in shots) == n
            assert all(a["end_frame"] + 1 == b["start_frame"] for a, b in zip(shots, shots[1:]))
            if verified:
                assert result["cache"]["detector_hit"]
            records.append(dict(detector=detector, mode=result["mode"], status="PASS", frames=n,
                                raw_candidates=result["raw_candidates"], decisions=result["decisions"],
                                shots=len(shots), cache=result["cache"], timings=result["timings"],
                                input_sha256=result["metadata"]["source_video_sha256"], model_sha256=result["model_sha256"]))
            print(detector, result["mode"], len(shots), "shots", flush=True)
    save_json(output, records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", default=str(root / "examples/inputs/sintel excerpt.mp4"))
    parser.add_argument("--detectors", nargs="+", default=["tinycut", "autoshot", "omnishotcut"])
    parser.add_argument("--output", default=str(root / "reports/real_video_smoke.json"))
    args = parser.parse_args()
    smoke(args.video, args.detectors, args.output)
