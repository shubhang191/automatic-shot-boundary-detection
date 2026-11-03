"""Exercise every supported mode on the bundled licensed sample (or an explicit video)."""
from pathlib import Path
import sys
import json
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sbd_app.config import ROOT
from sbd_app.pipeline import Pipeline


def run(video, cache):
    pipeline = Pipeline(cache)
    records = []
    for detector in ['tinycut','autoshot','omnishotcut','omnishotcut_legacy_v15']:
        for mode in ['raw','sbdv2'] + ([] if detector == 'tinycut' else ['legacy_python']):
            result = pipeline.analyze(video, detector, verification=mode)
            records.append(dict(detector=detector, mode=mode, shots=result['shots'], raw_candidates=result['raw_candidates'],
                decisions=result['decisions'], model_sha256=result['model_sha256'], verifier=result['verifier'],
                detector_details=result['detector_details'], metadata=result['metadata'], cache=result['cache'],
                export_zip=result['export_zip'], timings=result['timings']))
            print(f'{detector} / {mode}: {len(result["shots"])} shots', flush=True)
    return dict(status='PASS', video=str(video), records=records)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, default=ROOT/'examples/inputs/sintel excerpt.mp4')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/legacy/smoke.json')
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/legacy-smoke')
    args=parser.parse_args()
    report=run(args.video,args.cache)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
