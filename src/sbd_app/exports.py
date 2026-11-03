from pathlib import Path
import csv
import json
import zipfile
from .cache import save_json


def csv_write(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_export(dest, result):
    dest = Path(dest)
    shot_payload={"metadata":result["metadata"],"shots":result["shots"]}
    if 'provenance' in result: shot_payload['provenance']=result['provenance']
    save_json(dest / "shots.json", shot_payload)
    if 'provenance' in result: save_json(dest/'provenance.json',result['provenance'])
    # Exact post-ZIP wall duration is provided in the separate timings.json sidecar.
    save_json(dest / "metadata.json", {k: v for k, v in result.items() if k not in {"gallery", "export_zip", "timeline", "shots", "raw_shots"}})
    save_json(dest / "cuts.json", result["decisions"])
    save_json(dest / "raw_shots.json", result["raw_shots"])
    csv_shots=result['shots']
    if result.get('mode')=='legacy_python':
        provenance={k:v for k,v in result['provenance'].items() if k not in ['thresholds','display_convention']}
        provenance['thresholds_json']=json.dumps(result['provenance']['thresholds'],sort_keys=True)
        csv_shots=[dict(**row,**provenance) for row in csv_shots]
    csv_write(dest / "shots.csv", csv_shots, list(csv_shots[0]))
    fields = ["a", "b", "kind", "score", "p_keep", "status"]
    csv_decisions=result['decisions']
    if result.get('mode')=='legacy_python':
        csv_decisions=[dict(**row,**provenance) for row in csv_decisions]
        fields += sorted(set().union(*(row.keys() for row in csv_decisions))-set(fields)) if csv_decisions else list(provenance)
    csv_write(dest / "cuts.csv", csv_decisions, fields)
    bundle = dest / "shot_export.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(dest.rglob("*")):
            if f.is_file() and f != bundle and f.name != 'timings.json':
                name = f.relative_to(dest).as_posix()
                if name.startswith("/") or ".." in Path(name).parts or f.is_symlink():
                    raise ValueError("Unsafe export path")
                archive.write(f, name)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
    return str(bundle)
