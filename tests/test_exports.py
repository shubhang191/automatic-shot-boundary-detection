import csv
import json
import zipfile
from pathlib import PurePosixPath


def test_export_consistency(sample,pipeline):
    result=pipeline.analyze(sample,'tinycut',True)
    with zipfile.ZipFile(result['export_zip']) as z:
        assert z.testzip() is None
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in z.namelist())
        shots=json.loads(z.read('shots.json'))['shots']
        assert shots==result['shots']
        csv_rows=list(csv.DictReader(z.read('shots.csv').decode('utf-8-sig').splitlines()))
        assert len(csv_rows)==len(shots)
        assert sum(int(r['end_frame'])-int(r['start_frame'])+1 for r in csv_rows)==result['metadata']['decoded_frame_count']
        assert json.loads(z.read('cuts.json'))==result['decisions']
        meta=json.loads(z.read('metadata.json'))
        assert meta['model_sha256']==result['model_sha256']
        assert meta['verifier']['model_sha256']==result['verifier']['model_sha256']
        assert len([n for n in z.namelist() if n.startswith('thumbnails/')])==len(shots)
