"""Portability, integrity checks and diagnostics around the original functions.

No training, threshold selection, substitute model or prediction correction.
"""
from pathlib import Path
import importlib
import importlib.metadata
import hashlib
import sys
import os
import numpy as np
from sbd_app.config import ROOT, sha256
from sbd_app.errors import ModelError
from sbd_app.shot_schema import cuts_from_native_scenes

BUNDLES = {
    'autoshot': ('verifier_v15_idea12.joblib', 'b057ac00be1defafc0d7df5838f82e842a482324f5f073357c78d7fe66e61d46'),
    'omnishotcut': ('omni_verifier_v15.joblib', '3e18cde1c20f8240cf178889ebf392d5d9b4281a14ca0786683020b8d03d3090'),
}
MODULE_HASHES = {
    'sbd_refine.py': 'f2de4d3f1a55b373c4f95f2f112a05fe381a3c59ebd6070c193fb4cf13e41d97',
    'sbd_verifier.py': '14bc3cc2e8841b0843677461ccb00821fe1528b754f14a612467a31f1a2c55f3',
    'sbd_verifier2.py': '086307f06f94da2a5aa47c7956fefb32e0fbe3c50c659b764a6aef8e8097a6ea',
    'sbd_verifier3.py': 'ef5e3fc3c3f720729a41585b84642e48159c33a117943abee1e5a515180a3683',
    'sbd_omni_verifier.py': '9015e4ec1fc9d13d222df7e71702e27d29a1c46acac62f5c8b54d851e2bfa73e',
}


def family(detector):
    if detector == 'autoshot': return 'autoshot'
    if detector in {'omnishotcut', 'omnishotcut_legacy_v15'}: return 'omnishotcut'
    raise ModelError('Legacy Python verification supports AutoShot and OmniShotCut only. Choose Raw or SBDV2 for Clean TinyCut S.')


def checked_bundle_path(detector):
    name, digest = BUNDLES[family(detector)]
    path = ROOT / 'models/legacy_python' / name
    if not path.is_file() or sha256(path) != digest:
        raise ModelError(f'Missing or changed historical bundle {name}. Restore the release asset; no fallback is used.')
    return path


def modules():
    folder = Path(__file__).parent
    for name, digest in MODULE_HASHES.items():
        if sha256(folder/name) != digest:
            raise ModelError(f'Historical source integrity failed: {name}')
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
    return importlib.import_module('sbd_verifier3'), importlib.import_module('sbd_omni_verifier')


def load_bundle(detector):
    if importlib.metadata.version('scikit-learn') != '1.6.1':
        raise ModelError('Historical models require scikit-learn 1.6.1. Run setup.bat in this release.')
    # WMIC was removed from recent Windows; avoid loky's optional physical-core probe.
    os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
    import joblib
    # Only fixed, checksum-verified bundled assets can reach pickle deserialization.
    return joblib.load(checked_bundle_path(detector))


def scenes_to_shots(scenes, fps):
    """Preserve actual inclusive legacy scenes, including gradual-transition gaps."""
    out = []
    for i, (s, e) in enumerate(scenes):
        s, e = int(s), int(e)
        gap = s - int(scenes[i-1][1]) - 1 if i else 0
        out.append(dict(shot=i+1, start_frame=s, end_frame=e, start_s=s/fps,
            end_inclusive_s=e/fps, end_exclusive_s=(e+1)/fps, duration_s=(e-s+1)/fps,
            boundary_type='start' if i == 0 else 'gradual_candidate' if gap else 'hard_candidate'))
    return out


class LegacyVerifier:
    def __init__(self):
        self.bundles = {}

    def verify(self, detector, prediction, frames):
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (27, 48, 3) or len(frames) < 1:
            raise ModelError('Historical verifier requires nonempty (T,27,48,3) uint8 RGB frames.')
        fam = family(detector)
        path = checked_bundle_path(detector)
        s3, ov = modules()
        if fam not in self.bundles: self.bundles[fam] = load_bundle(detector)
        b = self.bundles[fam]
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            if fam == 'autoshot':
                if prediction.probabilities is None or len(prediction.probabilities) != len(frames):
                    raise ModelError('AutoShot probabilities are missing or misaligned; refusing to pad deployment inputs.')
                scenes, info = s3.verify3(prediction.probabilities, frames, b)
            else:
                ranges = prediction.native_ranges
                intra, inter = prediction.details['intra_labels'], prediction.details['inter_labels']
                if not len(ranges) == len(intra) == len(inter): raise ModelError('Omni native labels are misaligned.')
                scenes, info = ov.verify_omni(ranges, intra, inter, frames, b)
        scenes = np.asarray(scenes, dtype=int).reshape(-1, 2)
        kinds = ['gradual_candidate' if y[0]-x[1]>1 else 'hard_candidate' for x, y in zip(scenes, scenes[1:])]
        cuts = cuts_from_native_scenes(scenes.tolist(), len(frames), kinds)
        native = {}
        if fam == 'omnishotcut':
            indices = [i for i, lab in enumerate(intra) if lab == 'General']
            for left, right in zip(indices, indices[1:]):
                native[(int(ranges[left][1])-1, int(ranges[right][0]))] = dict(
                    native_inter_label=inter[right], native_intra_label=intra[right],
                    dropped_transition_labels=intra[left+1:right], n_dropped=right-left-1)
        decisions = []
        used_index = 0
        for d in info:
            accepted = d['kept']
            final = cuts[used_index] if accepted else None
            if accepted: used_index += 1
            threshold = float(b['tau'] if fam == 'autoshot' else b['tau_r'] if d['src'] == 'rescue' else b['tau_o'])
            p = float(d['p_keep'])
            decisions.append(dict(a=int(d['a']), b=int(d['b']), kind='gradual_candidate' if d['b']-d['a']>1 else 'hard_candidate',
                score=None, p_keep=p, status='KEEP' if accepted else 'DROP', source=d['src'], threshold=threshold,
                threshold_pass=p>=threshold, decision_reason='accepted' if accepted else 'below threshold' if p<threshold else 'overlapping candidate',
                rescued=d['src'] in {'rescue', 'low'}, snapped=bool(final and (final.a,final.b)!=(d['a'],d['b'])),
                original_a=int(d['a']), original_b=int(d['b']), final_a=final.a if final else None, final_b=final.b if final else None,
                **native.get((d['a'],d['b']), {})))
        assert used_index == len(cuts), 'Legacy diagnostics/scenes disagree'
        settings = {k: b[k] for k in ['tau','base_thr','low_thr','tau_o','tau_r','rescue','r_ratio','r_min','snap_w','variant','tag'] if k in b}
        source_identity = {name:sha256(Path(__file__).parent/name) for name in MODULE_HASHES}
        source_identity['integration_adapter.py']=sha256(Path(__file__))
        details = dict(implementation='authentic legacy Python', bundle=path.name, bundle_sha256=sha256(path),
            source_sha256=source_identity, settings=settings, rescue_enabled=bool(b.get('rescue',fam=='autoshot')),
            snap_enabled=bool(b.get('snap_w',0)), frame_format='uint8 RGB (T,27,48,3)',
            frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest(), native_scenes_inclusive=scenes.tolist(),
            dependencies={x:importlib.metadata.version(x) for x in ['scikit-learn','joblib','numpy','scipy','threadpoolctl']},
            benchmark_claim='No new SHOT200 accuracy claim. Saved final-fit model differs from historical out-of-fold evaluation.')
        return cuts, decisions, details, scenes.tolist()
