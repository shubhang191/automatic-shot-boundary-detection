import math
from ..config import ROOT, verified_path, KNOWN_HASHES
from ..detectors.base import import_file, validate_frames
from ..shot_schema import canonical_cuts
from ..errors import ModelError
from .build_status import library_path


class SBDV2Verifier:
    def __init__(self):
        if not library_path().is_file():
            raise ModelError("Native SBDV2 library missing. Run scripts/build_verifier_windows.bat or setup.bat.")
        module = import_file("sbd_cpp_wrapper", ROOT / "third_party/sbdv2/python/sbdv.py")
        self.model = module.Model(verified_path("verifier"), lib=str(library_path()))
        self.info = self.model.info()
        expected = dict(input="cuts", trees=60, nodes=1632, bytes=46359, rescue=False, snap=0)
        if any(self.info[k] != v for k, v in expected.items()) or not math.isclose(self.info['tau'], .4, abs_tol=1e-12):
            raise ModelError(f"Unexpected generic verifier model configuration: {self.info}")

    def verify(self, frames, cuts):
        validate_frames(frames)
        cuts = canonical_cuts(cuts, len(frames))
        if len(frames) == 1 and not cuts:
            return [], [], {"status": "no_candidates_single_frame", "model": self.info}
        result = self.model.verify(frames, cuts=[(c.a, c.b) for c in cuts], lenient=False)
        if result["n_invalid"] or result["n_bad"]:
            raise ModelError("C++ verifier rejected candidate indexing")
        cs = result["cands"]
        decisions = [dict(a=int(a), b=int(b), p_keep=float(p), status="KEEP" if k else "DROP")
                     for a, b, p, k in zip(cs["a"], cs["b"], cs["p_keep"], cs["kept"])]
        lookup = {(c.a, c.b): c for c in cuts}
        if {(d["a"], d["b"]) for d in decisions} != set(lookup):
            raise ModelError("Verifier candidates differ from detector input")
        kept = [lookup[(d["a"], d["b"])] for d in decisions if d["status"] == "KEEP"]
        return canonical_cuts(kept, len(frames)), decisions, dict(model=self.info, model_sha256=KNOWN_HASHES["verifier"],
                                                                timing_us=result["timing_us"], native_scenes=result["scenes"].tolist())
