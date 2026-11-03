import json
import pickle
import time
import numpy as np
from .base import Detection, import_file, validate_frames
from ..config import ROOT, verified_path
from ..shot_schema import CandidateCut, canonical_cuts
from ..errors import ModelError


class TinyCutDetector:
    name = "tinycut"

    def __init__(self):
        # The exact trusted project artifact is checked BEFORE pickle executes.
        checkpoint = verified_path("tinycut")
        self.decoder = json.loads(verified_path("decoder").read_text())["decoder"]
        self.lib = import_file("sbd_tinycut_original", ROOT / "third_party/tinycut/tricut2_lib.py")
        with checkpoint.open("rb") as stream:
            self.member = pickle.load(stream)
        m = self.member
        if m["cols"] != list(range(31)) or m["causal"] or self.lib.n_weights(m) != 18433:
            raise ModelError("Checkpoint is not the frozen clean single-member TinyCut S")
        if m["mu"].shape != (31,) or m["sd"].shape != (31,) or not np.all(m["sd"] > 0):
            raise ModelError("TinyCut scaler has invalid dimensions/scale")
        if self.decoder != dict(mode="runs", thr=.9, close=2, max_hard=2):
            raise ModelError("TinyCut frozen decoder mismatch")

    def features(self, frames):
        validate_frames(frames)
        l = self.lib
        return np.concatenate([l.base_numbers(frames), l.struct_features(l.grey(frames).astype(np.float32) / 255), l.consensus(frames)], 1)

    def predict(self, frames, video_path=None):
        t = time.perf_counter()
        features = self.features(frames)
        feature_s = time.perf_counter() - t
        t = time.perf_counter()
        probs = self.lib.member_probs(self.member, features, dev="cpu")
        if probs.shape != (len(frames),) or not np.isfinite(probs).all():
            raise ModelError("TinyCut did not return one finite probability per frame")
        network_s = time.perf_counter() - t
        bounds = self.lib.decode(probs, self.decoder)
        cuts = canonical_cuts([CandidateCut(a, b, "hard_candidate" if b == a + 1 else "gradual_candidate", float(probs[a + 1:b + 1].max())) for a, b in bounds], len(frames))
        return Detection(cuts, probs, self.lib.scenes_from_bounds(bounds, len(frames)).tolist(),
                         dict(parameters=18433, features=31, device="cpu", decoder=self.decoder, selected_epoch=10),
                         dict(feature_extraction_s=feature_s, network_inference_s=network_s))
