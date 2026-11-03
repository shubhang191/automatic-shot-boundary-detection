import sys
import time
import numpy as np
from .base import Detection
from ..config import ROOT, verified_path
from ..shot_schema import cuts_from_native_scenes
from ..video_io import decode_rgb
from ..errors import ModelError


def convert_native(ranges, intra, inter, count):
    """Exact Notebook4 to_gt_format conversion: General [s,e) -> [s,e-1]."""
    if not len(ranges) == len(intra) == len(inter):
        raise ValueError("Omni native label lengths differ")
    previous = 0
    for s, e in ranges:
        if not 0 <= s < e <= count or s != previous:
            raise ValueError("Omni ranges must be contiguous zero-based half-open ranges")
        previous = e
    clean = [[int(s), int(e) - 1] for (s, e), label in zip(ranges, intra) if label == "General"]
    # Native API can omit a final tail. We retain native output without inventing detections.
    # Display partitions below extend the terminal shot to T-1, and metadata states this.
    kinds = []
    general_indices = [i for i, label in enumerate(intra) if label == "General"]
    for i in general_indices[1:]:
        label = inter[i]
        kinds.append("hard_candidate" if label == "Hard_Cut" else "gradual_candidate" if "Transition" in label else "unknown")
    return cuts_from_native_scenes(clean, count, kinds), clean


class OmniShotCutDetector:
    name = "omnishotcut"

    def __init__(self):
        ckpt = verified_path("omnishotcut")
        sys.path.insert(0, str(ROOT / "third_party/omnishotcut"))
        from omnishotcut.engine import load_model, _run_on_numpy
        from omnishotcut.label_correspondence import intra_int2string, inter_int2string
        self.model, self.args = load_model(str(ckpt), device="cpu")
        self.run = _run_on_numpy
        self.intra_names = intra_int2string
        self.inter_names = inter_int2string

    def predict(self, frames, video_path=None):
        t = time.perf_counter()
        if len(frames) * self.args.process_width * self.args.process_height * 3 > 256 * 1024**2:
            raise ModelError("OmniShotCut's decoded input would exceed 256 MiB. Use a shorter video.")
        native_frames = decode_rgb(video_path, self.args.process_width, self.args.process_height)
        if len(native_frames) != len(frames):
            raise ModelError("Omni decode count differs from canonical analysis frames; cannot align cuts")
        decode_s = time.perf_counter() - t
        t = time.perf_counter()
        ranges, intra_int, inter_int = self.run(native_frames, self.model, self.args, 10)
        intra = [self.intra_names[int(x)] for x in intra_int]
        inter = [self.inter_names[int(x)] for x in inter_int]
        cuts, clean = convert_native(ranges, intra, inter, len(frames))
        return Detection(cuts, None, ranges,
                         dict(device="cpu", checkpoint_version="provided 41M checkpoint; not asserted to be v1.5",
                              parameters=sum(p.numel() for p in self.model.parameters()),
                              process_width=self.args.process_width, process_height=self.args.process_height,
                              overlap=10, intra_labels=intra, inter_labels=inter, native_general_inclusive=clean,
                              native_convention="zero-based half-open; General ranges only produce proposals"),
                         dict(model_decode_s=decode_s, feature_extraction_s=0.0, network_inference_s=time.perf_counter() - t))
