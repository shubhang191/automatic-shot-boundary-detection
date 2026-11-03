import sys
import time
import numpy as np
import torch
from .base import Detection, import_file, validate_frames
from ..config import ROOT, verified_path
from ..shot_schema import cuts_from_native_scenes


def window_indices(count):
    if count < 1:
        raise ValueError("Empty video")
    for offset in range(0, count, 50):
        yield np.clip(np.arange(offset - 25, offset + 75), 0, count - 1)


def official_scenes(predictions):
    # Exact upstream utils.predictions_to_scenes algorithm, isolated from plotting imports.
    scenes = []
    t, previous, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if previous == 1 and t == 0:
            start = i
        if previous == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        previous = t
    if t == 0:
        scenes.append([start, i])
    return scenes or [[0, len(predictions) - 1]]


class AutoShotDetector:
    name = "autoshot"

    def __init__(self):
        ckpt = verified_path("autoshot")
        path = ROOT / "third_party/autoshot"
        sys.path.insert(0, str(path))
        arch = import_file("sbd_autoshot_original", path / "supernet_flattransf_3_8_8_8_13_12_0_16_60.py")
        self.model = arch.TransNetV2Supernet()
        data = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = data.get("net", data)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()  # CPU float32 reference; GPU is not advertised until tested.

    def predict(self, frames, video_path=None):
        validate_frames(frames)
        t = time.perf_counter()
        chunks = []
        with torch.inference_mode():
            for indices in window_indices(len(frames)):
                x = torch.from_numpy(frames[indices].copy()).permute(3, 0, 1, 2).unsqueeze(0).float()
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # Architecture emits LOGITS: always sigmoid, never infer activation from values.
                p = torch.sigmoid(logits).reshape(-1).cpu().numpy()
                if len(p) != 100:
                    raise ValueError("AutoShot window must return exactly 100 probabilities")
                chunks.append(p[25:75])
        probs = np.concatenate(chunks)[:len(frames)]
        if len(probs) != len(frames) or not np.isfinite(probs).all():
            raise ValueError("AutoShot frame alignment failed")
        native = official_scenes((probs > .296).astype(np.uint8))
        cuts = cuts_from_native_scenes(native, len(frames))
        return Detection(cuts, probs, native,
                         dict(device="cpu", threshold=.296, output="sigmoid(logits)", window=100, center=50,
                              parameters=sum(p.numel() for p in self.model.parameters()), native_convention="official inclusive scene endpoints"),
                         dict(network_inference_s=time.perf_counter() - t, feature_extraction_s=0.0))
