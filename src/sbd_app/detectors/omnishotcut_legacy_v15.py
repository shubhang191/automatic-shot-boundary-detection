"""Pinned historical v1.5 weights and decoder, isolated from the supplied detector."""
import sys
import time
from .base import Detection
from .omnishotcut import convert_native
from ..config import ROOT, verified_path
from ..video_io import decode_rgb
from ..errors import ModelError


class LegacyOmniDetector:
    name = 'omnishotcut_legacy_v15'

    def __init__(self):
        ckpt = verified_path(self.name)
        sys.path.insert(0, str(ROOT/'third_party/omnishotcut_legacy_v15'))
        from sbd_omni_v15.engine import load_model, _run_on_numpy
        from sbd_omni_v15.label_correspondence import intra_int2string, inter_int2string
        self.model, self.args = load_model(str(ckpt))
        self.run = _run_on_numpy
        self.intra_names, self.inter_names = intra_int2string, inter_int2string

    def predict(self, frames, video_path=None):
        t = time.perf_counter()
        if len(frames)*self.args.process_width*self.args.process_height*3 > 256*1024**2:
            raise ModelError('OmniShotCut decoded input exceeds 256 MiB. Use a shorter video.')
        native = decode_rgb(video_path, self.args.process_width, self.args.process_height)
        if len(native) != len(frames): raise ModelError('Historical Omni decode count is misaligned.')
        decode_s = time.perf_counter()-t
        t = time.perf_counter()
        ranges, ia, ie = self.run(native, self.model, self.args, 10)
        intra = [self.intra_names[int(x)] for x in ia]
        inter = [self.inter_names[int(x)] for x in ie]
        cuts, clean = convert_native(ranges,intra,inter,len(frames))
        return Detection(cuts, None, ranges, dict(device='cpu', checkpoint_version='v1.5 pinned historical revision',
            checkpoint_revision='1bc678977ad48c65d57d600296d95f97c1be293f', source_commit='40ecf12',
            parameters=sum(p.numel() for p in self.model.parameters()), process_width=self.args.process_width,
            process_height=self.args.process_height, overlap=10, intra_labels=intra, inter_labels=inter,
            native_general_inclusive=clean, native_convention='zero-based half-open; General ranges only produce proposals'),
            dict(model_decode_s=decode_s,feature_extraction_s=0.,network_inference_s=time.perf_counter()-t))
