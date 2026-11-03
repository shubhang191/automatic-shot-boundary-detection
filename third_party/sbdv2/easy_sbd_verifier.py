# Easy offline SBD verification using the original compiled sbdv2 C++ engine.
# Install: python -m pip install numpy; cd sbdv2 && bash build.sh
# Example:
#   from easy_sbd_verifier import SBDVerifier
#   result=SBDVerifier().verify('sample.mp4',cuts=[(120,121),(251,252)])
#   print(result['scenes'])
# Default model: original generic_local.sbdm; opt into an experimental .sbdm explicitly.
# No automatic proposal generation. The verifier can only keep/drop supplied cut candidates.
# End of module header
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sbdv2'/'python'))
import sbdv

class SBDVerifier:
    def __init__(self, model=None):
        self.model=Path(model) if model else ROOT/'sbdv2'/'models'/'generic_local.sbdm'
        if not self.model.is_file():raise FileNotFoundError(self.model)
    def verify(self, video, *, cuts=None, scenes=None, times=None, fps=None, **kwargs):
        provided=sum(x is not None for x in (cuts,scenes,times))
        if provided!=1:raise ValueError('Supply exactly one of cuts, scenes, times.')
        if times is not None:
            if fps is None or float(fps)<=0:raise ValueError('fps>0 is required for seconds input')
            import sbdv_io
            times=sbdv_io.times_to_cuts(times,fps)
            return sbdv.verify_video(self.model,video,cuts=times,**kwargs)
        if cuts is not None:return sbdv.verify_video(self.model,video,cuts=cuts,**kwargs)
        return sbdv.verify_video(self.model,video,scenes=scenes,**kwargs)
    def verify_frames(self,frames,*,cuts=None,scenes=None):
        if (cuts is None)==(scenes is None):raise ValueError('Supply cuts or scenes')
        model=sbdv.Model(self.model)
        return model.verify(frames,cuts=cuts) if cuts is not None else model.verify_scenes(frames,scenes)
