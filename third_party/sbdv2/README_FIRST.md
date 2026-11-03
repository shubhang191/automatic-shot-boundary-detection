# SBD-Verifier: portable lightweight KEEP/DROP postprocessor

**What it does:** Accept proposed shot cuts from any detector, inspect 48×27 video frames around each cut and return fewer false boundaries where the trained classifier identifies a likely false alarm. It cannot invent or recover a missing proposal.

## Quick start (Linux x86-64 or compile from source)
```bash
cd sbdv2
bash build.sh
python -m pip install numpy
./bin/sbdv info models/generic_local.sbdm
./bin/sbdv verify --model models/generic_local.sbdm --video video.mp4 --cuts cuts.txt --out result.json
```
`cuts.txt` is one line per candidate: `a b`, where `a` is the zero-based last frame of the earlier shot and `b` is the zero-based first frame of the next shot. If using whole shot lists, choose `--scenes` and 0-based inclusive `start end`. For a model produced by a different detector, check its input format rather than assuming its raw probabilities are compatible.

### Minimal Python
```python
from easy_sbd_verifier import SBDVerifier
m=SBDVerifier()  # Default is the pre-existing, historically tested generic_local.sbdm
print(m.verify('video.mp4',cuts=[(50,51),(99,100)])['scenes'])
# You may also pass scenes=[(0,50),(51,99),(100,149)], or times=[2.04,4.0],fps=25.
```
Run from the release folder. Install FFmpeg on the computer's PATH for `video.mp4` decoding. The included Linux binary may need recompilation on another OS/CPU; sources and `build.sh` are provided.

### Which model?
- `models/generic_local.sbdm`: **baseline default**; unchanged and preserved from the historical verifier.
- `models/generic_original_labels_control.sbdm`: 69-video study overlap-label control, for reproduction.
- `models/generic_event_aware_candidate.sbdm`: **experimental**, if V39 showed evidence for improved labels. Explicitly choose it with `SBDVerifier(model=...)`. Evidence came from approved 69-video video-grouped training CV and reused 20-video diagnostic, not an independent benchmark. Do not state universal improvement.

### Limits
- Unlike CLAHE/Optuna, this is a fitted classifier: portability of the **API** does not imply accuracy gains on all video domains or detector types.
- Every cut decision uses a future-frame lookahead (historical local model needs up to 60 frames, about 2.4s at 25fps); this is algorithmic delay, not CPU runtime.
- Peak memory, total decoded-video throughput and energy use must be measured separately. No power data was collected.
- If detections use different frame-number conventions, consult `sbdv2/python/sbdv_io.py` or `sbdv2/README.md`. Never silently shift indices.
- AutoShot+classic train69 is *two proposal families* but a single underlying training-video source. It does not guarantee arbitrary detector transfer.

## Reproduce validation
- `V39...ipynb`: 69-video nested, grouped comparison of overlap versus event-aware labels using actual C++ 30 features; source integrity checks.
- `V40...ipynb`: model export, C++/sklearn parity, stream/RAM parity, original toolkit regression suite and reproducible ZIP. No SHOT200 benchmark refit or evaluation.
- See `V40_validation_manifest.json` for measured tests, SHA-256s and caveats.
