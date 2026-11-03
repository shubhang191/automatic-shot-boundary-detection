from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json
import threading
import time
import numpy as np
from .config import ROOT, CACHE_ROOT, KNOWN_HASHES, MODEL_PATHS, PREPROCESS_VERSION, verified_path, sha256, executable
from .cache import DiskCache, identity, save_json
from .detectors.base import Detection
from .shot_schema import CandidateCut, cut_dicts, canonical_cuts, shots_from_cuts
from .video_io import validate_upload, normalize, run, decode_rgb
from .visualizations import render
from .exports import write_export
from .errors import SBDError, ModelError


class Pipeline:
    def __init__(self, cache_root=CACHE_ROOT):
        self.cache = DiskCache(cache_root)
        self.lock = threading.RLock()
        from .profiling import install
        install()
        self._tool_identity = {}
        self.models = {}
        self.verifier = None
        self.legacy_verifier = None
        self.calls = {name: 0 for name in ["tinycut", "autoshot", "omnishotcut", "omnishotcut_legacy_v15"]}

    def detector(self, name):
        if name not in self.models:
            import torch
            torch.set_num_threads(2)
            if name == "tinycut":
                from .detectors.tinycut import TinyCutDetector
                cls = TinyCutDetector
            elif name == "autoshot":
                from .detectors.autoshot import AutoShotDetector
                cls = AutoShotDetector
            elif name == "omnishotcut":
                from .detectors.omnishotcut import OmniShotCutDetector
                cls = OmniShotCutDetector
            elif name == "omnishotcut_legacy_v15":
                from .detectors.omnishotcut_legacy_v15 import LegacyOmniDetector
                cls = LegacyOmniDetector
            else:
                raise ModelError(f"Unknown detector: {name}")
            self.models[name] = cls()
        return self.models[name]

    def _tool_version(self):
        from .profiling import span
        with span('tool_identity'):
            path = Path(executable('ffmpeg'))
            signature = (str(path), sha256(path))
            if signature not in self._tool_identity:
                self._tool_identity = {signature: run([str(path), '-version']).decode(errors='replace').splitlines()[0]}
            return self._tool_identity[signature]

    def _model_identity(self, detector):
        from .profiling import span
        with span('model_identity'):
            files = sorted((ROOT / 'src/sbd_app').rglob('*.py')) + sorted((ROOT / 'third_party').rglob('*.py'))
            # Preserve the original content-based invalidation, including same-size
            # source edits whose timestamps are retained by deployment/copy tools.
            return identity(dict(model=KNOWN_HASHES[detector], decoder=KNOWN_HASHES['decoder'] if detector == 'tinycut' else None,
                code=[sha256(p) for p in files], device='cpu-float32',
                versions={m: importlib.metadata.version(m) for m in ['torch','torchvision','numpy','scipy','opencv-python-headless']}))

    def analyze(self, path, detector="tinycut", verified=False, progress=None, verification=None):
        from .profiling import ACTIVE, Profile
        profile = ACTIVE.get()
        owned = profile is None
        if owned: profile = Profile()
        token = ACTIVE.set(profile)
        began = time.perf_counter()
        locked = False
        try:
            with profile.span('pipeline_request'):
                with profile.span('pipeline_lock_wait'):
                    self.lock.acquire()
                locked = True
                selected = verification or ('sbdv2' if verified else 'raw')
                if selected not in {'raw','sbdv2','legacy_python'}:
                    raise ValueError('Unknown verification mode; no fallback is used.')
                if selected == 'legacy_python':
                    from legacy_python.integration_adapter import family
                    family(detector)
                result = self._analyze(path, detector, selected, progress or (lambda *a, **k: None))
            result['timings']['total_wall_s'] = time.perf_counter() - began
            if owned:
                detail = dict(total_wall_s=result['timings']['total_wall_s'], stages=profile.summary(), trace=profile.roots,
                    cpu_definition='process CPU sums all threads; FFmpeg/FFprobe child CPU separately measured on Windows',
                    scope='callback pipeline through cache eviction; excludes browser upload, Gradio queue/postprocessing/download and writing this timing sidecar',
                    verifier_transport='Python in process' if result['mode']=='legacy_python' else 'ctypes DLL in process for SBDV2; raw has no verifier',
                    annotated_video=dict(status='NOT_GENERATED',reason='Existing release provides normalized preview, not an annotated-video renderer',wall_s=0.))
                result['timings']['stages'] = detail['stages']
                timing_path = Path(result['output_dir'])/'timings.json'
                save_json(timing_path, detail)
                result['timings_file'] = str(timing_path)
            return result
        except (SBDError, ValueError):
            raise
        except Exception as exc:
            raise ModelError(f"{detector} failed: {type(exc).__name__}: {exc}. Other detectors remain available.") from exc
        finally:
            if locked:
                self.lock.release()
            ACTIVE.reset(token)

    def _analyze(self, path, detector, verification, progress):
        start = time.perf_counter()
        path = validate_upload(path)
        verified_path(detector)
        if detector == "tinycut":
            verified_path("decoder")
        progress(.05, desc="Validating and normalizing video")
        tool_version = self._tool_version()
        video_key = identity(dict(input=sha256(path), preprocessing=PREPROCESS_VERSION, ffmpeg=tool_version))
        work = self.cache.directory(video_key)
        # Evict at request entry too: a failed/corrupt decode must not accumulate unbounded jobs.
        self.cache.prune(work)
        meta_path = work / "video.json"
        t = time.perf_counter()
        decode_hit = meta_path.is_file() and (work / "frames.npy").is_file() and (work / "analysis.mp4").is_file()
        if decode_hit:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            frames = np.load(work / "frames.npy", allow_pickle=False, mmap_mode="r")
            analysis_path = work / "analysis.mp4"
        else:
            analysis_path, frames, metadata = normalize(path, work)
            np.save(work / "frames.npy", frames, allow_pickle=False)
            save_json(meta_path, metadata)
        decode_s = time.perf_counter() - t
        metadata["original_name"] = path.name
        inference_frames, inference_path = frames, analysis_path
        frame_input = 'normalized CFR analysis video'
        if detector == 'omnishotcut_legacy_v15':
            # Historical inference decodes original RGB pictures. Use them only
            # when source PTS and canonical CFR frames have a one-to-one mapping.
            pts = metadata['source_pts_seconds']
            fps = metadata['fps']
            regular = len(pts)==len(frames) and metadata['analysis_to_source_frame']==list(range(len(frames)))
            regular = regular and all(abs((p-pts[0])*fps-i)<0.01 for i,p in enumerate(pts))
            if regular:
                original_frames = work/'legacy_source_frames.npy'
                if not original_frames.is_file(): np.save(original_frames,decode_rgb(path),allow_pickle=False)
                inference_frames=np.load(original_frames,allow_pickle=False,mmap_mode='r')
                if len(inference_frames)!=len(frames): raise ModelError('Historical source decode count differs from CFR mapping')
                inference_path=path
                frame_input='original source RGB; verified regular CFR one-to-one mapping'
        progress(.3, desc="Running detector (CPU; first load may take longer)")
        # Code and dependency identities prevent reuse across adapter/feature/decoder changes.
        model_key = self._model_identity(detector)
        # Keep the added detector's filesystem prefix short on Windows. The
        # complete detector name and full content identity remain in metadata.
        disk_name = 'omni15' if detector == 'omnishotcut_legacy_v15' else detector
        pred_path = work / (disk_name + "_" + model_key + ".json")
        prob_path = pred_path.with_suffix(".npy")
        detector_hit = pred_path.is_file()
        t = time.perf_counter()
        load_s = 0.0
        if detector_hit:
            obj = json.loads(pred_path.read_text(encoding="utf-8"))
            prediction = Detection([CandidateCut(**c) for c in obj["cuts"]],
                                  np.load(prob_path, allow_pickle=False) if prob_path.is_file() else None,
                                  obj["native_ranges"], obj["details"], obj["timings"])
        else:
            lt = time.perf_counter()
            model = self.detector(detector)
            load_s = time.perf_counter() - lt
            prediction = model.predict(inference_frames, inference_path)
            if detector == 'omnishotcut_legacy_v15':
                prediction.details['frame_input'] = frame_input
            self.calls[detector] += 1
            save_json(pred_path, dict(cuts=cut_dicts(prediction.cuts), native_ranges=prediction.native_ranges,
                                      details=prediction.details, timings=prediction.timings))
            if prediction.probabilities is not None:
                np.save(prob_path, prediction.probabilities, allow_pickle=False)
        detection_s = time.perf_counter() - t
        raw_cuts = canonical_cuts(prediction.cuts, len(frames))
        final_cuts = raw_cuts
        decisions = [dict(**asdict(c), p_keep=None, status="RAW") for c in raw_cuts]
        verifier_info = None
        legacy_scenes = None
        progress(.7, desc="Verifying candidates" if verification != 'raw' else "Preparing raw results")
        t = time.perf_counter()
        if verification == 'sbdv2':
            verified_path("verifier")
            if self.verifier is None:
                from .verifier.sbdv2_adapter import SBDV2Verifier
                self.verifier = SBDV2Verifier()
            final_cuts, keep_drop, verifier_info = self.verifier.verify(frames, raw_cuts)
            by_pair = {(d["a"], d["b"]): d for d in keep_drop}
            decisions = [dict(**asdict(c), p_keep=by_pair[(c.a, c.b)]["p_keep"], status=by_pair[(c.a, c.b)]["status"]) for c in raw_cuts]
        elif verification == 'legacy_python':
            from legacy_python.integration_adapter import LegacyVerifier
            from .profiling import span
            if self.legacy_verifier is None: self.legacy_verifier=LegacyVerifier()
            with span('legacy_python_verify'):
                final_cuts,decisions,verifier_info,legacy_scenes=self.legacy_verifier.verify(detector,prediction,inference_frames)
            verifier_info['frame_input']=frame_input
        verifier_s = time.perf_counter() - t
        raw_shots = shots_from_cuts(raw_cuts, len(frames), metadata["fps"])
        shots = shots_from_cuts(final_cuts, len(frames), metadata["fps"])
        if legacy_scenes is not None:
            from legacy_python.integration_adapter import scenes_to_shots
            shots=scenes_to_shots(legacy_scenes,metadata['fps'])
        if detector == 'omnishotcut_legacy_v15':
            from legacy_python.integration_adapter import scenes_to_shots
            native=prediction.details['native_general_inclusive']
            if native:
                raw_shots=scenes_to_shots(native,metadata['fps'])
                if verification == 'raw': shots=raw_shots
        mode = 'verified' if verification == 'sbdv2' else verification
        if verification == 'legacy_python':
            verifier_info['cache_identity']=identity(dict(video=video_key,detector=detector,checkpoint=KNOWN_HASHES[detector],
                model=model_key,bundle=verifier_info['bundle_sha256'],source=verifier_info['source_sha256'],
                settings=verifier_info['settings'],dependencies=verifier_info['dependencies'],frame_input=frame_input))
            # Verifier always recomputes. Only immutable detector predictions are reused.
            mode_dest=mode+'_'+verifier_info['cache_identity'][:16]
        else: mode_dest=mode
        dest = work / (disk_name + "_" + model_key[:12] + "_" + mode_dest)
        dest.mkdir(exist_ok=True)
        progress(.85, desc="Drawing timeline, extracting thumbnails, exporting")
        t = time.perf_counter()
        timeline, gallery = render(analysis_path, raw_shots, shots, decisions, metadata, dest)
        render_s = time.perf_counter() - t
        result = dict(metadata=metadata, detector=detector, mode=mode, model_sha256=KNOWN_HASHES[detector],
                      decoder_sha256=KNOWN_HASHES["decoder"] if detector == "tinycut" else None,
                      preprocessing=PREPROCESS_VERSION, model_cache_identity=model_key, detector_details=prediction.details,
                      native_ranges=prediction.native_ranges, raw_candidates=cut_dicts(raw_cuts), decisions=decisions,
                      verifier=verifier_info, raw_shots=raw_shots, shots=shots,
                      timings=dict(decode_normalize_s=decode_s, model_load_s=load_s, detector_request_s=detection_s,
                                   original_detector_timings=prediction.timings, verifier_s=verifier_s, render_s=render_s),
                      cache=dict(video_hit=decode_hit, detector_hit=detector_hit, policy="4 inputs / 4 GiB / 24h; serialized requests"))
        result['provenance']=dict(detector=detector,checkpoint_sha256=KNOWN_HASHES[detector],verification=mode,
            bundle_sha256=verifier_info.get('bundle_sha256') if verifier_info else None,
            native_verifier_model_sha256=verifier_info.get('model_sha256') if verifier_info else None,
            rescue_enabled=bool(verifier_info and verifier_info.get('rescue_enabled',False)),
            snap_enabled=bool(verifier_info and verifier_info.get('snap_enabled',False)),
            fps=metadata['fps'],decoded_frame_count=len(frames),preprocessing=PREPROCESS_VERSION,
            detector_frame_input=frame_input,detector_cached=detector_hit,
            thresholds=verifier_info.get('settings',verifier_info) if verifier_info else None,
            display_convention='inclusive native legacy scenes; transition gaps preserved' if legacy_scenes is not None or detector=='omnishotcut_legacy_v15' and verification=='raw' else metadata['transition_allocation'])
        t = time.perf_counter()
        # Metadata timing is measured before the ZIP is finalized and includes JSON/CSV generation.
        result["timings"]["total_before_export_s"] = time.perf_counter() - start
        export_zip = write_export(dest, result)
        result["timings"]["export_s"] = time.perf_counter() - t
        result["timings"]["total_wall_s"] = time.perf_counter() - start
        result.update(export_zip=export_zip, timeline=timeline, gallery=gallery, analysis_video=str(analysis_path), output_dir=str(dest))
        self.cache.prune(work)
        return result
