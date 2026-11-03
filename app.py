"""Local Gradio interface. Model and video logic live in src/sbd_app."""
from pathlib import Path
import argparse
import os
import socket
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import gradio as gr
from sbd_app.config import CACHE_ROOT, DETECTOR_NAMES, verified_path
from sbd_app.pipeline import Pipeline
from sbd_app.verifier.build_status import library_path

PIPELINE = Pipeline()
# Presentation labels only: backend IDs, config hashes and callback values stay stable.
UI_DETECTOR_LABELS = {
    'tinycut': 'Clean TinyCut S — Our Lightweight Detector',
    'autoshot': 'AutoShot — Pretrained Baseline',
    'omnishotcut': 'OmniShotCut Preview — 41M (Older Release)',
    'omnishotcut_legacy_v15': 'OmniShotCut v1.5 — 52.8M (Newer Version)',
}
UI_MODE_LABELS = {
    'Raw': 'Raw — No Verification',
    'Raw + SBDV2 (C++)': 'SBDV2 (C++) — Fast KEEP/DROP Filtering',
    'Raw + Legacy Python Verifier': 'Original Python Verifier — Learned Verification',
    'Raw + Legacy Python (Rescue + Snap)': 'Python Verifier — Higher Accuracy, Slower',
}
MODEL_DESCRIPTIONS = {
    'tinycut': '**Our Trained Lightweight Model.** Compact shot-boundary detector developed and trained in this project. Uses a small neural network with 18,433 parameters to predict shot boundaries directly.',
    'autoshot': '**External Pretrained Baseline.** Published shot-boundary detector used as a reference model for our SHOT benchmark and Python-verifier experiments. By Wentao Zhu and colleagues ([AutoShot](https://github.com/wentaozhu/AutoShot)).',
    'omnishotcut': '**Older Pretrained Preview — Approximately 41M Parameters.** Earlier OmniShotCut preview model. It can produce different shot candidates and transition labels from v1.5; accuracy depends on the video.',
    'omnishotcut_legacy_v15': '**Newer Recommended Pretrained Release — Approximately 52.8M Parameters.** Recommended by the original OmniShotCut authors and used in our original Review 1 research and historical Python-verifier experiments. We did not train this checkpoint; our contribution is the verifier and experimental evaluation.',
}
MODE_DESCRIPTIONS = {
    'Raw': "Use the detector's original predictions without our learned verifier.",
    'Raw + SBDV2 (C++)': 'Our lightweight native C++ verifier keeps or rejects existing candidate boundaries. It cannot independently recover boundaries the detector never proposed.',
    'Raw + Legacy Python Verifier': 'Our historical Python verifier evaluates original and lower-confidence AutoShot candidate boundaries.',
    'Raw + Legacy Python (Rescue + Snap)': 'Our historical learned verifier filters candidates, generates additional rescue candidates, and can adjust boundary positions using snapping.',
}
SHOT_HEADERS = ["Shot", "Start frame", "End frame", "Start (s)", "End exclusive (s)", "Duration (s)", "Boundary"]
CUT_HEADERS = ["Last old frame", "First new frame", "Detector score", "Verifier p_keep", "Decision", "Source", "Threshold", "Rescue proposal", "Snapped", "Final old frame", "Final new frame", "Native inter label", "Dropped transition labels"]


def mode_choices(detector):
    choices=['Raw','Raw + SBDV2 (C++)']
    if detector=='autoshot': choices.append('Raw + Legacy Python Verifier')
    elif detector in {'omnishotcut','omnishotcut_legacy_v15'}: choices.append('Raw + Legacy Python (Rescue + Snap)')
    return choices


def mode_options(detector):
    return [(UI_MODE_LABELS[value], value) for value in mode_choices(detector)]


def selection_description(detector, mode):
    text = MODE_DESCRIPTIONS[mode]
    if mode == 'Raw + SBDV2 (C++)':
        text += '\n\nOptimized C++ filtering pipeline. This is not identical to the historical Python rescue-and-snap method.'
    elif mode == 'Raw + Legacy Python (Rescue + Snap)':
        if detector == 'omnishotcut_legacy_v15':
            text += '\n\nOriginal Review 1 experimental configuration. Uses the v1.5 pretrained checkpoint and our historical Python verifier.'
        elif detector == 'omnishotcut':
            text += '\n\nCross-version experiment. Applies the Python verifier developed using v1.5 to the earlier 41M detector. Predictions may differ from the original Review 1 results.'
    return text


def detector_selection(detector):
    return (gr.Dropdown(choices=mode_options(detector), value='Raw'),
            MODEL_DESCRIPTIONS[detector], selection_description(detector, 'Raw'))


def shot_rows(shots):
    return [[r["shot"], r["start_frame"], r["end_frame"], round(r["start_s"], 6),
             round(r["end_exclusive_s"], 6), round(r["duration_s"], 6), r["boundary_type"]] for r in shots]


def readiness():
    lines = []
    for key, label in UI_DETECTOR_LABELS.items():
        try:
            verified_path(key)
            if key == "tinycut":
                verified_path("decoder")
            lines.append(f"**{label}:** checkpoint verified · CPU")
        except Exception as exc:
            lines.append(f"**{label}:** unavailable — {exc}")
    lines.append("**SBDV2:** " + ("native library present" if library_path().is_file() else "build required — run setup.bat"))
    from legacy_python.integration_adapter import checked_bundle_path
    for detector in ['autoshot','omnishotcut']:
        try: lines.append(f'**Legacy Python {detector}:** {checked_bundle_path(detector).name} · SHA-256 verified')
        except Exception as exc: lines.append(f'**Legacy Python {detector}:** unavailable — {exc}')
    return "\n\n".join(lines)


def analyze(video, detector, mode, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload an MP4, MOV or AVI video first.")
    try:
        modes={'Raw':'raw','Raw + SBDV2':'sbdv2','Raw + SBDV2 (C++)':'sbdv2',
               'Raw + Legacy Python Verifier':'legacy_python','Raw + Legacy Python (Rescue + Snap)':'legacy_python'}
        if mode not in modes: raise ValueError('Unknown verification choice; no fallback is used.')
        result = PIPELINE.analyze(video, detector, verification=modes[mode], progress=progress)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    m, t = result["metadata"], result["timings"]
    kept = sum(d["status"] == "KEEP" for d in result["decisions"])
    dropped = sum(d["status"] == "DROP" for d in result["decisions"])
    summary = (f"### {len(result['shots'])} shots · {m['duration_s']:.3f} seconds\n"
               f"{m['decoded_frame_count']} decoded frames · {m['fps_rational']} fps · {m['width']} × {m['height']}\n\n"
               f"Raw: **{len(result['raw_shots'])} shots** → Selected: **{len(result['shots'])} shots**. "
               + (f"Verifier kept {kept} and dropped {dropped} candidates." if mode != "Raw" else "Raw candidates shown; verifier not applied.")
               + f"\n\nDecode/normalize {t['decode_normalize_s']:.2f}s · Detector {t['detector_request_s']:.2f}s · "
                 f"Verifier {t['verifier_s']:.3f}s · Render/export {t['render_s'] + t['export_s']:.2f}s · Pipeline total {t['total_wall_s']:.2f}s\n\n"
               + ("Detector cache reused." if result["cache"]["detector_hit"] else "Detector computed on CPU.")
               + " Zero-based inclusive frames; displayed end time is exclusive. Transition frames stay with the preceding shot.")
    if result['mode']=='legacy_python':
        summary=summary.replace('Transition frames stay with the preceding shot.','Legacy scene intervals preserve gradual-transition gaps.')
        added=sum(d.get('rescued',False) and d['status']=='KEEP' for d in result['decisions'])
        snapped=sum(d.get('snapped',False) for d in result['decisions'])
        summary+=f"\n\nAccepted additional proposals: **{added}** · Snapped boundaries: **{snapped}**. "
        summary+='Omni Python uses native Omni labels and image features; AutoShot agreement is disabled. ' if detector!='autoshot' else ''
        summary+='Historical benchmark scores are not a new accuracy claim for this run.'
    provenance=result['provenance']
    summary+=f"\n\n**Detector:** {UI_DETECTOR_LABELS[detector]} · **Verification:** {UI_MODE_LABELS.get(mode, mode)}"
    summary+=f"\n\nCheckpoint SHA-256: `{result['model_sha256']}`"
    if provenance['bundle_sha256']: summary+=f"\n\nVerifier bundle SHA-256: `{provenance['bundle_sha256']}`"
    summary+=f"\n\nRescue: {provenance['rescue_enabled']} · Snap: {provenance['snap_enabled']} · Frame input: {provenance['detector_frame_input']}"
    if detector=='omnishotcut_legacy_v15' and result['mode']=='raw':
        summary=summary.replace('Transition frames stay with the preceding shot.','Legacy native scene intervals preserve gradual-transition gaps.')
    if detector=='omnishotcut' and result['mode']=='legacy_python':
        summary+='\n\nCross-version experiment: Preview + Python. Select OmniShotCut v1.5 + Original Python Verifier — Rescue + Snap for the original Review 1 configuration.'
    stages = t.get('stages', {})
    summary += "\n\n<details><summary>Detailed measured timings</summary>\n\n| Operation | Wall seconds | CPU core-seconds* |\n|---|---:|---:|\n"
    for label, key in [('Tool version / identity','tool_identity'), ('Code / package identity','model_identity'),
                       ('Checkpoint SHA verification','checkpoint_integrity'), ('Probe / transcode / decode overall','normalize_total'),
                       ('Source PTS probe','source_pts_probe'), ('Normalized MP4 validation','analysis_header_probe'),
                       ('Decode and resize','decode_resize'), ('Model load / initialization','model_load_initialize'),
                       ('TinyCut network initialization','tiny_network_initialize'),
                       ('Detector features / transforms','feature_extraction'), ('Neural forward','neural_forward'),
                       ('Verifier DLL load','verifier_load_initialize'), ('Verifier input preparation','verifier_prepare_transfer'),
                       ('C++ verifier call','verifier_native_call'), ('Legacy Python features / model / snap','legacy_python_verify'),
                       ('Thumbnails / timeline','render_total'), ('Exports','export_total')]:
        row = stages.get(key, {})
        wall_key, cpu_key = ('exclusive_wall_s', 'exclusive_cpu_s') if key == 'verifier_prepare_transfer' else ('wall_s', 'cpu_s')
        summary += f"| {label} | {row.get(wall_key, 0):.4f} | {row.get(cpu_key, 0):.4f} |\n"
    summary += ("\n*CPU sums all process threads; child FFmpeg CPU is recorded separately in the timing JSON. "
                "Rows can be nested; do not sum them. AutoShot's feature cues are included in its neural forward. "
                "The verifier uses an in-process DLL: process startup / pipe transfer are zero. "
                "This release generates a normalized preview, not an annotated video. "
                "Pipeline wall time excludes browser upload, Gradio queue/postprocessing and download.</details>")
    if result['mode']=='legacy_python':
        summary=summary.replace('The verifier uses an in-process DLL: process startup / pipe transfer are zero.',
            'The selected legacy verifier runs in Python in process; C++ call timings are zero for this mode.')
    cuts = [[d['a'],d['b'],d['score'],d['p_keep'],d['status'],d.get('source'),d.get('threshold'),
             d.get('rescued'),d.get('snapped'),d.get('final_a'),d.get('final_b'),d.get('native_inter_label'),
             ', '.join(d.get('dropped_transition_labels',[]))] for d in result['decisions']]
    return (summary, result["analysis_video"], result["timeline"], result["gallery"],
            shot_rows(result["shots"]), shot_rows(result["raw_shots"]), cuts, result["export_zip"],
            str(Path(result["output_dir"]) / "shots.csv"), str(Path(result["output_dir"]) / "shots.json"),
            str(Path(result["output_dir"]) / "cuts.csv"), result["timings_file"])


def create_app():
    with gr.Blocks(title="Automatic Shot Boundary Detection", analytics_enabled=False, delete_cache=(600, 86400)) as demo:
        gr.Markdown("# Automatic Shot Boundary Detection\nUpload a video. Compare our lightweight detector and third-party pretrained models. Inspect every verification decision.")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="1 · Upload video · MP4 / MOV / AVI · up to 5 minutes / 500 MiB", sources=["upload"], format=None)
                detector = gr.Dropdown([(label, key) for key, label in UI_DETECTOR_LABELS.items()], value="tinycut", label="2 · Choose detector", info='What model initially identifies potential shot transitions?')
                model_description = gr.Markdown(MODEL_DESCRIPTIONS['tinycut'])
                mode = gr.Dropdown(mode_options('tinycut'), value='Raw', label='Candidate verification', info="How should the detector's candidate boundaries be processed?", allow_custom_value=True)
                configuration = gr.Markdown(selection_description('tinycut', 'Raw'))
                button = gr.Button("3 · Analyze video", variant="primary")
                gr.Markdown("The progress estimate is approximate. Actual measured timings appear after processing.")
                with gr.Accordion('Understanding the models and verification modes', open=False):
                    gr.Markdown('A **detector** proposes transitions; a **verifier** processes those proposals. Raw retains detector predictions. C++ SBDV2 provides fast KEEP/DROP filtering. Original Python uses our fitted learned verifiers: AutoShot includes lower-confidence proposals; Omni includes rescue and three-frame snapping. TinyCut supports Raw and C++ only.\n\nBoth Omni checkpoints are pretrained models by Boyang Wang and colleagues ([official model zoo](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut#model-zoo-)); Preview is the older 41M release and v1.5 is the newer recommended 52.8M release. AutoShot is also third-party pretrained research. Our project developed TinyCut, the learned verifiers, C++ SBDV2, evaluation and integration.\n\nThe original Review 1 configuration uses **v1.5 + Original Python Verifier — Rescue + Snap**. Preview + Python is a cross-version experiment. Accuracy depends on the video; historical benchmark scores do not apply automatically to every combination.')
                with gr.Accordion("Model readiness", open=False):
                    gr.Markdown(readiness())
            with gr.Column(scale=2):
                summary = gr.Markdown("### Ready for a video\nClean TinyCut S is the default CPU detector.")
                preview = gr.Video(label="Analysis video · normalized CFR", format=None, interactive=False)
                timeline = gr.Image(label="Raw and selected shot timeline", type="filepath")
        gallery = gr.Gallery(label="Shot keyframes · middle frame", columns=4, height=310)
        with gr.Tab("Selected shots"):
            table = gr.Dataframe(headers=SHOT_HEADERS, interactive=False)
        with gr.Tab("Raw shots"):
            raw_table = gr.Dataframe(headers=SHOT_HEADERS, interactive=False)
        with gr.Tab("KEEP / DROP decisions"):
            cuts = gr.Dataframe(headers=CUT_HEADERS, interactive=False)
        with gr.Row():
            bundle = gr.File(label="All results (ZIP)")
            csv = gr.File(label="Shots CSV")
            json_file = gr.File(label="Shots JSON")
            cut_file = gr.File(label="Cuts CSV")
            timing_file = gr.File(label="Detailed timing JSON")
        gr.Markdown("Processing stays on this computer. The app uses a bounded local cache. Research metrics and limitations are documented in the release.")
        button.click(analyze, [video, detector, mode],
                     [summary, preview, timeline, gallery, table, raw_table, cuts, bundle, csv, json_file, cut_file, timing_file], api_name="analyze", concurrency_limit=1)
        detector.change(detector_selection, [detector], [mode, model_description, configuration], api_name=False)
        mode.change(selection_description, [detector, mode], [configuration], api_name=False)
    return demo


def free_port(preferred):
    for port in range(preferred, preferred + 20):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free local port in the requested range")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    port = free_port(args.port)
    print(f"Local application: http://127.0.0.1:{port}", flush=True)
    create_app().queue(max_size=4).launch(server_name="127.0.0.1", server_port=port, share=False,
                                         inbrowser=not args.no_browser, allowed_paths=[str(CACHE_ROOT)],
                                         max_file_size="500mb", theme=gr.themes.Soft(primary_hue="teal"))
