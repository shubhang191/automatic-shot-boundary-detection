from fractions import Fraction
from pathlib import Path
import bisect
import json
import subprocess
import numpy as np
from .config import MAX_BYTES, MAX_SECONDS, MAX_FRAMES, executable, sha256
from .errors import VideoError


def run(args, timeout=300):
    try:
        proc = subprocess.run([str(x) for x in args], capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoError(f"Video tool failed: {exc}") from exc
    if proc.returncode:
        raise VideoError(proc.stderr.decode("utf-8", errors="replace")[-1400:] or "Video tool failed")
    return proc.stdout


def probe(path, include_pts=False):
    args = [executable("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format"]
    if include_pts:
        args += ["-show_frames", "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time:stream:format"]
    try:
        obj = json.loads(run(args + ["-of", "json", str(path)], timeout=90))
        stream = obj["streams"][0]
        rate = Fraction(stream.get("avg_frame_rate", "0/1"))
        if rate <= 0:
            rate = Fraction(stream.get("r_frame_rate", "0/1"))
        duration = float(stream.get("duration", obj.get("format", {}).get("duration", "nan")))
        if not 1 <= float(rate) <= 60 or not np.isfinite(duration) or not 0 < duration <= MAX_SECONDS + .1:
            raise ValueError(f"FPS must be 1–60 and duration 0–{MAX_SECONDS}s (got {rate}, {duration}s)")
        if int(stream["width"]) * int(stream["height"]) > 3840 * 2160:
            raise ValueError("Maximum input resolution is 3840×2160")
        pts = [float(f["best_effort_timestamp_time"]) for f in obj.get("frames", []) if "best_effort_timestamp_time" in f]
        if include_pts and (not pts or len(pts) > MAX_FRAMES or any(b <= a for a, b in zip(pts, pts[1:]))):
            raise ValueError("Missing/non-monotonic timestamps or too many frames")
        return dict(codec=stream["codec_name"], width=int(stream["width"]), height=int(stream["height"]),
                    fps_rational=str(rate), fps=float(rate), time_base=stream.get("time_base"),
                    declared_frame_count=stream.get("nb_frames"), duration_s=duration, pts=pts)
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise VideoError(f"Unsupported or corrupt video: {exc}") from exc


def validate_upload(path):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
        raise VideoError("Upload a readable MP4, MOV or AVI video")
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise VideoError("Video must be nonempty and at most 500 MiB")
    return path


def decode_rgb(path, width=48, height=27):
    raw = run([executable("ffmpeg"), "-nostdin", "-v", "error", "-xerror", "-i", path,
               "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
               "-frames:v", str(MAX_FRAMES + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-"])
    size = width * height * 3
    if not raw or len(raw) % size or len(raw) // size > MAX_FRAMES:
        raise VideoError("Decode returned empty, partial or excessive frames")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def normalize(path, work):
    """Always materialize CFR analysis media. Preserve exact source PTS and FFmpeg's fps mapping."""
    path = validate_upload(path)
    source = probe(path, include_pts=True)
    rate = source["fps_rational"]
    dest = Path(work) / "analysis.mp4"
    # fps round=near selects the source frame whose rounded input timestamp most recently
    # crossed this output tick. Mapping below uses the identical rational timebase rounding.
    filt = f"setpts=PTS-STARTPTS,fps=fps={rate}:start_time=0:round=near,scale=-2:'min(720,trunc(ih/2)*2)'"
    run([executable("ffmpeg"), "-nostdin", "-v", "error", "-xerror", "-y", "-i", path,
         "-map", "0:v:0", "-map", "0:a:0?", "-vf", filt, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", dest])
    analysis = probe(dest)
    frames = decode_rgb(dest)
    fps = float(Fraction(rate))
    pts0 = source["pts"][0]
    ticks = [int((Fraction(str(p)) - Fraction(str(pts0))) * Fraction(rate) + Fraction(1, 2)) for p in source["pts"]]
    mapping = [max(0, bisect.bisect_right(ticks, i) - 1) for i in range(len(frames))]
    source_no_pts = {k: v for k, v in source.items() if k != "pts"}
    metadata = dict(source_video_sha256=sha256(path), original_name=path.name, source=source_no_pts,
                    source_pts_seconds=source["pts"], analysis_to_source_frame=mapping,
                    mapping_rule="FFmpeg fps round=near after PTS-STARTPTS; duplicate rounded PTS selects last input",
                    cfr_transcoded=True, fps=fps, fps_rational=rate, time_base=analysis["time_base"],
                    decoded_frame_count=len(frames), duration_s=len(frames) / fps,
                    width=analysis["width"], height=analysis["height"],
                    frame_convention="zero-based inclusive; end_exclusive_s=(end_frame+1)/fps",
                    transition_allocation="a+1..b-1 assigned to preceding display shot; original interval retained",
                    frames48_sha256=__import__("hashlib").sha256(frames.tobytes()).hexdigest())
    return dest, frames, metadata
