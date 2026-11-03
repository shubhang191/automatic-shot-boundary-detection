from pathlib import Path
import shutil
import hashlib
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image, ImageDraw


@lru_cache(maxsize=4)
def _image_identity(size, mtime, ctime):
    import PIL
    return hashlib.sha256(Path(__file__).read_bytes() + (cv2.__version__ + PIL.__version__).encode()).hexdigest()[:16]


def render(video, raw_shots, shots, decisions, metadata, dest):
    """Small timeline and middle-frame thumbnails. No unsupported accuracy/type annotations."""
    dest = Path(dest)
    thumbs = dest / "thumbnails"
    thumbs.mkdir(parents=True, exist_ok=True)
    st = Path(__file__).stat()
    keyframe_cache = dest.parent / ('keyframes_' + _image_identity(st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    keyframe_cache.mkdir(exist_ok=True)
    cap = None
    gallery = []
    try:
        for row in shots:
            i = (row["start_frame"] + row["end_frame"]) // 2
            path = thumbs / f"shot_{row['shot']:04d}.jpg"
            cached = keyframe_cache / f'frame_{i:08d}.jpg'
            if cached.is_file():
                shutil.copyfile(cached, path)
            else:
                if cap is None: cap = cv2.VideoCapture(str(video))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, bgr = cap.read()
                if not ok: raise ValueError(f"Cannot read thumbnail frame {i}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.thumbnail((320, 180))
                img.save(path, quality=85)
                temporary = cached.with_suffix('.tmp')
                shutil.copyfile(path, temporary)
                temporary.replace(cached)
            gallery.append((str(path), f"Shot {row['shot']} · {row['start_s']:.3f}–{row['end_exclusive_s']:.3f}s · frame {i}"))
    finally:
        if cap is not None: cap.release()
    timeline = Image.new("RGB", (1200, 205), "#101b2c")
    draw = ImageDraw.Draw(timeline)
    total = metadata["decoded_frame_count"]
    colors = ["#49c4c2", "#6b94ea", "#c99cda", "#f3c66b"]
    draw.text((30, 15), "SHOT TIMELINE  |  orange lines: dropped candidates", fill="#ffffff")
    for label, rows, y in [("Raw", raw_shots, 65), ("Selected", shots, 125)]:
        draw.text((15, y + 9), label, fill="#ffffff")
        for i, row in enumerate(rows):
            x0 = 100 + 1070 * row["start_frame"] / total
            x1 = 100 + 1070 * (row["end_frame"] + 1) / total
            draw.rectangle((x0, y, max(x0, x1 - 1), y + 30), fill=colors[i % len(colors)])
    for d in decisions:
        if d["status"] == "DROP":
            x = 100 + 1070 * d["b"] / total
            draw.line((x, 60, x, 160), fill="#ff924e", width=2)
    draw.text((100, 177), f"0 s                       Analysis CFR: {metadata['fps_rational']} fps                      {metadata['duration_s']:.3f} s", fill="#ffffff")
    path = dest / "timeline.png"
    timeline.save(path)
    return str(path), gallery
