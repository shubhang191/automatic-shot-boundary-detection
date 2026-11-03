"""Install a pinned official Windows FFmpeg tool archive locally; verify before extracting."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"
SHA = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"


def bootstrap(archive=None):
    dest = ROOT / "tools/ffmpeg"
    if (dest / "bin/ffmpeg.exe").is_file() and (dest / "bin/ffprobe.exe").is_file():
        print("Local FFmpeg tools already installed")
        return
    if not archive and shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("Using FFmpeg and FFprobe on PATH")
        return
    with tempfile.TemporaryDirectory(prefix="sbd_ffmpeg_") as td:
        downloaded = Path(archive) if archive else Path(td) / "ffmpeg.zip"
        if not archive:
            print("Downloading pinned FFmpeg (initial setup needs internet)", flush=True)
            # curl is shipped with Windows 10/11; it displays progress and enforces a deadline.
            subprocess.run(["curl.exe", "--fail", "-L", "--retry", "2", "--max-time", "600", "-o", str(downloaded), URL], check=True)
        got = hashlib.sha256(downloaded.read_bytes()).hexdigest()
        if got != SHA:
            raise RuntimeError("FFmpeg archive SHA mismatch. Do not use changed bytes; see docs/troubleshooting.md.")
        with zipfile.ZipFile(downloaded) as z:
            if z.testzip() is not None:
                raise RuntimeError("FFmpeg ZIP CRC failed")
            for name in z.namelist():
                leaf = Path(name).name
                if leaf in {"ffmpeg.exe", "ffprobe.exe", "LICENSE", "LICENSE.txt", "README.txt"}:
                    out = dest / ("bin/" + leaf if leaf.endswith(".exe") else leaf)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
    print("FFmpeg and FFprobe installed in tools/ffmpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", help="Use the exact pinned archive already downloaded (offline install)")
    bootstrap(parser.parse_args().archive)
