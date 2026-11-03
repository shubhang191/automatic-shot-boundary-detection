# Installation details

Follow the root README in order. A clean Windows x64 Python 3.13 virtual environment is required. requirements.txt deliberately uses the PyTorch CPU wheel index; requirements-lock.txt constrains transitive dependencies. Do not replace CPU pins with untested CUDA wheels. Internet is used for package installation, official model provisioning and FFmpeg bootstrap; normal inference runs locally afterward.

Manual prerequisite: obtain the exact AutoShot checkpoint from its authors, as explained in MODELS.md. setup.bat itself does not provision pretrained weights; prepare_models.py must run first. The included C++ binaries avoid compiler installation. FFmpeg is fetched rather than vendored, preserving the upstream distribution license/README.

The public repo intentionally omits pretrained .pth files, environments, caches, logs, private audits, SHOT/BBC/ad videos, credentials and obsolete machine-specific manifests. Small project-owned fitted bundles and synthetic numerical regression fixtures remain. The six-second Sintel sample is CC BY 3.0 with attribution in examples/README.md.
