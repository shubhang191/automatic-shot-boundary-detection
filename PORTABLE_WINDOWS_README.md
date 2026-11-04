# Portable Windows installation (v2)

This is the corrected setup/packaging release. Detection, model weights, verification rules and exports are unchanged. Older reports describe historical releases; use this guide for setup.

## Requirements

Windows x64; tested here on Windows 11 with CPython 3.13.5 x64. Other Python minor versions are rejected. Windows 10 is not independently tested in this repair. No CUDA is required: the unchanged dependency files pin CPU PyTorch 2.11.0+cpu and TorchVision 0.26.0+cpu. Gradio is 6.31.0, NumPy 2.1.3 and scikit-learn 1.6.1. Initial setup needs internet for pinned packages and FFmpeg if absent. Python itself is an external prerequisite; setup never installs it or changes system configuration. The faculty ZIP includes FFmpeg/FFprobe and the x64 C++ verifier binaries; their PE import tables reference Windows system/UCRT libraries. A compiler is not needed when using those supplied binaries. The public source checkout requires provisioning native tools as its existing documentation explains. Hardware memory requirements have not been measured. Allow several GB for locally installed dependencies and runtime results.

## Install and run

1. Extract the complete ZIP to a shallow writable directory. The full extracted application folder path must be at most **80 characters**, to accommodate long filenames inside pinned PyTorch wheels on Windows. Spaces are supported. Setup rejects longer paths before touching the environment. Do not run inside the ZIP viewer.
2. Install **Python 3.13 Windows installer (64-bit)** from https://www.python.org/downloads/windows/ if needed. Include the Python launcher. Reopen your terminal afterward. `py -3.13 --version` should work.
3. Double-click `setup.bat`, or invoke it from PowerShell using `& "PATH TO EXTRACTED APP\setup.bat"`. No activation is needed. Setup creates `.venv` on THIS computer and installs the unchanged pinned requirements.
4. Run `verify_installation.bat`. It checks imports, checkpoint hashes, native verifier, FFmpeg and writable cache. PASS here is installation verification, not inference proof.
5. Run `run.bat`. Open the printed http://127.0.0.1:7860 URL (or next available port).
6. Upload the bundled Sintel excerpt, choose a detector and supported verifier, then Analyze. TinyCut supports Raw and C++ only. The three other detectors additionally support Python verification.
7. Review intervals, keyframes and KEEP/DROP decisions. Download CSV, JSON or the results ZIP.

Do not transfer `.venv` between computers. The release archive never includes it. To move the app, extract the clean ZIP again and rerun setup locally.

## Safe repair and repeat setup

Setup executes the environment interpreter, checks Python 3.13 x64, verifies this folder's prefix and checks the base interpreter exists. It prefers `py -3.13`, then a supported `python.exe` on PATH, skipping Windows Store aliases. If no suitable interpreter is found, setup stops without moving the old environment.

If an environment is stale or incomplete, setup preserves it as `.venv.invalid-TIMESTAMP-UNIQUE`, then creates a new one using current-computer Python. Nothing is silently deleted. An environment explicitly marked INCOMPLETE from an interrupted setup is also backed up and recreated. Setup can be repeated; a valid completed environment is reused, dependencies checked again, and success is recorded only after diagnostics pass. `run.bat` refuses incomplete setup. Installation logs are in `installation_logs`. A Python virtual environment still requires its base Python to remain installed.

## Packaging

Use `python -B scripts/package_portable.py NEW_OUTPUT.zip`. It uses `distribution_files.json`, an explicit file allowlist. Virtual environments, backups, caches, bytecode, logs and nested copies are rejected if listed. Symlinks and unsafe archive paths are rejected. The script refuses to overwrite existing ZIPs, generates an archive hash manifest and verifies every CRC and file hash. To add a legitimate distributable file, review and update the allowlist; directory recursion is never used for packaging. The earlier 1.44 GB ZIP is superseded and must not be distributed.

Public source packages omit third-party weights and native tools; provision models exactly as docs/MODELS.md and the repository README describe before setup verification can pass. The local faculty repair package was tested with the exact frozen weights. Do not interpret those tests as proof that models are automatically downloadable from this public repository.

## Independent-computer acceptance check

A second Windows computer is NOT TESTED unless explicitly recorded in the repair report. On that computer: extract the new ZIP, confirm no `.venv` exists, install Python 3.13 x64, run setup, run verify_installation, run the app, process Sintel with all 11 selectable modes and download outputs. Send the installation log and exact error if any step fails. Repeating setup should reuse the local environment. Do not use the old 1.44 GB ZIP.


## Long Windows extraction paths

Always launch with run.bat. It selects a short, per-installation cache under the current account LOCALAPPDATA/SBD/c directory before starting the unchanged app.py. This avoids generated detector filenames exceeding the Windows path limit when the extracted application folder is deep. Diagnostics use the same location. Existing SBD_CACHE overrides are preserved; choose a short writable override if you use one. The cache identity, inference code and exported frame/timestamp rules are unchanged. Existing caches are not deleted. Directly invoking app.py bypasses this launcher safeguard.
