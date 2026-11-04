# Controlled Windows deployment audit

This local finishing repair preserves the academic folder name **209_ShubhangShrivats**. New faculty candidates use that exact top-level folder. The previously verified v2 ZIP is a protected reference and is not overwritten.

Use setup.bat, verify_installation.bat and run.bat. Python 3.13 x64, a writable application path at most 80 characters, and internet for initial dependency installation remain required. The precompiled faculty DLL/EXE are included: the professor does not need a compiler, Ninja, CMake or MSVC to run the app. Never copy .venv between machines.

Source builders now use a fresh unique output directory, deterministic artifact names, and x64 PE checks. CMake normalizes the Windows DLL prefix and statically links MinGW runtimes. Rebuilding is a developer operation; it does not replace the already verified faculty binaries during this audit.

Existing installation validation now includes a small deterministic DLL/CLI verification operation as well as loading the frozen model. The engine's public feature catalog has 57 names; the deployed frozen classifier uses 30 selected features and 60 trees. This does not change feature extraction or predictions. An unusable DLL or mismatched operation fails validation and prevents successful setup.

The packager retains an explicit file allowlist and now enforces safety with exceptions even under python -O. It rejects malformed/unsafe Windows paths, case-insensitive collisions, symlinks/junctions, incomplete manifests, missing required assets and corrupted content. Existing ZIPs cannot be overwritten. The public allowlist remains separate from the faculty distribution: private recordings and local research artifacts are not included.

The complete faculty archive and this public source repository have different distribution scopes. The archive includes provisioned runtime assets and private submission materials; it is not uploaded here. Follow the repository README and docs/MODELS.md for authorized checkpoint provisioning and tools. Packaging requires every explicitly listed asset and does not download missing files. Cloning the repository alone does not reproduce the complete faculty archive.

Local Windows validation passed; independent second-PC testing remains NOT TESTED. See docs/VALIDATION.md for test scope and limitations.
