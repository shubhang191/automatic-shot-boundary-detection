"""Build native engine; --compiler accepts a 64-bit clang++/g++ executable."""
from pathlib import Path
import argparse
import json
import platform
import shutil
import subprocess
import sys
import os
import tempfile
import struct

ROOT = Path(__file__).resolve().parents[1]


def select_artifact(directory, names):
    matches = [Path(directory)/name for name in names if (Path(directory)/name).is_file()]
    if len(matches) != 1:
        listing = ', '.join(sorted(p.name for p in Path(directory).iterdir()))
        raise RuntimeError(f"Expected exactly one of {names} in fresh build output {directory}; found {listing or '(empty)'}. Check CMake generator/target output logs. Import libraries (.dll.a) are not DLLs.")
    return matches[0]


def validate_windows_binary(path):
    with Path(path).open('rb') as f:
        if f.read(2) != b'MZ': raise RuntimeError(f'Not a Windows executable/DLL: {path}')
        f.seek(0x3c); offset=struct.unpack('<I',f.read(4))[0];f.seek(offset)
        if f.read(4)!=b'PE\0\0' or struct.unpack('<H',f.read(2))[0]!=0x8664:
            raise RuntimeError(f'Expected Windows x64 PE binary: {path}')


def build(compiler=None, output_dir=None, build_root=None, generator=None):
    src = ROOT / "third_party/sbdv2"
    dest = Path(output_dir).resolve() if output_dir else src / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    parent = Path(build_root).resolve() if build_root else ROOT / 'build/verifier'
    parent.mkdir(parents=True, exist_ok=True)
    builddir = Path(tempfile.mkdtemp(prefix='build-', dir=parent))
    fresh = builddir / 'out'
    fresh.mkdir()
    compiler = compiler or os.environ.get("SBD_CXX")
    if compiler:
        version = subprocess.check_output([compiler, "--version"], text=True)
        if os.name == "nt":
            commands = [
                [compiler, "-std=c++17", "-O2", "-static", "-shared", "-I" + str(src / "include"), str(src / "src/sbdv_capi.cpp"), "-o", str(fresh / "sbdv.dll")],
                [compiler, "-std=c++17", "-O2", "-static", str(src / "src/sbdv_cli_windows.cpp"), "-lshell32", "-o", str(fresh / "sbdv.exe")],
            ]
        else:
            commands = [[compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-I" + str(src / "include"), str(src / "src/sbdv_capi.cpp"), "-o", str(fresh / "libsbdv.so")],
                        [compiler, "-std=c++17", "-O2", str(src / "src/sbdv_cli.cpp"), "-o", str(fresh / "sbdv")]]
        for cmd in commands:
            subprocess.run(cmd, check=True)
    else:
        cmake = shutil.which("cmake")
        if not cmake:
            raise RuntimeError("Set SBD_CXX to a 64-bit LLVM-MinGW clang++.exe, or install CMake and MSVC Build Tools.")
        configure = [cmake, "-S", str(src), "-B", str(builddir), "-DCMAKE_BUILD_TYPE=Release"]
        if generator: configure += ['-G', generator]
        for kind in ['RUNTIME', 'LIBRARY']:
            configure += [f'-DCMAKE_{kind}_OUTPUT_DIRECTORY={fresh}', f'-DCMAKE_{kind}_OUTPUT_DIRECTORY_RELEASE={fresh}']
        subprocess.run(configure, check=True)
        subprocess.run([cmake, "--build", str(builddir), "--config", "Release"], check=True)
        version = subprocess.check_output([cmake, "--version"], text=True)
    artifacts = {'sbdv.dll': ('sbdv.dll', 'libsbdv.dll'), 'sbdv.exe': ('sbdv.exe',)} if os.name == 'nt' else {'libsbdv.so': ('libsbdv.so',), 'sbdv': ('sbdv',)}
    selected = {name: select_artifact(fresh, choices) for name, choices in artifacts.items()}
    if os.name == 'nt':
        for path in selected.values(): validate_windows_binary(path)
    for name, path in selected.items(): shutil.copy2(path, dest/name)
    version = '\n'.join(line for line in version.splitlines() if not line.startswith(('InstalledDir:', 'Configuration file:')))
    report = dict(platform=platform.platform(), architecture=platform.machine(), compiler=version,
                  core="Original V40; UTF-8 model path patch only", windows_cli="raw RGB RAM and stream transport")
    (dest / "BUILD_INFO.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler")
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--build-root', type=Path)
    parser.add_argument('--generator')
    args=parser.parse_args()
    build(args.compiler, args.output_dir, args.build_root, args.generator)
