"""Build native engine; --compiler accepts a 64-bit clang++/g++ executable."""
from pathlib import Path
import argparse
import json
import platform
import shutil
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]


def build(compiler=None):
    src = ROOT / "third_party/sbdv2"
    dest = src / "bin"
    dest.mkdir(exist_ok=True)
    compiler = compiler or os.environ.get("SBD_CXX")
    if compiler:
        version = subprocess.check_output([compiler, "--version"], text=True)
        if os.name == "nt":
            commands = [
                [compiler, "-std=c++17", "-O2", "-static", "-shared", "-I" + str(src / "include"), str(src / "src/sbdv_capi.cpp"), "-o", str(dest / "sbdv.dll")],
                [compiler, "-std=c++17", "-O2", "-static", str(src / "src/sbdv_cli_windows.cpp"), "-lshell32", "-o", str(dest / "sbdv.exe")],
            ]
        else:
            commands = [[compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-I" + str(src / "include"), str(src / "src/sbdv_capi.cpp"), "-o", str(dest / "libsbdv.so")],
                        [compiler, "-std=c++17", "-O2", str(src / "src/sbdv_cli.cpp"), "-o", str(dest / "sbdv")]]
        for cmd in commands:
            subprocess.run(cmd, check=True)
    else:
        cmake = shutil.which("cmake")
        if not cmake:
            raise RuntimeError("Set SBD_CXX to a 64-bit LLVM-MinGW clang++.exe, or install CMake and MSVC Build Tools.")
        builddir = ROOT / "build/verifier"
        subprocess.run([cmake, "-S", str(src), "-B", str(builddir), "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run([cmake, "--build", str(builddir), "--config", "Release"], check=True)
        for name in (["sbdv.dll", "sbdv.exe"] if os.name == "nt" else ["libsbdv.so", "sbdv"]):
            found = next(builddir.rglob(name))
            shutil.copy2(found, dest / name)
        version = subprocess.check_output([cmake, "--version"], text=True)
    version = '\n'.join(line for line in version.splitlines() if not line.startswith(('InstalledDir:', 'Configuration file:')))
    report = dict(platform=platform.platform(), architecture=platform.machine(), compiler=version,
                  core="Original V40; UTF-8 model path patch only", windows_cli="raw RGB RAM and stream transport")
    (dest / "BUILD_INFO.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler")
    build(parser.parse_args().compiler)
