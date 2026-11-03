import os
from ..config import ROOT


def library_path():
    return ROOT / "third_party/sbdv2/bin" / ("sbdv.dll" if os.name == "nt" else "libsbdv.so")


def binary_path():
    return ROOT / "third_party/sbdv2/bin" / ("sbdv.exe" if os.name == "nt" else "sbdv")
