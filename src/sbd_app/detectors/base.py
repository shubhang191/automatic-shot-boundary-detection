from dataclasses import dataclass, field
import importlib.util
import sys
import numpy as np
from ..errors import ModelError


@dataclass
class Detection:
    cuts: list
    probabilities: np.ndarray | None
    native_ranges: list
    details: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_frames(frames):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (27, 48, 3) or len(frames) < 1:
        raise ModelError("Expected nonempty RGB uint8 [T,27,48,3] frames")
