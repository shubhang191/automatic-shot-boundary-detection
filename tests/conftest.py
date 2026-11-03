from pathlib import Path
import pytest
import torch
from sbd_app.config import ROOT

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def sample():
    path = ROOT / "examples/inputs/sintel excerpt.mp4"
    assert path.is_file(), "Missing permitted real-footage sample"
    return path


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    from sbd_app.pipeline import Pipeline
    return Pipeline(tmp_path_factory.mktemp("pipeline cache"))
