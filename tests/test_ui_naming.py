"""Presentation choices retain the release's model IDs and callback values."""
import importlib.util
from pathlib import Path
import pytest
from sbd_app.config import ROOT, MODEL_PATHS, KNOWN_HASHES, DETECTOR_NAMES, sha256


@pytest.fixture(scope='module')
def ui():
    spec = importlib.util.spec_from_file_location('naming_ui_under_test', ROOT/'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('detector,filename,hash_prefix', [
    ('tinycut', 'clean_tinycut_s_member.pkl', 'b3f52ff1'),
    ('autoshot', 'ckpt_0_200_0.pth', '3e852905'),
    ('omnishotcut', 'OmniShotCut_ckpt.pth', '5948ea78'),
    ('omnishotcut_legacy_v15', 'OmniShotCut_ckpt.pth', '5bafe33d'),
])
def test_display_selection_keeps_verified_checkpoint(ui, detector, filename, hash_prefix):
    assert detector in ui.UI_DETECTOR_LABELS
    assert (ROOT/MODEL_PATHS[detector]).name == filename
    assert sha256(ROOT/MODEL_PATHS[detector]) == KNOWN_HASHES[detector]
    assert KNOWN_HASHES[detector].startswith(hash_prefix)
    options = ui.mode_options(detector)
    assert [value for label, value in options] == ui.mode_choices(detector)
    update, model_text, config_text = ui.detector_selection(detector)
    assert update.value == 'Raw'
    assert model_text == ui.MODEL_DESCRIPTIONS[detector]
    assert config_text == ui.selection_description(detector, 'Raw')


def test_original_mode_values_and_support_remain(ui):
    assert ui.mode_choices('tinycut') == ['Raw', 'Raw + SBDV2 (C++)']
    assert ui.mode_choices('autoshot')[-1] == 'Raw + Legacy Python Verifier'
    assert ui.mode_choices('omnishotcut')[-1] == 'Raw + Legacy Python (Rescue + Snap)'
    assert ui.mode_choices('omnishotcut_legacy_v15') == ui.mode_choices('omnishotcut')
    assert sum(len(ui.mode_options(key)) for key in ui.UI_DETECTOR_LABELS) == 11
    # The UI must not mutate the backend dictionary used by existing callers.
    assert DETECTOR_NAMES['omnishotcut'] == 'OmniShotCut (provided checkpoint)'
    assert DETECTOR_NAMES['omnishotcut_legacy_v15'] == 'OmniShotCut Legacy v1.5'


def test_configuration_explains_historical_and_cross_version_runs(ui):
    mode = 'Raw + Legacy Python (Rescue + Snap)'
    assert 'Original Review 1 experimental configuration.' in ui.selection_description('omnishotcut_legacy_v15', mode)
    assert 'Cross-version experiment.' in ui.selection_description('omnishotcut', mode)
    assert 'not identical' in ui.selection_description('omnishotcut', 'Raw + SBDV2 (C++)')
    assert '18,433' in ui.MODEL_DESCRIPTIONS['tinycut']
    assert 'did not train' in ui.MODEL_DESCRIPTIONS['omnishotcut_legacy_v15']


def test_actual_gradio_choice_labels_keep_stable_values(ui):
    demo = ui.create_app()
    dropdowns = [row['props'] for row in demo.get_config_file()['components'] if row['type'] == 'dropdown']
    detector, verification = dropdowns
    assert detector['choices'] == [(label, key) for key, label in ui.UI_DETECTOR_LABELS.items()]
    assert detector['value'] == 'tinycut'
    assert verification['choices'] == ui.mode_options('tinycut')
    assert verification['value'] == 'Raw'
