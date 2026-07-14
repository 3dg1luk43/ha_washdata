"""Phase B: provenance code derivation + trim edit marker."""
from custom_components.ha_washdata.const import QC_EDITED, QC_MANUAL, QC_RECORDING
from custom_components.ha_washdata.store import derive_qc, online_features_enabled


def test_derive_qc_recording():
    assert derive_qc({"meta": {"source": "recorder"}}) == QC_RECORDING
    assert derive_qc({"meta": {"original_samples": 500}}) == QC_RECORDING


def test_derive_qc_edited():
    assert derive_qc({"meta": {"edited": True}}) == QC_EDITED


def test_derive_qc_manual_default():
    assert derive_qc({}) == QC_MANUAL
    assert derive_qc({"meta": {}}) == QC_MANUAL
    assert derive_qc({"ml_review": {"golden": True}}) == QC_MANUAL


def test_recorder_precedence_over_edited():
    # A trimmed recording is still classed as a recording.
    assert derive_qc({"meta": {"source": "recorder", "edited": True}}) == QC_RECORDING


def test_online_features_gating_default_off():
    assert online_features_enabled(None) is False
    assert online_features_enabled({}) is False
    assert online_features_enabled({"enable_online_features": True}) is True
