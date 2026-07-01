"""Tests for Sample Ratio Mismatch detection."""

import pytest

from insightflow.core import detect_srm


def test_healthy_split_is_not_flagged():
    # 5000 vs 4950 on a 50/50 design is well within normal sampling noise.
    res = detect_srm({"control": 5000, "treatment": 4950})
    assert not res.mismatch_detected
    assert res.p_value > 0.001


def test_broken_split_is_flagged():
    # 6000 vs 4000 on a 50/50 design is a blatant mismatch.
    res = detect_srm({"control": 6000, "treatment": 4000})
    assert res.mismatch_detected
    assert res.p_value < 0.001


def test_expected_ratio_as_weights():
    # Designed 70/30 split; observed counts match it -> no SRM.
    res = detect_srm(
        {"control": 3000, "treatment": 7000}, expected_ratio=[0.3, 0.7]
    )
    assert not res.mismatch_detected


def test_expected_ratio_mapping():
    res = detect_srm(
        {"control": 3000, "treatment": 7000},
        expected_ratio={"control": 3, "treatment": 7},  # unnormalized weights ok
    )
    assert not res.mismatch_detected


def test_wrong_expected_ratio_triggers_srm():
    # Observed is 30/70 but we *expected* 50/50 -> that's a mismatch.
    res = detect_srm({"control": 3000, "treatment": 7000})
    assert res.mismatch_detected


def test_multi_arm_srm():
    # Three-arm equal split, one arm starved.
    res = detect_srm({"a": 3300, "b": 3300, "c": 3400})
    assert not res.mismatch_detected
    res_bad = detect_srm({"a": 5000, "b": 3000, "c": 2000})
    assert res_bad.mismatch_detected


def test_requires_two_arms():
    with pytest.raises(ValueError):
        detect_srm({"control": 5000})


def test_zero_total_raises():
    with pytest.raises(ValueError):
        detect_srm({"control": 0, "treatment": 0})


def test_summary_message_is_actionable():
    res = detect_srm({"control": 6000, "treatment": 4000})
    assert "SRM DETECTED" in res.summary()
