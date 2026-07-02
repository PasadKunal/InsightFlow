"""Tests for the simulation-based validation harness.

These are 'meta-tests': they confirm that when we run many simulated experiments, the
engine's *empirical* error rates match its theoretical promises. A fixed seed keeps
them deterministic; the tolerances absorb Monte-Carlo noise.
"""

from validation.simulation_harness import (
    validate_continuous,
    validate_engine,
    validate_proportion,
)


def test_proportion_type_i_error_controlled():
    type_i, _ = validate_proportion(n_experiments=300, seed=0)
    assert type_i.scenario == "Type I error"
    assert type_i.empirical <= type_i.target + type_i.tolerance
    assert type_i.passed


def test_proportion_power_meets_target():
    _, power = validate_proportion(n_experiments=300, seed=0)
    assert power.empirical >= power.target - power.tolerance
    assert power.passed


def test_continuous_type_i_and_power():
    type_i, power = validate_continuous(n_experiments=300, seed=1)
    assert type_i.passed
    assert power.passed


def test_full_report_passes():
    report = validate_engine(n_experiments=300, seed=0)
    assert len(report.results) == 4
    assert report.all_passed
    assert "ALL CHECKS PASSED" in report.summary()
