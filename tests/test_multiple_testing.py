"""Tests for multiple-testing correction, cross-checked against statsmodels."""

import pytest
from statsmodels.stats.multitest import multipletests

from insightflow.core import benjamini_hochberg, bonferroni, correct


def test_bonferroni_basic():
    # p=0.01 across 5 tests -> adjusted 0.05, which is NOT < 0.05.
    res = bonferroni([0.01, 0.2, 0.3, 0.4, 0.5], alpha=0.05)
    assert res.adjusted_p_values[0] == pytest.approx(0.05)
    assert not res.rejected[0]
    # A smaller p clears the stricter bar.
    res2 = bonferroni([0.005, 0.2, 0.3, 0.4, 0.5], alpha=0.05)
    assert res2.rejected[0]


def test_bonferroni_matches_statsmodels():
    p = [0.001, 0.008, 0.02, 0.04, 0.3, 0.7]
    ours = bonferroni(p, alpha=0.05)
    reject_ref, p_adj_ref, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
    assert list(ours.rejected) == list(reject_ref)
    assert list(ours.adjusted_p_values) == pytest.approx(list(p_adj_ref))


def test_bh_matches_statsmodels():
    p = [0.001, 0.008, 0.02, 0.04, 0.3, 0.7]
    ours = benjamini_hochberg(p, alpha=0.05)
    reject_ref, p_adj_ref, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert list(ours.rejected) == list(reject_ref)
    assert list(ours.adjusted_p_values) == pytest.approx(list(p_adj_ref))


def test_bh_is_more_powerful_than_bonferroni():
    # BH should reject at least as many hypotheses as Bonferroni.
    p = [0.001, 0.01, 0.02, 0.03, 0.04, 0.6, 0.7]
    bh = benjamini_hochberg(p, alpha=0.05)
    bon = bonferroni(p, alpha=0.05)
    assert bh.n_significant >= bon.n_significant


def test_labels_are_tracked():
    res = benjamini_hochberg(
        [0.001, 0.5, 0.02],
        labels=["revenue", "bounce_rate", "signups"],
    )
    sig = res.significant()
    assert "revenue" in sig
    assert "bounce_rate" not in sig


def test_correct_dispatch():
    p = [0.01, 0.02, 0.5]
    assert correct(p, method="bonferroni").method == "bonferroni"
    assert correct(p, method="benjamini-hochberg").method == "benjamini-hochberg"
    with pytest.raises(ValueError):
        correct(p, method="holm")  # unsupported


def test_invalid_pvalues_raise():
    with pytest.raises(ValueError):
        bonferroni([])
    with pytest.raises(ValueError):
        bonferroni([0.1, 1.5])  # out of range
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 0.2], labels=["only_one"])
