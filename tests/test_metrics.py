from __future__ import annotations

import numpy as np
import pytest

from sillage.eval.bootstrap import bootstrap_ci
from sillage.eval.metrics import bestseller_share, coverage, per_user_metrics, top_k


def test_top_k_respects_allowed_and_order():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    allowed = np.array([[False, True, True, True]])   # item 0 already seen
    assert top_k(scores, allowed, 2).tolist() == [[1, 3]]


def test_recall_and_ndcg_by_hand():
    topk = np.array([[2, 0, 1]])
    target = np.array([[True, False, False, True]])   # positives: items 0 and 3
    m = per_user_metrics(topk, target, k=3)
    assert m["recall"][0] == pytest.approx(0.5)
    # hit at rank 2: DCG = 1/log2(3); ideal = 1 + 1/log2(3)
    assert m["ndcg"][0] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_perfect_ranking_scores_one():
    m = per_user_metrics(np.array([[3, 0]]), np.array([[True, False, False, True]]), k=2)
    assert m["recall"][0] == 1.0 and m["ndcg"][0] == pytest.approx(1.0)


def test_coverage_and_bestseller_share():
    topk = np.array([[0, 1], [0, 2]])
    assert coverage(topk, 10) == pytest.approx(0.3)
    pop = np.arange(10, 0, -1).astype(float)          # item 0 most popular
    assert bestseller_share(topk, pop, np.ones(10, bool)) == pytest.approx(0.5)


def test_paired_bootstrap_difference():
    base = np.array([0.0, 0.0, 1.0, 1.0])
    better = base + 0.5
    ci = bootstrap_ci({"popularity": base, "m": better}, "popularity", 200, seed=0)
    assert ci["m"]["diff_vs_baseline"] == pytest.approx(0.5)
    assert ci["m"]["diff_ci"] == pytest.approx([0.5, 0.5])   # paired: difference is constant
    assert ci["popularity"]["diff_vs_baseline"] == 0
