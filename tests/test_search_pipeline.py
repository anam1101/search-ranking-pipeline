"""Tests for the ranking metrics and data loading."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_pipeline import (
    DCGEvaluator,
    DataLoader,
    Document,
    EvaluationMetrics,
    FeatureRanker,
    BM25Ranker,
)


def test_dcg_matches_hand_calculation():
    # rel=[3,2] -> (2^3-1)/log2(2) + (2^2-1)/log2(3) = 7 + 3/1.58496
    expected = 7.0 + 3.0 / math.log2(3)
    assert DCGEvaluator.dcg_at_k([3, 2]) == pytest.approx(expected)


def test_ndcg_is_one_for_ideal_ordering():
    assert DCGEvaluator.ndcg_at_k([3, 2, 1, 0]) == pytest.approx(1.0)


def test_ndcg_penalises_bad_ordering():
    assert DCGEvaluator.ndcg_at_k([0, 1, 2, 3]) < DCGEvaluator.ndcg_at_k([3, 2, 1, 0])


def test_ndcg_all_zero_relevance_is_zero():
    assert DCGEvaluator.ndcg_at_k([0, 0, 0]) == 0.0


def test_precision_at_k():
    assert EvaluationMetrics.precision_at_k([1, 0, 1, 0], 4) == 0.5
    assert EvaluationMetrics.precision_at_k([1, 1, 0, 0], 2) == 1.0
    assert EvaluationMetrics.precision_at_k([1, 1], 0) == 0.0


def test_recall_at_k():
    assert EvaluationMetrics.recall_at_k([1, 0, 1], 3, 4) == 0.5
    assert EvaluationMetrics.recall_at_k([1, 0], 2, 0) == 0.0


def test_average_precision_hand_calculation():
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert EvaluationMetrics.average_precision([1, 0, 1]) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2
    )


def test_average_precision_no_relevant_documents():
    assert EvaluationMetrics.average_precision([0, 0, 0]) == 0.0
    assert EvaluationMetrics.average_precision([]) == 0.0


def test_parse_line_extracts_relevance_qid_and_features():
    doc = DataLoader.parse_line("2 qid:46 1:0.5 11:120.0")
    assert doc.relevance == 2
    assert doc.qid == 46
    assert doc.get_feature(1) == 0.5
    assert doc.get_feature(999) == 0.0  # missing feature defaults to zero


def test_get_documents_by_qid_filters():
    docs = [Document(1, 1, {}), Document(0, 2, {}), Document(2, 1, {})]
    assert len(DataLoader.get_documents_by_qid(docs, 1)) == 2


def test_ideal_ranking_sorts_by_relevance():
    docs = [Document(0, 1, {}), Document(2, 1, {}), Document(1, 1, {})]
    ranked = DCGEvaluator.create_ideal_ranking(docs)
    assert [d.relevance for d in ranked] == [2, 1, 0]


def test_feature_ranker_orders_by_feature_value():
    docs = [Document(0, 1, {75: 0.1}), Document(0, 1, {75: 0.9})]
    ranked = FeatureRanker(75).rank_documents(docs)
    assert ranked[0].get_feature(75) == 0.9


def test_bm25_ranks_higher_term_frequency_first():
    low = Document(0, 1, {1: 1.0, 11: 100.0})
    high = Document(0, 1, {1: 20.0, 11: 100.0})
    ranked = BM25Ranker().rank_documents([low, high])
    assert ranked[0] is high
