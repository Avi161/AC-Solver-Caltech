"""Unit tests for value_search.feature_extraction.compute_features."""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from value_search.feature_extraction import compute_features


def _make_presentation(r1, r2, max_relator_length):
    """Pad two relator lists into a flat int8 presentation array."""
    pres = np.zeros(2 * max_relator_length, dtype=np.int8)
    pres[: len(r1)] = r1
    pres[max_relator_length : max_relator_length + len(r2)] = r2
    return pres


class TestComputeFeaturesBasics:
    def test_output_shape_and_dtype(self):
        pres = _make_presentation([1, 2], [-1, -2], 5)
        feats = compute_features(pres, max_relator_length=5)
        assert feats.shape == (14,)
        assert feats.dtype == np.float32

    def test_trivial_presentation(self):
        # r1=[1], r2=[2] is the trivial presentation <a,b | a, b>
        pres = _make_presentation([1], [2], 5)
        feats = compute_features(pres, max_relator_length=5)
        # total_length, len_r1, len_r2
        assert feats[0] == 2
        assert feats[1] == 1
        assert feats[2] == 1
        # x count in r1 = 1
        assert feats[3] == 1
        # y count in r2 = 1
        assert feats[9] == 1
        # exponent_sum_x = 1 (just one 'a')
        assert feats[11] == 1
        # length_ratio = 0.5
        assert feats[12] == pytest.approx(0.5)
        # max/min ratio = 1.0
        assert feats[13] == pytest.approx(1.0)

    def test_empty_presentation_zero_lengths(self):
        pres = np.zeros(10, dtype=np.int8)
        feats = compute_features(pres, max_relator_length=5)
        assert feats[0] == 0  # total length
        assert feats[1] == 0  # len_r1
        assert feats[2] == 0  # len_r2
        # length_ratio falls back to 0.5 when total_length == 0
        assert feats[12] == pytest.approx(0.5)
        # max/min ratio: min=0, max=0 → ratio = max_len = 0
        assert feats[13] == 0

    def test_character_counts(self):
        # r1 = [1, 1, -1, 2, -2, -2], r2 = [1, -1]
        pres = _make_presentation([1, 1, -1, 2, -2, -2], [1, -1], 8)
        feats = compute_features(pres, max_relator_length=8)
        assert feats[3] == 2   # count_x_r1: two +1s
        assert feats[4] == 1   # count_xinv_r1: one -1
        assert feats[5] == 1   # count_y_r1: one +2
        assert feats[6] == 2   # count_yinv_r1: two -2s
        assert feats[7] == 1   # count_x_r2: one +1
        assert feats[8] == 1   # count_xinv_r2: one -1
        assert feats[9] == 0   # count_y_r2
        assert feats[10] == 0  # count_yinv_r2
        # exponent_sum_x = (2 - 1) + (1 - 1) = 1
        assert feats[11] == 1

    def test_imbalanced_lengths_ratio(self):
        # r1 has length 6, r2 has length 2
        pres = _make_presentation([1, 1, 1, 1, 1, 1], [2, 2], 8)
        feats = compute_features(pres, max_relator_length=8)
        assert feats[1] == 6
        assert feats[2] == 2
        # length_ratio = 6/8 = 0.75
        assert feats[12] == pytest.approx(6 / 8)
        # max/min = 6/2 = 3
        assert feats[13] == pytest.approx(3.0)

    def test_one_empty_relator_no_zero_division(self):
        # r2 is empty (length 0), min_len = 0 → ratio should equal max_len
        pres = _make_presentation([1, 2, 1], [], 5)
        feats = compute_features(pres, max_relator_length=5)
        assert feats[1] == 3
        assert feats[2] == 0
        assert feats[13] == pytest.approx(3.0)

    def test_accepts_int8_and_list_inputs(self):
        as_list = [1, 2, 0, 0, 0, -1, -2, 0, 0, 0]
        as_array = np.array(as_list, dtype=np.int8)
        f1 = compute_features(as_list, max_relator_length=5)
        f2 = compute_features(as_array, max_relator_length=5)
        np.testing.assert_array_equal(f1, f2)
