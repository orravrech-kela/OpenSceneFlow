"""Unit tests for the DDP rank-merge math in eval_metric.

`BucketResultMatrix.all_reduce_()` reconciles per-rank accumulated metrics into
identical global state before validation logging. Each cell of the matrices
holds a *count-weighted average* EPE/range plus a count; merging across ranks
must reconstruct the weighted sums, add them, and re-divide by the total count.

These tests stub out `torch.distributed` so the math runs in a single process
with no real process group and no GPU. `all_reduce` is faked as a 2-rank SUM by
adding a precomputed rank-B contribution in the call order the implementation
uses (epe_weighted, range_weighted, count). The expected result is computed by
an independent numpy oracle of the intended formula, so the test does not just
re-derive the implementation.

Run:
    .venv/bin/python -m unittest tests.test_eval_metric_all_reduce
    .venv/bin/python tests/test_eval_metric_all_reduce.py
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.eval_metric import BucketResultMatrix, BucketedSpeedMatrix  # noqa: E402


# --- helpers ---------------------------------------------------------------

CLASSES = ["Static", "Dynamic"]
RANGES = [(0.0, 35.0), (35.0, np.inf)]


def _make_matrix(epe, rng, cnt, cls=BucketResultMatrix):
    """Build a matrix with its storage set directly to the given arrays."""
    # Positional: BucketResultMatrix takes (class_names, range_buckets) and
    # BucketedSpeedMatrix takes (class_names, speed_buckets) -- same positions.
    m = cls(CLASSES, RANGES)
    m.epe_storage_matrix = np.asarray(epe, dtype=np.float64)
    m.range_storage_matrix = np.asarray(rng, dtype=np.float64)
    m.count_storage_matrix = np.asarray(cnt, dtype=np.int64)
    return m


def _weighted_addends(epe, rng, cnt):
    """Rank-B's [epe_weighted, range_weighted, count] tensors, as the impl forms
    them: NaN cells contribute zero weight (nan_to_num -> 0) and the count is
    carried as float for the SUM all-reduce."""
    epe = np.nan_to_num(np.asarray(epe, dtype=np.float64), nan=0.0)
    rng = np.nan_to_num(np.asarray(rng, dtype=np.float64), nan=0.0)
    cnt = np.asarray(cnt, dtype=np.float64)
    return [
        torch.as_tensor(epe * cnt),
        torch.as_tensor(rng * cnt),
        torch.as_tensor(cnt),
    ]


def _fake_all_reduce_factory(addends):
    """Return an `all_reduce` stub that adds the next rank-B addend in place.

    The implementation calls all_reduce three times, in order: epe_weighted,
    range_weighted, count. We consume `addends` in that same order. If that call
    order ever changes, the merge would pair the wrong addend with the wrong
    quantity and these tests fail loudly -- which is the intended guard."""
    it = iter(addends)

    def fake_all_reduce(tensor, op=None):
        addend = next(it).to(device=tensor.device, dtype=tensor.dtype)
        tensor.add_(addend)
        return None  # real API returns a work handle; the impl ignores it

    return fake_all_reduce


def _expected_merge(epe_a, rng_a, cnt_a, epe_b, rng_b, cnt_b):
    """Independent numpy oracle of the intended cross-rank weighted merge."""
    epe_a0 = np.nan_to_num(np.asarray(epe_a, dtype=np.float64), nan=0.0)
    epe_b0 = np.nan_to_num(np.asarray(epe_b, dtype=np.float64), nan=0.0)
    rng_a0 = np.nan_to_num(np.asarray(rng_a, dtype=np.float64), nan=0.0)
    rng_b0 = np.nan_to_num(np.asarray(rng_b, dtype=np.float64), nan=0.0)
    ca = np.asarray(cnt_a, dtype=np.float64)
    cb = np.asarray(cnt_b, dtype=np.float64)

    tot = ca + cb
    epe_w = epe_a0 * ca + epe_b0 * cb
    rng_w = rng_a0 * ca + rng_b0 * cb
    with np.errstate(invalid="ignore", divide="ignore"):
        exp_epe = np.where(tot > 0, epe_w / np.clip(tot, 1.0, None), np.nan)
        exp_rng = np.where(tot > 0, rng_w / np.clip(tot, 1.0, None), np.nan)
    return exp_epe, exp_rng, tot.astype(np.int64)


def _run_all_reduce(matrix, rankB_addends, world_size=2):
    """Invoke matrix.all_reduce_() under a stubbed 2-rank process group."""
    with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
         mock.patch.object(torch.distributed, "get_world_size", return_value=world_size), \
         mock.patch.object(torch.distributed, "all_reduce",
                           side_effect=_fake_all_reduce_factory(rankB_addends)):
        matrix.all_reduce_()


# --- tests -----------------------------------------------------------------

class TestRankMergeMath(unittest.TestCase):

    def test_weighted_average_across_two_ranks(self):
        # Cell [0,0]: both ranks saw it with different epe and unequal counts ->
        #   the merged epe must be count-weighted, not a plain mean.
        # Cell [0,1]: only rank A saw it (rank B NaN / 0 count) -> keep A.
        # Cell [1,0]: only rank B saw it (rank A NaN / 0 count) -> take B.
        # Cell [1,1]: neither rank saw it -> stays NaN, count 0.
        epe_a = [[2.0, 4.0], [np.nan, np.nan]]
        rng_a = [[10.0, 20.0], [np.nan, np.nan]]
        cnt_a = [[3, 5], [0, 0]]

        epe_b = [[8.0, np.nan], [6.0, np.nan]]
        rng_b = [[40.0, np.nan], [30.0, np.nan]]
        cnt_b = [[9, 0], [7, 0]]

        m = _make_matrix(epe_a, rng_a, cnt_a)
        _run_all_reduce(m, _weighted_addends(epe_b, rng_b, cnt_b))

        exp_epe, exp_rng, exp_cnt = _expected_merge(
            epe_a, rng_a, cnt_a, epe_b, rng_b, cnt_b)

        np.testing.assert_allclose(m.epe_storage_matrix, exp_epe, equal_nan=True)
        np.testing.assert_allclose(m.range_storage_matrix, exp_rng, equal_nan=True)
        np.testing.assert_array_equal(m.count_storage_matrix, exp_cnt)

        # Spot-check the headline cell by hand: (2*3 + 8*9) / (3 + 9) = 6.5,
        # whereas an unweighted mean would be (2 + 8) / 2 = 5.0.
        self.assertAlmostEqual(m.epe_storage_matrix[0, 0], 6.5)
        self.assertNotAlmostEqual(m.epe_storage_matrix[0, 0], 5.0)
        # Single-rank cells survive unchanged.
        self.assertAlmostEqual(m.epe_storage_matrix[0, 1], 4.0)
        self.assertAlmostEqual(m.epe_storage_matrix[1, 0], 6.0)
        # Never-seen cell is NaN with zero count.
        self.assertTrue(np.isnan(m.epe_storage_matrix[1, 1]))
        self.assertEqual(m.count_storage_matrix[1, 1], 0)

    def test_count_matrix_is_summed_and_int64(self):
        epe_a = [[1.0, 1.0], [1.0, 1.0]]
        rng_a = [[1.0, 1.0], [1.0, 1.0]]
        cnt_a = [[1, 2], [3, 4]]
        epe_b = [[1.0, 1.0], [1.0, 1.0]]
        rng_b = [[1.0, 1.0], [1.0, 1.0]]
        cnt_b = [[10, 20], [30, 40]]

        m = _make_matrix(epe_a, rng_a, cnt_a)
        _run_all_reduce(m, _weighted_addends(epe_b, rng_b, cnt_b))

        np.testing.assert_array_equal(
            m.count_storage_matrix, np.array([[11, 22], [33, 44]]))
        self.assertEqual(m.count_storage_matrix.dtype, np.int64)

    def test_identical_ranks_preserve_values(self):
        # Merging a rank with an identical copy of itself must be a no-op on the
        # averages (weighted mean of x and x is x) while doubling the counts.
        epe = [[3.0, 7.0], [np.nan, 9.0]]
        rng = [[12.0, 14.0], [np.nan, 18.0]]
        cnt = [[4, 6], [0, 2]]

        m = _make_matrix(epe, rng, cnt)
        _run_all_reduce(m, _weighted_addends(epe, rng, cnt))

        np.testing.assert_allclose(
            m.epe_storage_matrix, np.asarray(epe, dtype=np.float64), equal_nan=True)
        np.testing.assert_allclose(
            m.range_storage_matrix, np.asarray(rng, dtype=np.float64), equal_nan=True)
        np.testing.assert_array_equal(
            m.count_storage_matrix, np.asarray(cnt, dtype=np.int64) * 2)

    def test_subclass_inherits_all_reduce(self):
        # OfficialMetrics.bucketedMatrix is a BucketedSpeedMatrix; the trainer
        # calls all_reduce_() on it, so the inherited method must apply.
        self.assertTrue(hasattr(BucketedSpeedMatrix, "all_reduce_"))
        epe = [[2.0, 4.0], [6.0, 8.0]]
        rng = [[1.0, 2.0], [3.0, 4.0]]
        cnt = [[1, 1], [1, 1]]
        m = _make_matrix(epe, rng, cnt, cls=BucketedSpeedMatrix)
        _run_all_reduce(m, _weighted_addends(epe, rng, cnt))
        np.testing.assert_allclose(
            m.epe_storage_matrix, np.asarray(epe, dtype=np.float64))
        np.testing.assert_array_equal(
            m.count_storage_matrix, np.asarray(cnt, dtype=np.int64) * 2)


class TestRankMergeNoOps(unittest.TestCase):
    """The guard clauses: single-process or uninitialized -> leave state alone."""

    def _assert_unchanged(self, **dist_patch):
        epe = [[2.0, 4.0], [np.nan, 8.0]]
        rng = [[10.0, 20.0], [np.nan, 40.0]]
        cnt = [[3, 5], [0, 7]]
        m = _make_matrix(epe, rng, cnt)
        # all_reduce is patched to blow up if it is ever called on these paths.
        with mock.patch.object(torch.distributed, "all_reduce",
                               side_effect=AssertionError("all_reduce called on no-op path")):
            for name, val in dist_patch.items():
                ctx = mock.patch.object(torch.distributed, name, return_value=val)
                ctx.start()
                self.addCleanup(ctx.stop)
            m.all_reduce_()
        np.testing.assert_allclose(
            m.epe_storage_matrix, np.asarray(epe, dtype=np.float64), equal_nan=True)
        np.testing.assert_array_equal(
            m.count_storage_matrix, np.asarray(cnt, dtype=np.int64))

    def test_not_initialized_is_noop(self):
        self._assert_unchanged(is_initialized=False)

    def test_single_rank_is_noop(self):
        self._assert_unchanged(is_initialized=True, get_world_size=1)


if __name__ == "__main__":
    unittest.main()
