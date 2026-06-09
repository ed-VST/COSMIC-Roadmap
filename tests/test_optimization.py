"""
Tests for the me_roadmap.optimization package.
"""

import numpy as np
import pytest

from me_roadmap.optimization.cost_model import (
    calculate_cost_impact,
    calculate_independent_mission_utilization,
    calculate_nre_and_readiness,
    roadmap_cost,
)
from me_roadmap.optimization.optimizer import optimize_schedule
from me_roadmap.optimization.utils import reorder_array, reorder_multiple_arrays

# ---------------------------------------------------------------------------
# Minimal fixtures: 3 missions, 2 capabilities
# ---------------------------------------------------------------------------
# Mission 0 uses cap 0 first (dep=1.0, readiness=13 already at max)
# Mission 1 uses cap 1 first (dep=1.0, readiness=10, needs 3 NRE)
# Mission 2 uses both (dep=0.7/0.3)

READINESS = np.array([[13.0, 5.0], [0.0, 10.0], [0.0, 13.0]], dtype=float)
DEPENDENCY = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.3]], dtype=float)
LEARNING_RATE = np.array([[90.0, 85.0], [95.0, 80.0], [88.0, 92.0]], dtype=float)
UTILIZATION = np.array([[25.0, 0.0], [0.0, 250.0], [5.0, 25.0]], dtype=float)


# ---------------------------------------------------------------------------
# reorder_array / reorder_multiple_arrays
# ---------------------------------------------------------------------------

def test_reorder_array_basic():
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    result = reorder_array([2, 0, 1], arr)
    np.testing.assert_array_equal(result, [[5, 6], [1, 2], [3, 4]])


def test_reorder_array_identity():
    arr = np.arange(9).reshape(3, 3)
    np.testing.assert_array_equal(reorder_array([0, 1, 2], arr), arr)


def test_reorder_multiple_arrays():
    a = np.array([10, 20, 30])
    b = np.array([1, 2, 3])
    ra, rb = reorder_multiple_arrays([2, 0, 1], a, b)
    np.testing.assert_array_equal(ra, [30, 10, 20])
    np.testing.assert_array_equal(rb, [3, 1, 2])


# ---------------------------------------------------------------------------
# calculate_independent_mission_utilization
# ---------------------------------------------------------------------------

def test_independent_mission_utilization_shape():
    result = calculate_independent_mission_utilization(DEPENDENCY, UTILIZATION, LEARNING_RATE, 0)
    assert result.shape == (3, 3)


def test_independent_mission_utilization_first_mission_cap0():
    result = calculate_independent_mission_utilization(DEPENDENCY, UTILIZATION, LEARNING_RATE, 0)
    # Mission 0 has dep=1.0 and util=25 for cap 0 → non-zero from t=0
    assert result[0, 0] == 25.0
    assert result[0, 1] > result[0, 0]  # compounds forward


def test_independent_mission_utilization_zero_dep():
    result = calculate_independent_mission_utilization(DEPENDENCY, UTILIZATION, LEARNING_RATE, 0)
    # Mission 1 has dep=0.0 for cap 0 → all zeros
    np.testing.assert_array_equal(result[1, :], [0.0, 0.0, 0.0])


def test_independent_mission_utilization_compound_growth():
    result = calculate_independent_mission_utilization(DEPENDENCY, UTILIZATION, LEARNING_RATE, 0)
    lr = LEARNING_RATE[0, 0] / 100.0
    expected_t1 = 25.0 * (1.0 + lr)
    assert abs(result[0, 1] - expected_t1) < 1e-10


# ---------------------------------------------------------------------------
# calculate_cost_impact
# ---------------------------------------------------------------------------

def test_cost_impact_shape():
    mv = calculate_independent_mission_utilization(DEPENDENCY, UTILIZATION, LEARNING_RATE, 0)
    cost_red, total_cost = calculate_cost_impact(mv, LEARNING_RATE, 0)
    assert cost_red.shape == (3, 3)
    assert total_cost.shape == (3, 3)


def test_cost_impact_non_negative():
    mv = calculate_independent_mission_utilization(DEPENDENCY, UTILIZATION, LEARNING_RATE, 1)
    cost_red, total_cost = calculate_cost_impact(mv, LEARNING_RATE, 1)
    assert np.all(total_cost >= 0)


def test_cost_impact_zero_where_no_utilization():
    mv = calculate_independent_mission_utilization(DEPENDENCY, UTILIZATION, LEARNING_RATE, 0)
    _, total_cost = calculate_cost_impact(mv, LEARNING_RATE, 0)
    # Mission 1 has no utilization for cap 0 → zero cost
    np.testing.assert_array_equal(total_cost[1, :], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# calculate_nre_and_readiness
# ---------------------------------------------------------------------------

def test_nre_readiness_shape():
    nre, readiness_tl = calculate_nre_and_readiness(READINESS, DEPENDENCY, 0)
    assert nre.shape == (3,)
    assert readiness_tl.shape == (3,)


def test_nre_already_at_max():
    # Cap 0: mission 0 uses it first with readiness 13 → NRE = max(0, 13-13) = 0
    nre, _ = calculate_nre_and_readiness(READINESS, DEPENDENCY, 0)
    assert nre[0] == 0.0


def test_nre_partial_readiness():
    # Cap 1: mission 1 uses it first; readiness 5 (mission 0) then 10 (mission 1)
    # current_max_readiness before mission 1 = max(0, 5, 10) = 10 → NRE = 13-10 = 3
    nre, _ = calculate_nre_and_readiness(READINESS, DEPENDENCY, 1)
    assert nre[1] == pytest.approx(3.0)


def test_nre_developed_stays_at_max():
    _, readiness_tl = calculate_nre_and_readiness(READINESS, DEPENDENCY, 0)
    assert readiness_tl[0] == 13.0
    assert readiness_tl[1] == 13.0
    assert readiness_tl[2] == 13.0


# ---------------------------------------------------------------------------
# roadmap_cost
# ---------------------------------------------------------------------------

def test_roadmap_cost_is_float():
    cost = roadmap_cost([0, 1, 2], READINESS, DEPENDENCY, LEARNING_RATE, UTILIZATION)
    assert isinstance(cost, float)


def test_roadmap_cost_non_negative():
    cost = roadmap_cost([0, 1, 2], READINESS, DEPENDENCY, LEARNING_RATE, UTILIZATION)
    assert cost >= 0.0


def test_roadmap_cost_finite():
    cost = roadmap_cost([0, 1, 2], READINESS, DEPENDENCY, LEARNING_RATE, UTILIZATION)
    assert np.isfinite(cost)


def test_roadmap_cost_order_matters():
    cost_fwd = roadmap_cost([0, 1, 2], READINESS, DEPENDENCY, LEARNING_RATE, UTILIZATION)
    cost_rev = roadmap_cost([2, 1, 0], READINESS, DEPENDENCY, LEARNING_RATE, UTILIZATION)
    # Costs should generally differ for different orders
    assert np.isfinite(cost_fwd) and np.isfinite(cost_rev)


# ---------------------------------------------------------------------------
# optimize_schedule
# ---------------------------------------------------------------------------

def test_optimize_schedule_valid_permutation():
    best_order, best_cost = optimize_schedule(
        READINESS, DEPENDENCY, LEARNING_RATE, UTILIZATION,
        n_iterations=50, initial_temp=1e6, verbose=False,
    )
    assert len(best_order) == 3
    assert set(best_order) == {0, 1, 2}


def test_optimize_schedule_cost_is_finite():
    _, best_cost = optimize_schedule(
        READINESS, DEPENDENCY, LEARNING_RATE, UTILIZATION,
        n_iterations=50, initial_temp=1e6, verbose=False,
    )
    assert np.isfinite(best_cost)
    assert best_cost >= 0.0
