"""
Wright's Law cost model for COSMIC roadmap optimization.

Implements per-capability utilization growth and cost reduction using Wright's Law,
plus NRE (Non-Recurring Engineering) cost calculation based on readiness levels.
"""

import numpy as np
from typing import Tuple

READINESS_MAX: float = 13.0  # fully-developed readiness level


def calculate_independent_mission_utilization(
    dependency_array: np.ndarray,
    utilization_array: np.ndarray,
    learning_rate_array: np.ndarray,
    cap_idx: int,
) -> np.ndarray:
    """Calculate independent utilization growth over time for each mission.

    Each mission that uses a capability contributes a stream of utilization that
    compounds forward (via the mission's learning rate) at every subsequent time step.

    Parameters:
        dependency_array: (num_missions, num_capabilities) – dependency levels [0..1].
        utilization_array: (num_missions, num_capabilities) – base utilization per mission.
        learning_rate_array: (num_missions, num_capabilities) – learning rates as percentages,
            used as the per-step compound growth rate for utilization.
        cap_idx: capability column index.

    Returns:
        np.ndarray: (num_missions, num_missions) matrix.
            Element [i, j] is the utilization contributed by mission i at time step j.
    """
    num_missions = dependency_array.shape[0]
    independent_utilization = np.zeros((num_missions, num_missions))

    for i in range(num_missions):
        dep = dependency_array[i, cap_idx]
        base_util = utilization_array[i, cap_idx]
        growth_rate = learning_rate_array[i, cap_idx] / 100.0

        if dep > 0:
            independent_utilization[i, i] = base_util
            for t in range(i + 1, num_missions):
                prior = independent_utilization[i, t - 1]
                independent_utilization[i, t] = prior * (1.0 + growth_rate)

    return independent_utilization


def calculate_cost_impact(
    independent_util_matrix: np.ndarray,
    learning_rate_array: np.ndarray,
    cap_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Wright's Law cost reduction and total cost for each mission over time.

    Parameters:
        independent_util_matrix: (num_missions, num_missions) output of
            calculate_independent_mission_utilization.
        learning_rate_array: (num_missions, num_capabilities) – learning rates as percentages.
            A value of 90 means costs drop to 90% with each doubling of cumulative volume.
        cap_idx: capability column index.

    Returns:
        Tuple of two (num_missions, num_missions) arrays:
            - cost_reduction_matrix: per-unit cost multiplier (< 1 means cheaper).
            - total_cost_matrix: mission utilization × cost multiplier at each time step.
    """
    num_missions = independent_util_matrix.shape[0]
    total_cumulative = np.sum(independent_util_matrix, axis=0)

    cost_reduction_matrix = np.zeros((num_missions, num_missions))
    total_cost_matrix = np.zeros((num_missions, num_missions))

    for i in range(num_missions):
        lr = learning_rate_array[i, cap_idx] / 100.0
        wrights_exponent = np.log2(lr) if lr > 0 else 0.0

        for t in range(num_missions):
            if independent_util_matrix[i, t] > 0:
                v_total = total_cumulative[t]
                cost_multiplier = v_total ** wrights_exponent if v_total >= 1.0 else 1.0
                cost_reduction_matrix[i, t] = cost_multiplier
                total_cost_matrix[i, t] = independent_util_matrix[i, t] * cost_multiplier

    return cost_reduction_matrix, total_cost_matrix


def calculate_nre_and_readiness(
    readiness_array: np.ndarray,
    dependency_array: np.ndarray,
    cap_idx: int,
    readiness_max: float = READINESS_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate NRE cost and readiness progression for a single capability.

    NRE cost is incurred at the first mission that depends on the capability and equals
    the gap between its current maximum readiness and readiness_max.  After that point
    the capability is considered fully developed and no further NRE is charged.

    Parameters:
        readiness_array: (num_missions, num_capabilities) – readiness levels [0..13].
        dependency_array: (num_missions, num_capabilities) – dependency levels [0..1].
        cap_idx: capability column index.
        readiness_max: target readiness level when a capability is fully developed.

    Returns:
        Tuple of two 1-D arrays of length num_missions:
            - nre_cost_array: NRE cost incurred at each step (non-zero only at first use).
            - readiness_over_time_array: capability readiness level at each step.
    """
    num_missions = readiness_array.shape[0]
    nre_cost = np.zeros(num_missions)
    readiness_over_time = np.zeros(num_missions)

    current_max_readiness = 0.0
    is_developed = False

    for i in range(num_missions):
        dep = dependency_array[i, cap_idx]
        mission_readiness = readiness_array[i, cap_idx]

        if not is_developed:
            current_max_readiness = max(current_max_readiness, mission_readiness)
            if dep > 0:
                nre_cost[i] = max(0.0, readiness_max - current_max_readiness)
                is_developed = True
                readiness_over_time[i] = readiness_max
            else:
                readiness_over_time[i] = current_max_readiness
        else:
            readiness_over_time[i] = readiness_max

    return nre_cost, readiness_over_time


def roadmap_cost(
    order,
    readiness_array: np.ndarray,
    dependency_array: np.ndarray,
    learning_rate_array: np.ndarray,
    utilization_array: np.ndarray,
) -> float:
    """Calculate the total cost (NRE + operational) for a given mission ordering.

    Parameters:
        order: sequence of integer indices specifying the mission execution order.
        readiness_array: (num_missions, num_capabilities).
        dependency_array: (num_missions, num_capabilities).
        learning_rate_array: (num_missions, num_capabilities) – percentages.
        utilization_array: (num_missions, num_capabilities).

    Returns:
        float: total cost in normalised units (billions).
    """
    from me_roadmap.optimization.utils import reorder_multiple_arrays

    r_ready, r_dep, r_lr, r_util = reorder_multiple_arrays(
        order, readiness_array, dependency_array, learning_rate_array, utilization_array
    )

    num_capabilities = dependency_array.shape[1]
    nre_total = 0.0
    op_total = 0.0

    for cap_idx in range(num_capabilities):
        these_nre, _ = calculate_nre_and_readiness(r_ready, r_dep, cap_idx)
        nre_total += float(np.sum(these_nre))

        mission_vectors = calculate_independent_mission_utilization(r_dep, r_util, r_lr, cap_idx)
        _, this_total_cost = calculate_cost_impact(mission_vectors, r_lr, cap_idx)
        op_total += float(np.sum(this_total_cost))

    return (nre_total * 1e9 + op_total) / 1e9
