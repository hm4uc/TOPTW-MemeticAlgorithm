"""
Population Initialization for the Hybrid Genetic Algorithm (HGA).

Implements two strategies from Botelho et al. (2010) and Labadie et al. (2012):
  • Strategy 1 – Randomized Insertion Heuristic  (80% of population, 80 individuals)
  • Strategy 2 – Pure Random Initialization        (20% of population, 20 individuals)

Total population size: 100 (fixed).
"""

import random
from typing import List

from app.models.domain import POI, Individual
from app.models.requests import UserPreferences
from app.core.config import POPULATION_SIZE, HEURISTIC_COUNT, RANDOM_COUNT, RCL_SIZE, URGENCY_ALPHA, URGENCY_CAP
from app.services.algorithm.fitness import (
    get_travel_time,
    try_add_poi,
)


# =============================================================================
#  Strategy 1: Randomized Insertion Heuristic  (Labadie desirability ratio)
# =============================================================================

def _labadie_ratio(poi: POI, current_location: POI,
                   user_prefs: UserPreferences,
                   current_time: float = 0.0,
                   use_urgency: bool = True) -> float:
    """
    Labadie desirability ratio — CẢI TIẾN với Time-Window Urgency.

    Công thức GỐC (Labadie 2012):
        ratio = (POI.score × interest_weight) / distance(current, POI)

    Công thức CẢI TIẾN (use_urgency=True):
        travel  = distance(current, POI)
        arrival = max(current_time + travel, POI.open_time)
        time_remaining = POI.close_time - arrival
        urgency_factor = min(1 + α / max(time_remaining, 1.0), CAP)
        ratio = (POI.score × interest_weight × urgency_factor) / travel

    Ý nghĩa:
      • POI sắp đóng cửa (time_remaining nhỏ) → urgency_factor cao → ưu tiên
      • POI mở cửa cả ngày (time_remaining lớn) → urgency_factor ≈ 1.0 → bình thường
      • urgency_factor bị cap ở URGENCY_CAP để tránh POI score thấp nhưng sắp đóng
        được ưu tiên quá mức so với POI score cao.

    Parameters
    ----------
    poi : POI
        Điểm đang xét.
    current_location : POI
        Vị trí hiện tại.
    user_prefs : UserPreferences
        Sở thích người dùng.
    current_time : float
        Thời điểm hiện tại (phút / Solomon time units).
    use_urgency : bool
        True → áp dụng urgency factor (mặc định).
        False → Labadie ratio gốc (ablation).
    """
    interest_weight = user_prefs.interest_weights.get(poi.category, 0.0)
    numerator = poi.base_score * interest_weight

    dist = get_travel_time(current_location, poi)
    if dist == 0:
        return float('inf')

    if use_urgency:
        # Tính thời điểm đến nơi (tính cả thời gian chờ nếu đến sớm)
        arrival = current_time + dist
        if arrival < poi.open_time:
            arrival = poi.open_time

        # Thời gian còn lại trước khi POI đóng cửa
        time_remaining = poi.close_time - arrival

        if time_remaining <= 0:
            # Đã quá giờ đóng cửa → ratio = 0 (không khả thi)
            return 0.0

        # Urgency factor: càng gấp → càng cao, nhưng bị cap
        urgency_factor = min(1.0 + URGENCY_ALPHA / max(time_remaining, 1.0),
                             URGENCY_CAP)

        return (numerator * urgency_factor) / dist
    else:
        # Labadie ratio gốc (ablation mode)
        return numerator / dist


def _create_heuristic_individual(
    pois: List[POI],
    depot: POI,
    user_prefs: UserPreferences,
    use_urgency: bool = True,
) -> Individual:
    """
    Build ONE individual using the Randomized Insertion Heuristic:
      1. Start with route = [Depot].
      2. Maintain a set of unvisited POIs (all non-depot POIs).
      3. Repeat:
         a. Filter unvisited POIs → keep only those passing `try_add_poi`.
         b. Compute Labadie ratio (with urgency) for each valid candidate.
         c. Sort descending → build RCL from Top-k.
         d. Pick one random POI from the RCL → append to route.
         e. Update current_time (travel + wait + service).
      4. When no more valid POIs can be added, append Depot and return.

    Parameters
    ----------
    use_urgency : bool
        True  → dùng Improved Labadie ratio với urgency factor (mặc định).
        False → dùng Labadie ratio gốc (ablation mode).
    """
    route: List[POI] = [depot]
    unvisited = {p.id for p in pois if p.id != depot.id}
    poi_map = {p.id: p for p in pois}

    # Track thời gian hiện tại để tính urgency chính xác
    current_time = user_prefs.start_time_minutes

    while unvisited:
        current = route[-1]

        # --- Filter: only POIs that can be feasibly inserted ---
        candidates = []
        for pid in list(unvisited):
            poi = poi_map[pid]
            if try_add_poi(route, poi, user_prefs):
                ratio = _labadie_ratio(poi, current, user_prefs,
                                       current_time, use_urgency)
                candidates.append((poi, ratio))

        if not candidates:
            break  # No feasible POI left

        # --- Sort by desirability ratio (descending) ---
        candidates.sort(key=lambda x: x[1], reverse=True)

        # --- Restricted Candidate List (Top-k) ---
        rcl = candidates[:RCL_SIZE]

        # --- Random pick from RCL ---
        chosen_poi, _ = random.choice(rcl)

        route.append(chosen_poi)
        unvisited.discard(chosen_poi.id)

        # --- Cập nhật current_time sau khi thêm POI ---
        travel = get_travel_time(current, chosen_poi)
        arrival = current_time + travel
        if arrival < chosen_poi.open_time:
            arrival = chosen_poi.open_time  # Chờ đến giờ mở
        departure = arrival + chosen_poi.duration
        current_time = departure

    # Close route at Depot
    route.append(depot)
    return Individual(route=route)


# =============================================================================
#  Strategy 2: Pure Random Initialization
# =============================================================================

def _create_random_individual(
    pois: List[POI],
    depot: POI,
    user_prefs: UserPreferences,
) -> Individual:
    """
    Build ONE individual using Pure Random insertion:
      1. Start with route = [Depot].
      2. Shuffle all non-depot POIs randomly.
      3. Iterate: if adding the POI satisfies constraints, append it.
      4. When done, append Depot and return.
    """
    route: List[POI] = [depot]
    candidates = [p for p in pois if p.id != depot.id]
    random.shuffle(candidates)

    for poi in candidates:
        if try_add_poi(route, poi, user_prefs):
            route.append(poi)

    # Close route at Depot
    route.append(depot)
    return Individual(route=route)


# =============================================================================
#  PUBLIC API: Generate Full Initial Population
# =============================================================================

def initialize_population(
    pois: List[POI],
    user_prefs: UserPreferences,
    use_heuristic_init: bool = True,
    use_urgency: bool = True,
) -> List[Individual]:
    """
    Generate the initial population of 100 individuals.

    Chế độ mặc định (use_heuristic_init=True):
      • 80 via Randomized Insertion Heuristic  (high quality + diversity)
      • 20 via Pure Random                     (exploration / diversity)

    Chế độ ablation (use_heuristic_init=False):
      • 100 via Pure Random  (để đánh giá đóng góp của heuristic init)

    Every route is guaranteed to:
      ✓ Start and end at the Depot (POI id == 0)
      ✓ Pass check_constraints before any POI is appended

    Parameters
    ----------
    pois : list[POI]
        All available Points of Interest (including the depot at index 0).
    user_prefs : UserPreferences
        User constraints (budget, time window, interests).
    use_heuristic_init : bool
        True → 80% Heuristic + 20% Random (mặc định).
        False → 100% Random (ablation study).
    use_urgency : bool
        True → Improved Labadie ratio với time-window urgency (mặc định).
        False → Labadie ratio gốc (ablation study).

    Returns
    -------
    list[Individual]
        Population of size 100.
    """
    depot = next((p for p in pois if p.id == 0), None)
    if depot is None:
        raise ValueError("Depot (POI id=0) not found in the POI list.")

    population: List[Individual] = []

    if use_heuristic_init:
        heuristic_count = HEURISTIC_COUNT
        random_count = RANDOM_COUNT
    else:
        heuristic_count = 0
        random_count = POPULATION_SIZE

    # --- Strategy 1: Heuristic individuals ---
    for i in range(heuristic_count):
        ind = _create_heuristic_individual(pois, depot, user_prefs, use_urgency)
        population.append(ind)

    # --- Strategy 2: Random individuals ---
    for i in range(random_count):
        ind = _create_random_individual(pois, depot, user_prefs)
        population.append(ind)

    assert len(population) == POPULATION_SIZE, (
        f"Expected {POPULATION_SIZE} individuals, got {len(population)}"
    )

    # --- Summary log ---
    if use_heuristic_init and heuristic_count > 0:
        heuristic_lens = [len(ind.route) for ind in population[:heuristic_count]]
        random_lens = [len(ind.route) for ind in population[heuristic_count:]]
        print(f"[Init] Population created: {POPULATION_SIZE} individuals")
        print(f"       Heuristic ({heuristic_count}): avg route length = "
              f"{sum(heuristic_lens)/len(heuristic_lens):.1f}")
        print(f"       Random    ({random_count}):  avg route length = "
              f"{sum(random_lens)/len(random_lens):.1f}")
    else:
        all_lens = [len(ind.route) for ind in population]
        print(f"[Init] Population created: {POPULATION_SIZE} individuals (100% Random)")
        print(f"       Avg route length = {sum(all_lens)/len(all_lens):.1f}")

    return population
