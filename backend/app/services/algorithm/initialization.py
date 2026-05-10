"""
Khởi tạo quần thể cho Thuật toán Memetic (MA).

Triển khai hai chiến lược từ Botelho et al. (2010) và Labadie et al. (2012):
  • Chiến lược 1 – Heuristic chèn ngẫu nhiên (80% quần thể, 80 cá thể)
  • Chiến lược 2 – Khởi tạo ngẫu nhiên thuần túy (20% quần thể, 20 cá thể)

Tổng kích thước quần thể: 100 (cố định).
"""

import random
from typing import List

from app.models.domain import POI, Individual
from app.models.requests import UserPreferences
from app.core.config import POPULATION_SIZE, HEURISTIC_COUNT, RANDOM_COUNT, RCL_SIZE
from app.services.algorithm.fitness import (
    get_travel_time,
    try_add_poi,
)


# =============================================================================
#  Chiến lược 1: Heuristic chèn ngẫu nhiên (Tỷ lệ ưu tiên Labadie)
# =============================================================================

def _labadie_ratio(
    poi: POI,
    current_location: POI,
    user_prefs: UserPreferences,
) -> float:
    """
    Tỷ lệ ưu tiên Labadie (baseline).

    Công thức GỐC (Labadie 2012):
        ratio = (POI.score × interest_weight) / distance(current, POI)

    Công thức:
        ratio = (POI.score × interest_weight) / distance(current, POI)

    Parameters
    ----------
    poi : POI
        Điểm đang xét.
    current_location : POI
        Vị trí hiện tại.
    user_prefs : UserPreferences
        Sở thích người dùng.
    """
    interest_weight = user_prefs.interest_weights.get(poi.category, 0.0)
    numerator = poi.base_score * interest_weight

    dist = get_travel_time(current_location, poi)
    if dist == 0:
        return float('inf')

    return numerator / dist


def _create_heuristic_individual(
    pois: List[POI],
    depot: POI,
    user_prefs: UserPreferences,
) -> Individual:
    """
    Xây dựng MỘT cá thể bằng Heuristic chèn ngẫu nhiên:
      1. Bắt đầu với lộ trình = [Depot].
      2. Duy trì một tập hợp các POI chưa ghé (tất cả các POI không phải depot).
      3. Lặp lại:
         a. Lọc các POI chưa ghé → chỉ giữ lại những POI vượt qua `try_add_poi`.
         b. Tính tỷ lệ Labadie cho mỗi ứng viên hợp lệ.
         c. Sắp xếp giảm dần → xây dựng RCL từ Top-k.
         d. Chọn ngẫu nhiên một POI từ RCL → thêm vào lộ trình.
         e. Cập nhật current_time (travel + wait + service).
      4. Khi không còn POI hợp lệ nào có thể thêm vào, thêm Depot và trả về.
    """
    route: List[POI] = [depot]
    unvisited = {p.id for p in pois if p.id != depot.id}
    poi_map = {p.id: p for p in pois}

    while unvisited:
        current = route[-1]

        # --- Filter: only POIs that can be feasibly inserted ---
        candidates = []
        for pid in list(unvisited):
            poi = poi_map[pid]
            if try_add_poi(route, poi, user_prefs):
                ratio = _labadie_ratio(poi, current, user_prefs)
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

    # Close route at Depot
    route.append(depot)
    return Individual(route=route)


# =============================================================================
#  Chiến lược 2: Khởi tạo ngẫu nhiên thuần túy
# =============================================================================

def _create_random_individual(
    pois: List[POI],
    depot: POI,
    user_prefs: UserPreferences,
) -> Individual:
    """
    Xây dựng MỘT cá thể bằng cách chèn ngẫu nhiên thuần túy:
      1. Bắt đầu với lộ trình = [Depot].
      2. Xáo trộn tất cả các POI không phải depot một cách ngẫu nhiên.
      3. Lặp lại: nếu việc thêm POI thỏa mãn các ràng buộc, hãy thêm nó.
      4. Khi hoàn tất, thêm Depot và trả về.
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
#  PUBLIC API: Tạo quần thể ban đầu đầy đủ
# =============================================================================

def initialize_population(
    pois: List[POI],
    user_prefs: UserPreferences,
    use_heuristic_init: bool = True,
) -> List[Individual]:
    """
    Tạo quần thể ban đầu gồm 100 cá thể.

    Chế độ mặc định (use_heuristic_init=True):
      • 80 thông qua Heuristic chèn ngẫu nhiên (chất lượng cao + đa dạng)
      • 20 thông qua ngẫu nhiên thuần túy (khám phá / đa dạng)

    Chế độ ablation (use_heuristic_init=False):
      • 100 thông qua ngẫu nhiên thuần túy (để đánh giá đóng góp của khởi tạo heuristic)

    Mọi lộ trình được đảm bảo:
      ✓ Bắt đầu và kết thúc tại Depot (POI id == 0)
      ✓ Vượt qua check_constraints trước khi bất kỳ POI nào được thêm vào

    Parameters
    ----------
    pois : list[POI]
        Tất cả các POI hiện có (bao gồm cả depot ở chỉ số 0).
    user_prefs : UserPreferences
        Ràng buộc người dùng (ngân sách, khung thời gian, sở thích).
    use_heuristic_init : bool
        True → 80% Heuristic + 20% Random (mặc định).
        False → 100% Random (ablation study).
    Returns
    -------
    list[Individual]
        Quần thể có kích thước 100.
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
        ind = _create_heuristic_individual(pois, depot, user_prefs)
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
