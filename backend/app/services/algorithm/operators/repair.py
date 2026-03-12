"""
Repair Operators — Smart Repair và Greedy Refill.

Smart Repair:
  Sửa chữa cá thể vi phạm ràng buộc bằng cách lần lượt xóa POI
  có tỷ lệ Score/TimeCost thấp nhất cho đến khi route khả thi.

Greedy Refill:
  Sau khi Repair cắt bớt POI, lấp đầy route bằng cách chèn thêm
  POI chưa ghé theo thứ tự score × interest_weight cao nhất,
  tại vị trí tốn ít thời gian nhất, miễn là vẫn thỏa constraints.

  ★ CẢI TIẾN: Khi use_urgency=True, thứ tự ưu tiên POI có tính đến
    thời gian còn lại trước khi đóng cửa (urgency factor).

Cả hai hàm hỗ trợ ablation flag để bật/tắt riêng từng cơ chế.
"""

from typing import List

from app.models.domain import POI, Individual
from app.models.requests import UserPreferences
from app.services.algorithm.fitness import get_travel_time, check_constraints
from app.core.config import URGENCY_ALPHA, URGENCY_CAP


def repair(
    individual: Individual,
    user_prefs: UserPreferences,
    use_smart_repair: bool = True,
) -> Individual:
    """
    Sửa chữa individual vi phạm ràng buộc bằng cách xóa POI xấu.

    Chế độ mặc định (use_smart_repair=True):
      ★ SMART REPAIR — xóa POI có tỷ lệ Score/TimeCost kém nhất.
      Tỷ lệ = (base_score × interest_weight) / time_cost_of_removal.

    Chế độ ablation (use_smart_repair=False):
      Simple Repair — luôn xóa POI áp chót (route[-2]).

    Parameters
    ----------
    individual : Individual
        Cá thể cần sửa chữa.
    user_prefs : UserPreferences
        Ràng buộc người dùng để kiểm tra tính khả thi.
    use_smart_repair : bool
        True → Smart Repair (mặc định).
        False → Simple Repair (ablation).

    Returns
    -------
    Individual
        Cá thể đã được sửa chữa (route luôn khả thi hoặc chỉ còn [Depot, Depot]).
    """
    route = individual.route
    weights = user_prefs.interest_weights

    while not check_constraints(route, user_prefs) and len(route) > 2:
        if use_smart_repair:
            # ── Smart Repair: tìm POI kém nhất (Score/TimeCost thấp nhất) ──
            worst_idx = -1
            worst_value = float('inf')

            for i in range(1, len(route) - 1):
                poi = route[i]
                score_value = poi.base_score * weights.get(poi.category, 0.0)

                prev_poi = route[i - 1]
                next_poi = route[i + 1]
                time_cost = (
                    get_travel_time(prev_poi, poi)
                    + poi.duration
                    + get_travel_time(poi, next_poi)
                    - get_travel_time(prev_poi, next_poi)
                )

                ratio = score_value / time_cost if time_cost > 0 else float('inf')

                if ratio < worst_value:
                    worst_value = ratio
                    worst_idx = i

            if worst_idx > 0:
                route.pop(worst_idx)
            else:
                route.pop(-2)
        else:
            # ── Simple Repair: luôn xóa POI áp chót ─────────────────────
            route.pop(-2)

    individual.route = route
    return individual


def greedy_refill(
    individual: Individual,
    pois: List[POI],
    user_prefs: UserPreferences,
    use_urgency: bool = True,
) -> Individual:
    """
    ★ GREEDY REFILL — Chèn thêm POI vào route sau khi Repair xóa bớt ★

    Sau khi Repair cắt POI vi phạm, route thường rất ngắn (1-2 POI).
    Bước này tìm POI chưa ghé, thử chèn vào vị trí tốt nhất (tốn ít
    thời gian nhất), miễn là vẫn thỏa constraints.

    ★ CẢI TIẾN (use_urgency=True):
      Thay vì sort chỉ theo score × weight, hàm tính urgency_score
      dựa trên thời gian còn lại trước khi đóng cửa (close_time).
      POI sắp đóng cửa sớm → urgency_score cao hơn → được chèn trước.
      Điều này tránh mất POI có time window hẹp.

    Parameters
    ----------
    individual : Individual
        Cá thể cần lấp đầy.
    pois : list[POI]
        Toàn bộ danh sách POI có thể thêm vào.
    user_prefs : UserPreferences
        Ràng buộc người dùng để kiểm tra tính khả thi.
    use_urgency : bool
        True → urgency-aware sorting (mặc định).
        False → chỉ sort theo score × weight (ablation).

    Returns
    -------
    Individual
        Cá thể sau khi đã chèn thêm POI.
    """
    route = list(individual.route)
    visited_ids = {p.id for p in route}
    unvisited = [p for p in pois if p.id not in visited_ids and p.id != 0]

    if not unvisited:
        individual.route = route
        return individual

    # Sắp xếp theo score cá nhân hóa (có hoặc không có urgency)
    weights = user_prefs.interest_weights

    if use_urgency:
        # ── Urgency-aware sorting ────────────────────────────────────────
        # Ước lượng current_time bằng cách simulate route hiện tại
        current_time = user_prefs.start_time_minutes
        for i in range(len(route) - 1):
            travel = get_travel_time(route[i], route[i + 1])
            arrival = current_time + travel
            if arrival < route[i + 1].open_time:
                arrival = route[i + 1].open_time
            current_time = arrival + route[i + 1].duration

        # Tính urgency_score cho mỗi POI chưa ghé
        def urgency_score(p: POI) -> float:
            base = p.base_score * weights.get(p.category, 0.0)
            time_remaining = p.close_time - user_prefs.start_time_minutes
            if time_remaining <= 0:
                return 0.0
            urgency_factor = min(1.0 + URGENCY_ALPHA / max(time_remaining, 1.0),
                                 URGENCY_CAP)
            return base * urgency_factor

        unvisited.sort(key=urgency_score, reverse=True)
    else:
        # ── Sorting gốc (ablation) ──────────────────────────────────────
        unvisited.sort(
            key=lambda p: p.base_score * weights.get(p.category, 0.0),
            reverse=True,
        )

    for candidate in unvisited:
        best_pos = -1
        best_cost_increase = float('inf')

        for pos in range(1, len(route)):
            prev_poi = route[pos - 1]
            next_poi = route[pos]

            old_travel = get_travel_time(prev_poi, next_poi)
            new_travel = (
                get_travel_time(prev_poi, candidate)
                + candidate.duration
                + get_travel_time(candidate, next_poi)
            )
            cost_increase = new_travel - old_travel

            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_pos = pos

        if best_pos > 0:
            test_route = list(route)
            test_route.insert(best_pos, candidate)
            if check_constraints(test_route, user_prefs):
                route = test_route

    individual.route = route
    return individual
