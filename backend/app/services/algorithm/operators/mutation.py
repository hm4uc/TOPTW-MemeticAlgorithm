"""
Mutation Operators — 2-opt, Swap, Insertion Mutation (Depot-Safe).

Nguyên tắc "Depot-Safe":
  Toán tử CHỈ thao tác trên "interior" = route[1:-1].
  Depot được gắn lại sau khi xử lý xong.

Chiến lược mặc định (use_insertion_mutation=True):
  • 2-opt     (30%) : đảo ngược đoạn con → giảm quãng đường.
  • Swap      (30%) : hoán đổi 2 POI → thay đổi thứ tự.
  • Insertion (40%) : tìm POI mới chưa đi, chèn vào vị trí tốt nhất.

Chiến lược ablation (use_insertion_mutation=False):
  • 2-opt     (50%)
  • Swap      (50%)
"""

import random
from typing import List

from app.models.domain import POI, Individual
from app.models.requests import UserPreferences
from app.services.algorithm.fitness import get_travel_time, check_constraints


def mutate(
    individual: Individual,
    depot: POI,
    pois: List[POI],
    user_prefs: UserPreferences,
    mutation_rate: float,
    use_insertion_mutation: bool = True,
    operator_probs: dict[str, float] | None = None,
    collect_stats: bool = False,
) -> Individual | tuple[Individual, str, bool]:
    """
    Áp dụng đột biến ngẫu nhiên cho một cá thể (Depot-Safe).

    Parameters
    ----------
    individual : Individual
        Cá thể cần đột biến.
    depot : POI
        Điểm depot để gắn lại sau khi đột biến.
    pois : list[POI]
        Toàn bộ danh sách POI (dùng cho Insertion Mutation).
    user_prefs : UserPreferences
        Ràng buộc người dùng (dùng cho Insertion Mutation).
    mutation_rate : float
        Xác suất áp dụng đột biến (0.0 → 1.0).
    use_insertion_mutation : bool
        True  → 2-opt(30%) / Swap(30%) / Insertion(40%).
        False → chỉ 2-opt(50%) + Swap(50%)  [ablation mode].
    operator_probs : dict, optional
        Xác suất toán tử động, dạng {"2opt": x, "swap": y, "insertion": z}.
        Nếu None -> dùng tỷ lệ mặc định như cũ.
    collect_stats : bool
        True -> trả thêm (op_name, op_success) cho engine theo dõi telemetry.

    Returns
    -------
    Individual
        Cá thể sau đột biến.
    """
    if random.random() > mutation_rate:
        return (individual, "skip", False) if collect_stats else individual

    interior = list(individual.route[1:-1])

    probs = _resolve_operator_probs(operator_probs, use_insertion_mutation)

    # Nếu interior quá ngắn, chỉ thử Insertion nếu được bật
    if len(interior) < 2:
        if use_insertion_mutation:
            individual, success = _insertion_mutation(individual, depot, pois, user_prefs)
            return (individual, "insertion", success) if collect_stats else individual
        return (individual, "skip", False) if collect_stats else individual

    roll = random.random()

    if roll < probs["2opt"]:
        i, j = sorted(random.sample(range(len(interior)), 2))
        interior[i:j + 1] = interior[i:j + 1][::-1]
        individual.route = [depot] + interior + [depot]
        return (individual, "2opt", True) if collect_stats else individual

    if roll < probs["2opt"] + probs["swap"]:
        i, j = random.sample(range(len(interior)), 2)
        interior[i], interior[j] = interior[j], interior[i]
        individual.route = [depot] + interior + [depot]
        return (individual, "swap", True) if collect_stats else individual

    individual, success = _insertion_mutation(individual, depot, pois, user_prefs)
    return (individual, "insertion", success) if collect_stats else individual


def _resolve_operator_probs(
    operator_probs: dict[str, float] | None,
    use_insertion_mutation: bool,
) -> dict[str, float]:
    """Chuẩn hóa xác suất toán tử để tổng = 1.0."""
    if not use_insertion_mutation:
        return {"2opt": 0.5, "swap": 0.5, "insertion": 0.0}

    if operator_probs is None:
        return {"2opt": 0.3, "swap": 0.3, "insertion": 0.4}

    p2 = max(0.0, float(operator_probs.get("2opt", 0.3)))
    ps = max(0.0, float(operator_probs.get("swap", 0.3)))
    pi = max(0.0, float(operator_probs.get("insertion", 0.4)))

    total = p2 + ps + pi
    if total <= 1e-12:
        return {"2opt": 0.3, "swap": 0.3, "insertion": 0.4}

    return {
        "2opt": p2 / total,
        "swap": ps / total,
        "insertion": pi / total,
    }


def _insertion_mutation(
    individual: Individual,
    depot: POI,
    pois: List[POI],
    user_prefs: UserPreferences,
) -> tuple[Individual, bool]:
    """
    ★ INSERTION MUTATION — Toán tử cốt lõi để phá Hội tụ sớm ★

    Tìm POI chưa ghé thăm, chèn vào vị trí tốn ít thời gian nhất.
    Duyệt TOÀN BỘ POI chưa thăm để không bỏ sót ứng viên tốt.

    Ưu tiên POI theo score × interest_weight (cao → thử trước).
    Có xáo trộn ngẫu nhiên trước khi sort để giữ đa dạng quần thể.
    """
    route = list(individual.route)
    visited_ids = {p.id for p in route}
    unvisited = [p for p in pois if p.id not in visited_ids and p.id != 0]

    if not unvisited:
        return individual, False

    # Sắp xếp theo score cá nhân hóa, xen kẽ random để giữ đa dạng
    weights = user_prefs.interest_weights
    random.shuffle(unvisited)
    unvisited.sort(
        key=lambda p: p.base_score * weights.get(p.category, 0.0),
        reverse=True,
    )

    inserted_any = False

    for candidate in unvisited:
        insert_options: list[tuple[int, float]] = []

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
            insert_options.append((pos, cost_increase))

        insert_options.sort(key=lambda x: x[1])

        for pos, _ in insert_options:
            test_route = list(route)
            test_route.insert(pos, candidate)
            if check_constraints(test_route, user_prefs):
                route = test_route
                inserted_any = True
                break

    individual.route = route
    return individual, inserted_any
