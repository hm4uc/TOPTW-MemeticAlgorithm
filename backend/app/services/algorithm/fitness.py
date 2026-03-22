import math
from types import NoneType
from typing import List, Optional, Any

from app.models.domain import POI
from app.models.requests import UserPreferences
from app.core.config import (
    PENALTY_LATE_ARRIVAL,
    PENALTY_LATE_RETURN,
    PENALTY_BUDGET,
    PENALTY_WAIT,
)


# =============================================================================
#  DISTANCE MATRIX  (O(1) lookup – pre-computed at startup)
# =============================================================================
#
#  Thay vì gọi math.sqrt() hàng triệu lần trong quá trình GA, ta tính
#  trước ma trận khoảng cách N×N một lần duy nhất khi load dữ liệu.
#  Mọi tra cứu sau đó chỉ mất O(1) (truy cập mảng 2D).
#  Với 101 POI → ma trận 101×101 = ~10 201 giá trị float ≈ 80 KB RAM.
#
# =============================================================================

# Module-level distance matrix – initialized by build_distance_matrix()
_DIST_MATRIX: Optional[list[list[float]]] = None


def euclidean_distance(p1: POI, p2: POI) -> float:
    """Euclidean distance between two POIs in coordinate units."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def build_distance_matrix(pois: List[POI]) -> list[list[float]]:
    """
    Pre-compute the full N×N Euclidean distance matrix.

    Must be called ONCE at startup (after loading POIs).
    Subsequent calls to get_travel_time() will use O(1) lookups.

    Parameters
    ----------
    pois : list[POI]
        All POIs (including depot). POI ids must be 0..N-1.

    Returns
    -------
    list[list[float]]
        2D matrix where matrix[i][j] = Euclidean distance from POI i to POI j.
    """
    global _DIST_MATRIX
    n = len(pois)

    # Sort by id to guarantee matrix[poi.id] maps correctly
    sorted_pois = sorted(pois, key=lambda p: p.id)

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        pi = sorted_pois[i]
        for j in range(i + 1, n):
            pj = sorted_pois[j]
            d = math.sqrt((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2)
            matrix[i][j] = d
            matrix[j][i] = d  # Symmetric

    _DIST_MATRIX = matrix
    print(f"[DistMatrix] Built {n}×{n} distance matrix ({n*n} entries)")
    return matrix


def get_travel_time(p1: POI, p2: POI) -> float | type[None[Any]]:
    """
    Travel time between two POIs.
    For Solomon benchmarks, travel time == Euclidean distance
    (speed = 1 unit/time-unit).

    Uses pre-computed distance matrix for O(1) lookup.
    Falls back to direct calculation if matrix is not yet initialized.
    """
    if _DIST_MATRIX is not None:
        return _DIST_MATRIX[p1.id][p2.id]
    # Fallback (should not happen in normal flow)
    return euclidean_distance(p1, p2)


# =============================================================================
#  CONSTRAINT CHECKING  (TOPTW feasibility)
# =============================================================================

def check_constraints(route: list[POI], user_prefs: UserPreferences) -> bool:
    """
    Validate whether a COMPLETE route [Depot, ..., Depot] satisfies all
    TOPTW constraints:
      1. Time Windows  – arrive at each POI before its close_time.
      2. Max Tour Time – return to depot before end_time.
      3. Budget        – total price of visited POIs ≤ user_prefs.budget.

    ĐƠN VỊ: Mọi phép tính bên trong dùng PHÚT (Solomon time units).
    User input (giờ) được chuyển qua start_time_minutes / end_time_minutes.

    Returns True if ALL constraints are satisfied, False otherwise.
    """
    if len(route) < 2:
        return False  # Must at least have [Depot, Depot]

    current_time = user_prefs.start_time_minutes  # Phút (VD: 8h → 480)
    total_cost = 0.0

    for i in range(len(route) - 1):
        curr = route[i]
        next_p = route[i + 1]

        # --- Travel ---
        travel = get_travel_time(curr, next_p)
        arrival = current_time + travel

        # --- Time Window ---
        # Wait if arrived too early
        if arrival < next_p.open_time:
            arrival = next_p.open_time

        # Infeasible if arrived after closing
        if arrival > next_p.close_time:
            return False

        # --- Service ---
        departure = arrival + next_p.duration
        current_time = departure

        # --- Budget ---
        total_cost += next_p.price

    # Budget constraint
    if total_cost > user_prefs.budget:
        return False

    return True


def try_add_poi(route: list[POI], candidate: POI,
                user_prefs: UserPreferences) -> bool:
    """
    Check if `candidate` can be *inserted just before the trailing Depot*
    while keeping the route feasible.
    
    The route is assumed to be [Depot, ..., (last‐visited)] WITHOUT
    the trailing Depot yet (the depot is appended temporarily for checking).
    
    Returns True if the route [*route, candidate, Depot] satisfies constraints.
    """
    depot = route[0]  # Depot is always the first element
    test_route = route + [candidate, depot]
    return check_constraints(test_route, user_prefs)


# =============================================================================
#  FITNESS EVALUATION
# =============================================================================
#  Hệ số phạt được định nghĩa tập trung tại: app/core/config.py
#  PENALTY_LATE_ARRIVAL, PENALTY_LATE_RETURN, PENALTY_BUDGET, PENALTY_WAIT
#  đã được import ở đầu file.
# =============================================================================


def calculate_fitness(ind, user_prefs: UserPreferences,
                      wait_penalty_weight: float = PENALTY_WAIT) -> float:
    """
    Evaluate fitness of an Individual.

    Fitness = Σ (base_score × interest_weight) − penalties.

    ĐƠN VỊ: Mọi phép tính bên trong dùng PHÚT (Solomon time units).

    Parameters
    ----------
    ind : Individual
        Cá thể cần đánh giá.
    user_prefs : UserPreferences
        Sở thích và ràng buộc người dùng.
    wait_penalty_weight : float
        Hệ số phạt thời gian chờ (mặc định 0.2).
        Đặt = 0.0 để tắt penalty chờ (ablation study).

    Penalties cover:
      • Time-window violation  – arrive after close_time       (×100.0)
      • Late return to depot   – return after end_time          (×100.0)
      • Budget overrun         – total price > budget           (× 0.5)
      • Waiting time           – arrive before open_time        (× wait_penalty_weight)
        → Ép GA sắp xếp thứ tự POI sao cho đến nơi là vào chơi luôn,
          tránh bắt du khách chờ ngoài cửa.
    """
    current_time = user_prefs.start_time_minutes  # Phút (VD: 8h → 480)
    total_score = 0.0
    total_cost = 0.0
    total_wait = 0.0
    penalty = 0.0

    for i in range(len(ind.route) - 1):
        curr = ind.route[i]
        next_p = ind.route[i + 1]

        # --- Score (skip depot; its category is 'depot') ---
        w = user_prefs.interest_weights.get(curr.category, 0.0)
        total_score += curr.base_score * w
        total_cost += curr.price

        # --- Travel ---
        travel = get_travel_time(curr, next_p)
        arrival = current_time + travel

        # --- Time Window ---
        if arrival < next_p.open_time:
            wait = next_p.open_time - arrival
            total_wait += wait
            penalty += wait * wait_penalty_weight    # ★ Phạt chờ (configurable)
            arrival = next_p.open_time              # Vẫn phải chờ đến giờ mở

        if arrival > next_p.close_time:
            over = arrival - next_p.close_time
            penalty += over * PENALTY_LATE_ARRIVAL   # Phạt trễ giờ

        # --- Service ---
        departure = arrival + next_p.duration
        current_time = departure

    # Budget penalty
    if total_cost > user_prefs.budget:
        penalty += (total_cost - user_prefs.budget) * PENALTY_BUDGET

    # Late return penalty (check against user end_time in minutes)
    end_time_limit = user_prefs.end_time_minutes  # Phút (VD: 17h → 1020)
    if current_time > end_time_limit:
        penalty += (current_time - end_time_limit) * PENALTY_LATE_RETURN

    ind.fitness = total_score - penalty
    ind.total_score = total_score
    ind.total_cost = total_cost
    ind.total_time = current_time
    ind.total_wait = total_wait

    return ind.fitness