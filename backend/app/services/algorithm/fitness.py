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
#  MA TRẬN KHOẢNG CÁCH (O(1) lookup – tính toán trước khi khởi động)
# =============================================================================
#
#  Thay vì gọi math.sqrt() hàng triệu lần trong quá trình GA, ta tính
#  trước ma trận khoảng cách N×N một lần duy nhất khi nạp dữ liệu.
#  Mọi tra cứu sau đó chỉ mất O(1) (truy cập mảng 2D).
#  Với 101 POI → ma trận 101×101 = ~10 201 giá trị float  80 KB RAM.
#
# =============================================================================

# Module-level distance matrix – initialized by build_distance_matrix()
_DIST_MATRIX: Optional[list[list[float]]] = None


def euclidean_distance(p1: POI, p2: POI) -> float:
    """Tính khoảng cách Euclidean giữa hai POI theo tọa độ."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def build_distance_matrix(pois: List[POI]) -> list[list[float]]:
    """
    Tính toán trước ma trận khoảng cách Euclidean N×N.

    Phải được gọi MỘT LẦN khi khởi động (sau khi nạp POI).
    Các lần gọi get_travel_time() sau đó sẽ sử dụng tra cứu O(1).

    Parameters
    ----------
    pois : list[POI]
        Tất cả các POI (bao gồm cả depot). ID của POI phải từ 0..N-1.

    Returns
    -------
    list[list[float]]
        Ma trận 2D trong đó matrix[i][j] là khoảng cách Euclidean từ POI i đến POI j.
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


def get_travel_time(p1: POI, p2: POI) -> float:
    """
    Thời gian di chuyển giữa hai POI.
    Đối với Solomon benchmarks, thời gian di chuyển == khoảng cách Euclidean
    (tốc độ = 1 đơn vị/đơn vị thời gian).

    Sử dụng ma trận khoảng cách đã tính toán trước để tra cứu O(1).
    Sẽ tính toán trực tiếp nếu ma trận chưa được khởi tạo.
    """
    if _DIST_MATRIX is not None:
        return _DIST_MATRIX[p1.id][p2.id]
    # Fallback (should not happen in normal flow)
    return euclidean_distance(p1, p2)


# =============================================================================
#  KIỂM TRA RÀNG BUỘC (Tính khả thi của TOPTW)
# =============================================================================

def check_constraints(route: list[POI], user_prefs: UserPreferences) -> bool:
    """
    Kiểm tra xem một lộ trình HOÀN CHỈNH [Depot, ..., Depot] có thỏa mãn tất cả
    các ràng buộc của TOPTW hay không:
      1. Khung thời gian (Time Windows) – đến mỗi POI trước giờ đóng cửa (close_time).
      2. Thời gian hành trình tối đa – quay lại depot trước giờ kết thúc (end_time).
      3. Ngân sách (Budget) – tổng giá vé các POI đã ghé ≤ user_prefs.budget.

    ĐƠN VỊ: Mọi phép tính bên trong dùng PHÚT (Solomon time units).
    Dữ liệu đầu vào (giờ) được chuyển qua start_time_minutes / end_time_minutes.

    Trả về True nếu TẤT CẢ các ràng buộc được thỏa mãn, ngược lại là False.
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
    Kiểm tra xem `candidate` có thể được *chèn ngay trước Depot cuối*
    mà vẫn giữ cho lộ trình khả thi hay không.
    
    Lộ trình được giả định là [Depot, ..., (điểm ghé cuối)] CHƯA có
    Depot kết thúc (depot sẽ được thêm tạm thời để kiểm tra).
    
    Trả về True nếu lộ trình [*route, candidate, Depot] thỏa mãn các ràng buộc.
    """
    depot = route[0]  # Depot is always the first element
    test_route = route + [candidate, depot]
    return check_constraints(test_route, user_prefs)


# =============================================================================
#  ĐÁNH GIÁ FITNESS
# =============================================================================
#  Hệ số phạt được định nghĩa tập trung tại: app/core/config.py
#  PENALTY_LATE_ARRIVAL, PENALTY_LATE_RETURN, PENALTY_BUDGET, PENALTY_WAIT
#  đã được import ở đầu file.
# =============================================================================


def calculate_fitness(ind, user_prefs: UserPreferences,
                      wait_penalty_weight: float = PENALTY_WAIT) -> float:
    """
    Đánh giá fitness của một cá thể (Individual).

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

    Các hình phạt (Penalties) bao gồm:
      • Vi phạm khung thời gian – đến sau close_time       (×100.0)
      • Về depot trễ          – quay lại sau end_time      (×100.0)
      • Vượt ngân sách        – tổng chi phí > budget      (× 0.5)
      • Thời gian chờ         – đến trước open_time        (× wait_penalty_weight)
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
            penalty += wait * wait_penalty_weight    #  Phạt chờ (configurable)
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