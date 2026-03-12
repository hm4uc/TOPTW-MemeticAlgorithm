"""
Cấu hình trung tâm cho toàn bộ hệ thống TOPTW-HybridGA.

Tập trung tất cả hằng số, hệ số phạt và tham số GA mặc định
để dễ dàng điều chỉnh mà không cần tìm rải rác trong nhiều file.
"""

# =============================================================================
#  GA POPULATION PARAMETERS
# =============================================================================

POPULATION_SIZE: int = 100    # Kích thước quần thể
HEURISTIC_COUNT: int = 80     # 80% → Randomized Insertion Heuristic
RANDOM_COUNT:    int = 20     # 20% → Pure Random
RCL_SIZE:        int = 3      # Top-k candidates trong Restricted Candidate List

# =============================================================================
#  GA TUNABLE DEFAULTS  (có thể override khi khởi tạo HybridGeneticAlgorithm)
# =============================================================================

DEFAULT_MUTATION_RATE:     float = 0.7
DEFAULT_GENERATIONS:       int   = 200
DEFAULT_STAGNATION_LIMIT:  int   = 25
DEFAULT_TOURNAMENT_K:      int   = 3
IMPROVEMENT_THRESHOLD:     float = 1e-4

# =============================================================================
#  PENALTY COEFFICIENTS  (Bảng hệ số phạt)
#
#  ┌─────────────────────────┬────────────┬──────────────────────────────────┐
#  │ Loại phạt               │  Hệ số     │  Mục đích                        │
#  ├─────────────────────────┼────────────┼──────────────────────────────────┤
#  │ Trễ giờ (> close_time)  │  100.0     │  Vi phạm ràng buộc CỨNG          │
#  │ Về depot trễ            │  100.0     │  Vi phạm ràng buộc CỨNG          │
#  │ Lố ngân sách            │    0.5     │  Ràng buộc mềm                   │
#  │ Thời gian chờ           │    0.2     │  Chất lượng trải nghiệm du lịch  │
#  └─────────────────────────┴────────────┴──────────────────────────────────┘
#
#  Ví dụ tác động phạt chờ:
#    Chờ 15'  → phạt  3.0   (bình thường, ghé cafe)
#    Chờ 30'  → phạt  6.0   (hơi khó chịu)
#    Chờ 60'  → phạt 12.0   (tệ → GA cố tránh)
#    Chờ 120' → phạt 24.0   (rất tệ → GA gần như chắc chắn tránh)
# =============================================================================

PENALTY_LATE_ARRIVAL: float = 100.0  # Đến sau close_time  (ràng buộc cứng)
PENALTY_LATE_RETURN:  float = 100.0  # Về depot trễ        (ràng buộc cứng)
PENALTY_BUDGET:       float =   0.5  # Vượt ngân sách      (ràng buộc mềm)
PENALTY_WAIT:         float =   0.2  # Thời gian chờ       (chất lượng trải nghiệm)

# =============================================================================
#  URGENCY HEURISTIC PARAMETERS
#
#  Cải tiến Labadie ratio: thêm yếu tố "độ khẩn cấp" (time-window urgency)
#  để ưu tiên POI sắp đóng cửa khi khởi tạo quần thể và greedy refill.
#
#  Công thức: urgency_factor = 1 + URGENCY_ALPHA / max(time_remaining, ε)
#  URGENCY_CAP giới hạn trên để urgency không quá áp đảo score/distance.
# =============================================================================

URGENCY_ALPHA: float = 10.0   # Hệ số khẩn cấp (α): càng cao → càng ưu tiên POI sắp đóng
URGENCY_CAP:   float = 5.0    # Giới hạn trên urgency_factor (tránh outlier)

