"""
Cấu hình trung tâm cho toàn bộ hệ thống TOPTW-MemeticAlgorithm.

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
#  GA TUNABLE DEFAULTS  (có thể override khi khởi tạo MemeticAlgorithm)
# =============================================================================

DEFAULT_MUTATION_RATE:     float = 0.3
DEFAULT_GENERATIONS:       int   = 200
DEFAULT_STAGNATION_LIMIT:  int   = 25
DEFAULT_TOURNAMENT_K:      int   = 3
IMPROVEMENT_THRESHOLD:     float = 1e-4

# =============================================================================
#  ADAPTIVE-LITE MUTATION PARAMETERS
#
#  Tầng 1 (schedule theo progress):
#    - Insert giảm dần: 0.45 -> 0.15
#    - 2-opt tăng dần : 0.25 -> 0.55
#    - Swap = phần còn lại
#
#  Tầng 2 (feedback):
#    - Stagnation cao  -> tăng 2-opt
#    - Diversity thấp  -> tăng swap
#    - Insert fail cao -> giảm insert, dồn sang 2-opt
# =============================================================================

USE_ADAPTIVE_MUTATION_DEFAULT: bool = True
ADAPTIVE_INSERT_START: float = 0.45
ADAPTIVE_INSERT_END: float = 0.15
ADAPTIVE_2OPT_START: float = 0.25
ADAPTIVE_2OPT_END: float = 0.55

ADAPTIVE_STAGNATION_TRIGGER: int = 8
ADAPTIVE_LOW_DIVERSITY_THRESHOLD: float = 0.35
ADAPTIVE_INSERT_FAIL_TRIGGER: float = 0.60
ADAPTIVE_INSERT_FAIL_WINDOW: int = 5

ADAPTIVE_MIN_PROB: float = 0.10
ADAPTIVE_MAX_PROB: float = 0.80

# =============================================================================
#  PENALTY COEFFICIENTS
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


