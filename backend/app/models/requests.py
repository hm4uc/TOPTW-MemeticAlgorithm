"""
Request schemas (dữ liệu đầu vào từ người dùng).

Định nghĩa UserPreferences — Pydantic model cho request body của API /optimize.
Bao gồm validation logic (field validators, model validators) và
các property tiện ích chuyển đổi đơn vị Giờ → Phút (Solomon time units).
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict

# =============================================================================
#  Hằng số validation
# =============================================================================

MIN_TOUR_DURATION_HOURS: float = 1.0  # Tối thiểu 1 giờ để lập lịch trình

# Bảng ánh xạ: số sao người dùng chọn → trọng số dùng trong thuật toán
STAR_TO_WEIGHT: Dict[int, float] = {
    1: 0.1,   # Không quan tâm
    2: 0.5,   # Ít quan tâm
    3: 1.0,   # Trung bình (mức cơ sở)
    4: 1.5,   # Quan tâm nhiều
    5: 2.0,   # Rất quan tâm / ưu tiên cao nhất
}

# Danh sách category hợp lệ
VALID_CATEGORIES = {
    'history_culture', 'nature_parks', 'food_drink', 'shopping', 'entertainment'
}

# Danh sách Solomon instances hỗ trợ trong API/pipeline
VALID_INSTANCES = {"C101", "C201", "R101", "R201", "RC101", "RC201"}


# =============================================================================
#  Request Model
# =============================================================================

class UserPreferences(BaseModel):
    """Sở thích và ràng buộc của người dùng cho chuyến du lịch."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instance_name": "C101",
                    "budget": 500000,
                    "start_time": 8.0,
                    "end_time": 17.0,
                    "start_node_id": 0,
                    "interests": {
                        "history_culture": 5,
                        "nature_parks": 3,
                        "food_drink": 4,
                        "shopping": 1,
                        "entertainment": 2
                    }
                }
            ]
        }
    }

    instance_name: str = Field(
        "C101",
        description="Tên Solomon instance (C101, C201, R101, R201, RC101, RC201)",
    )
    budget: float = Field(..., description="Ngân sách tối đa cho chuyến đi (VND)")
    start_time: float = Field(8.0, description="Thời gian bắt đầu chuyến đi (giờ, VD: 8.0 = 8:00)")
    end_time: float = Field(17.0, description="Thời gian kết thúc chuyến đi (giờ, VD: 17.0 = 17:00)")
    start_node_id: int = Field(..., description="ID điểm xuất phát (depot), thường là 0")
    interests: Dict[str, int] = Field(
        ...,
        description=(
            "Mức độ quan tâm của người dùng với từng loại hình địa điểm, "
            "tính bằng số sao từ 1 đến 5. "
            "Key phải thuộc: history_culture, nature_parks, food_drink, shopping, entertainment. "
            "1 sao = không quan tâm (w=0.1), 2 sao = ít quan tâm (w=0.5), "
            "3 sao = trung bình (w=1.0), 4 sao = quan tâm nhiều (w=1.5), "
            "5 sao = rất quan tâm (w=2.0)."
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  Field Validators
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator('instance_name')
    @classmethod
    def validate_instance_name(cls, v: str) -> str:
        """Chuẩn hóa instance_name về uppercase và kiểm tra hợp lệ."""
        normalized = v.strip().upper()
        if normalized not in VALID_INSTANCES:
            raise ValueError(
                f"instance_name '{v}' không hợp lệ. "
                f"Các giá trị hợp lệ: {sorted(VALID_INSTANCES)}"
            )
        return normalized

    @field_validator('budget')
    @classmethod
    def validate_budget(cls, v: float) -> float:
        """Ngân sách phải dương."""
        if v <= 0:
            raise ValueError(
                f"Ngân sách phải lớn hơn 0, nhận được: {v}"
            )
        return v

    @field_validator('interests')
    @classmethod
    def validate_interests(cls, v: Dict[str, int]) -> Dict[str, int]:
        """
        Kiểm tra:
          1. Phải có đủ 5 category bắt buộc.
          2. Không có key lạ.
          3. Giá trị sao nằm trong [1, 5].
        """
        missing = VALID_CATEGORIES - set(v.keys())
        if missing:
            raise ValueError(
                f"Thiếu đánh giá cho các category: {sorted(missing)}. "
                f"Phải cung cấp đủ 5 category: {sorted(VALID_CATEGORIES)}"
            )

        for category, stars in v.items():
            if category not in VALID_CATEGORIES:
                raise ValueError(
                    f"Category '{category}' không hợp lệ. "
                    f"Phải thuộc: {sorted(VALID_CATEGORIES)}"
                )
            if stars not in STAR_TO_WEIGHT:
                raise ValueError(
                    f"Số sao cho '{category}' phải là số nguyên từ 1 đến 5, "
                    f"nhận được: {stars}"
                )
        return v

    # ─────────────────────────────────────────────────────────────────────────
    #  Model Validator  (cross-field validation)
    # ─────────────────────────────────────────────────────────────────────────

    @model_validator(mode='after')
    def validate_time_range(self) -> 'UserPreferences':
        """
        Edge Case 3: start_time >= end_time → vô lý, reject ngay.
        Edge Case 1: end_time - start_time < MIN_TOUR_DURATION_HOURS → quá ngắn.
        """
        if self.start_time >= self.end_time:
            raise ValueError(
                f"Thời gian bắt đầu ({self.start_time}h) phải nhỏ hơn "
                f"thời gian kết thúc ({self.end_time}h)."
            )

        duration = self.end_time - self.start_time
        if duration < MIN_TOUR_DURATION_HOURS:
            raise ValueError(
                f"Khung thời gian quá ngắn để lập lịch trình. "
                f"Thời lượng tối thiểu: {MIN_TOUR_DURATION_HOURS} giờ, "
                f"nhận được: {duration:.1f} giờ "
                f"({self.start_time}h → {self.end_time}h)."
            )

        return self

    # ─────────────────────────────────────────────────────────────────────────
    #  Chuyển đổi đơn vị: Giờ (user input) → Phút (Solomon time units)
    # ─────────────────────────────────────────────────────────────────────────
    #  Solomon benchmark dùng đơn vị PHÚT cho tất cả time fields:
    #    READY TIME, DUE DATE, SERVICE TIME, travel_time (Euclidean distance).
    #  User nhập giờ (8.0 = 8AM, 17.0 = 5PM) → cần nhân 60 để khớp đơn vị.
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def start_time_minutes(self) -> float:
        """Thời gian bắt đầu quy đổi sang phút (Solomon time units)."""
        return self.start_time * 60.0

    @property
    def end_time_minutes(self) -> float:
        """Thời gian kết thúc quy đổi sang phút (Solomon time units)."""
        return self.end_time * 60.0

    @property
    def interest_weights(self) -> Dict[str, float]:
        """
        Chuyển đổi số sao → trọng số float ĐÃ CHUẨN HÓA để thuật toán sử dụng.

        ★ NORMALIZATION (Edge Case 2) ★
          Khi tất cả category có cùng số sao (VD: toàn bộ 1★ hoặc toàn bộ 5★),
          raw weights sẽ giống nhau → thuật toán không phân biệt được category.

          Giải pháp: Normalize trọng số sao cho:
            • Tổng trọng số = len(interests) (trung bình = 1.0 mỗi category)
            • Nếu tất cả bằng nhau → mỗi cái = 1.0 → GA vẫn phân biệt POI
              bằng base_score (DEMAND), đảm bảo công bằng.
            • Nếu có sự khác biệt → tỷ lệ giữ nguyên, chỉ scale lại.
        """
        raw = {cat: STAR_TO_WEIGHT[stars] for cat, stars in self.interests.items()}

        total = sum(raw.values())
        if total == 0:
            return {cat: 1.0 for cat in raw}

        n = len(raw)
        scale = n / total
        return {cat: w * scale for cat, w in raw.items()}
