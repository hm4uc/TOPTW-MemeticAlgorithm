"""
Response schemas (dữ liệu đầu ra trả về cho client).

Định nghĩa ItineraryItem và OptimizationResponse — Pydantic models
dùng làm response_model cho endpoint POST /api/optimize.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ItineraryItem(BaseModel):
    """Thông tin một điểm tham quan trong lộ trình."""

    order:           int            = Field(...,  description="Thứ tự trong lộ trình (bắt đầu từ 1)")
    id:              int            = Field(...,  description="ID điểm tham quan")
    name:            str            = Field(...,  description="Tên điểm tham quan")
    category:        Optional[str]  = Field(None, description="Loại điểm tham quan (history_culture, nature_parks, food_drink, shopping, entertainment, nightlife_wellness, depot)")
    travel_distance: Optional[float]= Field(None, description="Khoảng cách từ điểm trước đó (đơn vị khoảng cách)")
    travel_time:     Optional[int]  = Field(None, description="Thời gian di chuyển từ điểm trước đó (phút)")
    arrival:         Optional[str]  = Field(None, description="Thời gian đến nơi (HH:MM)")
    wait:            Optional[int]  = Field(0,    description="Thời gian chờ mở cửa (phút)")
    start:           Optional[str]  = Field(None, description="Thời gian bắt đầu tham quan (HH:MM)")
    leave:           Optional[str]  = Field(None, description="Thời gian rời đi (HH:MM)")
    cost:            float          = Field(...,  description="Chi phí tham quan tại điểm này (VND)")
    score:           float          = Field(...,  description="Điểm đạt được tại điểm tham quan (đã tính trọng số sở thích)")


class OptimizationResponse(BaseModel):
    """Kết quả tối ưu hóa lộ trình du lịch."""

    total_score:    float              = Field(..., description="Tổng điểm đạt được của toàn bộ lộ trình")
    total_cost:     float              = Field(..., description="Tổng chi phí chuyến đi (VND)")
    total_distance: float              = Field(..., description="Tổng quãng đường di chuyển (đơn vị khoảng cách)")
    total_duration: float              = Field(..., description="Tổng thời gian chuyến đi (giờ)")
    route:          List[ItineraryItem]= Field(..., description="Danh sách các điểm tham quan theo thứ tự (bao gồm Depot đầu và cuối)")
    execution_time: float              = Field(..., description="Thời gian chạy thuật toán (giây)")
