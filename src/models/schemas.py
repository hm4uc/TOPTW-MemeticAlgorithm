"""
[DEPRECATED] schemas.py — Compatibility shim.

Các module mới nên import trực tiếp từ:
  • models.requests  → UserPreferences
  • models.responses → OptimizationResponse, ItineraryItem

File này chỉ re-export để tránh ImportError trong trường hợp còn code cũ
chưa được cập nhật. Sẽ xóa trong phiên bản tiếp theo.
"""

from models.requests import (   # noqa: F401
    UserPreferences,
    STAR_TO_WEIGHT,
    VALID_CATEGORIES,
    MIN_TOUR_DURATION_HOURS,
)
from models.responses import (  # noqa: F401
    ItineraryItem,
    OptimizationResponse,
)
