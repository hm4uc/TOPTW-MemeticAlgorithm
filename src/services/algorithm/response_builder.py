"""
Response Builder — Chuyển đổi Individual tốt nhất thành OptimizationResponse.

Tách biệt hoàn toàn logic format output (API layer) khỏi engine GA.
Engine chỉ cần gọi build_response() sau khi kết thúc quá trình tiến hóa.

ĐƠN VỊ:
  • Bên trong tính bằng PHÚT (Solomon time units).
  • Output arrival/start/leave → format HH:MM.
  • Output total_duration → giờ (để user dễ đọc).
"""

from models.domain import Individual
from models.requests import UserPreferences
from models.responses import OptimizationResponse, ItineraryItem
from services.algorithm.fitness import get_travel_time


def _format_time(minutes: float) -> str:
    """
    Chuyển đổi thời gian dạng PHÚT (Solomon time units) sang chuỗi HH:MM.

    Examples
    --------
    >>> _format_time(480.0)
    '08:00'
    >>> _format_time(612.5)
    '10:12'
    """
    total_minutes = int(round(minutes))
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"{hours:02d}:{mins:02d}"


def build_response(
    best: Individual,
    user_prefs: UserPreferences,
    execution_time: float,
) -> OptimizationResponse:
    """
    Chuyển đổi Individual tốt nhất thành OptimizationResponse gửi về client.

    Mô phỏng lại timeline trên route để tính chính xác:
      • travel_distance, travel_time giữa 2 điểm liên tiếp
      • arrival, wait, start, leave tại mỗi POI
      • total_score, total_cost, total_distance, total_duration

    Parameters
    ----------
    best : Individual
        Cá thể tốt nhất sau khi GA kết thúc.
    user_prefs : UserPreferences
        Sở thích người dùng (dùng để lấy interest_weights và start_time).
    execution_time : float
        Thời gian chạy thuật toán (giây), để đưa vào response.

    Returns
    -------
    OptimizationResponse
        Response đã được format sẵn để trả về client.
    """
    route = best.route
    weights = user_prefs.interest_weights

    current_time = user_prefs.start_time_minutes  # Phút (VD: 8h → 480)
    items: list[ItineraryItem] = []
    total_cost = 0.0
    total_score = 0.0
    total_distance = 0.0

    for order, poi in enumerate(route):
        if order == 0:
            # Depot xuất phát — không có travel (điểm bắt đầu)
            items.append(ItineraryItem(
                order=1,
                id=poi.id,
                name="Điểm xuất phát (Depot)",
                category="depot",
                travel_distance=None,
                travel_time=None,
                arrival=_format_time(current_time),
                wait=0,
                start=_format_time(current_time),
                leave=_format_time(current_time),
                cost=0.0,
                score=0.0,
            ))
            continue

        # Tính khoảng cách và thời gian di chuyển từ điểm trước
        prev_poi = route[order - 1]
        travel = get_travel_time(prev_poi, poi)
        total_distance += travel
        travel_time_minutes = int(round(travel))

        arrival_raw = current_time + travel  # Thời điểm đến (phút)

        # Chờ nếu đến sớm hơn giờ mở cửa
        wait_minutes = 0
        if arrival_raw < poi.open_time:
            wait_minutes = int(round(poi.open_time - arrival_raw))
            arrival_effective = poi.open_time
        else:
            arrival_effective = arrival_raw

        start_service = arrival_effective
        leave_time = start_service + poi.duration
        current_time = leave_time

        # Tính điểm theo trọng số sở thích
        w = weights.get(poi.category, 0.0)
        score = poi.base_score * w

        if order == len(route) - 1:
            # Depot cuối (trở về) — không tính vào score/cost
            items.append(ItineraryItem(
                order=order + 1,
                id=poi.id,
                name="Trở về (Depot)",
                category="depot",
                travel_distance=round(travel, 2),
                travel_time=travel_time_minutes,
                arrival=_format_time(arrival_raw),
                wait=0,
                start=None,
                leave=None,
                cost=0.0,
                score=0.0,
            ))
        else:
            total_cost += poi.price
            total_score += score
            items.append(ItineraryItem(
                order=order + 1,
                id=poi.id,
                name=f"POI-{poi.id} ({poi.category})",
                category=poi.category,
                travel_distance=round(travel, 2),
                travel_time=travel_time_minutes,
                arrival=_format_time(arrival_raw),
                wait=wait_minutes,
                start=_format_time(start_service),
                leave=_format_time(leave_time),
                cost=poi.price,
                score=round(score, 2),
            ))

    # total_duration: phút → giờ (output cho user)
    total_duration_hours = (current_time - user_prefs.start_time_minutes) / 60.0

    return OptimizationResponse(
        total_score=round(total_score, 2),
        total_cost=round(total_cost, 2),
        total_distance=round(total_distance, 2),
        total_duration=round(total_duration_hours, 2),
        route=items,
        execution_time=round(execution_time, 4),
    )
