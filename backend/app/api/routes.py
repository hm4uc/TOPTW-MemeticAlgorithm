from fastapi import APIRouter, HTTPException
import logging
from app.models.requests import UserPreferences
from app.models.responses import OptimizationResponse
from app.services.algorithm.hga_engine import HybridGeneticAlgorithm
from app.services.data_loader import load_solomon_instance

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    summary="Tối ưu hóa lộ trình du lịch",
    description=(
        "Nhận sở thích người dùng (ngân sách, khung giờ, mức quan tâm 5 loại hình) "
        "và trả về lộ trình tối ưu sử dụng thuật toán Di truyền Lai (HGA).\n\n"
        "**Quy trình xử lý:**\n"
        "1. Pydantic validation: kiểm tra instance_name, budget, khung thời gian, interests → 422 nếu sai định dạng.\n"
        "2. Business validation: kiểm tra start_node_id có tồn tại trong instance đã chọn → 400 nếu không hợp lệ.\n"
        "3. Chạy HGA tối ưu lộ trình → 500 nếu lỗi hệ thống.\n"
        "4. Kiểm tra kết quả: route rỗng hoặc chỉ có Depot → 404.\n\n"
        "**Loại hình điểm tham quan (interests):**\n"
        "- `history_culture`: Lịch sử - Văn hóa\n"
        "- `nature_parks`: Thiên nhiên - Công viên\n"
        "- `food_drink`: Ẩm thực\n"
        "- `shopping`: Mua sắm\n"
        "- `entertainment`: Giải trí\n"
        "- `nightlife_wellness`: Nightlife & Wellness\n\n"
        "**Thang đánh giá (1-5 sao):**\n"
        "- 1 sao = Không quan tâm (weight 0.1)\n"
        "- 2 sao = Ít quan tâm (weight 0.5)\n"
        "- 3 sao = Trung bình (weight 1.0)\n"
        "- 4 sao = Quan tâm nhiều (weight 1.5)\n"
        "- 5 sao = Rất quan tâm (weight 2.0)"
    ),
    responses={
        200: {
            "description": "Lộ trình tối ưu được tìm thấy thành công.",
        },
        400: {
            "description": "Dữ liệu đầu vào không hợp lệ (VD: start_node_id không tồn tại).",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Điểm xuất phát (start_node_id=999) không tồn tại trong dataset. ID hợp lệ: 0 đến 100."
                    }
                }
            },
        },
        404: {
            "description": "Không tìm được lộ trình khả thi với ràng buộc đã cho.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Không thể ghé thăm bất kỳ điểm nào trong khung thời gian và ngân sách đã cho."
                    }
                }
            },
        },
        422: {
            "description": "Lỗi validation dữ liệu (thiếu trường, sai kiểu, giá trị ngoài phạm vi).",
        },
        500: {
            "description": "Lỗi hệ thống trong quá trình chạy thuật toán.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Đã xảy ra lỗi trong quá trình tối ưu hóa lộ trình."
                    }
                }
            },
        },
    },
)
async def optimize_itinerary(request: UserPreferences):
    try:
        logger.info("Received optimization request with preferences: %s", request)

        # ── Load đúng instance theo request + validate start_node_id ──────
        pois = load_solomon_instance(request.instance_name)
        if not pois:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Không tải được dữ liệu cho instance '{request.instance_name}'. "
                    "Hãy kiểm tra dữ liệu benchmark trong backend/data/solomon_instances."
                ),
            )

        valid_ids = {p.id for p in pois}
        if request.start_node_id not in valid_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Điểm xuất phát (start_node_id={request.start_node_id}) "
                    f"không tồn tại trong instance {request.instance_name}. "
                    f"ID hợp lệ: 0 đến {max(valid_ids)}."
                ),
            )

        # ── Run HGA ───────────────────────────────────────────────────────
        hga_solver = HybridGeneticAlgorithm(
            request,
            pois=pois,
            instance_name=request.instance_name,
        )
        result = hga_solver.run()

        # ── Edge Case 7: GA trả về route rỗng [Depot, Depot] ─────────────
        if not result:
            raise HTTPException(
                status_code=404,
                detail="Không tìm được lộ trình khả thi với các tùy chọn đã cho.",
            )

        # Kiểm tra route chỉ có Depot (không ghé được POI nào)
        if hasattr(result, 'route') and len(result.route) <= 2:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Không thể ghé thăm bất kỳ điểm nào trong khung thời gian "
                    f"và ngân sách đã cho ({request.start_time}h → {request.end_time}h, "
                    f"budget={request.budget:,.0f}). "
                    "Hãy thử mở rộng khung giờ hoặc tăng ngân sách."
                ),
            )

        return result

    except HTTPException:
        # Re-raise HTTPExceptions (đã có status code rõ ràng)
        raise

    except Exception as e:
        logger.error("Error during optimization: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Đã xảy ra lỗi trong quá trình tối ưu hóa lộ trình.",
        )