class POI:
    """
    Đại diện cho một Điểm tham quan (POI) hoặc Depot (khi id == 0).
    Tọa độ sử dụng hệ tọa độ Euclidean x/y khớp với định dạng benchmark Solomon.
    Các trường thời gian tính bằng đơn vị thời gian nguyên của Solomon.
    """

    def __init__(self, id: int, x: float, y: float, score: float,
                 open_time: float, close_time: float,
                 duration: float, category: str, price: float = 0.0):
        self.id = id
        self.x = x                  # Tọa độ X (Euclidean – Solomon)
        self.y = y                  # Tọa độ Y (Euclidean – Solomon)
        self.base_score = score     # DEMAND tương ứng trong Solomon
        self.open_time = open_time  # Giờ mở cửa (đơn vị thời gian Solomon)
        self.close_time = close_time  # Giờ đóng cửa (đơn vị thời gian Solomon)
        self.price = price          # Chi phí tham quan
        self.duration = duration    # Thời gian tham quan (đơn vị thời gian Solomon = SERVICE TIME)
        self.category = category    # Loại điểm (depot, history_culture, food_drink, ...)

    def __repr__(self):
        return f"POI(id={self.id}, cat={self.category}, score={self.base_score})"


class Individual:
    """
    Đại diện cho một giải pháp (nhiễm sắc thể) trong quần thể GA.
    Một lộ trình (route) là danh sách thứ tự các đối tượng POI: [Depot, POI_a, POI_b, ..., Depot].
    """

    def __init__(self, route: list[POI] = None):
        self.route: list[POI] = route if route is not None else []
        self.fitness: float = 0.0
        self.total_score: float = 0.0
        self.total_cost: float = 0.0
        self.total_time: float = 0.0
        self.total_wait: float = 0.0   # Tổng thời gian chờ tại các POI

    def __repr__(self):
        ids = [p.id for p in self.route]
        return f"Individual(fitness={self.fitness:.2f}, route_ids={ids})"

    def __len__(self):
        return len(self.route)
