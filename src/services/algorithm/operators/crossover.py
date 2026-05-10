"""
Crossover Operators - Order Crossover (OX1).

Nguyên tắc "Depot-Safe":
  Toán tử CHỈ thao tác trên "interior" = route[1:-1].
  Depot được gắn lại sau khi xử lý xong.

Reference: Davis (1985) - "Applying adaptive algorithms to epistatic domains."
"""

import random
from typing import Optional

from models.domain import POI, Individual


def crossover_ox1(
    parent1: Individual,
    parent2: Individual,
    depot: POI,
) -> Individual:
    """
    Áp dụng Order Crossover (OX1) trên phân đoạn nội thất (interior).

    OX1 là toán tử dựa trên thứ tự, bảo tồn các chuỗi POI từ cha mẹ.
    Phù hợp cho các bài toán định tuyến vì nó giữ lại cấu trúc lộ trình cục bộ.

    Thuật toán:
      1. Cắt ngẫu nhiên đoạn [cut1, cut2] từ parent1 → giữ nguyên vào child.
      2. Điền các POI còn lại từ parent2 theo đúng thứ tự xuất hiện của chúng.

    Parameters
    ----------
    parent1, parent2 : Individual
        Hai cá thể cha-mẹ.
    depot : POI
        Điểm depot để gắn vào đầu và cuối route con.

    Returns
    -------
    Individual
        Cá thể con mới.
    """
    r1 = parent1.route[1:-1]
    r2 = parent2.route[1:-1]

    size = min(len(r1), len(r2))
    if size < 2:
        return Individual(route=list(parent1.route))

    r1, r2 = r1[:size], r2[:size]

    cut1, cut2 = sorted(random.sample(range(size), 2))

    child_interior: list[Optional[POI]] = [None] * size

    # Giữ nguyên đoạn [cut1, cut2] từ parent1
    segment_ids = set()
    for k in range(cut1, cut2 + 1):
        child_interior[k] = r1[k]
        segment_ids.add(r1[k].id)

    # Điền phần còn lại theo thứ tự parent2
    remaining = [poi for poi in r2 if poi.id not in segment_ids]
    empty_positions = [k for k in range(size) if child_interior[k] is None]
    for pos, poi in zip(empty_positions, remaining):
        child_interior[pos] = poi

    child_interior_clean = [p for p in child_interior if p is not None]

    new_route = [depot] + child_interior_clean + [depot]
    return Individual(route=new_route)
