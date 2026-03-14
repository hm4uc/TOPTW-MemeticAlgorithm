"""
Data Loader — Đọc Solomon benchmark instances.

Hỗ trợ 2 chế độ:
  1. load_solomon_instance(name)  → Đọc từ extended CSV (có CATEGORY + PRICE sẵn)
  2. load_solomon_c101()          → Legacy API, tương thích ngược
"""

import csv
import copy
import random
from pathlib import Path
from typing import List
from app.models.domain import POI

# --- DANH SÁCH CATEGORY CHUẨN ---
CATEGORIES = [
    'history_culture',  # Lăng Bác, Văn Miếu, Hoàng Thành, Chùa Một Cột, ...
    'nature_parks',     # Công viên Thống Nhất, Hồ Gươm, Vườn hoa Lý Thái Tổ, ...
    'food_drink',       # Phở Thìn, Bún Chả Hương Liên, Cà phê Giảng, ...
    'shopping',         # Chợ Đồng Xuân, Vincom Bà Triệu, Tràng Tiền Plaza, ...
    'entertainment'     # Rạp chiếu phim Quốc gia, Nhà hát Lớn Hà Nội, ...
]

# Tỷ lệ xuất hiện giả lập (Mô phỏng đặc thù du lịch Hà Nội)
CATEGORY_WEIGHTS = [0.35, 0.15, 0.25, 0.15, 0.10]

# ── BẢNG GIÁ THEO LOẠI HÌNH (Pricing Tiers) ────────────────────────────────
CATEGORY_PRICE_TIERS: dict[str, list[float]] = {
    'nature_parks':    [0.0],
    'history_culture': [30_000.0, 50_000.0, 100_000.0],
    'entertainment':   [100_000.0, 200_000.0, 500_000.0],
    'food_drink':      [50_000.0, 150_000.0, 300_000.0, 800_000.0],
    'shopping':        [0.0, 100_000.0, 300_000.0, 500_000.0],
}


# =============================================================================
#  IN-MEMORY CACHE  (Singleton Pattern — per instance)
# =============================================================================
_INSTANCE_CACHE: dict[str, List[POI]] = {}
AVAILABLE_INSTANCES = ("C101", "C201", "R101", "R201", "RC101", "RC201")

# backend/app/services/data_loader.py -> parents[2] = backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SOLOMON_DIR = _BACKEND_DIR / "data" / "solomon_instances"
_EXTENDED_DIR = _SOLOMON_DIR / "extended"


def _parse_solomon_csv(file_path: str, use_extended: bool = False) -> List[POI]:
    """
    Internal: Đọc file Solomon CSV và parse thành list[POI].

    Parameters
    ----------
    file_path : str
        Đường dẫn đến file CSV.
    use_extended : bool
        True nếu file có cột CATEGORY + PRICE sẵn (extended CSV).
        False nếu cần random gán (legacy).
    """
    pois = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cust_no = int(row.get('CUST NO.', 0))
                pid = cust_no - 1  # Remap: CUST NO. 1 → Depot (id=0)

                if use_extended:
                    # Extended CSV — đọc CATEGORY + PRICE trực tiếp
                    cat = row.get('CATEGORY', 'depot')
                    price = float(row.get('PRICE', 0.0))
                else:
                    # Legacy — random gán (seed = pid cho reproducibility)
                    if pid == 0:
                        cat = "depot"
                        price = 0.0
                    else:
                        rng = random.Random(pid)
                        cat = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
                        price = rng.choice(CATEGORY_PRICE_TIERS[cat])

                poi = POI(
                    id=pid,
                    x=float(row.get('XCOORD.', 0)),
                    y=float(row.get('YCOORD.', 0)),
                    score=float(row.get('DEMAND', 0)),
                    open_time=float(row.get('READY TIME', 0)),
                    close_time=float(row.get('DUE DATE', 0)),
                    duration=float(row.get('SERVICE TIME', 0)),
                    category=cat,
                    price=price,
                )
                pois.append(poi)
    except Exception as e:
        print(f"[DataLoader] Error reading file {file_path}: {e}")
        return []

    print(f"[DataLoader] Loaded {len(pois)} POIs from {Path(file_path).name} "
          f"(Depot id=0 at ({pois[0].x}, {pois[0].y}))")
    return pois


def load_solomon_instance(instance_name: str = "C101") -> List[POI]:
    """
    Load Solomon benchmark instance từ extended CSV — CÓ CACHE.

    Extended CSV đã có sẵn CATEGORY + PRICE cố định (reproducible).
    Cache theo tên instance.

    Parameters
    ----------
    instance_name : str
        Tên instance: "C101", "C201", "R101", "R201", "RC101", "RC201"

    Returns
    -------
    list[POI]
        Deep copy of POI list. POI with id=0 is always the Depot.
    """
    global _INSTANCE_CACHE
    instance_name = instance_name.strip().upper()

    if instance_name not in AVAILABLE_INSTANCES:
        raise ValueError(
            f"instance_name '{instance_name}' không hợp lệ. "
            f"Các giá trị hợp lệ: {', '.join(AVAILABLE_INSTANCES)}"
        )

    if instance_name not in _INSTANCE_CACHE:
        # Tìm file extended CSV
        file_path = _EXTENDED_DIR / f'{instance_name}_extended.csv'

        if not file_path.exists():
            # Fallback: thử từ thư mục gốc (legacy format)
            file_path = _SOLOMON_DIR / f'{instance_name}.csv'
            _INSTANCE_CACHE[instance_name] = _parse_solomon_csv(str(file_path), use_extended=False)
        else:
            _INSTANCE_CACHE[instance_name] = _parse_solomon_csv(str(file_path), use_extended=True)

        print(f"[DataLoader] Cache initialized for {instance_name}: "
              f"{len(_INSTANCE_CACHE[instance_name])} POIs")
    else:
        print(f"[DataLoader] Cache HIT — {instance_name}: "
              f"{len(_INSTANCE_CACHE[instance_name])} POIs from RAM")

    return copy.deepcopy(_INSTANCE_CACHE[instance_name])


def load_solomon_c101() -> List[POI]:
    """
    Legacy API — tương thích ngược.
    Load Solomon C101, sử dụng extended CSV nếu có.
    """
    return load_solomon_instance("C101")