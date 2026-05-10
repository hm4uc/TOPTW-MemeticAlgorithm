"""
Generate Extended CSV — Thêm CATEGORY + PRICE cố định vào Solomon instances.

Chạy 1 lần duy nhất. Output là 6 file CSV cố định.
Mọi thí nghiệm sau đều đọc từ file này → đảm bảo Reproducibility.

Quy tắc gán Category:
  - POI có READY TIME ≥ 75% Depot DUE DATE → nightlife_wellness (mở ban đêm)
  - POI còn lại → random theo phân phối xác suất (seed = CUST_NO)

Seed = CUST_NO (POI ID) → cùng 1 POI luôn có cùng category + price.

Usage:
    cd backend
    python -m experiments.generate_extended_data
"""

import csv
import os
import random

# - Category Configuration (6 categories) -
# 5 category "ban ngày" — gán theo phân phối xác suất
DAYTIME_CATEGORIES = [
    'history_culture',      # Lăng Bác, Văn Miếu, Hoàng Thành, ...
    'nature_parks',         # Công viên Thống Nhất, Hồ Gươm, ...
    'food_drink',           # Phở Thìn, Bún Chả, Cà phê Giảng, ...
    'shopping',             # Chợ Đồng Xuân, Vincom, ...
    'entertainment',        # Nhà hát Lớn, Rạp chiếu phim, ...
]

# Tỷ lệ phân bố cho 5 category ban ngày (đặc thù du lịch Hà Nội)
DAYTIME_WEIGHTS = [0.30, 0.15, 0.25, 0.15, 0.15]

# Category đặc biệt — gán DETERMINISTIC theo Time Window
NIGHTLIFE_CATEGORY = 'nightlife_wellness'  # Tạ Hiện, Rooftop bar, Spa, Chợ đêm, ...

# Ngưỡng: POI có READY TIME ≥ NIGHTLIFE_THRESHOLD * Depot_DUE_DATE → nightlife
NIGHTLIFE_THRESHOLD = 0.75

# Danh sách đầy đủ 6 categories (dùng cho thống kê)
ALL_CATEGORIES = DAYTIME_CATEGORIES + [NIGHTLIFE_CATEGORY]

# - Bảng giá theo loại hình (VND) -
CATEGORY_PRICE_TIERS: dict[str, list[float]] = {
    'nature_parks':         [0.0],
    'history_culture':      [30_000.0, 50_000.0, 100_000.0],
    'entertainment':        [100_000.0, 200_000.0, 500_000.0],
    'food_drink':           [50_000.0, 150_000.0, 300_000.0, 800_000.0],
    'shopping':             [0.0, 100_000.0, 300_000.0, 500_000.0],
    'nightlife_wellness':   [100_000.0, 200_000.0, 500_000.0],
}

INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]


def generate_extended(instance_name: str, base_dir: str) -> None:
    """Đọc Solomon CSV gốc, thêm CATEGORY + PRICE, lưu file extended."""
    input_path = os.path.join(base_dir, 'data', 'solomon_instances', f'{instance_name}.csv')
    output_dir = os.path.join(base_dir, 'data', 'solomon_instances', 'extended')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{instance_name}_extended.csv')

    # --- Bước 1: Đọc toàn bộ file gốc để tìm Depot DUE DATE ---
    raw_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)

    # Depot là dòng đầu tiên (CUST NO. = 1, pid = 0)
    depot_due_date = int(raw_rows[0]['DUE DATE'])
    nightlife_ready_threshold = depot_due_date * NIGHTLIFE_THRESHOLD

    # --- Bước 2: Gán CATEGORY + PRICE ---
    rows = []
    for row in raw_rows:
        cust_no = int(row['CUST NO.'])
        pid = cust_no - 1  # Remap: 1-based → 0-based (CUST 1 = Depot id=0)
        ready_time = int(row['READY TIME'])

        if pid == 0:
            cat = "depot"
            price = 0.0
        elif ready_time >= nightlife_ready_threshold:
            #  DETERMINISTIC: POI mở muộn → nightlife_wellness
            cat = NIGHTLIFE_CATEGORY
            rng = random.Random(pid)
            price = rng.choice(CATEGORY_PRICE_TIERS[NIGHTLIFE_CATEGORY])
        else:
            # Random gán theo phân phối xác suất (seed cố định)
            rng = random.Random(pid)
            cat = rng.choices(DAYTIME_CATEGORIES, weights=DAYTIME_WEIGHTS, k=1)[0]
            price = rng.choice(CATEGORY_PRICE_TIERS[cat])

        row['CATEGORY'] = cat
        row['PRICE'] = price
        rows.append(row)

    # --- Bước 3: Ghi ra CSV mới ---
    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Log category distribution ---
    cats = [r['CATEGORY'] for r in rows if r['CATEGORY'] != 'depot']
    dist = {c: cats.count(c) for c in ALL_CATEGORIES}
    print(f"  [OK] {instance_name}_extended.csv -- {len(rows)} nodes "
          f"(nightlife threshold: READY >= {nightlife_ready_threshold:.0f})")
    print(f"     Category distribution: {dist}")


def main():
    """Generate extended CSV cho tat ca 6 instances."""
    # Tim thu muc backend (chay tu backend/)
    base_dir = os.getcwd()
    if not os.path.exists(os.path.join(base_dir, 'data', 'solomon_instances')):
        # Thu len 1 cap
        base_dir = os.path.dirname(base_dir)

    print("=" * 70)
    print("  GENERATE EXTENDED CSV - 6 Categories + Nightlife (Time-Window Based)")
    print("=" * 70)

    for inst in INSTANCES:
        generate_extended(inst, base_dir)

    print(f"\n{'=' * 70}")
    print(f"  [OK] ALL {len(INSTANCES)} EXTENDED CSV FILES CREATED.")
    output_dir = os.path.join(base_dir, 'data', 'solomon_instances', 'extended')
    print(f"  Dir: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
