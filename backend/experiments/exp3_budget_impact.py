"""
Thí nghiệm 3: Đánh giá Tác động Ràng buộc Ngân sách (Budget Impact).

★ Mục tiêu:
  - Chứng minh hệ thống "liệu cơm gắp mắm": tự động điều chỉnh lộ trình
    theo ngân sách người dùng.
  - Backpacker (200k) → ưu tiên công viên/đi dạo (free) thay vì nhà hàng đắt.
  - Standard (500k) → cân bằng giữa trả phí và free.
  - Luxury (∞) → thoải mái chọn mọi điểm, bao gồm entertainment/food đắt đỏ.

★ CHIẾN LƯỢC:
  - Giữ cố định 1 profile (explorer — balanced nhưng thích nature)
  - Thay đổi 3 mức ngân sách trên instance RC201 (mixed TW, nhiều POI)
  - So sánh: Score, Số POI, Phân bổ category, Tổng chi phí
  - Chạy trên 1 instance: RC201 (mixed TW) để đánh giá tính thích ứng ngân sách

Usage:
    cd backend
    py -m experiments.exp3_budget_impact
    py -m experiments.exp3_budget_impact --instances C201 --num-runs 5
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_runner import run_batch, INSTANCE_CONFIGS, parse_instances_arg
from app.models.requests import UserPreferences

# ══════════════════════════════════════════════════════════════════════════════
#  Cấu hình
# ══════════════════════════════════════════════════════════════════════════════
INSTANCES = ["RC201"]
NUM_RUNS = 10
OUTPUT_DIR = "experiments/results/exp3_budget_impact"

# ── Profile cố định: Explorer (thích nature + entertainment, ít shopping) ────
FIXED_INTERESTS = {
    "history_culture": 3,
    "nature_parks": 4,
    "food_drink": 3,
    "shopping": 2,
    "entertainment": 4,
    "nightlife_wellness": 3,
}

# ── 3 mức ngân sách ──────────────────────────────────────────────────────────
# Mức backpacker: chỉ đủ cho ~2-3 điểm trả phí rẻ
# Mức standard:   đủ cho ~5-6 điểm trung bình
# Mức luxury:     vô hạn — không bị giới hạn bởi budget
BUDGET_TIERS = {
    "backpacker_200k": 200_000,
    "standard_500k":   500_000,
    "luxury_unlimited": 999_999_999,
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Thí nghiệm tác động ngân sách lên lộ trình du lịch"
    )
    parser.add_argument(
        "--instances",
        default=",".join(INSTANCES),
        help="Danh sách instance, cách nhau bởi dấu phẩy. Ví dụ: C201,RC201",
    )
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        instances = parse_instances_arg(args.instances)
    except ValueError as e:
        raise SystemExit(f"\nLỗi tham số --instances: {e}\n")

    total_runs = args.num_runs * len(instances) * len(BUDGET_TIERS)
    print("=" * 76)
    print("  THÍ NGHIỆM 3: TÁC ĐỘNG RÀNG BUỘC NGÂN SÁCH (BUDGET IMPACT)")
    print(f"  Instances: {instances}")
    print(f"  Budget tiers: {list(BUDGET_TIERS.keys())}")
    print(f"  Runs/tier/instance: {args.num_runs}")
    print(f"  Tổng số runs: {total_runs}")
    print("=" * 76)

    # all_results[tier_name] = [(instance, df), ...]
    all_results: dict[str, list[tuple[str, pd.DataFrame]]] = {
        t: [] for t in BUDGET_TIERS
    }

    for inst in instances:
        cfg = INSTANCE_CONFIGS[inst]

        for tier_name, budget in BUDGET_TIERS.items():
            print(f"\n{'#' * 76}")
            print(f"  INSTANCE: {inst} | BUDGET: {tier_name} ({budget:,} VND)")
            print(f"  Profile: Explorer (fixed)")
            print(f"{'#' * 76}")

            prefs = UserPreferences(
                instance_name=inst,
                budget=budget,
                start_time=0.0,
                end_time=cfg["depot_due"] / 60.0,
                start_node_id=0,
                interests=FIXED_INTERESTS,
            )
            df = run_batch(
                instance_name=inst,
                user_prefs=prefs,
                num_runs=args.num_runs,
                output_dir=args.output_dir,
                label=tier_name,
            )
            all_results[tier_name].append((inst, df))

    # ── Bảng tổng hợp ───────────────────────────────────────────────────────
    cat_cols = ["cat_history_culture", "cat_nature_parks", "cat_food_drink",
                "cat_shopping", "cat_entertainment", "cat_nightlife_wellness"]
    cat_short = ["Hist", "Nat", "Food", "Shop", "Ent", "Night"]

    print(f"\n\n{'=' * 130}")
    print("  BẢNG TỔNG HỢP — TÁC ĐỘNG NGÂN SÁCH LÊN LỘ TRÌNH")
    print(f"{'=' * 130}")

    # Header
    header_parts = [
        f"{'Budget Tier':<22}",
        f"{'Score':>8}", f"{'±Std':>7}",
        f"{'POIs':>6}", f"{'Cost':>12}",
        f"{'Free%':>7}",
    ]
    for s in cat_short:
        header_parts.append(f"{'│':>2}")
        header_parts.append(f"{s:>5}")
        header_parts.append(f"{'(%)':>5}")
    print("".join(header_parts))
    print("─" * 130)

    summary_rows = []
    for tier_name, budget in BUDGET_TIERS.items():
        items = all_results[tier_name]
        if not items:
            continue

        # Merge tất cả instances
        merged_df = pd.concat([df for _, df in items], ignore_index=True)

        score = merged_df['total_score'].mean()
        std = merged_df['total_score'].std()
        pois = merged_df['num_pois'].mean()
        cost = merged_df['total_cost'].mean()

        cats = {}
        for col in cat_cols:
            cats[col] = merged_df[col].mean() if col in merged_df.columns else 0

        total_pois = sum(cats.values())

        # Tính % free POI (nature_parks luôn free, một số shopping cũng free)
        free_pois = cats.get("cat_nature_parks", 0)
        free_pct = (free_pois / total_pois * 100) if total_pois > 0 else 0

        # Build row
        row_parts = [
            f"{tier_name:<22}",
            f"{score:8.1f}", f"{std:7.1f}",
            f"{pois:6.1f}", f"{cost:12,.0f}",
            f"{free_pct:6.1f}%",
        ]
        for col in cat_cols:
            v = cats[col]
            pct = (v / total_pois * 100) if total_pois > 0 else 0
            row_parts.append(f"{'│':>2}")
            row_parts.append(f"{v:5.1f}")
            row_parts.append(f"{pct:4.0f}%")
        print("".join(row_parts))

        summary_rows.append({
            "Budget_Tier": tier_name,
            "Budget_VND": budget,
            "Score_Avg": round(score, 2),
            "Score_Std": round(std, 2),
            "POIs_Avg": round(pois, 2),
            "Cost_Avg": round(cost, 2),
            "Free_Pct": round(free_pct, 2),
            **{f"Cat_{s}": round(cats[cat_cols[i]], 2) for i, s in enumerate(cat_short)},
            **{f"Cat_{s}%": round(
                (cats[cat_cols[i]] / total_pois * 100) if total_pois > 0 else 0, 2
            ) for i, s in enumerate(cat_short)},
        })

    # ── Phân tích chuyển dịch danh mục ────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  PHÂN TÍCH CHUYỂN DỊCH DANH MỤC KHI THAY ĐỔI NGÂN SÁCH")
    print(f"{'=' * 90}")

    if len(summary_rows) >= 2:
        bp = next((r for r in summary_rows if "backpacker" in r["Budget_Tier"]), None)
        lux = next((r for r in summary_rows if "luxury" in r["Budget_Tier"]), None)
        if bp and lux:
            print(f"\n  So sánh Backpacker vs Luxury:")
            print(f"    Score:     {bp['Score_Avg']:.1f} → {lux['Score_Avg']:.1f} (Δ={lux['Score_Avg']-bp['Score_Avg']:+.1f})")
            print(f"    POIs:      {bp['POIs_Avg']:.1f} → {lux['POIs_Avg']:.1f}")
            print(f"    Cost:      {bp['Cost_Avg']:,.0f} → {lux['Cost_Avg']:,.0f}")
            print(f"    Free%:     {bp['Free_Pct']:.1f}% → {lux['Free_Pct']:.1f}%")
            for s in cat_short:
                bp_pct = bp.get(f'Cat_{s}%', 0)
                lux_pct = lux.get(f'Cat_{s}%', 0)
                delta = lux_pct - bp_pct
                arrow = "↑" if delta > 1 else ("↓" if delta < -1 else "≈")
                print(f"    {s:>8}:  {bp_pct:5.1f}% → {lux_pct:5.1f}% ({arrow}{delta:+.1f}%)")

    # Lưu CSV
    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("experiments/results/summary", exist_ok=True)
    out_path = "experiments/results/summary/exp3_budget_impact.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  📄 Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
