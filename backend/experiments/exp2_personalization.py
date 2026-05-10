"""
Thí nghiệm 2: Đánh giá Giá trị Cá nhân hóa (Personalization Value).

 CHIẾN LƯỢC 
  • Chạy trên RC201 (Mixed TW — nhiều POI, đa dạng)
  • Budget nới rộng (2_000_000) → isolate preference from financial constraints
  • 5 profiles: baseline, history_buff, foodie, explorer, shopper
  • 10 runs per profile

Usage:
    cd backend
    py -m experiments.exp2_personalization
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_runner import run_batch, INSTANCE_CONFIGS, parse_instances_arg
from app.models.requests import UserPreferences

# 
#  Cấu hình
# 
INSTANCE = "RC201"
NUM_RUNS = 10
OUTPUT_DIR = "experiments/results/exp2_personalization"

# 5 user profiles khác nhau
USER_PROFILES = {
    "baseline": {
        "history_culture": 3, "nature_parks": 3,
        "food_drink": 3, "shopping": 3, "entertainment": 3,
        "nightlife_wellness": 3,
    },
    "history_buff": {
        "history_culture": 5, "nature_parks": 2,
        "food_drink": 3, "shopping": 1, "entertainment": 1,
        "nightlife_wellness": 1,
    },
    "foodie": {
        "history_culture": 1, "nature_parks": 2,
        "food_drink": 5, "shopping": 1, "entertainment": 3,
        "nightlife_wellness": 2,
    },
    "explorer": {
        "history_culture": 3, "nature_parks": 5,
        "food_drink": 2, "shopping": 1, "entertainment": 4,
        "nightlife_wellness": 3,
    },
    "shopper": {
        "history_culture": 1, "nature_parks": 1,
        "food_drink": 3, "shopping": 5, "entertainment": 2,
        "nightlife_wellness": 2,
    },
}

# Budget nới rộng → isolate preference effect
BUDGET = 2_000_000


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Thí nghiệm giá trị cá nhân hóa trên một hoặc nhiều instance"
    )
    parser.add_argument(
        "--instances",
        default=INSTANCE,
        help="Danh sách instance, cách nhau bởi dấu phẩy. Ví dụ: C101,R101",
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

    print("=" * 70)
    print("  EXPERIMENT 2: PERSONALIZATION VALUE EVALUATION")
    print(f"  Instances: {instances} | Budget: {BUDGET:,}")
    print(f"  Profiles: {len(USER_PROFILES)} | Runs/profile/instance: {args.num_runs}")
    print("=" * 70)

    # all_results[profile] = [df_instance_1, df_instance_2, ...]
    all_results = {k: [] for k in USER_PROFILES}

    for inst in instances:
        cfg = INSTANCE_CONFIGS[inst]

        for profile_name, interests in USER_PROFILES.items():
            print(f"\n{'#' * 70}")
            print(f"  INSTANCE: {inst} | PROFILE: {profile_name}")
            print(f"  Interests: {interests}")
            print(f"{'#' * 70}")

            prefs = UserPreferences(
                instance_name=inst,
                budget=BUDGET,
                start_time=0.0,
                end_time=cfg["depot_due"] / 60.0,  # Solomon time → giờ
                start_node_id=0,
                interests=interests,
            )
            df = run_batch(
                instance_name=inst,
                user_prefs=prefs,
                num_runs=args.num_runs,
                output_dir=args.output_dir,
                label=profile_name,
            )
            all_results[profile_name].append(df)

    # - Bảng tổng hợp -
    cat_cols = ["cat_history_culture", "cat_nature_parks", "cat_food_drink",
                "cat_shopping", "cat_entertainment", "cat_nightlife_wellness"]
    cat_short = ["Hist", "Nat", "Food", "Shop", "Ent", "Night"]

    print(f"\n\n{'=' * 110}")
    print("  SUMMARY TABLE - CATEGORY DISTRIBUTION BY PROFILE (Count + %)")
    print(f"{'=' * 110}")

    # Header
    header_parts = [f"{'Profile':<15}", f"{'Score':>8}", f"{'±Std':>6}",
                    f"{'POIs':>5}", f"{'Cost':>10}"]
    for s in cat_short:
        header_parts.append(f"{'│':>2}")
        header_parts.append(f"{s:>5}")
        header_parts.append(f"{'(%)':>5}")
    print("".join(header_parts))
    print("-" * 110)

    for name, df_list in all_results.items():
        if not df_list:
            continue
        df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, ignore_index=True)
        score = df['total_score'].mean()
        std = df['total_score'].std()
        pois = df['num_pois'].mean()
        cost = df['total_cost'].mean()

        cats = {}
        for col in cat_cols:
            cats[col] = df[col].mean() if col in df.columns else 0

        total_pois = sum(cats.values())

        # Build row
        row_parts = [f"{name:<15}", f"{score:8.1f}", f"{std:6.1f}",
                     f"{pois:5.1f}", f"{cost:10.0f}"]
        for col in cat_cols:
            v = cats[col]
            pct = (v / total_pois * 100) if total_pois > 0 else 0
            row_parts.append(f"{'│':>2}")
            row_parts.append(f"{v:5.1f}")
            row_parts.append(f"{pct:4.0f}%")
        print("".join(row_parts))

    print(f"\n   Instances: {instances} - Budget: {BUDGET:,}")
    print(f"   % Ratio = (Category POIs / Total POIs in route) * 100")


if __name__ == "__main__":
    main()
