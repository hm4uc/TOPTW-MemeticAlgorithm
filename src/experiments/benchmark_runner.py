"""
Benchmark Runner — Chạy batch thí nghiệm tự động + lưu kết quả CSV.

Cung cấp:
  • run_single()  — Chạy 1 lần MA, trả về dict metrics.
  • run_batch()   — Chạy N lần, lưu CSV, in summary.
  • create_fixed_prefs() — Tạo UserPreferences cho chế độ Fixed Score.

Usage:
    cd backend
    py -m experiments.benchmark_runner
"""

import os
import sys
import json
import argparse
import pandas as pd

# - Thêm backend vào path -
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.data_loader import load_solomon_instance
from services.algorithm.ma_engine import MemeticAlgorithm
from models.requests import UserPreferences


# 
#  Depot Time Windows cho từng Instance
# 
INSTANCE_CONFIGS = {
    "C101":  {"depot_due": 1236, "service_time": 90},
    "C201":  {"depot_due": 3390, "service_time": 90},
    "R101":  {"depot_due": 230,  "service_time": 10},
    "R201":  {"depot_due": 1000, "service_time": 10},
    "RC101": {"depot_due": 240,  "service_time": 10},
    "RC201": {"depot_due": 960,  "service_time": 10},
}

# 6 categories chuẩn
ALL_CATEGORIES = [
    "history_culture", "nature_parks", "food_drink", "shopping",
    "entertainment", "nightlife_wellness"
]


def parse_instances_arg(instances_arg: str) -> list[str]:
    """
    Parse và validate chuỗi --instances (vd: "C101,R101").

    Raises
    ------
    ValueError
        Nếu rỗng hoặc chứa mã instance không hợp lệ.
    """
    valid_instances = list(INSTANCE_CONFIGS.keys())

    raw_items = [s.strip().upper() for s in instances_arg.split(",")]
    instances = [s for s in raw_items if s]

    if not instances:
        raise ValueError(
            "--instances đang rỗng. Ví dụ hợp lệ: --instances C101,R101"
        )

    invalid = [s for s in instances if s not in INSTANCE_CONFIGS]
    if invalid:
        raise ValueError(
            "Mã instance không hợp lệ: "
            f"{', '.join(invalid)}. "
            f"Các mã hợp lệ: {', '.join(valid_instances)}"
        )

    # Giữ thứ tự người dùng nhập, loại trùng để tránh chạy lặp không cần thiết.
    deduped: list[str] = []
    seen = set()
    for inst in instances:
        if inst not in seen:
            deduped.append(inst)
            seen.add(inst)

    return deduped


def create_fixed_prefs(instance_name: str) -> UserPreferences:
    """
    Tạo UserPreferences cho chế độ Fixed Score (so sánh Labadie).

    Tất cả interests = 3 → weight = 1.0 sau normalization
    → score = base_score × 1.0 = DEMAND gốc trong Solomon.
    Budget = vô hạn (Labadie không có budget constraint).
    Time window = depot time window.
    """
    cfg = INSTANCE_CONFIGS[instance_name]
    return UserPreferences(
        budget=999_999_999,
        start_time=0.0,
        end_time=cfg["depot_due"] / 60.0,   # Solomon time units → giờ
        start_node_id=0,
        interests={k: 3 for k in ALL_CATEGORIES},
    )


def run_single(
    instance_name: str,
    user_prefs: UserPreferences,
    pois: list = None,
    ablation_flags: dict = None,
    ga_params: dict = None,
) -> dict:
    """
    Chạy 1 lần MA, trả về dict metrics.

    Parameters
    ----------
    instance_name : str
        Tên instance Solomon (VD: "C101").
    user_prefs : UserPreferences
        Sở thích người dùng.
    ablation_flags : dict, optional
        Flags tắt/bật thành phần (VD: {"use_smart_repair": False}).
    ga_params : dict, optional
        Override tham số GA (VD: {"population_size": 100}).

    Returns
    -------
    dict
        Metrics của lần chạy.
    """
    if pois is None:
        pois = load_solomon_instance(instance_name)
    flags = ablation_flags or {}
    params = ga_params or {}

    ma = MemeticAlgorithm(
        user_prefs=user_prefs,
        pois=pois,
        instance_name=instance_name,
        **flags,
        **params,
    )
    response = ma.run()

    best = ma.best_individual
    route_ids = [p.id for p in best.route]

    # Đếm category distribution trong route
    cat_counts = {cat: 0 for cat in ALL_CATEGORIES}
    for p in best.route[1:-1]:  # Bỏ 2 depot
        if p.category in cat_counts:
            cat_counts[p.category] += 1

    return {
        "total_score": best.total_score,
        "fitness": best.fitness,
        "num_pois": len(best.route) - 2,
        "total_distance": response.total_distance,
        "total_wait": best.total_wait,
        "total_cost": best.total_cost,
        "total_duration": response.total_duration,
        "execution_time": response.execution_time,
        "generations_run": ma.actual_gens,
        "route_ids": json.dumps(route_ids),
        "convergence_log": json.dumps(ma.convergence_log),
        **{f"cat_{cat}": cnt for cat, cnt in cat_counts.items()},
    }


def run_batch(
    instance_name: str,
    user_prefs: UserPreferences,
    pois: list = None,
    num_runs: int = 30,
    output_dir: str = "experiments/results",
    label: str = "",
    ablation_flags: dict = None,
    ga_params: dict = None,
) -> pd.DataFrame:
    """
    Chạy N lần MA, lưu kết quả CSV, in summary.

    Returns
    -------
    pd.DataFrame
        DataFrame với kết quả tất cả các lần chạy.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i in range(num_runs):
        print(f"\n{'=' * 60}")
        print(f"  RUN {i + 1}/{num_runs} — {instance_name} [{label}]")
        print(f"{'=' * 60}")

        # Thêm seed cố định theo run_id để fair comparison giữa các param
        import random
        random.seed(42 + i)

        result = run_single(instance_name, user_prefs, pois, ablation_flags, ga_params)
        result["run_id"] = i + 1
        result["instance"] = instance_name
        result["label"] = label
        results.append(result)

    df = pd.DataFrame(results)

    # Lưu CSV
    safe_label = label.replace(" ", "_") if label else "default"
    filename = f"{instance_name}_{safe_label}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    # - Print Summary -
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {instance_name} [{label}] ({num_runs} runs)")
    print(f"{'=' * 60}")

    summary_cols = ["total_score", "fitness", "num_pois",
                    "execution_time", "total_wait", "total_distance"]
    for col in summary_cols:
        if col in df.columns:
            vals = df[col]
            print(f"  {col:20s}: "
                  f"Mean={vals.mean():10.2f}  "
                  f"Std={vals.std():8.2f}  "
                  f"Best={vals.max():10.2f}  "
                  f"Worst={vals.min():10.2f}")

    # Category distribution (average)
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    if cat_cols:
        print(f"\n  Category Distribution (avg):")
        for col in cat_cols:
            print(f"    {col}: {df[col].mean():.1f}")

    print(f"\n   Saved to: {filepath}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark runner cho MA-TOPTW")
    parser.add_argument("--instance", default="C101", choices=list(INSTANCE_CONFIGS.keys()))
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--label", default="cli")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    print(f"=== QUICK TEST: {args.num_runs} run(s) on {args.instance} (fixed scores) ===")
    prefs = create_fixed_prefs(args.instance)

    if args.num_runs == 1:
        result = run_single(args.instance, prefs)
        print(f"\nResult: score={result['total_score']:.2f}, "
              f"pois={result['num_pois']}, "
              f"time={result['execution_time']:.3f}s")
    else:
        run_batch(
            instance_name=args.instance,
            user_prefs=prefs,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            label=args.label,
        )
