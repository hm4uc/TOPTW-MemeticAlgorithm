"""
Thí nghiệm 1: So sánh MA vs Labadie (2012) trên 6 Solomon instances.

Chế độ: Fixed Scores (all weights = 1.0), Budget = ∞
Mục đích: Chứng minh MA cạnh tranh với state-of-the-art.

Usage:
    cd backend
    py -m experiments.exp1_benchmark
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_runner import run_batch, create_fixed_prefs, parse_instances_arg

# ══════════════════════════════════════════════════════════════════════════════
#  Cấu hình
# ══════════════════════════════════════════════════════════════════════════════
INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
NUM_RUNS = 30
OUTPUT_DIR = "experiments/results/exp1_benchmark"
FAIR_BENCHMARK_FLAGS = {"use_wait_penalty": False}

# ══════════════════════════════════════════════════════════════════════════════
#  Best-Known Solutions từ Labadie (2012)
#  ĐỌC TỪ PAPER labadie2012.pdf → Section "Computational Results"
#  Điền giá trị BKS (Total Score) cho từng instance.
# ══════════════════════════════════════════════════════════════════════════════
LABADIE_BKS = {
    "C101":  320,
    "C201":  870,
    "R101":  198,
    "R201":  797,
    "RC101": 219,
    "RC201": 795,
}

# GVNS results: (Min, Avg, Max, Gap%, Time_Avg)
LABADIE_GVNS = {
    "C101":  {"min": 320,  "avg": 320.0,  "max": 320,  "gap": 0.0, "time_avg": 0.2},
    "C201":  {"min": 850,  "avg": 850.0,  "max": 850,  "gap": 2.3, "time_avg": 0.1},
    "R101":  {"min": 197,  "avg": 197.0,  "max": 197,  "gap": 0.5, "time_avg": 0.2},
    "R201":  {"min": 765,  "avg": 775.6,  "max": 785,  "gap": 2.7, "time_avg": 6.7},
    "RC101": {"min": 219,  "avg": 219.0,  "max": 219,  "gap": 0.0, "time_avg": 2.1},
    "RC201": {"min": 778,  "avg": 784.0,  "max": 788,  "gap": 1.4, "time_avg": 3.7},
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark MA vs Labadie (2012) trên các Solomon instances"
    )
    parser.add_argument(
        "--instances",
        default=",".join(INSTANCES),
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
    print("  THÍ NGHIỆM 1: SO SÁNH MA vs LABADIE (2012)")
    print("  Chế độ: Fixed Scores | Budget = ∞")
    print(f"  Instances: {instances}")
    print(f"  Số lần chạy mỗi instance: {args.num_runs}")
    print("=" * 70)

    all_results = {}

    for inst in instances:
        print(f"\n{'#' * 70}")
        print(f"  INSTANCE: {inst}")
        print(f"{'#' * 70}")

        prefs = create_fixed_prefs(inst)
        df = run_batch(
            instance_name=inst,
            user_prefs=prefs,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            label="fixed",
            # Đồng bộ fairness benchmark với Labadie GVNS
            ablation_flags=FAIR_BENCHMARK_FLAGS,
        )
        all_results[inst] = df

    # ── Bảng tổng hợp cuối cùng ─────────────────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("  BẢNG TỔNG HỢP - SO SÁNH MA vs LABADIE GVNS (2012)")
    print(f"{'=' * 110}")
    print(f"{'Instance':<10} {'BKS':>6} | {'GVNS Best':>9} {'GVNS Avg':>9} {'GVNS Gap%':>9} {'GVNS T(s)':>9} | "
          f"{'MA Best':>9} {'MA Avg':>9} {'MA Std':>8} {'MA Gap%':>9} {'MA T(s)':>9}")
    print("-" * 110)

    for inst in instances:
        df = all_results[inst]
        bks = LABADIE_BKS[inst]
        gvns = LABADIE_GVNS[inst]

        ma_best = df['total_score'].max()
        ma_avg = df['total_score'].mean()
        ma_std = df['total_score'].std()
        ma_time = df['execution_time'].mean()
        ma_gap = (bks - ma_avg) / bks * 100 if bks > 0 else 0

        print(f"{inst:<10} {bks:>6} | "
              f"{gvns['min']:>9.1f} {gvns['avg']:>9.1f} {gvns['gap']:>8.1f}% {gvns['time_avg']:>9.1f} | "
              f"{ma_best:>9.1f} {ma_avg:>9.1f} {ma_std:>8.1f} {ma_gap:>8.1f}% {ma_time:>9.3f}")


if __name__ == "__main__":
    main()
