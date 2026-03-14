"""
Thí nghiệm 3: Ablation Study — Đánh giá đóng góp từng thành phần.

★ CHIẾN LƯỢC ★
  • Chạy trên TOÀN BỘ 6 instances (như exp1 và exp4)
  • Mỗi instance chạy 5 runs × 5 variants = 25 runs/instance
  • Chuẩn hóa score qua BKS → robust, không thiên lệch

Usage:
    cd backend
    py -m experiments.exp3_ablation
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_runner import run_batch, create_fixed_prefs, parse_instances_arg

# ══════════════════════════════════════════════════════════════════════════════
#  Cấu hình
# ══════════════════════════════════════════════════════════════════════════════
INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
NUM_RUNS = 5    # 5 runs × 6 instances × 5 variants = 150 runs tổng
OUTPUT_DIR = "experiments/results/exp3_ablation"
FAIR_BENCHMARK_FLAGS = {"use_wait_penalty": False}

# BKS để chuẩn hóa score
BKS = {
    "C101": 320, "C201": 870, "R101": 198,
    "R201": 797, "RC101": 219, "RC201": 795,
}

# ══════════════════════════════════════════════════════════════════════════════
#  Ablation Variants
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_VARIANTS = {
    "full_hga":           {},                                    # Đầy đủ (baseline)
    "no_smart_repair":    {"use_smart_repair": False},           # Simple Repair
    "no_insertion_mut":   {"use_insertion_mutation": False},      # Chỉ 2-opt + Swap
    "no_heuristic_init":  {"use_heuristic_init": False},         # 100% Random Init
    "no_diversity_check": {"use_diversity_check": False},        # Bỏ check duplicate
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study HGA trên nhiều Solomon instances"
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

    total_runs = args.num_runs * len(instances) * len(ABLATION_VARIANTS)
    print("=" * 70)
    print("  THÍ NGHIỆM 3: ABLATION STUDY (Normalized)")
    print(f"  Instances: {instances}")
    print(f"  Variants: {len(ABLATION_VARIANTS)} | Runs/variant/instance: {args.num_runs}")
    print(f"  Tổng số runs: {total_runs}")
    print("=" * 70)

    # Dict lưu: {variant_name: {instance: df, ...}}
    all_results = {}

    for variant_name, flags in ABLATION_VARIANTS.items():
        all_results[variant_name] = {}

        print(f"\n{'#' * 70}")
        print(f"  VARIANT: {variant_name}")
        print(f"  Flags: {flags if flags else 'FULL (no changes)'}")
        print(f"{'#' * 70}")

        for inst in instances:
            print(f"\n  --- {variant_name} on {inst} ---")
            prefs = create_fixed_prefs(inst)

            df = run_batch(
                instance_name=inst,
                user_prefs=prefs,
                num_runs=args.num_runs,
                output_dir=args.output_dir,
                # run_batch đã prefix instance vào filename, label chỉ cần tên variant
                label=variant_name,
                # Đồng bộ fairness benchmark với BKS trong toàn bộ exp3
                ablation_flags={**flags, **FAIR_BENCHMARK_FLAGS},
            )
            all_results[variant_name][inst] = df

    # ══════════════════════════════════════════════════════════════════════════
    #  Bảng tổng hợp — Normalized Score (% of BKS)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print("  BẢNG TỔNG HỢP — ABLATION STUDY (Normalized % of BKS)")
    print(f"{'=' * 100}")

    print(f"  {'Variant':<22} {'Norm%':>8} {'±Std':>8} "
          f"{'Time(s)':>10} {'Gens':>8} {'Diff%':>8}  Per-Instance Norm%")
    print(f"  {'-' * 90}")

    summary_rows = []
    baseline_norm = None

    for variant_name in ABLATION_VARIANTS:
        dfs = all_results[variant_name]
        norm_scores = []
        per_inst = {}

        for inst, df in dfs.items():
            bks = BKS[inst]
            inst_norm = df['total_score'].mean() / bks * 100
            norm_scores.extend((df['total_score'] / bks * 100).tolist())
            per_inst[inst] = inst_norm

        avg_norm = sum(norm_scores) / len(norm_scores)
        std_norm = pd.Series(norm_scores).std()
        avg_time = sum(df['execution_time'].mean() for df in dfs.values()) / len(dfs)
        avg_gens = sum(df['generations_run'].mean() for df in dfs.values()) / len(dfs)

        if variant_name == "full_hga":
            baseline_norm = avg_norm
            diff_str = "baseline"
        else:
            diff_pct = avg_norm - baseline_norm
            diff_str = f"{diff_pct:+.2f}%"

        per_str = " | ".join(f"{inst[:4]}={per_inst[inst]:.1f}" for inst in instances)

        print(f"  {variant_name:<22} {avg_norm:8.2f}% {std_norm:8.2f} "
              f"{avg_time:10.3f} {avg_gens:8.1f} {diff_str:>8}  [{per_str}]")

        summary_rows.append({
            "Variant": variant_name,
            "Norm_Score%": round(avg_norm, 2),
            "Norm_Std": round(std_norm, 2),
            "Time(s)": round(avg_time, 3),
            "Gens": round(avg_gens, 1),
            "Diff%": diff_str,
            **{f"Norm_{inst}": round(per_inst[inst], 2) for inst in instances},
        })

    # Lưu CSV summary
    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("experiments/results/summary", exist_ok=True)
    out_path = "experiments/results/summary/exp3_ablation_normalized.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  📄 Normalized summary saved to: {out_path}")


if __name__ == "__main__":
    main()
