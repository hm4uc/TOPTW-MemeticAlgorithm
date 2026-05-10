"""
Thí nghiệm 4: Ablation Study — Smart Repair & Local Search.

 Mục tiêu:
  Chứng minh "trái tim" của MA — Smart Repair + Greedy Refill — là thành phần
  THIẾT YẾU giúp thuật toán xử lý vi phạm Time Window hiệu quả.

 CHIẾN LƯỢC:
  So sánh 3 variants:
    1. full_ma          — Đầy đủ (Smart Repair + Greedy Refill)
    2. no_smart_repair  — Tắt Smart Repair → dùng Simple Repair (xóa cuối)
                          → Repair vẫn hoạt động nhưng kém thông minh
    3. no_local_search  — Tắt CẢ Repair lẫn Refill
                          → Con cái vi phạm TW bị phạt nặng trong fitness
                          → Thuật toán phải HOÀN TOÀN dựa vào GA operators

  Output kỳ vọng:
    • full_ma hội tụ NHANH hơn + score CAO hơn hẳn
    • no_local_search bị mắc kẹt ở local optima (đường cong hội tụ lẹt đẹt)
    • Feasibility Rate: full_ma  100%, no_local_search << 100%

Usage:
    cd backend
    py -m experiments.exp4_ablation_repair
    py -m experiments.exp4_ablation_repair --instances C101,R101 --num-runs 5
"""

import os
import sys
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_runner import run_batch, create_fixed_prefs, parse_instances_arg

# 
#  Cấu hình
# 
INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
NUM_RUNS = 5
OUTPUT_DIR = "experiments/results/exp4_ablation_repair"

# Fair benchmark: tắt wait penalty cho tất cả variants (đồng bộ BKS)
FAIR_BENCHMARK_FLAGS = {"use_wait_penalty": False}

# BKS để chuẩn hóa score
BKS = {
    "C101": 320, "C201": 870, "R101": 198,
    "R201": 797, "RC101": 219, "RC201": 795,
}

# 
#  Ablation Variants
# 
ABLATION_VARIANTS = {
    "full_ma": {},
    # Tắt Smart Repair → dùng Simple Repair (xóa cuối thay vì xóa POI kém nhất)
    # Greedy Refill vẫn hoạt động bình thường
    "no_smart_repair": {"use_smart_repair": False},
    # Tắt HOÀN TOÀN Repair + Refill → con cái vi phạm TW bị phạt fitness
    # Thuật toán phải HOÀN TOÀN dựa vào GA operators
    "no_local_search": {"use_local_search": False},
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study: Smart Repair & Local Search"
    )
    parser.add_argument(
        "--instances",
        default=",".join(INSTANCES),
        help="Danh sách instance, cách nhau bởi dấu phẩy.",
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
    print("=" * 76)
    print("  EXPERIMENT 4: ABLATION STUDY - SMART REPAIR & LOCAL SEARCH")
    print(f"  Instances: {instances}")
    print(f"  Variants: {list(ABLATION_VARIANTS.keys())}")
    print(f"  Runs per variant per instance: {args.num_runs}")
    print(f"  Total runs: {total_runs}")
    print("=" * 76)

    # all_results[variant] = {instance: df}
    all_results: dict[str, dict[str, pd.DataFrame]] = {
        v: {} for v in ABLATION_VARIANTS
    }

    for variant_name, flags in ABLATION_VARIANTS.items():
        print(f"\n{'#' * 76}")
        print(f"  VARIANT: {variant_name}")
        print(f"  Flags: {flags if flags else 'FULL (no changes)'}")
        print(f"{'#' * 76}")

        for inst in instances:
            print(f"\n  --- {variant_name} on {inst} ---")
            prefs = create_fixed_prefs(inst)

            df = run_batch(
                instance_name=inst,
                user_prefs=prefs,
                num_runs=args.num_runs,
                output_dir=args.output_dir,
                label=variant_name,
                ablation_flags={**flags, **FAIR_BENCHMARK_FLAGS},
            )
            all_results[variant_name][inst] = df

    # 
    #  Bảng 1: Per-Instance Normalized Score
    # 
    print(f"\n\n{'=' * 120}")
    print("  TABLE 1: NORMALIZED SCORE PER-INSTANCE (% of BKS)")
    print(f"{'=' * 120}")

    print(f"  {'Variant':<22} {'Overall%':>9} {'±Std':>8} "
          f"{'Time(s)':>10} {'Gens':>8} {'Diff%':>8}  Per-Instance Norm%")
    print(f"  {'-' * 110}")

    summary_rows = []
    baseline_norm = None
    convergence_data: dict[str, dict] = {}

    for variant_name in ABLATION_VARIANTS:
        dfs = all_results[variant_name]
        norm_scores = []
        per_inst = {}
        time_vals = []
        gen_vals = []

        for inst, df in dfs.items():
            bks = BKS[inst]
            inst_norm = df['total_score'].mean() / bks * 100
            norm_scores.extend((df['total_score'] / bks * 100).tolist())
            per_inst[inst] = inst_norm
            time_vals.extend(df['execution_time'].tolist())
            gen_vals.extend(df['generations_run'].tolist())

        avg_norm = sum(norm_scores) / len(norm_scores)
        std_norm = pd.Series(norm_scores).std()
        avg_time = float(pd.Series(time_vals).mean())
        avg_gens = float(pd.Series(gen_vals).mean())

        if variant_name == "full_ma":
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

        # - Thu thập convergence data cho so sánh -
        all_best_per_gen: dict[int, list[float]] = {}
        for inst, df in dfs.items():
            bks = BKS[inst]
            for _, row in df.iterrows():
                try:
                    log = json.loads(row["convergence_log"])
                except Exception:
                    continue
                for entry in log:
                    g = entry["gen"]
                    norm_best = entry["best_fitness"] / bks * 100
                    all_best_per_gen.setdefault(g, []).append(norm_best)

        convergence_data[variant_name] = all_best_per_gen

    # 
    #  Bảng 2: Convergence Comparison Summary
    # 
    print(f"\n\n{'=' * 90}")
    print("  TABLE 2: CONVERGENCE COMPARISON (Norm Score at key generations)")
    print(f"{'=' * 90}")

    key_gens = [1, 5, 10, 25, 50]
    header = f"  {'Variant':<22}"
    for g in key_gens:
        header += f" {'Gen'+str(g):>8}"
    print(header)
    print(f"  {'-' * 80}")

    for variant_name, gen_data in convergence_data.items():
        row = f"  {variant_name:<22}"
        for g in key_gens:
            if g in gen_data:
                val = sum(gen_data[g]) / len(gen_data[g])
                row += f" {val:8.2f}%"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # 
    #  Lưu results
    # 
    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("experiments/results/summary", exist_ok=True)
    out_path = "experiments/results/summary/exp4_ablation_repair.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n   Summary saved to: {out_path}")

    # Lưu convergence data dạng CSV
    conv_rows = []
    for variant_name, gen_data in convergence_data.items():
        for g, vals in sorted(gen_data.items()):
            conv_rows.append({
                "Variant": variant_name,
                "Generation": g,
                "Norm_Best_Mean": round(sum(vals) / len(vals), 4),
                "Norm_Best_Std": round(pd.Series(vals).std(), 4),
                "N_Runs": len(vals),
            })
    conv_df = pd.DataFrame(conv_rows)
    conv_path = os.path.join(args.output_dir, "convergence_comparison.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    conv_df.to_csv(conv_path, index=False)
    print(f"   Convergence data saved to: {conv_path}")


if __name__ == "__main__":
    main()
