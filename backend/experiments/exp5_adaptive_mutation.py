"""
Thí nghiệm 5: So sánh Adaptive-Lite Mutation vs Static Mutation.

Mục tiêu:
  - Đối sánh trực tiếp cơ chế mutation mới (2 tầng) với mutation tĩnh cũ.
  - Dùng cùng seed/run_id để đảm bảo fair comparison.

Usage:
    cd backend
    py -m experiments.exp5_adaptive_mutation
    py -m experiments.exp5_adaptive_mutation --instances C101,R101 --num-runs 3
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_runner import run_batch, create_fixed_prefs, parse_instances_arg

INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
NUM_RUNS = 5
OUTPUT_DIR = "experiments/results/exp5_adaptive_mutation"

# Fair benchmark theo BKS: tắt wait penalty cho cả 2 phía
FAIR_BENCHMARK_FLAGS = {"use_wait_penalty": False}

BKS = {
    "C101": 320,
    "C201": 870,
    "R101": 198,
    "R201": 797,
    "RC101": 219,
    "RC201": 795,
}

VARIANTS = {
    "static_mutation": {"use_adaptive_mutation": False},
    "adaptive_lite_2tier": {"use_adaptive_mutation": True},
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="So sánh Adaptive-Lite mutation với static mutation"
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

    total_runs = args.num_runs * len(instances) * len(VARIANTS)
    print("=" * 76)
    print("  THÍ NGHIỆM 5: ADAPTIVE-LITE MUTATION vs STATIC MUTATION")
    print(f"  Instances: {instances}")
    print(f"  Runs/variant/instance: {args.num_runs}")
    print(f"  Tổng số runs: {total_runs}")
    print("=" * 76)

    all_results: dict[str, dict[str, pd.DataFrame]] = {v: {} for v in VARIANTS}

    for variant_name, flags in VARIANTS.items():
        print(f"\n{'#' * 76}")
        print(f"  VARIANT: {variant_name}")
        print(f"  Flags: {flags}")
        print(f"{'#' * 76}")

        for inst in instances:
            prefs = create_fixed_prefs(inst)
            print(f"\n  --- {variant_name} on {inst} ---")
            df = run_batch(
                instance_name=inst,
                user_prefs=prefs,
                num_runs=args.num_runs,
                output_dir=args.output_dir,
                label=variant_name,
                ablation_flags={**FAIR_BENCHMARK_FLAGS, **flags},
            )
            all_results[variant_name][inst] = df

    # ── Summary per instance ───────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  BẢNG 1: ĐỐI SÁNH PER-INSTANCE")
    print(f"{'=' * 120}")
    print(
        f"  {'Instance':<8} {'Variant':<22} {'Norm%':>9} {'Score':>9} {'Wait':>9} {'Time(s)':>9} {'Gens':>8}"
    )
    print(f"  {'-' * 112}")

    detail_rows = []
    for inst in instances:
        for variant_name in VARIANTS:
            df = all_results[variant_name][inst]
            norm_vals = (df["total_score"] / BKS[inst] * 100.0)
            row = {
                "Instance": inst,
                "Variant": variant_name,
                "Norm%": round(norm_vals.mean(), 2),
                "Score": round(df["total_score"].mean(), 2),
                "Wait": round(df["total_wait"].mean(), 2),
                "Time(s)": round(df["execution_time"].mean(), 3),
                "Gens": round(df["generations_run"].mean(), 2),
            }
            detail_rows.append(row)
            print(
                f"  {row['Instance']:<8} {row['Variant']:<22} {row['Norm%']:>8.2f}% {row['Score']:>9.2f}"
                f" {row['Wait']:>9.2f} {row['Time(s)']:>9.3f} {row['Gens']:>8.2f}"
            )

    # ── Overall delta ──────────────────────────────────────────────────────
    static_dfs = [all_results["static_mutation"][inst] for inst in instances]
    adapt_dfs = [all_results["adaptive_lite_2tier"][inst] for inst in instances]

    static_norm = []
    adapt_norm = []
    for inst, s_df, a_df in zip(instances, static_dfs, adapt_dfs):
        static_norm.extend((s_df["total_score"] / BKS[inst] * 100.0).tolist())
        adapt_norm.extend((a_df["total_score"] / BKS[inst] * 100.0).tolist())

    static_wait_vals = [v for df in static_dfs for v in df["total_wait"].tolist()]
    adapt_wait_vals = [v for df in adapt_dfs for v in df["total_wait"].tolist()]
    static_time_vals = [v for df in static_dfs for v in df["execution_time"].tolist()]
    adapt_time_vals = [v for df in adapt_dfs for v in df["execution_time"].tolist()]
    static_gens_vals = [v for df in static_dfs for v in df["generations_run"].tolist()]
    adapt_gens_vals = [v for df in adapt_dfs for v in df["generations_run"].tolist()]

    static_wait_mean = float(pd.Series(static_wait_vals).mean())
    adapt_wait_mean = float(pd.Series(adapt_wait_vals).mean())
    static_time_mean = float(pd.Series(static_time_vals).mean())
    adapt_time_mean = float(pd.Series(adapt_time_vals).mean())
    static_gens_mean = float(pd.Series(static_gens_vals).mean())
    adapt_gens_mean = float(pd.Series(adapt_gens_vals).mean())

    overall = {
        "static_norm": round(float(pd.Series(static_norm).mean()), 2),
        "adaptive_norm": round(float(pd.Series(adapt_norm).mean()), 2),
        "delta_norm": round(float(pd.Series(adapt_norm).mean() - pd.Series(static_norm).mean()), 2),
        "static_wait": round(static_wait_mean, 2),
        "adaptive_wait": round(adapt_wait_mean, 2),
        "delta_wait": round(adapt_wait_mean - static_wait_mean, 2),
        "static_time": round(static_time_mean, 3),
        "adaptive_time": round(adapt_time_mean, 3),
        "delta_time": round(adapt_time_mean - static_time_mean, 3),
        "static_gens": round(static_gens_mean, 2),
        "adaptive_gens": round(adapt_gens_mean, 2),
        "delta_gens": round(adapt_gens_mean - static_gens_mean, 2),
    }

    print(f"\n\n{'=' * 120}")
    print("  BẢNG 2: KẾT QUẢ TỔNG HỢP OVERALL")
    print(f"{'=' * 120}")
    print(f"  Norm%      : static={overall['static_norm']:.2f}% | adaptive={overall['adaptive_norm']:.2f}% | Δ={overall['delta_norm']:+.2f}%")
    print(f"  Wait       : static={overall['static_wait']:.2f}   | adaptive={overall['adaptive_wait']:.2f}   | Δ={overall['delta_wait']:+.2f}")
    print(f"  Time(s)    : static={overall['static_time']:.3f}  | adaptive={overall['adaptive_time']:.3f}  | Δ={overall['delta_time']:+.3f}")
    print(f"  Gens       : static={overall['static_gens']:.2f}  | adaptive={overall['adaptive_gens']:.2f}  | Δ={overall['delta_gens']:+.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(args.output_dir, "exp5_detail.csv")
    detail_df.to_csv(detail_path, index=False)

    summary_df = pd.DataFrame([overall])
    summary_path = os.path.join(args.output_dir, "exp5_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  📄 Detail saved to: {detail_path}")
    print(f"  📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()



