"""
Thí nghiệm 5: Đánh giá Improved Insertion Heuristic (Time-Window Urgency).

★ MỤC ĐÍCH ★
  So sánh hiệu quả của Improved Insertion Heuristic (có urgency factor)
  so với Labadie ratio gốc (không urgency) trên toàn bộ 6 Solomon instances.

★ PHƯƠNG PHÁP ★
  • 2 variants: "with_urgency" (mặc định mới) vs "no_urgency" (Labadie gốc)
  • Chạy trên TOÀN BỘ 6 instances (C101, C201, R101, R201, RC101, RC201)
  • Mỗi instance chạy 10 runs × 2 variants = 20 runs/instance
  • Chuẩn hóa score qua BKS → Normalized Score (% of BKS)
  • So sánh: Total Score, Fitness, Thời gian chờ, Số POI, Thời gian chạy
  • Kiểm định thống kê Wilcoxon signed-rank test (p-value)

★ CHỈ SỐ ĐÁNH GIÁ ★
  1. Normalized Score (% of BKS) — chất lượng lời giải
  2. Total Wait Time — thời gian du khách phải chờ
  3. Number of POIs visited — độ phong phú lộ trình
  4. Convergence speed — tốc độ hội tụ (generations to best)
  5. Statistical significance — Wilcoxon p-value

Usage:
    cd backend
    py -m experiments.exp5_urgency
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_runner import run_batch, create_fixed_prefs

# ══════════════════════════════════════════════════════════════════════════════
#  Cấu hình
# ══════════════════════════════════════════════════════════════════════════════
INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
NUM_RUNS = 10    # 10 runs × 6 instances × 2 variants = 120 runs tổng
OUTPUT_DIR = "experiments/results/exp5_urgency"

# BKS để chuẩn hóa score (từ Labadie 2012)
BKS = {
    "C101": 320, "C201": 870, "R101": 198,
    "R201": 797, "RC101": 219, "RC201": 795,
}

# ══════════════════════════════════════════════════════════════════════════════
#  Urgency Variants
# ══════════════════════════════════════════════════════════════════════════════
VARIANTS = {
    "with_urgency": {"use_urgency": True},    # ★ CẢI TIẾN: có urgency factor
    "no_urgency":   {"use_urgency": False},   # Labadie ratio gốc (baseline)
}


def _extract_convergence_speed(df: pd.DataFrame) -> float:
    """
    Tính tốc độ hội tụ trung bình: thế hệ mà best fitness đạt 95% giá trị cuối.
    Thấp hơn = hội tụ nhanh hơn = tốt hơn.
    """
    speeds = []
    for _, row in df.iterrows():
        try:
            log = json.loads(row['convergence_log'])
            if not log:
                continue
            final_best = log[-1]['best_fitness']
            threshold = final_best * 0.95
            for entry in log:
                if entry['best_fitness'] >= threshold:
                    speeds.append(entry['gen'])
                    break
        except (json.JSONDecodeError, KeyError):
            continue
    return np.mean(speeds) if speeds else float('nan')


def main():
    total_runs = NUM_RUNS * len(INSTANCES) * len(VARIANTS)
    print("=" * 70)
    print("  THÍ NGHIỆM 5: IMPROVED INSERTION HEURISTIC (URGENCY)")
    print(f"  So sánh: with_urgency vs no_urgency (Labadie gốc)")
    print(f"  Instances: {INSTANCES}")
    print(f"  Runs/variant/instance: {NUM_RUNS}")
    print(f"  Tổng số runs: {total_runs}")
    print("=" * 70)

    # Dict lưu: {variant_name: {instance: df, ...}}
    all_results = {}

    for variant_name, flags in VARIANTS.items():
        all_results[variant_name] = {}

        print(f"\n{'#' * 70}")
        print(f"  VARIANT: {variant_name}")
        print(f"  Flags: {flags}")
        print(f"{'#' * 70}")

        for inst in INSTANCES:
            print(f"\n  --- {variant_name} on {inst} ---")
            prefs = create_fixed_prefs(inst)

            df = run_batch(
                instance_name=inst,
                user_prefs=prefs,
                num_runs=NUM_RUNS,
                output_dir=OUTPUT_DIR,
                label=f"{variant_name}_{inst}",
                ablation_flags={
                    **flags,
                    # Tắt wait penalty để so sánh công bằng với BKS
                    "use_wait_penalty": False,
                },
            )
            all_results[variant_name][inst] = df

    # ══════════════════════════════════════════════════════════════════════════
    #  BẢNG 1: So sánh chi tiết từng Instance
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 120}")
    print("  BẢNG 1: SO SÁNH CHI TIẾT — WITH URGENCY vs NO URGENCY (per Instance)")
    print(f"{'=' * 120}")

    header = (f"  {'Instance':<10} │ {'Variant':<16} │ "
              f"{'Best':>7} {'Avg':>8} {'Std':>7} {'Norm%':>7} │ "
              f"{'POIs':>5} {'Wait':>7} {'Time(s)':>8} {'Gens':>5} │ "
              f"{'Conv95%':>8}")
    print(header)
    print(f"  {'─' * 115}")

    detail_rows = []

    for inst in INSTANCES:
        for variant_name in VARIANTS:
            df = all_results[variant_name][inst]
            bks = BKS[inst]

            best_score = df['total_score'].max()
            avg_score = df['total_score'].mean()
            std_score = df['total_score'].std()
            norm_pct = avg_score / bks * 100
            avg_pois = df['num_pois'].mean()
            avg_wait = df['total_wait'].mean()
            avg_time = df['execution_time'].mean()
            avg_gens = df['generations_run'].mean()
            conv_speed = _extract_convergence_speed(df)

            print(f"  {inst:<10} │ {variant_name:<16} │ "
                  f"{best_score:7.1f} {avg_score:8.1f} {std_score:7.1f} "
                  f"{norm_pct:6.1f}% │ "
                  f"{avg_pois:5.1f} {avg_wait:7.1f} {avg_time:8.3f} "
                  f"{avg_gens:5.0f} │ {conv_speed:8.1f}")

            detail_rows.append({
                "Instance": inst,
                "Variant": variant_name,
                "BKS": bks,
                "Best_Score": best_score,
                "Avg_Score": round(avg_score, 2),
                "Std_Score": round(std_score, 2),
                "Norm%": round(norm_pct, 2),
                "Avg_POIs": round(avg_pois, 1),
                "Avg_Wait": round(avg_wait, 1),
                "Avg_Time(s)": round(avg_time, 3),
                "Avg_Gens": round(avg_gens, 1),
                "Conv_95%_Gen": round(conv_speed, 1),
            })

        print(f"  {'─' * 115}")

    # ══════════════════════════════════════════════════════════════════════════
    #  BẢNG 2: Tổng hợp Normalized Score (% of BKS)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print("  BẢNG 2: TỔNG HỢP — NORMALIZED SCORE (% of BKS)")
    print(f"{'=' * 100}")

    print(f"  {'Variant':<18} {'Avg Norm%':>10} {'±Std':>8} "
          f"{'Avg Wait':>10} {'Avg POIs':>10} {'Avg Time':>10} "
          f"{'Conv95%':>8}  │ Δ vs baseline")
    print(f"  {'─' * 95}")

    summary_rows = []
    baseline_norm = None
    baseline_wait = None

    for variant_name in VARIANTS:
        dfs = all_results[variant_name]
        all_norm_scores = []
        all_waits = []
        all_pois_counts = []
        all_times = []
        all_conv = []
        per_inst = {}

        for inst, df in dfs.items():
            bks = BKS[inst]
            inst_norm = df['total_score'].mean() / bks * 100
            all_norm_scores.extend((df['total_score'] / bks * 100).tolist())
            all_waits.extend(df['total_wait'].tolist())
            all_pois_counts.extend(df['num_pois'].tolist())
            all_times.extend(df['execution_time'].tolist())
            all_conv.append(_extract_convergence_speed(df))
            per_inst[inst] = inst_norm

        avg_norm = np.mean(all_norm_scores)
        std_norm = np.std(all_norm_scores)
        avg_wait = np.mean(all_waits)
        avg_pois = np.mean(all_pois_counts)
        avg_time = np.mean(all_times)
        avg_conv = np.nanmean(all_conv)

        if variant_name == "no_urgency":
            baseline_norm = avg_norm
            baseline_wait = avg_wait
            diff_str = "baseline"
        else:
            diff_norm = avg_norm - baseline_norm if baseline_norm else 0
            diff_wait = avg_wait - baseline_wait if baseline_wait else 0
            diff_str = f"Score {diff_norm:+.2f}%, Wait {diff_wait:+.1f}"

        print(f"  {variant_name:<18} {avg_norm:10.2f}% {std_norm:8.2f} "
              f"{avg_wait:10.1f} {avg_pois:10.1f} {avg_time:10.3f} "
              f"{avg_conv:8.1f}  │ {diff_str}")

        summary_rows.append({
            "Variant": variant_name,
            "Avg_Norm%": round(avg_norm, 2),
            "Std_Norm": round(std_norm, 2),
            "Avg_Wait": round(avg_wait, 1),
            "Avg_POIs": round(avg_pois, 1),
            "Avg_Time(s)": round(avg_time, 3),
            "Conv_95%_Gen": round(avg_conv, 1),
            **{f"Norm_{inst}": round(per_inst[inst], 2) for inst in INSTANCES},
        })

    # ══════════════════════════════════════════════════════════════════════════
    #  BẢNG 3: Kiểm định thống kê Wilcoxon (per Instance)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 90}")
    print("  BẢNG 3: KIỂM ĐỊNH THỐNG KÊ — WILCOXON SIGNED-RANK TEST")
    print(f"{'=' * 90}")

    try:
        from scipy.stats import wilcoxon

        print(f"  {'Instance':<10} │ {'Metric':<15} │ "
              f"{'Urgency Avg':>12} {'No-Urg Avg':>12} │ "
              f"{'W-stat':>8} {'p-value':>10} {'Significant?':>13}")
        print(f"  {'─' * 85}")

        stat_rows = []

        for inst in INSTANCES:
            df_urg = all_results["with_urgency"][inst]
            df_no  = all_results["no_urgency"][inst]

            # So sánh Total Score
            scores_urg = df_urg['total_score'].values
            scores_no  = df_no['total_score'].values

            diff = scores_urg - scores_no
            if np.all(diff == 0):
                w_stat, p_val = 0, 1.0
            else:
                try:
                    w_stat, p_val = wilcoxon(scores_urg, scores_no,
                                             alternative='greater')
                except ValueError:
                    w_stat, p_val = 0, 1.0

            sig = "✅ Yes (p<0.05)" if p_val < 0.05 else "❌ No"
            print(f"  {inst:<10} │ {'Total Score':<15} │ "
                  f"{scores_urg.mean():12.1f} {scores_no.mean():12.1f} │ "
                  f"{w_stat:8.1f} {p_val:10.4f} {sig:>13}")

            stat_rows.append({
                "Instance": inst,
                "Metric": "Total_Score",
                "Urgency_Avg": round(scores_urg.mean(), 2),
                "No_Urgency_Avg": round(scores_no.mean(), 2),
                "W_stat": round(w_stat, 2),
                "p_value": round(p_val, 4),
                "Significant": p_val < 0.05,
            })

            # So sánh Num POIs
            pois_urg = df_urg['num_pois'].values
            pois_no  = df_no['num_pois'].values

            diff = pois_urg - pois_no
            if np.all(diff == 0):
                w_stat, p_val = 0, 1.0
            else:
                try:
                    w_stat, p_val = wilcoxon(pois_urg, pois_no,
                                             alternative='greater')
                except ValueError:
                    w_stat, p_val = 0, 1.0

            sig = "✅ Yes (p<0.05)" if p_val < 0.05 else "❌ No"
            print(f"  {'':<10} │ {'Num POIs':<15} │ "
                  f"{pois_urg.mean():12.1f} {pois_no.mean():12.1f} │ "
                  f"{w_stat:8.1f} {p_val:10.4f} {sig:>13}")

            stat_rows.append({
                "Instance": inst,
                "Metric": "Num_POIs",
                "Urgency_Avg": round(pois_urg.mean(), 2),
                "No_Urgency_Avg": round(pois_no.mean(), 2),
                "W_stat": round(w_stat, 2),
                "p_value": round(p_val, 4),
                "Significant": p_val < 0.05,
            })

            print(f"  {'─' * 85}")

        # Tổng hợp (tất cả instances)
        all_scores_urg = pd.concat(
            [all_results["with_urgency"][inst]['total_score'] for inst in INSTANCES]
        ).values
        all_scores_no = pd.concat(
            [all_results["no_urgency"][inst]['total_score'] for inst in INSTANCES]
        ).values

        # Normalize trước khi test (vì scale khác nhau giữa instances)
        all_norm_urg = []
        all_norm_no = []
        for inst in INSTANCES:
            bks = BKS[inst]
            all_norm_urg.extend(
                (all_results["with_urgency"][inst]['total_score'] / bks * 100).tolist()
            )
            all_norm_no.extend(
                (all_results["no_urgency"][inst]['total_score'] / bks * 100).tolist()
            )

        all_norm_urg = np.array(all_norm_urg)
        all_norm_no = np.array(all_norm_no)

        diff = all_norm_urg - all_norm_no
        if np.all(diff == 0):
            w_stat, p_val = 0, 1.0
        else:
            try:
                w_stat, p_val = wilcoxon(all_norm_urg, all_norm_no,
                                         alternative='greater')
            except ValueError:
                w_stat, p_val = 0, 1.0

        sig = "✅ Yes (p<0.05)" if p_val < 0.05 else "❌ No"
        print(f"  {'OVERALL':<10} │ {'Norm Score%':<15} │ "
              f"{all_norm_urg.mean():12.2f}% {all_norm_no.mean():12.2f}% │ "
              f"{w_stat:8.1f} {p_val:10.4f} {sig:>13}")

        stat_rows.append({
            "Instance": "OVERALL",
            "Metric": "Norm_Score%",
            "Urgency_Avg": round(all_norm_urg.mean(), 2),
            "No_Urgency_Avg": round(all_norm_no.mean(), 2),
            "W_stat": round(w_stat, 2),
            "p_value": round(p_val, 4),
            "Significant": p_val < 0.05,
        })

        # Lưu statistical test results
        stat_df = pd.DataFrame(stat_rows)
        stat_path = os.path.join(OUTPUT_DIR, "wilcoxon_test_results.csv")
        stat_df.to_csv(stat_path, index=False)
        print(f"\n  📄 Wilcoxon results saved to: {stat_path}")

    except ImportError:
        print("  ⚠ scipy chưa cài đặt → bỏ qua kiểm định Wilcoxon.")
        print("    Cài: pip install scipy")

    # ══════════════════════════════════════════════════════════════════════════
    #  BẢNG 4: Win / Tie / Lose analysis (per Instance)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("  BẢNG 4: WIN / TIE / LOSE ANALYSIS")
    print(f"{'=' * 70}")

    wins, ties, losses = 0, 0, 0
    for inst in INSTANCES:
        urg_avg = all_results["with_urgency"][inst]['total_score'].mean()
        no_avg  = all_results["no_urgency"][inst]['total_score'].mean()

        if urg_avg > no_avg + 0.5:
            result = "🏆 WIN (urgency better)"
            wins += 1
        elif no_avg > urg_avg + 0.5:
            result = "❌ LOSE (no-urgency better)"
            losses += 1
        else:
            result = "🤝 TIE"
            ties += 1

        print(f"  {inst:<10}: urgency={urg_avg:.1f} vs baseline={no_avg:.1f}  → {result}")

    print(f"\n  Kết quả tổng: {wins} Wins, {ties} Ties, {losses} Losses "
          f"(trên {len(INSTANCES)} instances)")

    # ══════════════════════════════════════════════════════════════════════════
    #  Lưu CSV
    # ══════════════════════════════════════════════════════════════════════════
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Detail CSV
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(OUTPUT_DIR, "exp5_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"\n  📄 Detail saved to: {detail_path}")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "exp5_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  📄 Summary saved to: {summary_path}")

    # Copy to summary_to_reports
    os.makedirs("experiments/results/summary_to_reports", exist_ok=True)
    report_path = "experiments/results/summary_to_reports/exp5_urgency_summary.csv"
    summary_df.to_csv(report_path, index=False)
    print(f"  📄 Report summary saved to: {report_path}")

    print(f"\n{'=' * 70}")
    print("  ★ THÍ NGHIỆM 5 HOÀN TẤT ★")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

