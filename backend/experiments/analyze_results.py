"""Analyze Results — tự động tổng hợp kết quả theo naming/flow hiện tại.

Script này đọc CSV từ các thư mục `experiments/results/exp*` và xuất summary,
hạn chế hardcode để giảm thao tác chỉnh tay khi naming thay đổi nhẹ.
"""

from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, cast
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
RESULTS_DIR = BASE_DIR / "experiments" / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

LABADIE_BKS = {
    "C101": 320, "C201": 870, "R101": 198,
    "R201": 797, "RC101": 219, "RC201": 795,
}

LABADIE_GVNS = {
    "C101": {"min": 320, "avg": 320.0, "max": 320, "gap": 0.0, "time": 0.2},
    "C201": {"min": 850, "avg": 850.0, "max": 850, "gap": 2.3, "time": 0.1},
    "R101": {"min": 197, "avg": 197.0, "max": 197, "gap": 0.5, "time": 0.2},
    "R201": {"min": 765, "avg": 775.6, "max": 785, "gap": 2.7, "time": 6.7},
    "RC101": {"min": 219, "avg": 219.0, "max": 219, "gap": 0.0, "time": 2.1},
    "RC201": {"min": 778, "avg": 784.0, "max": 788, "gap": 1.4, "time": 3.7},
}

INSTANCES = list(LABADIE_BKS.keys())
SENSITIVITY_PARAMS = {"population_size", "mutation_rate", "tournament_k", "stagnation_limit"}


def _instance_from_stem(stem: str) -> str | None:
    head = stem.split("_", 1)[0]
    return head if head in LABADIE_BKS else None


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  ⚠ Bỏ qua {path.name}: {e}")
        return None


def analyze_exp1_benchmark() -> pd.DataFrame:
    print("\n" + "=" * 110)
    print("  THÍ NGHIỆM 1: SO SÁNH HGA vs LABADIE GVNS")
    print("=" * 110)

    exp_dir = RESULTS_DIR / "exp1_benchmark"
    rows = []

    for csv_path in sorted(exp_dir.glob("*_fixed.csv")):
        inst = _instance_from_stem(csv_path.stem)
        if not inst:
            continue
        df = _safe_read_csv(csv_path)
        if df is None or df.empty:
            continue

        bks = LABADIE_BKS[inst]
        gvns = LABADIE_GVNS[inst]
        hga_avg = df["total_score"].mean()

        rows.append({
            "Instance": inst,
            "BKS": bks,
            "GVNS_Best": gvns["min"],
            "GVNS_Avg": gvns["avg"],
            "GVNS_Gap%": gvns["gap"],
            "GVNS_Time(s)": gvns["time"],
            "HGA_Best": df["total_score"].max(),
            "HGA_Avg": round(hga_avg, 2),
            "HGA_Std": round(df["total_score"].std(), 2),
            "HGA_Gap%": round((bks - hga_avg) / bks * 100, 2),
            "HGA_Time(s)": round(df["execution_time"].mean(), 3),
            "Runs": len(df),
        })

    summary_df = pd.DataFrame(rows).sort_values("Instance") if rows else pd.DataFrame()
    out_path = SUMMARY_DIR / "exp1_benchmark_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  📄 Saved: {out_path}")
    return summary_df


def analyze_exp2_personalization() -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("  THÍ NGHIỆM 2: GIÁ TRỊ CÁ NHÂN HÓA")
    print("=" * 100)

    exp_dir = RESULTS_DIR / "exp2_personalization"
    grouped: DefaultDict[str, list[pd.DataFrame]] = defaultdict(list)

    for csv_path in sorted(exp_dir.glob("*.csv")):
        inst = _instance_from_stem(csv_path.stem)
        if not inst:
            continue
        profile = csv_path.stem.split("_", 1)[1]
        df = _safe_read_csv(csv_path)
        if df is not None and not df.empty:
            grouped[profile].append(df)

    cat_cols = [
        "cat_history_culture", "cat_nature_parks", "cat_food_drink",
        "cat_shopping", "cat_entertainment", "cat_nightlife_wellness",
    ]
    cat_short = ["Hist", "Nat", "Food", "Shop", "Ent", "Night"]

    rows = []
    for profile, dfs in sorted(grouped.items()):
        if not dfs:
            continue
        merged_df = cast(pd.DataFrame, pd.concat(dfs, ignore_index=True))
        row = {
            "Profile": profile,
            "Score_Avg": round(float(merged_df["total_score"].mean()), 2),
            "Score_Std": round(float(merged_df["total_score"].std()), 2),
            "POIs_Avg": round(float(merged_df["num_pois"].mean()), 2),
            "Cost_Avg": round(float(merged_df["total_cost"].mean()), 2),
            "Rows": len(merged_df),
        }
        total_cat = 0.0
        means = []
        for col in cat_cols:
            v = float(merged_df[col].mean()) if col in merged_df.columns else 0.0
            means.append(v)
            total_cat += v

        for short, v in zip(cat_short, means):
            pct = (v / total_cat * 100.0) if total_cat > 0 else 0.0
            row[short] = round(v, 2)
            row[f"{short}%"] = round(pct, 2)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = SUMMARY_DIR / "exp2_personalization_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  📄 Saved: {out_path}")
    return summary_df


def analyze_exp3_budget_impact() -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("  THÍ NGHIỆM 3: TÁC ĐỘNG NGÂN SÁCH")
    print("=" * 100)

    exp_dir = RESULTS_DIR / "exp3_budget_impact"
    budget_tiers = {"backpacker_200k", "standard_500k", "luxury_unlimited"}
    cat_cols = [
        "cat_history_culture", "cat_nature_parks", "cat_food_drink",
        "cat_shopping", "cat_entertainment", "cat_nightlife_wellness",
    ]
    grouped: DefaultDict[str, list[pd.DataFrame]] = defaultdict(list)

    for csv_path in sorted(exp_dir.glob("*.csv")):
        stem = csv_path.stem
        inst = _instance_from_stem(stem)
        if not inst:
            continue
        tier = stem.split("_", 1)[1]
        if tier not in budget_tiers:
            continue
        df = _safe_read_csv(csv_path)
        if df is not None and not df.empty:
            grouped[tier].append(df)

    rows = []
    for tier in sorted(grouped.keys()):
        merged = cast(pd.DataFrame, pd.concat(grouped[tier], ignore_index=True))
        row = {
            "Budget_Tier": tier,
            "Score_Avg": round(float(merged["total_score"].mean()), 2),
            "Score_Std": round(float(merged["total_score"].std()), 2),
            "POIs_Avg": round(float(merged["num_pois"].mean()), 2),
            "Cost_Avg": round(float(merged["total_cost"].mean()), 2),
            "Rows": len(merged),
        }
        for col in cat_cols:
            row[col] = round(float(merged[col].mean()), 2) if col in merged.columns else 0.0
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = SUMMARY_DIR / "exp3_budget_impact_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  📄 Saved: {out_path}")
    return summary_df


def analyze_exp4_ablation_repair() -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("  THÍ NGHIỆM 4: ABLATION STUDY — SMART REPAIR & LOCAL SEARCH")
    print("=" * 100)

    exp_dir = RESULTS_DIR / "exp4_ablation_repair"
    variants = {"full_hga", "no_smart_repair", "no_local_search"}
    grouped: DefaultDict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)

    for csv_path in sorted(exp_dir.glob("*.csv")):
        stem = csv_path.stem
        inst = _instance_from_stem(stem)
        if not inst:
            continue
        variant = stem.split("_", 1)[1]
        if variant not in variants:
            continue
        df = _safe_read_csv(csv_path)
        if df is not None and not df.empty:
            grouped[variant].append((inst, df))

    rows = []
    baseline_norm = None
    for variant in ["full_hga", "no_smart_repair", "no_local_search"]:
        if variant not in grouped:
            continue
        norm_scores = []
        time_vals = []
        gen_vals = []

        for inst, df in grouped[variant]:
            bks = LABADIE_BKS[inst]
            norm_scores.extend((df["total_score"] / bks * 100.0).tolist())
            time_vals.extend(df["execution_time"].tolist())
            gen_vals.extend(df["generations_run"].tolist())

        avg_norm = float(pd.Series(norm_scores).mean())
        if variant == "full_hga":
            baseline_norm = avg_norm
            diff_str = "baseline"
        elif baseline_norm is not None:
            diff_str = f"{avg_norm - baseline_norm:+.2f}%"
        else:
            diff_str = "N/A"

        rows.append({
            "Variant": variant,
            "Norm_Score%": round(avg_norm, 2),
            "Norm_Std": round(float(pd.Series(norm_scores).std()), 2),
            "Time(s)": round(float(pd.Series(time_vals).mean()), 3),
            "Gens_Avg": round(float(pd.Series(gen_vals).mean()), 2),
            "Diff%": diff_str,
            "Rows": len(norm_scores),
        })

    summary_df = pd.DataFrame(rows)
    out_path = SUMMARY_DIR / "exp4_ablation_repair_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  📄 Saved: {out_path}")
    return summary_df


def analyze_exp5_sensitivity() -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("  THÍ NGHIỆM 5: SENSITIVITY ANALYSIS")
    print("=" * 100)

    exp_dir = RESULTS_DIR / "exp5_sensitivity"
    grouped: DefaultDict[tuple[str, str], list[tuple[str, pd.DataFrame]]] = defaultdict(list)

    for csv_path in sorted(exp_dir.glob("*.csv")):
        stem = csv_path.stem
        inst = _instance_from_stem(stem)
        if not inst:
            continue
        rest = stem.split("_", 1)[1]

        matched_param = None
        matched_value = None
        for p in SENSITIVITY_PARAMS:
            prefix = f"{p}_"
            if rest.startswith(prefix):
                matched_param = p
                matched_value = rest[len(prefix):]
                break

        if not matched_param:
            continue

        df = _safe_read_csv(csv_path)
        if df is not None and not df.empty:
            grouped[(matched_param, matched_value)].append((inst, df))

    rows = []
    for (param, value), items in sorted(grouped.items()):
        norm_scores = []
        time_vals = []
        gen_vals = []
        per_inst = {}

        for inst, df in items:
            bks = LABADIE_BKS[inst]
            inst_norm = (df["total_score"] / bks * 100.0)
            norm_scores.extend(inst_norm.tolist())
            per_inst[inst] = float(inst_norm.mean())
            time_vals.extend(df["execution_time"].tolist())
            gen_vals.extend(df["generations_run"].tolist())

        row = {
            "Parameter": param,
            "Value": value,
            "Norm_Score%": round(float(pd.Series(norm_scores).mean()), 2),
            "Norm_Std": round(float(pd.Series(norm_scores).std()), 2),
            "Time(s)": round(float(pd.Series(time_vals).mean()), 3),
            "Gens_Avg": round(float(pd.Series(gen_vals).mean()), 2),
            "Rows": len(norm_scores),
        }
        for inst in INSTANCES:
            row[f"Norm_{inst}"] = round(per_inst[inst], 2) if inst in per_inst else None
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = SUMMARY_DIR / "exp5_sensitivity_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  📄 Saved: {out_path}")
    return summary_df





def main():
    print("╔" + "═" * 78 + "╗")
    print("║  PHÂN TÍCH KẾT QUẢ THỰC NGHIỆM HGA-TOPTW  —  TỰ ĐỘNG THEO FLOW MỚI     ║")
    print("╚" + "═" * 78 + "╝")

    analyze_exp1_benchmark()
    analyze_exp2_personalization()
    analyze_exp3_budget_impact()
    analyze_exp4_ablation_repair()
    analyze_exp5_sensitivity()

    print(f"\n{'=' * 90}")
    print(f"  ✅ Đã lưu toàn bộ summary vào: {SUMMARY_DIR}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()

