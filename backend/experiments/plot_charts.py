"""
Visualization — Vẽ đồ thị cho báo cáo khóa luận.

Tạo:
  1. Convergence Curve: Best/Avg Fitness vs Generation
  2. Ablation Boxplot: So sánh Total Score giữa các variant
  3. Route Scatter Plot: So sánh lộ trình Foodie vs History Buff
  4. Personalization Bar Chart: Category distribution theo profile
  5. Sensitivity Line Chart: Score vs Parameter value
  6. Exp6 Boxplot: Static vs Adaptive-Lite (Normalized Score)
  7. Exp6 Operator Curves: p_insert / p_2opt theo generation

Usage:
    cd backend
    py -m experiments.plot_charts
"""

import os
import sys
import json
import glob
from typing import cast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ── Cấu hình font tiếng Việt-safe ──────────────────────────────────────────
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Unified chart template (EN-only) ───────────────────────────────────────
TITLE_SIZE = 16
SUBTITLE_SIZE = 14
LABEL_SIZE = 12
LEGEND_SIZE = 10
LINE_MAIN = 2.2
LINE_AUX = 1.6
MARKER_SIZE = 7

PALETTE = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "gray": "#90A4AE",
    "gold": "#FFD700",
}

CHART_DIR = "experiments/results/charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Convergence Curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_convergence(csv_path: str, output_name: str = "convergence"):
    """
    Vẽ Convergence Curve từ convergence_log trong CSV.
    Lấy run đầu tiên làm đại diện.
    Hiển thị: Best, Median, Avg Fitness.
    """
    df = pd.read_csv(csv_path)
    if 'convergence_log' not in df.columns:
        print(f"  ⚠ Không tìm thấy convergence_log trong {csv_path}")
        return

    # Lấy convergence log từ run đầu tiên
    log_raw = df["convergence_log"].iloc[0] if "convergence_log" in df.columns else "[]"
    log = json.loads(str(log_raw))
    gens = [r["gen"] for r in log]
    best = [r["best_fitness"] for r in log]
    avg = [r["avg_fitness"] for r in log]
    # median_fitness có thể chưa tồn tại trong data cũ
    median = [r.get("median_fitness", r["avg_fitness"]) for r in log]

    instance = str(df["instance"].iloc[0]) if "instance" in df.columns else "Unknown"
    label = str(df["label"].iloc[0]) if "label" in df.columns else ""

    plt.figure(figsize=(10, 6))
    plt.plot(gens, best, '-', label='Best Fitness', linewidth=LINE_MAIN, color=PALETTE["blue"])
    plt.plot(gens, median, '-.', label='Median Fitness', linewidth=LINE_AUX, alpha=0.8, color=PALETTE["green"])
    plt.plot(gens, avg, '--', label='Avg Fitness', alpha=0.7, linewidth=LINE_AUX, color=PALETTE["red"])
    plt.fill_between(gens, avg, best, alpha=0.08, color=PALETTE["blue"])
    plt.xlabel('Generation', fontsize=LABEL_SIZE)
    plt.ylabel('Fitness', fontsize=LABEL_SIZE)
    plt.title(f'Convergence Curve — {instance} [{label}]', fontsize=SUBTITLE_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, f"{output_name}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. Ablation Boxplot
# ══════════════════════════════════════════════════════════════════════════════
def plot_ablation_boxplot(results_dir: str = "experiments/results/exp3_ablation"):
    """Boxplot so sánh Normalized Score (% of BKS) giữa các variant ablation (6 instances)."""
    bks = {"C101": 320, "C201": 870, "R101": 198,
           "R201": 797, "RC101": 219, "RC201": 795}

    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csv_files:
        print(f"  ⚠ Không tìm thấy CSV trong {results_dir}")
        return

    # Naming hiện tại của exp3 là: instance_label.csv (VD: C101_full_hga.csv)
    variants = ["full_hga", "no_smart_repair", "no_insertion_mut",
                "no_heuristic_init", "no_diversity_check"]

    rows = []
    for f in csv_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue

        inst, variant = parts[0], parts[1]
        if inst not in bks:
            continue

        df = pd.read_csv(f)
        if "total_score" not in df.columns:
            continue

        tmp = df[["total_score"]].copy()
        tmp["instance"] = inst
        tmp["variant"] = variant
        tmp["norm_score"] = tmp["total_score"] / bks[inst] * 100
        rows.append(tmp)

    if not rows:
        print("  ⚠ Không đủ dữ liệu exp3 theo naming instance_label.csv")
        return

    all_data = cast(pd.DataFrame, pd.concat(rows, ignore_index=True))
    existing_order = [v for v in variants if v in all_data['variant'].unique()]
    if not existing_order:
        existing_order = sorted(all_data['variant'].unique())

    plot_data = [all_data[all_data['variant'] == v]['norm_score'].values for v in existing_order]

    fig, ax = plt.subplots(figsize=(12, 7))
    bp = ax.boxplot(plot_data, patch_artist=True, labels=existing_order)

    colors = [PALETTE["green"], PALETTE["orange"], PALETTE["blue"], PALETTE["purple"], PALETTE["red"], "#8D6E63"]
    for patch, color in zip(bp['boxes'], colors[:len(existing_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Variant', fontsize=LABEL_SIZE)
    ax.set_ylabel('Normalized Score (% of BKS)', fontsize=LABEL_SIZE)
    ax.set_title('Ablation Study — Component Contribution', fontsize=SUBTITLE_SIZE)
    ax.set_xticklabels(existing_order, rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "ablation_boxplot.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Route Scatter Plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_route_comparison(results_dir: str = "experiments/results/exp2_personalization"):
    """Vẽ 2 route cạnh nhau: Foodie vs History Buff."""
    from app.services.data_loader import load_solomon_instance

    # Ưu tiên C101, fallback C201
    instance = "C101"
    pois = load_solomon_instance(instance)
    poi_map = {p.id: p for p in pois}

    # Đọc route_ids từ CSV (lấy run 1 = best representative)
    profiles = {}
    for profile_name in ["foodie", "history_buff"]:
        csv_path = os.path.join(results_dir, f"{instance}_{profile_name}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(results_dir, f"C201_{profile_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  ⚠ Không tìm thấy {csv_path}")
            return
        df = pd.read_csv(csv_path)
        # Lấy run có score cao nhất
        best_idx = df['total_score'].idxmax()
        route_raw = df.loc[best_idx, 'route_ids'] if 'route_ids' in df.columns else '[]'
        route_ids = json.loads(str(route_raw))
        # Đảm bảo điểm 0 (Depot) luôn có mặt ở đầu và cuối để vẽ không bị lỗi
        route = [poi_map[0]] + [poi_map[pid] for pid in route_ids if pid in poi_map and pid != 0] + [poi_map[0]]
        profiles[profile_name] = route

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    configs = [
        (axes[0], profiles["foodie"], "Foodie", 'tomato'),
        (axes[1], profiles["history_buff"], "History Buff", 'steelblue'),
    ]

    for ax, route, title, color in configs:
        # Vẽ tất cả POI (xám nhạt)
        all_x = [p.x for p in pois if p.id != 0]
        all_y = [p.y for p in pois if p.id != 0]
        ax.scatter(all_x, all_y, c='lightgray', s=30, zorder=1,
                   label='Not visited', alpha=0.6)

        # Vẽ route (màu đậm + nối đường)
        rx = [p.x for p in route]
        ry = [p.y for p in route]
        ax.plot(rx, ry, '-', color=color, linewidth=LINE_AUX, alpha=0.6, zorder=2)
        ax.scatter(rx[1:-1], ry[1:-1], c=color, s=80, zorder=3,
                   label=f'Visited ({len(route)-2} POIs)', edgecolors='white')

        # Depot
        depot = route[0]
        ax.scatter([depot.x], [depot.y], c='gold', s=200, marker='*',
                   zorder=4, label='Depot', edgecolors='black')

        ax.set_title(f'{title}', fontsize=SUBTITLE_SIZE, fontweight='bold')
        ax.set_xlabel('X coordinate', fontsize=LABEL_SIZE)
        ax.set_ylabel('Y coordinate', fontsize=LABEL_SIZE)
        ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'Route Comparison ({instance}): Personalization creates different routes',
                 fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "route_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. Personalization Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_personalization_bars(results_dir: str = "experiments/results/exp2_personalization"):
    """Stacked Bar Chart: Category % distribution theo profile."""
    profiles = ["baseline", "history_buff", "foodie", "explorer", "shopper"]
    cats = ["cat_history_culture", "cat_nature_parks", "cat_food_drink",
            "cat_shopping", "cat_entertainment"]
    cat_labels = ["History", "Nature", "Food", "Shopping", "Entertain"]
    colors = ['#4169E1', '#228B22', '#FF6347', '#DAA520', '#9370DB']

    data = {}
    pois_count = {}
    for name in profiles:
        csv_path = os.path.join(results_dir, f"C101_{name}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(results_dir, f"C201_{name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            vals = [df[c].mean() if c in df.columns else 0 for c in cats]
            total = sum(vals)
            pois_count[name] = total
            # Convert to percentages
            data[name] = [(v / total * 100) if total > 0 else 0 for v in vals]

    if not data:
        print("  ⚠ Không tìm thấy dữ liệu personalization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: Stacked Percentage Bar Chart ──────────────────────────────────
    ax1 = axes[0]
    x = range(len(data))
    profile_names = list(data.keys())
    bottom = [0] * len(profile_names)

    for cat_idx, cat_label in enumerate(cat_labels):
        vals = [data[name][cat_idx] for name in profile_names]
        bars = ax1.bar(x, vals, bottom=bottom, label=cat_label,
                       color=colors[cat_idx], alpha=0.85, width=0.6)

        # Add % labels inside bars (chỉ khi > 8%)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 8:
                ax1.text(i, b + v / 2, f'{v:.0f}%', ha='center', va='center',
                         fontsize=9, fontweight='bold', color='white')

        bottom = [b + v for b, v in zip(bottom, vals)]

    ax1.set_xlabel('User Profile', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Category Distribution (%)', fontsize=LABEL_SIZE)
    ax1.set_title('Category Distribution by Profile (%)', fontsize=SUBTITLE_SIZE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(profile_names, rotation=20, ha='right')
    ax1.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.2, axis='y')

    # ── Right: Grouped Bar (absolute count) ─────────────────────────────────
    ax2 = axes[1]
    width = 0.15
    for i, (name, pcts) in enumerate(data.items()):
        total = pois_count[name]
        abs_vals = [pct / 100 * total for pct in pcts]
        offset = (i - len(data) / 2 + 0.5) * width
        ax2.bar([xi + offset for xi in range(len(cat_labels))], abs_vals, width,
                label=name, alpha=0.85)

    ax2.set_xlabel('Category', fontsize=LABEL_SIZE)
    ax2.set_ylabel('Average Number of POIs', fontsize=LABEL_SIZE)
    ax2.set_title('Absolute POI Count by Category', fontsize=SUBTITLE_SIZE, fontweight='bold')
    ax2.set_xticks(range(len(cat_labels)))
    ax2.set_xticklabels(cat_labels)
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    path = os.path.join(CHART_DIR, "personalization_bars.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. Sensitivity Line Charts
# ══════════════════════════════════════════════════════════════════════════════
def plot_sensitivity(results_dir: str = "experiments/results/exp4_sensitivity"):
    """Line charts: Normalized Score vs Execution Time Trade-off."""
    instances = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
    bks = {"C101": 320, "C201": 870, "R101": 198,
           "R201": 797, "RC101": 219, "RC201": 795}

    param_grid = {
        "population_size": [50, 100, 150, 200],
        "mutation_rate": [0.3, 0.5, 0.7, 0.9],
        "tournament_k": [2, 3, 5, 7],
        "stagnation_limit": [15, 25, 40, 60],
    }
    defaults = {"population_size": 100, "mutation_rate": 0.7,
                "tournament_k": 3, "stagnation_limit": 25}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, (param, values) in enumerate(param_grid.items()):
        ax1 = axes[idx]
        ax2 = ax1.twinx()  # Tạo trục Y thứ 2 bên phải

        means_score, stds_score, means_time = [], [], []

        for val in values:
            norm_scores = []
            times = []
            for inst in instances:
                csv_path = os.path.join(results_dir, f"{inst}_{param}_{val}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    inst_norm = df['total_score'] / bks[inst] * 100
                    norm_scores.extend(inst_norm.tolist())
                    times.extend(df['execution_time'].tolist())

            if norm_scores:
                means_score.append(sum(norm_scores) / len(norm_scores))
                stds_score.append(pd.Series(norm_scores).std())
                means_time.append(sum(times) / len(times))
            else:
                means_score.append(0)
                stds_score.append(0)
                means_time.append(0)

        # Vẽ đường Điểm số (Score) - Trục trái (Màu xanh)
        l1 = ax1.errorbar(values, means_score, yerr=stds_score, marker='o', capsize=5,
                          linewidth=LINE_MAIN, markersize=MARKER_SIZE, color=PALETTE["blue"], label='Normalized Score')

        # Vẽ đường Thời gian (Time) - Trục phải (Màu cam gạch ngang)
        l2, = ax2.plot(values, means_time, marker='s', linestyle='--', 
                       linewidth=LINE_MAIN, markersize=MARKER_SIZE, color=PALETTE["orange"], label='Execution Time')

        # Đánh dấu Default
        if defaults[param] in values:
            di = values.index(defaults[param])
            ax1.scatter([defaults[param]], [means_score[di]], color=PALETTE["red"],
                        s=180, zorder=5, marker='D', label=f'Default ({defaults[param]})')

        # Formatting
        ax1.set_xlabel(param.replace('_', ' ').title(), fontsize=LABEL_SIZE)
        ax1.set_ylabel('Normalized Score (% of BKS)', fontsize=LABEL_SIZE, color=PALETTE["blue"])
        ax2.set_ylabel('Execution Time (seconds)', fontsize=LABEL_SIZE, color=PALETTE["orange"])

        ax1.tick_params(axis='y', labelcolor=PALETTE["blue"])
        ax2.tick_params(axis='y', labelcolor=PALETTE["orange"])
        
        ax1.set_title(f'Trade-off: {param}', fontsize=SUBTITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Gộp legend của cả 2 trục
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='lower right' if param != 'stagnation_limit' else 'lower right', fontsize=LEGEND_SIZE)

    plt.suptitle('Sensitivity Analysis — Trade-off Between Score and Execution Time',
                 fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "sensitivity_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. Exp6 Boxplot — Static vs Adaptive
# ═══════════════════════════════════════════════════════════════════���══════════
def plot_exp6_boxplot(results_dir: str = "experiments/results/exp6_adaptive_mutation"):
    """Boxplot so sánh Normalized Score (%BKS) giữa static_mutation và adaptive_lite_2tier."""
    bks = {"C101": 320, "C201": 870, "R101": 198,
           "R201": 797, "RC101": 219, "RC201": 795}
    variants = ["static_mutation", "adaptive_lite_2tier"]

    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csv_files:
        print(f"  ⚠ Không tìm thấy CSV trong {results_dir}")
        return

    variant_norm = {v: [] for v in variants}

    for f in csv_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        inst, label = parts[0], parts[1]
        if inst not in bks or label not in variant_norm:
            continue

        df = pd.read_csv(f)
        if "total_score" not in df.columns:
            continue
        norm_vals = (df["total_score"] / bks[inst] * 100.0).tolist()
        variant_norm[label].extend(norm_vals)

    available = [v for v in variants if variant_norm[v]]
    if not available:
        print("  ⚠ Không đủ dữ liệu exp6 để vẽ boxplot")
        return

    data = [variant_norm[v] for v in available]
    labels = ["Static" if v == "static_mutation" else "Adaptive-Lite" for v in available]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    colors = ["#90A4AE", "#42A5F5"]
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel('Normalized Score (% of BKS)', fontsize=LABEL_SIZE)
    ax.set_title('Exp6 — Static vs Adaptive-Lite Mutation', fontsize=SUBTITLE_SIZE)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "exp6_adaptive_boxplot.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. Exp6 Operator Curves — p_insert / p_2opt
# ══════════════════════════════════════════════════════════════════════════════
def plot_exp6_operator_curves(results_dir: str = "experiments/results/exp6_adaptive_mutation"):
    """Vẽ xu hướng trung bình p_insert và p_2opt theo generation từ convergence_log của adaptive_lite_2tier."""
    pattern = os.path.join(results_dir, "*_adaptive_lite_2tier.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        print(f"  ⚠ Không tìm thấy dữ liệu adaptive_lite_2tier trong {results_dir}")
        return

    rows = []
    for f in csv_files:
        df = pd.read_csv(f)
        if "convergence_log" not in df.columns:
            continue
        for _, r in df.iterrows():
            try:
                log = json.loads(r["convergence_log"])
            except Exception:
                continue
            for entry in log:
                if "gen" not in entry:
                    continue
                rows.append({
                    "gen": entry.get("gen"),
                    "p_insert": entry.get("p_insert"),
                    "p_2opt": entry.get("p_2opt"),
                    "p_swap": entry.get("p_swap"),
                })

    if not rows:
        print("  ⚠ Không có convergence_log hợp lệ để vẽ operator curves")
        return

    hist_df = pd.DataFrame(rows)
    grouped = hist_df.groupby("gen", as_index=False).mean(numeric_only=True)

    plt.figure(figsize=(10, 6))
    plt.plot(grouped["gen"], grouped["p_insert"], label="p_insert", color=PALETTE["red"], linewidth=LINE_MAIN)
    plt.plot(grouped["gen"], grouped["p_2opt"], label="p_2opt", color=PALETTE["blue"], linewidth=LINE_MAIN)
    if "p_swap" in grouped.columns:
        plt.plot(grouped["gen"], grouped["p_swap"], label="p_swap", color=PALETTE["green"], linestyle='--', linewidth=LINE_AUX, alpha=0.9)

    plt.xlabel('Generation', fontsize=LABEL_SIZE)
    plt.ylabel('Operator Probability', fontsize=LABEL_SIZE)
    plt.title('Exp6 — Adaptive Mutation Policy Over Generations', fontsize=SUBTITLE_SIZE)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "exp6_operator_curves.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main — Vẽ tất cả
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  VẼ ĐỒ THỊ CHO BÁO CÁO")
    print("=" * 60)

    # 1. Convergence Curves
    print("\n[1] Convergence Curves...")
    exp1_dir = "experiments/results/exp1_benchmark"
    if os.path.exists(exp1_dir):
        for f in sorted(glob.glob(os.path.join(exp1_dir, "*.csv"))):
            inst = os.path.basename(f).replace("_fixed.csv", "")
            plot_convergence(f, f"convergence_{inst}")

    # 2. Ablation Boxplot
    print("\n[2] Ablation Boxplot...")
    plot_ablation_boxplot()

    # 3. Route Comparison
    print("\n[3] Route Comparison (Foodie vs History Buff)...")
    plot_route_comparison()

    # 4. Personalization Bars
    print("\n[4] Personalization Category Distribution...")
    plot_personalization_bars()

    # 5. Sensitivity Analysis
    print("\n[5] Sensitivity Analysis...")
    plot_sensitivity()

    # 6. Exp6 Boxplot
    print("\n[6] Exp6 Static vs Adaptive Boxplot...")
    plot_exp6_boxplot()

    # 7. Exp6 Operator Curves
    print("\n[7] Exp6 Operator Probability Curves...")
    plot_exp6_operator_curves()

    print(f"\n{'=' * 60}")
    print(f"  ✅ TẤT CẢ ĐỒ THỊ ĐÃ ĐƯỢC LƯU VÀO: {CHART_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
