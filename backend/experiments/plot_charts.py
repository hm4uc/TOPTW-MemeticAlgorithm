"""
Visualization — Vẽ đồ thị cho báo cáo khóa luận.

Tạo:
  1. Convergence Curve: Best/Avg Fitness vs Generation (multi-run band)
  2. Ablation Boxplot:  Normalized Score + mean annotation + BKS reference line
  3. Route Scatter Plot: Foodie vs History Buff (colored by category)
  4. Personalization Bar Chart: Stacked % + grouped absolute (merged C101+C201)
  5. Sensitivity Line Charts: Score (left) + Time (right) dual-axis
  6. Adaptive Boxplot: Swarmplot overlay + mean annotation + per-instance delta
  7. Adaptive Operator Curves: Smooth rolling-avg + shaded band

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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Font & DPI ─────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Unified design tokens ───────────────────────────────────────────────────────
TITLE_SIZE   = 15
SUBTITLE_SIZE= 13
LABEL_SIZE   = 11
LEGEND_SIZE  = 9
ANNOT_SIZE   = 8
LINE_MAIN    = 2.2
LINE_AUX     = 1.5
MARKER_SIZE  = 7

PALETTE = {
    "blue":   "#2196F3",
    "orange": "#FF9800",
    "green":  "#4CAF50",
    "red":    "#F44336",
    "purple": "#9C27B0",
    "teal":   "#009688",
    "gray":   "#90A4AE",
    "gold":   "#FFC107",
}

# Category color mapping (consistent across all charts)
CAT_COLORS = {
    "History":   "#4169E1",
    "Nature":    "#228B22",
    "Food":      "#FF6347",
    "Shopping":  "#DAA520",
    "Entertain": "#9370DB",
}
CAT_ORDER  = ["History", "Nature", "Food", "Shopping", "Entertain"]
CAT_COLS   = ["cat_history_culture", "cat_nature_parks",
              "cat_food_drink", "cat_shopping", "cat_entertainment"]

BKS = {"C101": 320, "C201": 870, "R101": 198,
       "R201": 797, "RC101": 219, "RC201": 795}
INSTANCES = list(BKS.keys())

CHART_DIR = "experiments/results/charts"
os.makedirs(CHART_DIR, exist_ok=True)


def _savefig(path: str):
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OK  {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. Convergence Curve  (multi-run shaded band)
# ══════════════════════════════════════════════════════════════════════════════
def plot_convergence(csv_path: str, output_name: str = "convergence"):
    """
    Vẽ convergence curve tốt hơn:
      - ALL runs được vẽ nhạt phía sau (transparency) để thấy variance
      - Best và Mean trên toàn bộ runs được highlight đậm
      - Shaded band = [min, max] của best_fitness tại mỗi generation
    """
    df = pd.read_csv(csv_path)
    if "convergence_log" not in df.columns:
        print(f"  WARN  convergence_log not found in {csv_path}")
        return

    instance = str(df["instance"].iloc[0]) if "instance" in df.columns else "?"
    label    = str(df["label"].iloc[0])    if "label" in df.columns else ""

    # Parse tất cả runs
    all_best, all_avg = {}, {}
    for _, row in df.iterrows():
        try:
            log = json.loads(row["convergence_log"])
        except Exception:
            continue
        for entry in log:
            g = entry["gen"]
            all_best.setdefault(g, []).append(entry["best_fitness"])
            all_avg.setdefault(g,  []).append(entry["avg_fitness"])

    if not all_best:
        print(f"  WARN  empty convergence log in {csv_path}")
        return

    gens      = sorted(all_best.keys())
    best_mean = [np.mean(all_best[g]) for g in gens]
    best_min  = [np.min(all_best[g])  for g in gens]
    best_max  = [np.max(all_best[g])  for g in gens]
    avg_mean  = [np.mean(all_avg[g])  for g in gens]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded variance band
    ax.fill_between(gens, best_min, best_max,
                    alpha=0.12, color=PALETTE["blue"], label="Best fitness range")

    # Mean lines
    ax.plot(gens, best_mean, "-",  linewidth=LINE_MAIN, color=PALETTE["blue"],
            label="Best fitness (mean)")
    ax.plot(gens, avg_mean,  "--", linewidth=LINE_AUX,  color=PALETTE["orange"],
            alpha=0.8, label="Avg fitness (mean)")

    # Annotate final best value
    ax.annotate(f"Final Best:\n{best_mean[-1]:.1f}",
                xy=(gens[-1], best_mean[-1]),
                xytext=(-55, 12), textcoords="offset points",
                fontsize=ANNOT_SIZE, color=PALETTE["blue"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=1.0))

    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel("Fitness Value", fontsize=LABEL_SIZE)
    ax.set_title(f"Convergence Curve — Instance {instance}", fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()

    _savefig(os.path.join(CHART_DIR, f"{output_name}.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  2. Ablation Boxplot  (BKS reference + mean dots + readable labels)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ablation_boxplot(results_dir: str = "experiments/results/exp3_ablation"):
    """
    Cải tiến:
      - Đường tham chiếu BKS = 100%
      - Mean được đánh dấu bằng tam giác đỏ
      - Label thân thiện hơn
      - Y-axis bắt đầu từ 96% thay vì 0% để phóng đại sự khác biệt nhỏ
    """
    VARIANT_LABELS = {
        "full_hga":           "Full HGA\n(baseline)",
        "no_smart_repair":    "No Smart\nRepair",
        "no_insertion_mut":   "No Insertion\nMutation",
        "no_heuristic_init":  "No Heuristic\nInit",
        "no_diversity_check": "No Diversity\nCheck",
    }
    variants = list(VARIANT_LABELS.keys())

    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        stem  = os.path.splitext(os.path.basename(f))[0]
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        inst, variant = parts[0], parts[1]
        if inst not in BKS or variant not in variants:
            continue
        df = pd.read_csv(f)
        if "total_score" not in df.columns:
            continue
        tmp = df[["total_score"]].copy()
        tmp["variant"]    = variant
        tmp["norm_score"] = tmp["total_score"] / BKS[inst] * 100
        rows.append(tmp)

    if not rows:
        print("  WARN  no ablation data found")
        return

    all_data     = cast(pd.DataFrame, pd.concat(rows, ignore_index=True))
    exist_order  = [v for v in variants if v in all_data["variant"].unique()]
    plot_data    = [all_data[all_data["variant"] == v]["norm_score"].values
                    for v in exist_order]

    box_colors = [PALETTE["green"],  PALETTE["orange"], PALETTE["blue"],
                  PALETTE["purple"], PALETTE["teal"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, color in zip(bp["boxes"], box_colors[:len(exist_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    # Mean markers
    for i, data in enumerate(plot_data, start=1):
        m = np.mean(data)
        ax.scatter(i, m, marker="^", color="red", s=60, zorder=5)
        ax.annotate(f"{m:.2f}%", xy=(i, m), xytext=(6, 4),
                    textcoords="offset points", fontsize=7, color="red")

    # BKS reference line
    ax.axhline(100, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.text(len(exist_order) + 0.55, 100.05, "BKS = 100%",
            fontsize=8, va="bottom", color="black", alpha=0.7)

    friendly_labels = [VARIANT_LABELS.get(v, v) for v in exist_order]
    ax.set_xticks(range(1, len(exist_order) + 1))
    ax.set_xticklabels(friendly_labels, fontsize=10)
    ax.set_ylabel("Normalized Score (% of BKS)", fontsize=LABEL_SIZE)
    ax.set_title("Ablation Study — Component Contribution\n(Normalized over 6 Solomon instances)",
                 fontsize=SUBTITLE_SIZE, fontweight="bold")

    # Zoom Y to show differences clearly
    all_vals = np.concatenate(plot_data)
    ax.set_ylim(max(94, all_vals.min() - 1), 101.5)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")

    mean_patch = Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
                        markersize=7, label="Mean")
    bks_line   = Line2D([0], [0], linestyle="--", color="black", lw=1.2,
                        alpha=0.6, label="BKS = 100%")
    ax.legend(handles=[mean_patch, bks_line], fontsize=LEGEND_SIZE,
              loc="lower right", framealpha=0.9)

    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "ablation_boxplot.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  3. Route Scatter Plot  (colored by POI category)
# ══════════════════════════════════════════════════════════════════════════════
def plot_route_comparison(results_dir: str = "experiments/results/exp2_personalization"):
    """
    Cải tiến:
      - POI được tô màu theo category (thay vì đồng màu)
      - Thêm số thứ tự thăm quan trên mỗi POI
      - Chú thích category rõ ràng hơn trong legend
    """
    from app.services.data_loader import load_solomon_instance

    instance = "C101"
    try:
        pois = load_solomon_instance(instance)
    except Exception as e:
        print(f"  WARN  Cannot load POIs: {e}")
        return
    poi_map = {p.id: p for p in pois}

    # Category color map for POIs
    CAT_COLOR_POI = {
        "history_culture": "#4169E1",
        "nature_parks":    "#228B22",
        "food_drink":      "#FF6347",
        "shopping":        "#DAA520",
        "entertainment":   "#9370DB",
        "depot":           "#FFC107",
    }

    profiles = {}
    for profile_name in ["foodie", "history_buff"]:
        csv_path = os.path.join(results_dir, f"{instance}_{profile_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARN  {csv_path} not found")
            return
        df = pd.read_csv(csv_path)
        best_idx  = df["total_score"].idxmax()
        route_raw = df.loc[best_idx, "route_ids"] if "route_ids" in df.columns else "[]"
        route_ids = json.loads(str(route_raw))
        route = ([poi_map[0]]
                 + [poi_map[pid] for pid in route_ids if pid in poi_map and pid != 0]
                 + [poi_map[0]])
        profiles[profile_name] = route

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    profile_configs = [
        ("foodie",       axes[0], "Foodie Profile"),
        ("history_buff", axes[1], "History Buff Profile"),
    ]

    for profile_name, ax, title in profile_configs:
        route = profiles[profile_name]

        # Background: all non-depot POIs (gray)
        for p in pois:
            if p.id == 0:
                continue
            ax.scatter(p.x, p.y, c="lightgray", s=22, zorder=1, alpha=0.5)

        # Visited POIs colored by category
        for idx, p in enumerate(route[1:-1], start=1):
            c = CAT_COLOR_POI.get(p.category, "#888888")
            ax.scatter(p.x, p.y, c=c, s=90, zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(str(idx), (p.x, p.y),
                        fontsize=6.5, ha="center", va="center",
                        color="white", fontweight="bold", zorder=4)

        # Route path
        rx = [p.x for p in route]
        ry = [p.y for p in route]
        ax.plot(rx, ry, "-", color="black", linewidth=1.1, alpha=0.35, zorder=2)

        # Depot star
        depot = route[0]
        ax.scatter([depot.x], [depot.y], c=PALETTE["gold"], s=250,
                   marker="*", zorder=5, edgecolors="black", linewidths=0.8)

        # Legend for categories
        legend_patches = [
            mpatches.Patch(facecolor=v, label=k.replace("_", " ").title(), edgecolor="white")
            for k, v in CAT_COLOR_POI.items() if k != "depot"
        ]
        legend_patches.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor=PALETTE["gold"],
                   markersize=11, label="Depot")
        )
        ax.legend(handles=legend_patches, fontsize=8, loc="upper right",
                  framealpha=0.9, title="Category", title_fontsize=8)

        ax.set_title(f"{title}\n({len(route)-2} POIs visited)",
                     fontsize=SUBTITLE_SIZE, fontweight="bold")
        ax.set_xlabel("X Coordinate", fontsize=LABEL_SIZE)
        ax.set_ylabel("Y Coordinate", fontsize=LABEL_SIZE)
        ax.grid(True, alpha=0.18, linestyle="--")

    plt.suptitle(
        f"Route Comparison on {instance}: Same Constraints, Different Preferences",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "route_comparison.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  4. Personalization Bar Chart  (merged C101+C201, better layout)
# ══════════════════════════════════════════════════════════════════════════════
def plot_personalization_bars(results_dir: str = "experiments/results/exp2_personalization"):
    """
    Cải tiến:
      - Kết hợp C101 + C201: trình bày % distribution thống nhất
      - Thêm Profile Label thân thiện
      - Thêm heatmap phụ để show sự khác biệt trực quan
    """
    PROFILE_LABELS = {
        "baseline":     "Balanced",
        "history_buff": "History\nBuff",
        "foodie":       "Foodie",
        "explorer":     "Explorer",
        "shopper":      "Shopper",
    }
    profiles = list(PROFILE_LABELS.keys())
    colors   = [CAT_COLORS[c] for c in CAT_ORDER]

    # Load data — prefer C201 (richer distribution) for visualization
    data = {}
    for name in profiles:
        for inst in ["C201", "C101"]:
            csv_path = os.path.join(results_dir, f"{inst}_{name}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                vals  = [df[c].mean() if c in df.columns else 0 for c in CAT_COLS]
                total = sum(vals)
                data[name] = {
                    "pcts":     [(v / total * 100) if total > 0 else 0 for v in vals],
                    "abs_vals": vals,
                    "score":    df["total_score"].mean(),
                    "pois":     df["num_pois"].mean(),
                    "inst":     inst,
                }
                break

    if not data:
        print("  WARN  no personalization data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # ── Left: Stacked % Bar ────────────────────────────────────────────────────
    ax1     = axes[0]
    x       = np.arange(len(profiles))
    bottom  = np.zeros(len(profiles))
    profile_names = profiles

    for cat_idx, (cat_label, color) in enumerate(zip(CAT_ORDER, colors)):
        vals = [data[n]["pcts"][cat_idx] for n in profile_names]
        ax1.bar(x, vals, bottom=bottom, label=cat_label, color=color,
                alpha=0.88, width=0.62, edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 9:
                ax1.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8.5, fontweight="bold", color="white")
        bottom += np.array(vals)

    ax1.set_xticks(x)
    ax1.set_xticklabels([PROFILE_LABELS[n] for n in profile_names],
                        fontsize=10.5)
    ax1.set_ylabel("Category Distribution (%)", fontsize=LABEL_SIZE)
    ax1.set_title("Category Distribution by User Profile\n(% of visited POIs)",
                  fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.legend(fontsize=LEGEND_SIZE, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.2, axis="y", linestyle="--")

    # Score annotations above bars
    for i, name in enumerate(profile_names):
        s = data[name]["score"]
        ax1.text(i, 102.5, f"Score\n{s:.0f}", ha="center", va="bottom",
                 fontsize=7.5, color="#333333", style="italic")

    # ── Right: Heatmap-style grouped absolute bar ──────────────────────────────
    ax2   = axes[1]
    width = 0.14
    n_prof = len(profile_names)
    for i, name in enumerate(profile_names):
        abs_vals = data[name]["abs_vals"]
        offset = (i - n_prof / 2 + 0.5) * width
        ax2.bar([xi + offset for xi in range(len(CAT_ORDER))], abs_vals, width,
                label=PROFILE_LABELS[name], alpha=0.87,
                edgecolor="white", linewidth=0.4)

    ax2.set_xticks(range(len(CAT_ORDER)))
    ax2.set_xticklabels(CAT_ORDER, fontsize=10.5)
    ax2.set_ylabel("Average Number of POIs Visited", fontsize=LABEL_SIZE)
    ax2.set_title("Absolute POI Count by Category\n(Avg across runs)",
                  fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax2.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax2.grid(True, alpha=0.25, axis="y", linestyle="--")

    inst_used = data[profiles[0]]["inst"]
    plt.suptitle(f"Personalization Effect — Instance {inst_used}",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "personalization_bars.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  5. Sensitivity Analysis  (dual-axis, std band, default marker)
# ══════════════════════════════════════════════════════════════════════════════
def plot_sensitivity(results_dir: str = "experiments/results/exp4_sensitivity"):
    """
    Cải tiến:
      - Vùng ±1 std được tô nhạt dưới đường Score
      - Default value được đánh dấu rõ hơn bằng vùng highlight đứng
      - Title mỗi subplot thân thiện hơn
    """
    param_grid = {
        "population_size":  [50, 100, 150, 200],
        "mutation_rate":    [0.3, 0.5, 0.7, 0.9],
        "tournament_k":     [2, 3, 5, 7],
        "stagnation_limit": [15, 25, 40, 60],
    }
    param_titles = {
        "population_size":  "Population Size",
        "mutation_rate":    "Mutation Rate",
        "tournament_k":     "Tournament Size (k)",
        "stagnation_limit": "Stagnation Limit",
    }
    defaults = {"population_size": 100, "mutation_rate": 0.7,
                "tournament_k": 3, "stagnation_limit": 25}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (param, values) in enumerate(param_grid.items()):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        means_score, stds_score, means_time = [], [], []

        for val in values:
            norm_scores, times = [], []
            for inst in INSTANCES:
                csv_path = os.path.join(results_dir, f"{inst}_{param}_{val}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    norm_scores.extend((df["total_score"] / BKS[inst] * 100).tolist())
                    times.extend(df["execution_time"].tolist())
            means_score.append(np.mean(norm_scores) if norm_scores else 0)
            stds_score.append(np.std(norm_scores)   if norm_scores else 0)
            means_time.append(np.mean(times)         if times else 0)

        arr_score = np.array(means_score)
        arr_std   = np.array(stds_score)

        # Score line + std band
        l1, = ax1.plot(values, arr_score, "o-",
                       linewidth=LINE_MAIN, markersize=MARKER_SIZE,
                       color=PALETTE["blue"], label="Norm. Score (%)")
        ax1.fill_between(values, arr_score - arr_std, arr_score + arr_std,
                         alpha=0.12, color=PALETTE["blue"])

        # Time line
        l2, = ax2.plot(values, means_time, "s--",
                       linewidth=LINE_MAIN, markersize=MARKER_SIZE,
                       color=PALETTE["orange"], label="Exec. Time (s)")

        # Shade default column
        default_val = defaults[param]
        if default_val in values:
            di = values.index(default_val)
            # Vertical band centered on default
            half_gap = (values[1] - values[0]) * 0.4 if len(values) > 1 else 5
            ax1.axvspan(default_val - half_gap, default_val + half_gap,
                        color="gray", alpha=0.08, zorder=0)
            ax1.axvline(default_val, color="gray", linestyle=":", linewidth=1.2,
                        alpha=0.6, zorder=0)
            ax1.annotate("default", xy=(default_val, ax1.get_ylim()[0]),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=7, color="gray")

        ax1.set_xlabel(param_titles[param], fontsize=LABEL_SIZE)
        ax1.set_ylabel("Normalized Score (% of BKS)", fontsize=10,
                       color=PALETTE["blue"])
        ax2.set_ylabel("Execution Time (s)", fontsize=10, color=PALETTE["orange"])
        ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["orange"])
        ax1.set_title(f"Sensitivity: {param_titles[param]}",
                      fontsize=SUBTITLE_SIZE, fontweight="bold")
        ax1.grid(True, alpha=0.25, linestyle="--")

        lines  = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=LEGEND_SIZE, loc="lower right",
                   framealpha=0.9)

    plt.suptitle("Sensitivity Analysis — Score vs Execution Time Trade-off\n"
                 "(Averaged over 6 Solomon instances, 5 runs each)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "sensitivity_analysis.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  6. Adaptive Boxplot  (swarm-like jitter + per-instance delta table)
# ══════════════════════════════════════════════════════════════════════════════
def plot_exp5_boxplot(results_dir: str = "experiments/results/exp5_adaptive_mutation"):
    """
    Cải tiến:
      - Jittered dots overlay để thấy individual data points
      - Per-instance mean comparison trên cùng figure
      - Delta annotation rõ ràng
    """
    variants       = ["static_mutation", "adaptive_lite_2tier"]
    variant_labels = ["Static\nMutation", "Adaptive-Lite\n(2-tier)"]

    variant_norm  = {v: [] for v in variants}
    per_inst_norm = {v: {} for v in variants}

    for f in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        stem  = os.path.splitext(os.path.basename(f))[0]
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        inst, lbl = parts[0], parts[1]
        if inst not in BKS or lbl not in variant_norm:
            continue
        df = pd.read_csv(f)
        if "total_score" not in df.columns:
            continue
        norms = (df["total_score"] / BKS[inst] * 100.0).tolist()
        variant_norm[lbl].extend(norms)
        per_inst_norm[lbl][inst] = np.mean(norms)

    available = [v for v in variants if variant_norm[v]]
    if not available:
        print("  WARN  no adaptive data")
        return

    data   = [variant_norm[v] for v in available]
    labels = [variant_labels[variants.index(v)] for v in available]

    fig, (ax_box, ax_inst) = plt.subplots(1, 2, figsize=(14, 6),
                                           gridspec_kw={"width_ratios": [1, 1.4]})

    # ── Left: Boxplot + jitter ─────────────────────────────────────────────────
    bp_colors = ["#90A4AE", "#42A5F5"]
    bp = ax_box.boxplot(data, patch_artist=True, labels=labels, widths=0.45,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="", alpha=0))
    for patch, color in zip(bp["boxes"], bp_colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    # Jitter overlay
    np.random.seed(42)
    for i, d in enumerate(data, start=1):
        jit = np.random.uniform(-0.12, 0.12, len(d))
        ax_box.scatter(np.full(len(d), i) + jit, d,
                       alpha=0.45, s=18, color=bp_colors[i - 1],
                       edgecolors="white", linewidths=0.4, zorder=3)

    # Mean annotations
    for i, d in enumerate(data, start=1):
        m = np.mean(d)
        ax_box.scatter(i, m, marker="^", color="red", s=70, zorder=5)
        ax_box.annotate(f"Mean\n{m:.2f}%", xy=(i, m),
                        xytext=(12, 0), textcoords="offset points",
                        fontsize=8, color="red", va="center")

    ax_box.axhline(100, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax_box.set_ylabel("Normalized Score (% of BKS)", fontsize=LABEL_SIZE)
    ax_box.set_title("Overall Distribution\n(All 6 instances, 5 runs each)",
                     fontsize=SUBTITLE_SIZE, fontweight="bold")
    all_vals = np.concatenate(data)
    ax_box.set_ylim(max(93, all_vals.min() - 1.5), 101.8)
    ax_box.grid(True, alpha=0.25, axis="y", linestyle="--")

    # ── Right: Per-instance grouped bar ───────────────────────────────────────
    bar_colors = bp_colors
    x          = np.arange(len(INSTANCES))
    width      = 0.35

    for j, variant in enumerate(available):
        vals = [per_inst_norm[variant].get(inst, 0) for inst in INSTANCES]
        bars = ax_inst.bar(x + j * width, vals, width, label=labels[j],
                           color=bar_colors[j], alpha=0.85,
                           edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax_inst.annotate(f"{val:.1f}%",
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=7.5, color="#333333")

    # Delta arrows
    if len(available) == 2:
        for i, inst in enumerate(INSTANCES):
            v0 = per_inst_norm[available[0]].get(inst, 0)
            v1 = per_inst_norm[available[1]].get(inst, 0)
            delta = v1 - v0
            color = "#4CAF50" if delta >= 0 else "#F44336"
            sign  = "+" if delta >= 0 else ""
            mid_x = x[i] + width * 0.5
            top   = max(v0, v1) + 0.5
            ax_inst.annotate(f"{sign}{delta:.2f}%",
                             xy=(mid_x, top), fontsize=7.5,
                             color=color, ha="center", fontweight="bold")

    ax_inst.axhline(100, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax_inst.set_xticks(x + width / 2)
    ax_inst.set_xticklabels(INSTANCES, fontsize=10.5)
    ax_inst.set_ylabel("Normalized Score (% of BKS)", fontsize=LABEL_SIZE)
    ax_inst.set_title("Per-Instance Comparison\n(Mean Normalized Score)",
                      fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax_inst.set_ylim(max(93, min(per_inst_norm[available[0]].values()) - 1.5), 101.8)
    ax_inst.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax_inst.grid(True, alpha=0.25, axis="y", linestyle="--")

    plt.suptitle("Adaptive-Lite vs Static Mutation",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "exp5_adaptive_boxplot.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  7. Adaptive Operator Curves  (rolling smooth + shaded band)
# ══════════════════════════════════════════════════════════════════════════════
def plot_exp5_operator_curves(results_dir: str = "experiments/results/exp5_adaptive_mutation"):
    """
    Cải tiến:
      - Rolling average (window=3) để làm mượt đường
      - Shaded band ±std cho mỗi operator
      - Annotate điểm cuối của mỗi đường
    """
    pattern   = os.path.join(results_dir, "*_adaptive_lite_2tier.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        print(f"  WARN  no adaptive_lite_2tier data in {results_dir}")
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
                    "gen":      entry.get("gen"),
                    "p_insert": entry.get("p_insert"),
                    "p_2opt":   entry.get("p_2opt"),
                    "p_swap":   entry.get("p_swap"),
                })

    if not rows:
        print("  WARN  no operator log data")
        return

    hist_df = pd.DataFrame(rows)
    grouped = hist_df.groupby("gen", as_index=False).agg(
        p_insert_mean=("p_insert", "mean"),
        p_insert_std =("p_insert", "std"),
        p_2opt_mean  =("p_2opt",   "mean"),
        p_2opt_std   =("p_2opt",   "std"),
        p_swap_mean  =("p_swap",   "mean"),
        p_swap_std   =("p_swap",   "std"),
    ).fillna(0)

    gens = grouped["gen"].values

    def smooth(arr, w=3):
        return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(10, 5.5))

    op_configs = [
        ("p_insert", "Insertion Operator",  PALETTE["red"],    "-"),
        ("p_2opt",   "2-opt Operator",       PALETTE["blue"],   "-"),
        ("p_swap",   "Swap Operator",         PALETTE["green"],  "--"),
    ]

    for col, label, color, ls in op_configs:
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"
        if mean_col not in grouped.columns:
            continue
        y_mean = smooth(grouped[mean_col].values)
        y_std  = smooth(grouped[std_col].values)

        ax.plot(gens, y_mean, ls, linewidth=LINE_MAIN, color=color, label=label)
        ax.fill_between(gens,
                        np.clip(y_mean - y_std, 0, 1),
                        np.clip(y_mean + y_std, 0, 1),
                        alpha=0.10, color=color)
        # Annotate final value
        ax.annotate(f"{y_mean[-1]:.2f}",
                    xy=(gens[-1], y_mean[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center", fontweight="bold")

    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel("Operator Selection Probability", fontsize=LABEL_SIZE)
    ax.set_title("Adaptive Mutation — Operator Probability Over Generations\n"
                 "(Mean ± Std across all runs & instances)",
                 fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "exp5_operator_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  PLOT CHARTS FOR THESIS REPORT  (improved version)")
    print("=" * 62)

    print("\n[1] Convergence Curves ...")
    exp1_dir = "experiments/results/exp1_benchmark"
    if os.path.exists(exp1_dir):
        for f in sorted(glob.glob(os.path.join(exp1_dir, "*.csv"))):
            inst = os.path.basename(f).replace("_fixed.csv", "")
            plot_convergence(f, f"convergence_{inst}")

    print("\n[2] Ablation Boxplot ...")
    plot_ablation_boxplot()

    print("\n[3] Route Comparison (Foodie vs History Buff) ...")
    plot_route_comparison()

    print("\n[4] Personalization Category Distribution ...")
    plot_personalization_bars()

    print("\n[5] Sensitivity Analysis ...")
    plot_sensitivity()

    print("\n[6] Adaptive vs Static Mutation Boxplot ...")
    plot_exp5_boxplot()

    print("\n[7] Adaptive Operator Probability Curves ...")
    plot_exp5_operator_curves()

    print(f"\n{'=' * 62}")
    print(f"  All charts saved to: {CHART_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()
