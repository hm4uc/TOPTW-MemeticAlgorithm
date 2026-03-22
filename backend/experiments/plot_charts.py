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


# (Ablation study chart removed — replaced by Adaptive Mutation in exp3)
def plot_ablation_boxplot(results_dir: str = "experiments/results/exp3_ablation"):
    """
    Redesigned ablation chart — hai panel riêng biệt để tránh noise khi pool instances:

    Root causes của noise cũ:
      1) Pool tất cả instances vào một boxplot → scale khác nhau (BKS C101=320 vs C201=870)
         gây ra phân tán nhân tạo. Dù normalize thì vẫn bị ảnh hưởng vì các instances
         dễ/khó khác nhau (C101 luôn đạt 100%, RC201 biến động nhiều).
      2) Outlier từ RC201 (std ~0.5-1%) kéo whisker xuống thấp cho mọi variant.
      3) Filename duplicates (C101_full_hga_C101.csv) có thể load nhầm.

    Giải pháp — Hiển thị hai panel:
      LEFT:  Grouped bar theo từng instance → thấy CHÍNH XÁC variant nào tệ ở đâu
      RIGHT: Delta bar (variant - full_hga) per instance → thấy IMPACT thực sự
             — Không còn noise do scale/pooling
    """
    VARIANT_INFO = {
        "full_hga":           ("Full HGA",          PALETTE["green"]),
        "no_smart_repair":    ("No Smart Repair",    PALETTE["orange"]),
        "no_insertion_mut":   ("No Ins. Mutation",   PALETTE["blue"]),
        "no_heuristic_init":  ("No Heuristic Init",  PALETTE["purple"]),
        "no_diversity_check": ("No Diversity Check", PALETTE["teal"]),
    }
    variants = list(VARIANT_INFO.keys())

    # ── Load data — canonical files only (no duplicate suffix) ─────────────────
    # Canonical: {INST}_{variant}.csv where variant does NOT contain inst name again
    data = {}  # variant -> {inst -> mean_norm}
    for variant in variants:
        data[variant] = {}
        for inst, bks in BKS.items():
            canonical = os.path.join(results_dir, f"{inst}_{variant}.csv")
            if not os.path.exists(canonical):
                continue
            df = pd.read_csv(canonical)
            if "total_score" not in df.columns:
                continue
            data[variant][inst] = np.mean(df["total_score"].values) / bks * 100.0

    exist_variants = [v for v in variants if data[v]]
    if not exist_variants:
        print("  WARN  no ablation data found")
        return

    # ── Build delta matrix: variant - full_hga, per instance ───────────────────
    baseline = data.get("full_hga", {})
    non_baseline = [v for v in exist_variants if v != "full_hga"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={"width_ratios": [1.6, 1.4]}
    )

    # ══ LEFT PANEL: Per-instance grouped bar ═══════════════════════════════════
    n_inst    = len(INSTANCES)
    n_var     = len(exist_variants)
    x         = np.arange(n_inst)
    bar_w     = 0.8 / n_var

    for j, variant in enumerate(exist_variants):
        label, color = VARIANT_INFO[variant]
        means = [data[variant].get(inst, np.nan) for inst in INSTANCES]
        offset = (j - n_var / 2 + 0.5) * bar_w
        bars = ax_left.bar(x + offset, means, bar_w * 0.92,
                           label=label, color=color, alpha=0.85,
                           edgecolor="white", linewidth=0.4)
        # Annotate bars that are meaningfully below BKS
        for bar_obj, m in zip(bars, means):
            if not np.isnan(m) and m < 99.9:
                ax_left.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                             bar_obj.get_height() - 0.05,
                             f"{m:.2f}", ha="center", va="top",
                             fontsize=6, color="white", fontweight="bold")

    ax_left.axhline(100, color="black", linewidth=1.2, linestyle="--", alpha=0.5)

    # Annotate BKS line
    ax_left.text(n_inst - 0.5, 100.05, "BKS = 100%",
                 fontsize=8, color="gray", va="bottom", ha="right")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(INSTANCES, fontsize=11)
    ax_left.set_ylabel("Normalized Score (% of BKS)", fontsize=LABEL_SIZE)
    ax_left.set_title("Per-Instance Score by Variant\n(Each bar = mean of 5 runs)",
                      fontsize=SUBTITLE_SIZE, fontweight="bold")

    # Dynamic Y range: zoom into the actual spread (highlight differences)
    all_means = [v for variant in exist_variants
                 for v in data[variant].values() if not np.isnan(v)]
    spread = max(all_means) - min(all_means)
    # Margin = 20% of spread below the min, to show bars clearly
    margin = max(0.3, spread * 0.2)
    ymin = max(96.5, min(all_means) - margin)
    ax_left.set_ylim(ymin, 101.2)
    ax_left.legend(fontsize=LEGEND_SIZE, loc="lower right", framealpha=0.9,
                   ncol=2)
    ax_left.grid(True, alpha=0.22, axis="y", linestyle="--")

    # ══ RIGHT PANEL: Delta bar (mean degradation vs full_hga) ══════════════════
    # For each non-baseline variant: compute mean delta across all instances
    deltas      = {}   # variant -> [delta_per_inst]
    for variant in non_baseline:
        d = []
        for inst in INSTANCES:
            b = baseline.get(inst)
            v = data[variant].get(inst)
            if b is not None and v is not None:
                d.append(v - b)
        deltas[variant] = d

    x2    = np.arange(len(non_baseline))
    bar_w2 = 0.55

    inst_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
                   PALETTE["red"],  PALETTE["purple"], PALETTE["teal"]]

    # Stacked bar where each segment = one instance's delta
    bottoms = np.zeros(len(non_baseline))   # not used for grouped; we do grouped
    # Use grouped dots + bar instead — one bar set per instance
    bar_width_inst = bar_w2 / len(INSTANCES)
    for k, inst in enumerate(INSTANCES):
        vals = [deltas[v][k] if k < len(deltas[v]) else 0
                for v in non_baseline]
        offset = (k - len(INSTANCES)/2 + 0.5) * bar_width_inst
        ax_right.bar(x2 + offset, vals, bar_width_inst * 0.9,
                     label=inst, color=inst_colors[k], alpha=0.80,
                     edgecolor="white", linewidth=0.3)

    # Mean delta line on top
    mean_deltas = [np.mean(deltas[v]) for v in non_baseline]
    ax_right.plot(x2, mean_deltas, "D--", color="black", linewidth=1.8,
                  markersize=8, zorder=10, label="Mean Δ")
    for xi, md in zip(x2, mean_deltas):
        color = "#D32F2F" if md < 0 else "#388E3C"
        sign  = "+" if md >= 0 else ""
        ax_right.annotate(f"{sign}{md:.3f}%",
                          xy=(xi, md),
                          xytext=(0, 10 if md >= 0 else -14),
                          textcoords="offset points",
                          ha="center", fontsize=9,
                          color=color, fontweight="bold")

    ax_right.axhline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.35)
    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(
        [VARIANT_INFO[v][0].replace(" ", "\n") for v in non_baseline],
        fontsize=9.5
    )
    ax_right.set_ylabel("Score Δ vs Full HGA (%)", fontsize=LABEL_SIZE)
    ax_right.set_title("Component Impact\n(Δ = variant mean − Full HGA mean, per instance)",
                       fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax_right.legend(fontsize=LEGEND_SIZE, framealpha=0.9, title="Instance",
                    title_fontsize=8, loc="lower left", ncol=2)
    ax_right.grid(True, alpha=0.22, axis="y", linestyle="--")

    # Shade the negative zone
    ylim2 = ax_right.get_ylim()
    ax_right.axhspan(ylim2[0], 0, color="red",   alpha=0.04, zorder=0)
    ax_right.axhspan(0, ylim2[1], color="green", alpha=0.04, zorder=0)

    plt.suptitle("Ablation Study — Component Contribution Analysis",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "ablation_boxplot.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  2. Route Scatter Plot  (colored by POI category)
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
#  3. Personalization Bar Chart  (merged C101+C201, better layout)
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

    # Load data — prefer RC201 (richer distribution) for visualization
    data = {}
    for name in profiles:
        for inst in ["RC201", "C201", "C101"]:
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
#  4. Sensitivity Analysis  (dual-axis, std band, default marker)
# ══════════════════════════════════════════════════════════════════════════════
def plot_sensitivity(results_dir: str = "experiments/results/exp5_sensitivity"):
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
    defaults = {"population_size": 100, "mutation_rate": 0.3,
                "tournament_k": 3, "stagnation_limit": 25}

    # ── Pre-collect all data to find global scales ───────────────────────────
    all_data = {}
    global_max_time = 0

    for param, values in param_grid.items():
        all_data[param] = {"values": values, "means_score": [], "stds_score": [], "means_time": []}
        for val in values:
            norm_scores, times = [], []
            for inst in INSTANCES:
                csv_path = os.path.join(results_dir, f"{inst}_{param}_{val}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    norm_scores.extend((df["total_score"] / BKS[inst] * 100).tolist())
                    times.extend(df["execution_time"].tolist())
            
            m_score = np.mean(norm_scores) if norm_scores else 0
            s_score = np.std(norm_scores)   if norm_scores else 0
            m_time  = np.mean(times)         if times else 0
            
            all_data[param]["means_score"].append(m_score)
            all_data[param]["stds_score"].append(s_score)
            all_data[param]["means_time"].append(m_time)
            
            if m_time > global_max_time:
                global_max_time = m_time

    # Buffer for y2 axis
    y2_max = (int(global_max_time / 5) + 1) * 5 if global_max_time > 0 else 50

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (param, data) in enumerate(all_data.items()):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        
        values = data["values"]
        arr_score = np.array(data["means_score"])
        arr_std   = np.array(data["stds_score"])
        means_time = data["means_time"]

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
            # Vertical band centered on default
            # Use original values for spacing even if x-axis is categorical
            try:
                # Find index of default_val
                v_idx = values.index(default_val)
                # Estimate half-gap based on actual neighbor values
                if len(values) > 1:
                    if v_idx < len(values) - 1:
                        half_gap = (values[v_idx+1] - values[v_idx]) * 0.4
                    else:
                        half_gap = (values[v_idx] - values[v_idx-1]) * 0.4
                else:
                    half_gap = 1.0

                ax1.axvspan(default_val - half_gap, default_val + half_gap,
                            color="gray", alpha=0.08, zorder=0)
                ax1.axvline(default_val, color="gray", linestyle=":", linewidth=1.2,
                            alpha=0.6, zorder=0)
                ax1.annotate("default", xy=(default_val, ax1.get_ylim()[0]),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=7, color="gray")
            except ValueError:
                pass

        ax1.set_xlabel(param_titles[param], fontsize=LABEL_SIZE)
        ax1.set_ylabel("Normalized Score (% of BKS)", fontsize=10,
                       color=PALETTE["blue"])
        ax2.set_ylabel("Execution Time (s)", fontsize=10, color=PALETTE["orange"])
        ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["orange"])
        
        # Synchronize Y2 Axis
        ax2.set_ylim(0, y2_max)
        if y2_max <= 50:
            ax2.set_yticks(np.arange(0, y2_max + 1, 5))
        elif y2_max <= 200:
            ax2.set_yticks(np.arange(0, y2_max + 1, 20))
        
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
#  5. Adaptive Boxplot  (swarm-like jitter + per-instance delta table)
# ══════════════════════════════════════════════════════════════════════════════
def plot_exp3_boxplot(results_dir: str = "experiments/results/exp3_adaptive_mutation"):
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
    _savefig(os.path.join(CHART_DIR, "exp3_adaptive_boxplot.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  6. Adaptive Operator Curves  (rolling smooth + shaded band)
# ══════════════════════════════════════════════════════════════════════════════
def plot_exp3_operator_curves(results_dir: str = "experiments/results/exp3_adaptive_mutation"):
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
    _savefig(os.path.join(CHART_DIR, "exp3_operator_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  5b. Budget Impact — Category Distribution & Score (Exp3)
# ══════════════════════════════════════════════════════════════════════════════
def plot_exp3_budget_bars(
    summary_csv: str = "experiments/results/summary/exp3_budget_impact.csv",
):
    """
    Vẽ 2-panel chart:
      Left:  Stacked bar chart — Category Distribution (%) per budget tier
      Right: Score + POIs comparison bar chart
    """
    if not os.path.exists(summary_csv):
        print(f"  SKIP  {summary_csv} not found")
        return

    df = pd.read_csv(summary_csv)

    # Sắp xếp theo budget tăng dần
    tier_order = ["backpacker_200k", "standard_500k", "luxury_unlimited"]
    tier_labels = ["Backpacker\n(200K)", "Standard\n(500K)", "Luxury\n(Unlimited)"]
    df["sort_key"] = df["Budget_Tier"].map({t: i for i, t in enumerate(tier_order)})
    df = df.sort_values("sort_key").reset_index(drop=True)

    cat_pct_cols = ["Cat_Hist%", "Cat_Nat%", "Cat_Food%", "Cat_Shop%", "Cat_Ent%"]
    cat_labels = ["History", "Nature", "Food", "Shopping", "Entertainment"]
    cat_colors = ["#4169E1", "#228B22", "#FF6347", "#DAA520", "#9370DB"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})

    # ── LEFT: Stacked Bar — Category Distribution ────────────────────────
    x = np.arange(len(df))
    bar_width = 0.55
    bottom = np.zeros(len(df))

    for i, (col, label, color) in enumerate(zip(cat_pct_cols, cat_labels, cat_colors)):
        vals = df[col].values
        bars = ax1.bar(x, vals, bar_width, bottom=bottom, label=label,
                       color=color, edgecolor="white", linewidth=0.5)
        # Annotate % nếu > 5%
        for j, v in enumerate(vals):
            if v > 5:
                ax1.text(x[j], bottom[j] + v / 2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=ANNOT_SIZE,
                         fontweight="bold", color="white")
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_labels, fontsize=LABEL_SIZE)
    ax1.set_ylabel("Category Distribution (%)", fontsize=LABEL_SIZE)
    ax1.set_title("POI Category Distribution by Budget",
                   fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax1.legend(fontsize=LEGEND_SIZE, loc="upper right", ncol=2,
               framealpha=0.9, edgecolor="#CCCCCC")
    ax1.set_ylim(0, 105)

    # ── RIGHT: Score & POIs Grouped Bar ──────────────────────────────────
    x2 = np.arange(len(df))
    w = 0.3

    color_score = PALETTE["blue"]
    color_pois = PALETTE["teal"]

    bars1 = ax2.bar(x2 - w/2, df["Score_Avg"].values, w, label="Score",
                    color=color_score, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Total Score", fontsize=LABEL_SIZE, color=color_score)

    # Annotate score
    for bar, val in zip(bars1, df["Score_Avg"].values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:.0f}", ha="center", va="bottom",
                 fontsize=ANNOT_SIZE + 1, fontweight="bold", color=color_score)

    # Second axis for POIs
    ax2b = ax2.twinx()
    bars2 = ax2b.bar(x2 + w/2, df["POIs_Avg"].values, w, label="POIs",
                     color=color_pois, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2b.set_ylabel("Number of POIs in Route", fontsize=LABEL_SIZE, color=color_pois)

    # Annotate POIs
    for bar, val in zip(bars2, df["POIs_Avg"].values):
        ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  f"{val:.1f}", ha="center", va="bottom",
                  fontsize=ANNOT_SIZE + 1, fontweight="bold", color=color_pois)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(tier_labels, fontsize=LABEL_SIZE)
    ax2.set_title("Score & POIs by Budget",
                   fontsize=SUBTITLE_SIZE, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               fontsize=LEGEND_SIZE, loc="upper left", framealpha=0.9)

    fig.suptitle("Impact of Budget Constraints — Instance RC201",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "exp3_budget_impact.png")
    _savefig(out)



# ══════════════════════════════════════════════════════════════════════════════
#  5. Ablation Study — Convergence Comparison (Exp4)
# ══════════════════════════════════════════════════════════════════════════════
def plot_exp4_ablation_convergence(
    conv_csv: str = "experiments/results/exp4_ablation_repair/convergence_comparison.csv",
):
    """
    Vẽ đường cong hội tụ so sánh 3 ablation variants.
    Trục X = Generation, trục Y = Normalized Best Score (% BKS).
    Band ±1σ cho mỗi variant.
    """
    if not os.path.exists(conv_csv):
        print(f"  SKIP  {conv_csv} not found")
        return

    df = pd.read_csv(conv_csv)

    variant_style = {
        "full_hga":         {"color": PALETTE["green"],  "label": "Full HGA",         "ls": "-"},
        "no_smart_repair":  {"color": PALETTE["orange"], "label": "No Smart Repair",  "ls": "--"},
        "no_local_search":  {"color": PALETTE["red"],    "label": "No Local Search",  "ls": "-."},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Chỉ vẽ đến gen mà N_Runs >= 5 (đủ tin cậy)
    min_runs = 5

    for variant, style in variant_style.items():
        vdf = df[(df["Variant"] == variant) & (df["N_Runs"] >= min_runs)].copy()
        if vdf.empty:
            continue

        gens = vdf["Generation"].values
        means = vdf["Norm_Best_Mean"].values
        stds = vdf["Norm_Best_Std"].fillna(0).values

        # Đường chính
        ax.plot(gens, means,
                color=style["color"], linestyle=style["ls"],
                linewidth=LINE_MAIN, label=style["label"], zorder=3)

        # Band ±1σ
        ax.fill_between(gens, means - stds, means + stds,
                        color=style["color"], alpha=0.12, zorder=1)

        # Annotate final value
        final_gen = gens[-1]
        final_val = means[-1]
        ax.annotate(f"{final_val:.1f}%",
                    xy=(final_gen, final_val),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=ANNOT_SIZE + 1, fontweight="bold",
                    color=style["color"],
                    va="center")

    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel("Normalized Best Score (% BKS)", fontsize=LABEL_SIZE)
    ax.set_title("Ablation Study — Convergence Comparison",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=12)

    # Đường tham chiếu 100% BKS
    ax.axhline(y=100, color="#999999", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(1, 100.3, "BKS (100%)", fontsize=ANNOT_SIZE, color="#999999")

    ax.legend(fontsize=LEGEND_SIZE + 1, loc="lower right",
              framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=82, top=102)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out = os.path.join(CHART_DIR, "exp4_ablation_convergence.png")
    _savefig(out)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
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

    print("\n[2] Route Comparison (Foodie vs History Buff) ...")
    plot_route_comparison()

    print("\n[3] Personalization Category Distribution ...")
    plot_personalization_bars()

    print("\n[4] Sensitivity Analysis ...")
    plot_sensitivity()

    print("\n[5] Budget Impact (Exp3) ...")
    plot_exp3_budget_bars()

    print("\n[6] Ablation Convergence Comparison ...")
    plot_exp4_ablation_convergence()

    print(f"\n{'=' * 62}")
    print(f"  All charts saved to: {CHART_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()

