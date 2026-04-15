"""
Visualization - Vẽ đồ thị cho báo cáo khóa luận.

Tạo:
  1. Convergence Curve: Best/Avg Fitness vs Generation (multi-run band)
  2. Route Scatter Plot: Foodie vs History Buff (colored by category)
  3. Personalization Bar Chart: Stacked % + grouped absolute (merged C101+C201)
  4. Sensitivity Line Charts: Score (left) + Time (right) dual-axis
  5. Budget Impact (Exp3): Category distribution + Score/POIs
  6. Ablation Convergence (Exp4): Normalized best score vs generation

Usage:
    cd backend
    py -m experiments.plot_charts
"""

import os
import sys
import json
import glob

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.artist import Artist
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
    "Nightlife": "#E91E63",
}
CAT_ORDER  = ["History", "Nature", "Food", "Shopping", "Entertain", "Nightlife"]
CAT_COLS   = ["cat_history_culture", "cat_nature_parks",
              "cat_food_drink", "cat_shopping", "cat_entertainment",
              "cat_nightlife_wellness"]

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
    ax.set_ylabel("Fitness value", fontsize=LABEL_SIZE)
    ax.set_title(f"Convergence curve - Instance {instance}", fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()

    _savefig(os.path.join(CHART_DIR, f"{output_name}.png"))


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
        "nightlife_wellness": "#E91E63",
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
        legend_handles: list[Artist] = [
            mpatches.Patch(facecolor=v, label=k.replace("_", " ").title(), edgecolor="white")
            for k, v in CAT_COLOR_POI.items() if k != "depot"
        ]
        legend_handles.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor=PALETTE["gold"],
                   markersize=11, label="Depot")
        )
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
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
    plt.suptitle(f"Personalization effect - Instance {inst_used}",
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

    plt.suptitle("Sensitivity analysis - Score vs Execution time Trade-off\n"
                 "(Averaged over 6 Solomon instances, 5 runs each)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    _savefig(os.path.join(CHART_DIR, "sensitivity_analysis.png"))



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

    cat_pct_cols = ["Cat_Hist%", "Cat_Nat%", "Cat_Food%", "Cat_Shop%", "Cat_Ent%", "Cat_Night%"]
    cat_labels = ["History", "Nature", "Food", "Shopping", "Entertainment", "Nightlife"]
    cat_colors = ["#4169E1", "#228B22", "#FF6347", "#DAA520", "#9370DB", "#E91E63"]

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

    fig.suptitle("Impact of Budget constraints - Instance RC201",
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
    ax.set_title("Ablation study - Convergence comparison",
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

