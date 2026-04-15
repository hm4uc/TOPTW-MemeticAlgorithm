# TOPTW-HybridGA

**A Memetic Algorithm for the Team Orienteering Problem with Time Windows (TOPTW) — Personalized Travel Itinerary Optimization**

> Thesis project — Faculty of Information Technology, VNU University of Engineering and Technology

## Overview

This system solves the **Team Orienteering Problem with Time Windows (TOPTW)** with user-dependent scoring. Unlike classic TOPTW where each POI has a fixed profit, our model computes personalized scores based on user preferences across 6 interest categories, subject to budget and time window constraints.

The solver implements a **Memetic Algorithm (MA)** — a Genetic Algorithm hybridized with local search operators — exposed via a REST API.

## Algorithm Design

| Component | Details |
|---|---|
| **Encoding** | Permutation of POI IDs (depot-sandwiched) |
| **Initialization** | 80% Randomized Insertion Heuristic + 20% Pure Random |
| **Selection** | Tournament Selection (k=3) |
| **Crossover** | Order Crossover (OX1) |
| **Mutation** | Adaptive-Lite 3-operator: 2-opt, Swap, Insertion |
| **Local Search** | Smart Repair (remove worst score/time POI) + Greedy Refill |
| **Replacement** | (μ+λ) Merged Elitist with duplicate elimination |
| **Stopping** | Early stopping after 25 stagnant generations |

### Key Features

- **User-Dependent Scoring**: POI scores weighted by 6 interest categories (1–5 stars each)
- **Budget Constraint**: Ticket prices per POI category, based on Hanoi tourism statistics
- **Adaptive Mutation**: Operator probabilities shift dynamically based on search progress, diversity, and insertion failure rate
- **Ablation-Ready**: Every component can be toggled on/off for controlled experiments

## Interest Categories

| Category | Description | Example (Hanoi) |
|---|---|---|
| `history_culture` | Historical & cultural sites | Temple of Literature, Ho Chi Minh Mausoleum |
| `nature_parks` | Nature & parks | Hoan Kiem Lake, West Lake |
| `food_drink` | Food & dining | Old Quarter street food, Bun Cha |
| `shopping` | Shopping & markets | Dong Xuan Market |
| `entertainment` | Entertainment venues | Cinema, Opera House |
| `nightlife_wellness` | Nightlife & wellness | Ta Hien Street, Spa, Night market |

## Project Structure

```
TOPTW-HybridGA/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entry point
│   │   ├── api/
│   │   │   └── routes.py              # POST /api/optimize endpoint
│   │   ├── core/
│   │   │   └── config.py             # Central configuration & GA parameters
│   │   ├── models/
│   │   │   ├── domain.py             # POI, Individual data classes
│   │   │   ├── requests.py           # UserPreferences (Pydantic)
│   │   │   └── responses.py          # OptimizationResponse (Pydantic)
│   │   └── services/
│   │       ├── data_loader.py        # Solomon instance loader & cache
│   │       └── algorithm/
│   │           ├── hga_engine.py     # MA main loop (selection → crossover → mutation → repair)
│   │           ├── initialization.py # Population initialization strategies
│   │           ├── fitness.py        # Fitness evaluation & distance matrix
│   │           ├── response_builder.py # Build API response from solution
│   │           └── operators/
│   │               ├── crossover.py  # Order Crossover (OX1)
│   │               ├── mutation.py   # 2-opt, Swap, Insertion mutation
│   │               └── repair.py    # Smart Repair + Greedy Refill
│   ├── data/
│   │   └── solomon_instances/        # Solomon benchmark datasets (6 instances)
│   │       ├── C101.csv, C201.csv    # Clustered instances
│   │       ├── R101.csv, R201.csv    # Random instances
│   │       ├── RC101.csv, RC201.csv  # Mixed instances
│   │       └── extended/             # Extended CSVs with category & price
│   └── experiments/
│       ├── benchmark_runner.py       # Batch experiment runner
│       ├── generate_extended_data.py # Generate category & price data
│       ├── exp1_benchmark.py         # Exp1: HGA vs Labadie GVNS (2012)
│       ├── exp2_personalization.py   # Exp2: Personalization value
│       ├── exp3_budget_impact.py     # Exp3: Budget constraint impact
│       ├── exp4_ablation_repair.py   # Exp4: Ablation study
│       ├── exp5_sensitivity.py       # Exp5: Parameter sensitivity
│       ├── analyze_results.py        # Auto-generate summary CSVs
│       └── plot_charts.py            # Publication-ready charts
├── LICENSE
└── README.md
```

## API

### `POST /api/optimize`

**Request:**

```json
{
  "instance_name": "C101",
  "budget": 500000,
  "start_time": 8.0,
  "end_time": 17.0,
  "start_node_id": 0,
  "interests": {
    "history_culture": 5,
    "nature_parks": 3,
    "food_drink": 4,
    "shopping": 1,
    "entertainment": 2,
    "nightlife_wellness": 3
  }
}
```

**Response:** Optimized itinerary with total score, cost, distance, duration, execution time, and a detailed ordered list of POIs with arrival/departure times.

Interactive API docs available at `http://localhost:8000/docs` after starting the server.

## Getting Started

### Prerequisites

- Python 3.11+

### Install & Run

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`.

### Run Experiments

```bash
cd backend

# Generate extended dataset (category + price)
py -m experiments.generate_extended_data

# Benchmark vs Labadie (2012) — 30 runs × 6 instances
py -m experiments.exp1_benchmark

# Personalization value — 10 runs × 5 profiles
py -m experiments.exp2_personalization

# Budget impact — 10 runs × 3 tiers
py -m experiments.exp3_budget_impact

# Ablation study — 10 runs × 3 variants × 6 instances
py -m experiments.exp4_ablation_repair

# Parameter sensitivity — 5 runs × 16 configs × 6 instances
py -m experiments.exp5_sensitivity

# Analyze & plot
py -m experiments.analyze_results
py -m experiments.plot_charts
```

## Benchmark Results (Solomon TOPTW)

Comparison with Labadie et al. (2012) GVNS on 6 Solomon instances, single-vehicle (m=1):

| Instance | BKS | GVNS Avg | HGA Best | HGA Avg | HGA Gap% |
|---|---|---|---|---|---|
| C101 | 320 | 320.0 | 320 | — | — |
| C201 | 870 | 850.0 | — | — | — |
| R101 | 198 | 197.0 | — | — | — |
| R201 | 797 | 775.6 | — | — | — |
| RC101 | 219 | 219.0 | — | — | — |
| RC201 | 795 | 784.0 | — | — | — |

> Results will be updated after running `exp1_benchmark.py` with the latest code.

## License

Distributed under the MIT License. See `LICENSE` for details.

## Author

**Hoang Minh Duc** — Faculty of Information Technology, VNU University of Engineering and Technology (VNU-UET)
