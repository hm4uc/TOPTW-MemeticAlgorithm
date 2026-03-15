"""
Hybrid Genetic Algorithm Engine (HGA) — TOPTW Solver

Luồng chính mỗi thế hệ:
  1. Elitism   – giữ nguyên N cá thể tốt nhất.
  2. Selection – Tournament Selection chọn cặp cha-mẹ.
  3. Crossover – Order Crossover OX1 (trên interior).
  4. Mutation  – 2-opt / Swap / Insertion (trên interior).
  5. Smart Repair – xóa POI có tỷ lệ Score/Time kém nhất.
  6. Diversity – loại con trùng lặp, thay bằng cá thể random.
  7. Evaluate  – tính fitness cho con mới.
  8. Replace   – thay thế quần thể, sắp xếp, lặp lại.

Nguyên tắc "Depot-Safe":
  Mọi toán tử GA đều CHỈ thao tác trên "interior" = route[1:-1].
  Depot được gắn lại sau khi xử lý xong.

★ ABLATION FLAGS ★
  Hỗ trợ tắt/bật từng thành phần để đánh giá đóng góp (Ablation Study).
"""

import random
import time
from typing import Optional, List

from app.models.domain import POI, Individual
from app.models.requests import UserPreferences
from app.models.responses import OptimizationResponse
from app.services.data_loader import load_solomon_instance
from app.services.algorithm.initialization import (
    initialize_population,
    _create_random_individual,
)
from app.services.algorithm.fitness import (
    calculate_fitness,
    build_distance_matrix,
)
from app.services.algorithm.operators import (
    crossover_ox1,
    mutate,
    repair,
    greedy_refill,
)
from app.services.algorithm.response_builder import build_response
from app.core.config import (
    POPULATION_SIZE,
    PENALTY_WAIT,
    DEFAULT_MUTATION_RATE,
    DEFAULT_GENERATIONS,
    DEFAULT_STAGNATION_LIMIT,
    DEFAULT_TOURNAMENT_K,
    IMPROVEMENT_THRESHOLD,
    USE_ADAPTIVE_MUTATION_DEFAULT,
    ADAPTIVE_INSERT_START,
    ADAPTIVE_INSERT_END,
    ADAPTIVE_2OPT_START,
    ADAPTIVE_2OPT_END,
    ADAPTIVE_STAGNATION_TRIGGER,
    ADAPTIVE_LOW_DIVERSITY_THRESHOLD,
    ADAPTIVE_INSERT_FAIL_TRIGGER,
    ADAPTIVE_INSERT_FAIL_WINDOW,
    ADAPTIVE_MIN_PROB,
    ADAPTIVE_MAX_PROB,
)


class HybridGeneticAlgorithm:
    def __init__(
        self,
        user_prefs: UserPreferences,
        pois: Optional[List[POI]] = None,
        instance_name: str = "C101",
        # ── Ablation Flags ──────────────────────────────────────────
        use_smart_repair: bool = True,        # False → Simple Repair (xóa cuối)
        use_insertion_mutation: bool = True,   # False → chỉ 2-opt + Swap
        use_wait_penalty: bool = True,         # False → PENALTY_WAIT = 0
        use_heuristic_init: bool = True,       # False → 100% Random Init
        use_diversity_check: bool = True,      # False → không check duplicate
        use_urgency: bool = True,              # False → Labadie ratio gốc (không urgency)
        use_adaptive_mutation: bool = USE_ADAPTIVE_MUTATION_DEFAULT,
        # ── Tunable Parameters ──────────────────────────────────────
        population_size: int = POPULATION_SIZE,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        generations: int = DEFAULT_GENERATIONS,
        stagnation_limit: int = DEFAULT_STAGNATION_LIMIT,
        tournament_k: int = DEFAULT_TOURNAMENT_K,
    ):
        self.user_prefs = user_prefs

        # ── Load dữ liệu ────────────────────────────────────────────────────
        self.pois = pois if pois is not None else load_solomon_instance(instance_name)

        # ── Pre-compute Distance Matrix ──────────────────────────────────────
        build_distance_matrix(self.pois)
        self.depot: Optional[POI] = next((p for p in self.pois if p.id == 0), None)
        self.poi_map = {p.id: p for p in self.pois}

        # ── Ablation Flags ───────────────────────────────────────────────────
        self.use_smart_repair = use_smart_repair
        self.use_insertion_mutation = use_insertion_mutation
        self.use_wait_penalty = use_wait_penalty
        self.use_heuristic_init = use_heuristic_init
        self.use_diversity_check = use_diversity_check
        self.use_urgency = use_urgency
        self.use_adaptive_mutation = use_adaptive_mutation

        # ── GA Parameters ────────────────────────────────────────────────────
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.stagnation_limit = stagnation_limit
        self.improvement_threshold = IMPROVEMENT_THRESHOLD
        self.tournament_k = tournament_k
        self.population: list[Individual] = []

        # ── Wait penalty weight (configurable for ablation) ──────────────────
        self.wait_penalty_weight = PENALTY_WAIT if use_wait_penalty else 0.0

        # ── Results Tracking ─────────────────────────────────────────────────
        self.convergence_log: list[dict] = []
        self.actual_gens: int = 0
        self.best_individual: Optional[Individual] = None
        self.insert_fail_history: list[float] = []

    def _count_unique_routes(self, population: list[Individual]) -> int:
        """Đếm số route unique trong quần thể theo tập POI nội bộ."""
        return len({frozenset(p.id for p in ind.route[1:-1]) for ind in population})

    def _avg_insert_fail_rate(self) -> float:
        """Lấy fail-rate trung bình trong cửa sổ gần đây của insertion mutation."""
        if not self.insert_fail_history:
            return 0.0
        return sum(self.insert_fail_history) / len(self.insert_fail_history)

    def _update_insert_fail_rate(self, attempts: int, success: int) -> None:
        """Cập nhật rolling window insert fail-rate."""
        if attempts <= 0:
            return
        fail_rate = 1.0 - (success / attempts)
        self.insert_fail_history.append(fail_rate)
        if len(self.insert_fail_history) > ADAPTIVE_INSERT_FAIL_WINDOW:
            self.insert_fail_history.pop(0)

    def _append_convergence_log(
        self,
        gen: int,
        all_fitnesses: list[float],
        unique_routes: int,
        operator_probs: dict[str, float],
        avg_insert_fail: float,
    ) -> None:
        """Ghi log hội tụ mỗi thế hệ với telemetry mutation."""
        best_fit = all_fitnesses[0]
        avg_fit = sum(all_fitnesses) / len(all_fitnesses)

        self.convergence_log.append({
            "gen": gen,
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "median_fitness": all_fitnesses[len(all_fitnesses) // 2],
            "worst_fitness": all_fitnesses[-1],
            "unique_routes": unique_routes,
            "wait_time": self.population[0].total_wait,
            "best_score": self.population[0].total_score,
            "p_2opt": round(operator_probs["2opt"], 4),
            "p_swap": round(operator_probs["swap"], 4),
            "p_insert": round(operator_probs["insertion"], 4),
            "insert_fail_rate": round(avg_insert_fail, 4),
        })

    def _print_gen_status(
        self,
        gen: int,
        best_fit: float,
        avg_fit: float,
        unique_routes: int,
        operator_probs: dict[str, float],
        gens_without_improvement: int,
        duplicates_replaced: int,
    ) -> None:
        print(
            f"[HGA] Gen {gen:>3}/{self.generations} | "
            f"Best = {best_fit:8.2f} | "
            f"Avg = {avg_fit:8.2f} | "
            f"Unique = {unique_routes:>2}/{self.population_size} | "
            f"Wait = {self.population[0].total_wait:6.1f} | "
            f"P(2O/S/I) = {operator_probs['2opt']:.2f}/{operator_probs['swap']:.2f}/{operator_probs['insertion']:.2f} | "
            f"Stag = {gens_without_improvement:>2}/{self.stagnation_limit} | "
            f"Dup = {duplicates_replaced}"
        )

    def _build_operator_probs(
        self,
        gen: int,
        gens_without_improvement: int,
        unique_routes: int,
        insert_fail_rate: float,
    ) -> dict[str, float]:
        """Adaptive-Lite 2 tầng: schedule theo progress + feedback trạng thái quần thể."""
        if not self.use_adaptive_mutation:
            if self.use_insertion_mutation:
                return {"2opt": 0.3, "swap": 0.3, "insertion": 0.4}
            return {"2opt": 0.5, "swap": 0.5, "insertion": 0.0}

        if not self.use_insertion_mutation:
            return {"2opt": 0.5, "swap": 0.5, "insertion": 0.0}

        progress = gen / max(self.generations - 1, 1)

        p_insert = ADAPTIVE_INSERT_START + (ADAPTIVE_INSERT_END - ADAPTIVE_INSERT_START) * progress
        p_2opt = ADAPTIVE_2OPT_START + (ADAPTIVE_2OPT_END - ADAPTIVE_2OPT_START) * progress
        p_swap = max(0.0, 1.0 - p_insert - p_2opt)

        # Tầng 2 - feedback theo stagnation
        if gens_without_improvement >= ADAPTIVE_STAGNATION_TRIGGER:
            p_2opt += 0.10
            p_swap -= 0.05
            p_insert -= 0.05

        # Tầng 2 - feedback theo diversity
        diversity_ratio = unique_routes / max(self.population_size, 1)
        if diversity_ratio < ADAPTIVE_LOW_DIVERSITY_THRESHOLD:
            p_swap += 0.10
            p_2opt -= 0.05
            p_insert -= 0.05

        # Tầng 2 - feedback theo insert fail-rate gần đây
        if insert_fail_rate >= ADAPTIVE_INSERT_FAIL_TRIGGER:
            p_insert -= 0.10
            p_2opt += 0.10

        # Clamp và normalize
        p_2opt = min(max(p_2opt, ADAPTIVE_MIN_PROB), ADAPTIVE_MAX_PROB)
        p_swap = min(max(p_swap, ADAPTIVE_MIN_PROB), ADAPTIVE_MAX_PROB)
        p_insert = min(max(p_insert, ADAPTIVE_MIN_PROB), ADAPTIVE_MAX_PROB)

        total = p_2opt + p_swap + p_insert
        return {
            "2opt": p_2opt / total,
            "swap": p_swap / total,
            "insertion": p_insert / total,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 1: Population Initialization
    # ══════════════════════════════════════════════════════════════════════════
    def _initialize_population(self) -> list[Individual]:
        self.population = initialize_population(
            self.pois, self.user_prefs,
            use_heuristic_init=self.use_heuristic_init,
            use_urgency=self.use_urgency,
        )
        # ★ Repair + Greedy Refill cho MỖI cá thể khởi tạo
        for i, ind in enumerate(self.population):
            ind = repair(ind, self.user_prefs, self.use_smart_repair)
            ind = greedy_refill(ind, self.pois, self.user_prefs, self.use_urgency)
            calculate_fitness(ind, self.user_prefs, self.wait_penalty_weight)
            self.population[i] = ind
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        print("[HGA] Population initialized and evaluated.")
        print(f"      Best fitness  = {self.population[0].fitness:.2f}")
        print(f"      Worst fitness = {self.population[-1].fitness:.2f}")
        return self.population

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 2: Fitness Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    def _evaluate_fitness(self, individual: Individual) -> float:
        return calculate_fitness(individual, self.user_prefs, self.wait_penalty_weight)

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 3: Parent Selection — Tournament
    # ══════════════════════════════════════════════════════════════════════════
    def _select_parents(
        self, population: list[Individual]
    ) -> tuple[Individual, Individual]:
        """Tournament Selection: chọn k cá thể ngẫu nhiên, lấy cá thể tốt nhất."""
        def tournament(pop: list[Individual]) -> Individual:
            contestants = random.sample(pop, min(self.tournament_k, len(pop)))
            return max(contestants, key=lambda ind: ind.fitness)

        p1 = tournament(population)

        if len(population) <= 1:
            return p1, p1

        p2 = tournament(population)
        retries = 3
        while p1 == p2 and retries > 0:
            p2 = tournament(population)
            retries -= 1

        return p1, p2

    # ══════════════════════════════════════════════════════════════════════════
    #  Diversity Check (Chống đồng huyết)
    # ══════════════════════════════════════════════════════════════════════════
    def _is_duplicate(self, child: Individual, population: list[Individual]) -> bool:
        """Kiểm tra `child` có trùng với bất kỳ cá thể nào trong `population`."""
        child_ids = frozenset(p.id for p in child.route[1:-1])
        return any(
            child_ids == frozenset(p.id for p in ind.route[1:-1])
            for ind in population
        )

    def _create_diverse_individual(self) -> Individual:
        """
        Tạo 1 cá thể mới khi phát hiện bản sao.

        ★ Cải tiến: Thay vì random thuần (fitness rất thấp), ta:
          1. Tạo random cơ bản
          2. Repair nếu vi phạm
          3. Greedy Refill để lấp đầy route → chất lượng cao hơn nhiều
        """
        ind = _create_random_individual(self.pois, self.depot, self.user_prefs)
        ind = repair(ind, self.user_prefs, self.use_smart_repair)
        ind = greedy_refill(ind, self.pois, self.user_prefs, self.use_urgency)
        calculate_fitness(ind, self.user_prefs, self.wait_penalty_weight)
        return ind

    # ══════════════════════════════════════════════════════════════════════════
    #  Main Loop — Early Stopping + Convergence Logging
    # ══════════════════════════════════════════════════════════════════════════
    def run(self) -> OptimizationResponse:
        """
        Chạy vòng lặp tiến hóa chính của Hybrid GA.

        ★ EARLY STOPPING ★
          Nếu best fitness không cải thiện (>= threshold) trong
          `stagnation_limit` thế hệ liên tiếp → dừng sớm.

        ★ CONVERGENCE LOGGING ★
          Lưu metrics mỗi thế hệ vào self.convergence_log để vẽ đồ thị.

        ★ ABLATION FLAGS ★
          Các toán tử có thể tắt/bật qua flags trong __init__.
        """
        start_time = time.perf_counter()

        self._initialize_population()
        best_ever = self.population[0]
        gens_without_improvement = 0
        actual_gens = 0

        for gen in range(self.generations):
            actual_gens = gen + 1
            duplicates_replaced = 0
            unique_routes_before = self._count_unique_routes(self.population)
            avg_insert_fail = self._avg_insert_fail_rate()
            operator_probs = self._build_operator_probs(
                gen,
                gens_without_improvement,
                unique_routes_before,
                avg_insert_fail,
            )
            insertion_attempts = 0
            insertion_success = 0

            # ── Tạo con cái ──────────────────────────────────────────────────
            children: list[Individual] = []
            while len(children) < self.population_size:
                p1, p2 = self._select_parents(self.population)
                child = crossover_ox1(p1, p2, self.depot)
                child, op_name, op_success = mutate(
                    child, self.depot, self.pois, self.user_prefs,
                    self.mutation_rate, self.use_insertion_mutation,
                    operator_probs=operator_probs,
                    collect_stats=True,
                )

                if op_name == "insertion":
                    insertion_attempts += 1
                    if op_success:
                        insertion_success += 1

                child = repair(child, self.user_prefs, self.use_smart_repair)
                child = greedy_refill(child, self.pois, self.user_prefs, self.use_urgency)
                calculate_fitness(child, self.user_prefs, self.wait_penalty_weight)
                children.append(child)

            self._update_insert_fail_rate(insertion_attempts, insertion_success)

            # ── Merged Replacement — giữ best từ (parents + children) ────────
            merged = list(self.population) + children
            merged.sort(key=lambda ind: ind.fitness, reverse=True)

            new_population: list[Individual] = []
            for ind in merged:
                if self.use_diversity_check and self._is_duplicate(ind, new_population):
                    duplicates_replaced += 1
                    continue
                new_population.append(ind)
                if len(new_population) >= self.population_size:
                    break

            # Nếu chưa đủ (do loại trùng), bổ sung cá thể đa dạng
            while len(new_population) < self.population_size:
                new_population.append(self._create_diverse_individual())
                duplicates_replaced += 1

            new_population.sort(key=lambda ind: ind.fitness, reverse=True)
            self.population = new_population

            # ── Cập nhật Best Ever + Early Stopping ──────────────────────────
            improvement = self.population[0].fitness - best_ever.fitness
            if improvement > self.improvement_threshold:
                best_ever = self.population[0]
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            # ── Enhanced Logging ──────────────────────────────────────────────
            all_fitnesses = sorted([ind.fitness for ind in self.population], reverse=True)
            best_fit = all_fitnesses[0]
            avg_fit = sum(all_fitnesses) / len(all_fitnesses)
            unique_routes = self._count_unique_routes(self.population)

            self._append_convergence_log(
                gen=gen + 1,
                all_fitnesses=all_fitnesses,
                unique_routes=unique_routes,
                operator_probs=operator_probs,
                avg_insert_fail=avg_insert_fail,
            )
            self._print_gen_status(
                gen=gen + 1,
                best_fit=best_fit,
                avg_fit=avg_fit,
                unique_routes=unique_routes,
                operator_probs=operator_probs,
                gens_without_improvement=gens_without_improvement,
                duplicates_replaced=duplicates_replaced,
            )

            # ── Early Stopping Check ──────────────────────────────────────────
            if gens_without_improvement >= self.stagnation_limit:
                print(
                    f"\n[HGA] ★ EARLY STOPPING ★ "
                    f"Best fitness không cải thiện trong "
                    f"{self.stagnation_limit} thế hệ liên tiếp. "
                    f"Dừng tại gen {gen + 1}/{self.generations}."
                )
                break

        elapsed = time.perf_counter() - start_time

        # ── Store results ─────────────────────────────────────────────────────
        self.actual_gens = actual_gens
        self.best_individual = best_ever

        print(f"\n[HGA] ═══ KẾT QUẢ CUỐI CÙNG ═══")
        print(f"      Generations run   = {actual_gens}/{self.generations}")
        print(f"      Best-ever fitness = {best_ever.fitness:.2f}")
        print(f"      Total wait time   = {best_ever.total_wait:.1f}")
        print(f"      Route IDs   : {[p.id for p in best_ever.route]}")
        print(f"      Route length: {len(best_ever.route)} nodes "
              f"({len(best_ever.route) - 2} POIs + 2 Depot)")
        print(f"      Execution time: {elapsed:.4f}s")

        return build_response(best_ever, self.user_prefs, elapsed)