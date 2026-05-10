# Tài liệu Backend - TOPTW Memetic Algorithm

## 1) Mục tiêu và phạm vi

Tài liệu này mô tả toàn bộ backend trong `backend/`: kiến trúc, luồng dữ liệu, API, thuật toán Memetic Algorithm (MA), hệ sinh thái thực nghiệm và các hướng cải tiến.

- Ngôn ngữ: Python 3.11+ / FastAPI (`backend/app`).
- Bài toán: Team Orienteering Problem with Time Windows (TOPTW), mở rộng theo hướng cá nhân hóa và ràng buộc ngân sách.
- Dữ liệu: Solomon benchmark instances CSV (`backend/data/solomon_instances`).

## 2) Kiến trúc tổng quan

### 2.1 Các lớp chính

- **API layer:**
  - `app/main.py` - Khởi tạo FastAPI app, CORS, mount router với prefix `/api`.
  - `app/api/routes.py` - Endpoint `POST /api/optimize`.

- **Contract layer:**
  - `app/models/requests.py` - `UserPreferences` + validation + quy đổi giờ -> phút.
  - `app/models/responses.py` - `OptimizationResponse`, `ItineraryItem`.
  - `app/models/domain.py` - Domain object `POI`, `Individual`.

- **Data layer:**
  - `app/services/data_loader.py` - Load và cache Solomon instance (ưu tiên file extended).

- **Algorithm layer:**
  - `app/services/algorithm/ma_engine.py` - Điều phối vòng lặp MA chính.
  - `app/services/algorithm/initialization.py` - Khởi tạo quần thể (80 heuristic + 20 random).
  - `app/services/algorithm/operators/crossover.py` - Order Crossover (OX1).
  - `app/services/algorithm/operators/mutation.py` - 2-opt, Swap, Insertion mutation.
  - `app/services/algorithm/operators/repair.py` - Smart Repair + Greedy Refill.
  - `app/services/algorithm/fitness.py` - Ma trận khoảng cách, feasibility, fitness.
  - `app/services/algorithm/response_builder.py` - Format kết quả API.

- **Cấu hình trung tâm:**
  - `app/core/config.py` - Hệ số phạt, tham số GA mặc định, tham số adaptive mutation.

### 2.2 Luồng dữ liệu runtime (từ request đến response)

1. Client gọi `POST /api/optimize`.
2. Pydantic validate `UserPreferences` (`instance_name`, budget, khung giờ, interests đủ 6 nhóm).
3. Route layer load POIs theo `instance_name` qua `load_solomon_instance(...)`, kiểm tra `start_node_id` tồn tại.
4. Tạo `MemeticAlgorithm(user_prefs)` -> `run()`.
5. Engine khởi tạo quần thể, lai ghép / đột biến / sửa chữa / bổ sung, tính fitness, early stopping.
6. Chọn cá thể tốt nhất -> `build_response(...)` trả về timeline HH:MM + tổng chỉ số.

## 3) API contract và validation

### 3.1 Endpoint

- `POST /api/optimize` (`app/api/routes.py`).
- Response model: `OptimizationResponse`.
- Các nhóm lỗi được mô tả rõ: 400/404/422/500.

### 3.2 Request (`UserPreferences`)

Trường bắt buộc:
- `instance_name` (mặc định `C101`), hợp lệ trong: `C101`, `C201`, `R101`, `R201`, `RC101`, `RC201`.
- `budget` > 0 (đơn vị VND).
- `start_time`, `end_time` (đơn vị giờ), với `start_time < end_time`.
- Thời lượng tối thiểu `end_time - start_time >= 1.0` giờ.
- `start_node_id` (được kiểm tra tồn tại trong dataset tại route layer).
- `interests` phải có đủ 6 key:
  - `history_culture`, `nature_parks`, `food_drink`, `shopping`, `entertainment`, `nightlife_wellness`.

Bảng ánh xạ số sao -> trọng số:

| Số sao | Trọng số | Mô tả |
|---|---|---|
| 1 | 0.1 | Không quan tâm |
| 2 | 0.5 | Ít quan tâm |
| 3 | 1.0 | Trung bình (mức cơ sở) |
| 4 | 1.5 | Quan tâm nhiều |
| 5 | 2.0 | Rất quan tâm / ưu tiên cao nhất |

Chuẩn hóa trọng số:
- `interest_weights` được scale sao cho tổng trọng số = số category (6).
- Nếu người dùng cho các mức sao đồng đều, mỗi category về ~1.0, tránh méo score.

### 3.3 Response (`OptimizationResponse`)

- Tổng quan: `total_score`, `total_cost`, `total_distance`, `total_duration`, `execution_time`.
- Chi tiết route: danh sách `ItineraryItem` với `arrival`, `wait`, `start`, `leave`, thông tin di chuyển, score/cost mỗi điểm.
- Category của mỗi điểm: `history_culture`, `nature_parks`, `food_drink`, `shopping`, `entertainment`, `nightlife_wellness`, `depot`.
- Định dạng thời gian output là `HH:MM` (xử lý tại `response_builder.py`).

## 4) Domain model và bất biến quan trọng

### 4.1 Đơn vị thời gian

- Đầu vào người dùng: giờ (VD: 8.0 = 8:00 AM).
- Bên trong thuật toán: phút (Solomon time units).
- Quy đổi tập trung qua `UserPreferences.start_time_minutes` và `end_time_minutes`.

### 4.2 Bất biến route

- Mọi `Individual.route` phải có dạng `[Depot, ..., Depot]`.
- Depot có `id == 0`.
- Các toán tử theo nguyên tắc "Depot-safe": chỉ thao tác trên `route[1:-1]`, rồi gắn lại depot.

### 4.3 Nạp dữ liệu và tính tái lập

- `load_solomon_instance(instance_name)` ưu tiên `extended/*_extended.csv` có `CATEGORY`, `PRICE` cố định.
- Nếu không có extended, fallback CSV gốc + random có seed theo `pid`.
- Có cache RAM `_INSTANCE_CACHE`, trả về `deepcopy` để tránh side-effect giữa các lần chạy.
- Đường dẫn dữ liệu chuẩn hóa bằng `Path(__file__)` trong `data_loader.py`.

### 4.4 Phân loại (Category) và giá vé (Price)

6 categories được gán cho POI theo 2 cơ chế:
- **Extended CSV (chính)**: `generate_extended_data.py` tạo file `*_extended.csv` với category và price cố định.
- **Logic nightlife**: POI có READY TIME >= 75% depot DUE DATE -> tự động gán `nightlife_wellness`.
- **Bảng giá vé** (theo thống kê du lịch Hà Nội):

| Loại hình | Mức giá (VND) |
|---|---|
| `nature_parks` | 0 (miễn phí) |
| `history_culture` | 30,000 / 50,000 / 100,000 |
| `entertainment` | 100,000 / 200,000 / 500,000 |
| `food_drink` | 50,000 / 150,000 / 300,000 / 800,000 |
| `shopping` | 0 / 100,000 / 300,000 / 500,000 |
| `nightlife_wellness` | 100,000 / 200,000 / 500,000 |

## 5) Memetic Algorithm - chi tiết triển khai

### 5.1 Khởi tạo quần thể

- Mặc định 100 cá thể (`POPULATION_SIZE` trong `config.py`).
- `use_heuristic_init=True` (mặc định):
  - 80 cá thể từ Randomized Insertion Heuristic (Labadie ratio).
  - 20 cá thể từ Pure Random.
- `use_heuristic_init=False`: 100% random (phục vụ ablation study).
- Tham số GA mặc định (trong `config.py`):
  - `POPULATION_SIZE = 100`
  - `mutation_rate = 0.3`
  - `tournament_k = 3`
  - `stagnation_limit = 25`
  - `generations = 200`

### 5.2 Hàm fitness và ràng buộc

Fitness tại `fitness.py`:
- `fitness = total_score - penalty`.
- Thành phần penalty:
  - Đến trễ sau `close_time` - `PENALTY_LATE_ARRIVAL = 100.0`
  - Về depot trễ - `PENALTY_LATE_RETURN = 100.0`
  - Vượt ngân sách - `PENALTY_BUDGET = 0.5`
  - Thời gian chờ đợi - `PENALTY_WAIT = 0.2` (có thể tắt bằng flag)

Kiểm tra tính khả thi (`check_constraints`):
- Cửa sổ thời gian của từng POI.
- Ràng buộc ngân sách.
- Tổng lịch trình về depot đúng hạn.

### 5.3 Vòng lặp tiến hóa (`ma_engine.py`)

Mỗi thế hệ (generation):
1. **Chọn lọc (Selection)**: Tournament (k=3).
2. **Lai ghép (Crossover)**: OX1 trên interior (depot-safe).
3. **Đột biến (Mutation)**: 2-opt / Swap / Insertion (xác suất adaptive).
4. **Sửa chữa (Repair)**: Smart Repair (loại POI có tỷ lệ Score/Time kém nhất) hoặc Simple Repair (loại POI cuối).
5. **Bổ sung (Greedy Refill)**: Chèn thêm POI chưa thăm nếu hợp lệ.
6. **Kiểm tra đa dạng (Diversity)**: Loại bỏ route trùng lặp, bổ sung cá thể mới đã được repair/refill.
7. **Đánh giá + Sắp xếp**: Tính fitness, sắp xếp giảm dần, early stopping theo `stagnation_limit`.

Chiến lược thay thế: (mu+lambda) Merged Elitist - gộp parents + children, sắp xếp, lấy top N cá thể duy nhất.

### 5.4 Adaptive-Lite Mutation (2 tầng)

**Tầng 1 - Lịch trình theo tiến trình (progress schedule):**
- Insertion: 0.45 -> 0.15 (giảm dần theo generation)
- 2-opt: 0.25 -> 0.55 (tăng dần theo generation)
- Swap: phần còn lại

**Tầng 2 - Phản hồi theo trạng thái quần thể (population feedback):**
- Stagnation cao (>= 8 thế hệ không cải thiện) -> tăng 2-opt +0.10
- Đa dạng thấp (< 35% route duy nhất) -> tăng swap +0.10
- Tỷ lệ insertion thất bại cao (>= 60%) -> giảm insert, tăng 2-opt

Giới hạn: mỗi toán tử trong khoảng [0.10, 0.80], chuẩn hóa tổng = 1.0.

### 5.5 Ablation flags (hỗ trợ hạng nhất)

`MemeticAlgorithm.__init__` hỗ trợ bật/tắt từng thành phần:

| Flag | True (mặc định) | False |
|---|---|---|
| `use_smart_repair` | Loại POI có Score/Time kém nhất | Loại POI cuối cùng (simple) |
| `use_local_search` | Chạy Repair + Greedy Refill | Tắt hoàn toàn |
| `use_insertion_mutation` | 3 toán tử (2-opt, Swap, Insertion) | Chỉ 2-opt + Swap |
| `use_wait_penalty` | Phạt thời gian chờ | PENALTY_WAIT = 0 |
| `use_heuristic_init` | 80/20 Heuristic/Random | 100% Random |
| `use_diversity_check` | Loại bỏ route trùng lặp | Không kiểm tra |
| `use_adaptive_mutation` | Adaptive-Lite 2 tầng | Tỷ lệ tĩnh |

### 5.6 Theo dõi hội tụ (Convergence Tracking)

- `convergence_log` lưu mỗi thế hệ: best/avg/median/worst fitness, số route duy nhất, thời gian chờ.
- Khi bật adaptive mutation, log thêm: `p_2opt`, `p_swap`, `p_insert`, `insert_fail_rate`.
- `actual_gens`, `best_individual` phục vụ benchmark.

## 6) Hệ sinh thái thực nghiệm (`experiments/`)

### 6.1 Core runner

- `benchmark_runner.py` - Hàm nền `run_single`, `run_batch`, `create_fixed_prefs`, `parse_instances_arg()`.
- `generate_extended_data.py` - Tạo bộ extended CSV có tính tái lập (category + price + nightlife logic).

### 6.2 Các thí nghiệm

| Script | Mục đích | Cấu hình |
|---|---|---|
| `exp1_benchmark.py` | So sánh MA vs Labadie GVNS (2012) | 30 runs x 6 instances, fixed scores, use_wait_penalty=False |
| `exp2_personalization.py` | Đo giá trị cá nhân hóa | 10 runs x 5 profiles (baseline, history, foodie, active, culture) |
| `exp3_budget_impact.py` | Đo tác động ngân sách | 10 runs x 3 mức (low/medium/high) |
| `exp4_ablation_repair.py` | Ablation study các thành phần | 10 runs x 3 biến thể x 6 instances |
| `exp5_sensitivity.py` | Đo độ nhạy tham số GA | 5 runs x 16 cấu hình x 6 instances |

### 6.3 Phân tích và biểu đồ

- `analyze_results.py` - Tự động đọc CSV từ `results/`, tạo bảng tổng hợp.
- `plot_charts.py` - Vẽ biểu đồ chất lượng xuất bản cho toàn bộ 5 thí nghiệm.
- `tune_params.py` - Tinh chỉnh tham số trên single instance.

### 6.4 Lưu ý workflow

- Tất cả script chạy từ `cd backend` với `py -m experiments.<tên_script>`.
- Dữ liệu kết quả lưu trong `experiments/results/*`.
- Các script có `--instances` đều báo lỗi thân thiện nếu nhập sai mã instance.
- Quy ước tên file CSV: `instance_label.csv` (VD: `C101_full_ma.csv`, `C101_fixed.csv`).
- Công bằng benchmark: exp1 và exp4 đều dùng `use_wait_penalty=False` khi so sánh với BKS.
- Trên Windows, cần set `$env:PYTHONIOENCODING="utf-8"` trước khi chạy để tránh lỗi encoding.

## 7) Hướng dẫn chạy nhanh

### Chạy API server

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Mở tài liệu API: `http://localhost:8000/docs`

### Chạy thực nghiệm

```powershell
cd backend
$env:PYTHONIOENCODING="utf-8"

# Tạo dữ liệu extended
py -m experiments.generate_extended_data

# Thí nghiệm 1: Benchmark vs Labadie (2012)
py -m experiments.exp1_benchmark

# Thí nghiệm 2: Giá trị cá nhân hóa
py -m experiments.exp2_personalization

# Thí nghiệm 3: Tác động ngân sách
py -m experiments.exp3_budget_impact

# Thí nghiệm 4: Ablation study
py -m experiments.exp4_ablation_repair

# Thí nghiệm 5: Độ nhạy tham số
py -m experiments.exp5_sensitivity

# Tổng hợp và vẽ biểu đồ
py -m experiments.analyze_results
py -m experiments.plot_charts
```

## 8) Độ bám sát thuật toán gốc

### 8.1 Điểm bám sát

- Sử dụng khung GA cho TOPTW, route [depot...depot], OX1, tournament, repair, kiểm tra khả thi theo time-window.
- Khởi tạo dựa trên hướng heuristic insertion + random (tham chiếu Botelho/Labadie trong code).
- Smart Repair + Greedy Refill tạo thành thành phần Local Search -> đủ điều kiện gọi là Memetic Algorithm.

### 8.2 Khác biệt so với bản gốc (mở rộng có chủ đích)

- Score được cá nhân hóa theo interest weight (không còn fixed score thuần benchmark).
- Thêm ràng buộc ngân sách (TOPTW gốc thường không có chi phí tiền tệ).
- Thêm phạt thời gian chờ để tránh lịch trình có idle time dài.
- Thêm insertion mutation + diversity control + early stopping.
- Thêm adaptive mutation 2 tầng (progress schedule + population feedback).

Kết luận:
- Không có vi phạm logic cốt lõi của TOPTW/GA.
- Đây là biến thể Memetic Algorithm với scoring cá nhân hóa; khi so sánh benchmark cần nêu rõ khác biệt setting.

## 9) Thư viện phụ thuộc

```
fastapi==0.128.0
uvicorn==0.40.0
numpy==2.4.2
pandas==3.0.0
pydantic==2.12.5
matplotlib>=3.9
```
