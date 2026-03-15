# Tài liệu Backend - TOPTW Hybrid GA

## 1) Mục tiêu và phạm vi
Tài liệu này mô tả toàn bộ backend trong `backend/` theo code hiện có: kiến trúc, luồng dữ liệu, API, thuật toán Hybrid GA, hệ sinh thái thực nghiệm, mức độ hoàn thiện ở góc nhìn nghiên cứu, mức độ bám sát thuật toán gốc, và các hướng cải tiến thuật toán.

- Ngôn ngữ/chính: Python + FastAPI (`backend/app`).
- Bài toán: Team Orienteering Problem with Time Windows (TOPTW), mở rộng theo hướng cá nhân hóa và ràng buộc ngân sách.
- Dữ liệu: Solomon instances CSV (`backend/data/solomon_instances`).

## 2) Kiến trúc tổng quan
### 2.1 Các lớp chính
- API layer:
  - `backend/app/main.py`: khởi tạo FastAPI app, CORS, mount router với prefix `/api`.
  - `backend/app/api/routes.py`: endpoint `POST /api/optimize`.
- Contract layer:
  - `backend/app/models/requests.py`: `UserPreferences` + validation + quy đổi giờ -> phút.
  - `backend/app/models/responses.py`: `OptimizationResponse`, `ItineraryItem`.
  - `backend/app/models/domain.py`: domain object `POI`, `Individual`.
- Data layer:
  - `backend/app/services/data_loader.py`: load và cache Solomon instance (ưu tiên file extended).
- Algorithm layer:
  - `backend/app/services/algorithm/hga_engine.py`: điều phối vòng lặp HGA.
  - `backend/app/services/algorithm/initialization.py`: khởi tạo quần thể (80 heuristic + 20 random).
  - `backend/app/services/algorithm/operators/*`: crossover, mutation, repair/refill.
  - `backend/app/services/algorithm/fitness.py`: matrix khoảng cách, feasibility, fitness.
  - `backend/app/services/algorithm/response_builder.py`: format kết quả API.
- Cấu hình trung tâm:
  - `backend/app/core/config.py`: penalty, tham số GA mặc định, urgency params.

### 2.2 Dataflow runtime (từ request đến response)
1. Client gọi `POST /api/optimize`.
2. Pydantic validate `UserPreferences` (`instance_name`, budget, time range, interests đủ 5 nhóm).
3. Route layer load POIs theo `instance_name` qua `load_solomon_instance(...)`, check `start_node_id` tồn tại trong đúng instance đó.
4. Tạo `HybridGeneticAlgorithm(request)` -> `run()`.
5. Engine khởi tạo quần thể, lai ghép/đột biến/sửa/bổ sung, tính fitness, early stopping.
6. Chọn cá thể tốt nhất -> `build_response(...)` trả về timeline HH:MM + tổng chỉ số.

## 3) API contract và validation
### 3.1 Endpoint
- `POST /api/optimize` (`backend/app/api/routes.py`).
- Response model: `OptimizationResponse`.
- Các nhóm lỗi được mô tả rõ: 400/404/422/500.

### 3.2 Request (`UserPreferences`)
Trường bắt buộc/chính:
- `instance_name` (mặc định `C101`), hợp lệ trong: `C101`, `C201`, `R101`, `R201`, `RC101`, `RC201`.
- `budget` > 0.
- `start_time`, `end_time` (đơn vị giờ), với `start_time < end_time`.
- Thời lượng tối thiểu `end_time - start_time >= 1.0` giờ.
- `start_node_id` (được check tồn tại trong dataset tại route layer).
- `interests` phải đúng exact 5 key:
  - `history_culture`, `nature_parks`, `food_drink`, `shopping`, `entertainment`.

Star -> weight map:
- 1 -> 0.1, 2 -> 0.5, 3 -> 1.0, 4 -> 1.5, 5 -> 2.0.

Normalization:
- `interest_weights` scale để tổng trọng số = số category.
- Nếu user cho các mức sao đồng đều, mỗi category về ~1.0, tránh méo score.

### 3.3 Response (`OptimizationResponse`)
- Tổng quan: `total_score`, `total_cost`, `total_distance`, `total_duration`, `execution_time`.
- Chi tiết route: danh sách `ItineraryItem` có `arrival`, `wait`, `start`, `leave`, travel info, score/cost mỗi điểm.
- Format thời gian output là `HH:MM` (xử lý tại `response_builder.py`).

## 4) Domain model và bất biến quan trọng
### 4.1 Đơn vị thời gian
- User input: giờ.
- Solver internals: phút (Solomon time units).
- Quy đổi tập trung qua `UserPreferences.start_time_minutes` và `end_time_minutes`.

### 4.2 Bất biến route
- Mọi `Individual.route` phải có dạng `[Depot, ..., Depot]`.
- Depot có `id == 0`.
- Các operator theo nguyên tắc "Depot-safe": chỉ sửa `route[1:-1]`, rồi gắn lại depot.

### 4.3 Data loading và reproducibility
- `load_solomon_instance(instance_name)` ưu tiên `extended/*_extended.csv` có `CATEGORY`, `PRICE` cố định.
- Nếu không có extended, fallback CSV gốc + random có seed theo `pid`.
- Có cache RAM `_INSTANCE_CACHE`, return `deepcopy` để tránh side-effect giữa runs.
- Đường dẫn dữ liệu đã chuẩn hóa bằng `Path(__file__)` trong `data_loader.py` (không còn phụ thuộc thư mục chạy lệnh như trước).

## 5) Hybrid GA implementation chi tiết
### 5.1 Khởi tạo
- Mặc định 100 cá thể (`POPULATION_SIZE`).
- `use_heuristic_init=True`:
  - 80 cá thể từ Randomized Insertion Heuristic (Labadie ratio + urgency tùy chọn).
  - 20 cá thể random.
- `use_heuristic_init=False`: 100% random (ablation).

### 5.2 Fitness và ràng buộc
Fitness tại `fitness.py`:
- `fitness = total_score - penalty`.
- Penalty gồm:
  - đến trễ sau `close_time` (`PENALTY_LATE_ARRIVAL`),
  - về depot trễ (`PENALTY_LATE_RETURN`),
  - vượt budget (`PENALTY_BUDGET`),
  - chờ đợi (`PENALTY_WAIT`, có thể tắt bằng flag).

Feasibility (`check_constraints`) kiểm tra route hoàn chỉnh theo:
- time windows,
- budget,
- tổng lịch trình về depot đúng hạn.

### 5.3 Vòng lặp tiến hóa (`hga_engine.py`)
Mỗi generation:
1. Selection: tournament.
2. Crossover: OX1 trên interior.
3. Mutation: 2-opt / swap / insertion mutation.
4. Repair: smart repair hoặc simple repair.
5. Greedy refill: chèn thêm POI chưa thăm nếu hợp lệ.
6. Diversity check: loại duplicate route, bổ sung cá thể mới đã được repair/refill.
7. Evaluate + sort + early stopping theo `stagnation_limit`.

Tracking:
- `convergence_log` lưu best/avg/median/worst/unique routes/wait per generation.
- Khi bật adaptive mutation, log có thêm `p_2opt`, `p_swap`, `p_insert`, `insert_fail_rate`.
- `actual_gens`, `best_individual` phục vụ benchmark.
- `hga_engine.py` đã tách thêm helper nội bộ (unique routes, rolling insert fail-rate, convergence logging, generation status) để giảm độ phức tạp của `run()` và dễ bảo trì.

### 5.4 Ablation flags (first-class)
`HybridGeneticAlgorithm.__init__` hỗ trợ:
- `use_smart_repair`
- `use_insertion_mutation`
- `use_wait_penalty`
- `use_heuristic_init`
- `use_diversity_check`
- `use_urgency`
- `use_adaptive_mutation`

Đây là điểm mạnh cho nghiên cứu/thực nghiệm vì có thể bật/tắt từng thành phần trong cùng code path.

## 6) Hệ sinh thái thực nghiệm (`backend/experiments`)
- `benchmark_runner.py`: hàm nền `run_single`, `run_batch`, tạo fixed prefs, và `parse_instances_arg()` để validate `--instances` dùng chung.
- `exp1_benchmark.py`: so sánh HGA vs BKS/GVNS (6 instances).
- `exp2_personalization.py`: đo tác động profile sở thích (đã hỗ trợ `--instances`, `--num-runs`, `--output-dir`).
- `exp3_ablation.py`: đánh giá đóng góp từng thành phần.
- `exp4_sensitivity.py`: đo độ nhạy tham số GA.
- `exp5_urgency.py`: so sánh urgency vs no-urgency + Wilcoxon nếu có scipy.
- `exp6_adaptive_mutation.py`: so sánh mutation tĩnh vs Adaptive-Lite 2 tầng.
- `generate_extended_data.py`: tạo bộ extended CSV reproducible.
- `analyze_results.py`, `plot_charts.py`, `tune_params.py`: tổng hợp, vẽ biểu đồ, tuning.
  - `plot_charts.py` hiện đã có chart cho exp6:
    - Boxplot `static_mutation` vs `adaptive_lite_2tier` theo Normalized Score.
    - Curve `p_insert` / `p_2opt` (và `p_swap`) theo generation từ `convergence_log`.
  - `plot_ablation_boxplot` đã đồng bộ naming exp3 hiện tại (`instance_label.csv`) để normalize đúng theo BKS.
  - Chart style template đã được chuẩn hóa đồng nhất trong code: `title=16`, `label=12`, `legend=10`, line width/marker/palette dùng chung.

Lưu ý workflow:
- Nhiều script giả định chạy từ `cd backend` để path dữ liệu đúng.
- Dữ liệu benchmark và báo cáo đều bám vào `experiments/results/*`.
- Các script có `--instances` đều báo lỗi thân thiện nếu nhập sai mã instance (vd: `C101,R101` là hợp lệ; nhập sai sẽ được gợi ý danh sách mã đúng).
- Naming file CSV đã đồng bộ theo dạng `instance_label.csv` (ví dụ: `C101_full_hga.csv`, `C101_with_urgency.csv`).
- Fairness benchmark đã đồng bộ ở nhóm `exp1/exp3/exp4/exp5`: khi so với BKS đều dùng `use_wait_penalty=False`.
- `backend/requirements.txt` đã gồm `matplotlib` (vẽ biểu đồ) và `scipy` (Wilcoxon trong exp5).

## 7) Hướng dẫn chạy nhanh
```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Mở API docs: `http://localhost:8000/docs`

Chạy một vài script nghiên cứu:
```powershell
cd backend
py -m experiments.benchmark_runner --instance C101 --num-runs 1
py -m experiments.exp1_benchmark --instances C101,R101 --num-runs 3
py -m experiments.exp2_personalization --instances C101,R101 --num-runs 2
py -m experiments.exp3_ablation --instances C101 --num-runs 2
py -m experiments.exp4_sensitivity --instances C101,RC101 --num-runs 2
py -m experiments.exp5_urgency --instances C101 --num-runs 3
py -m experiments.exp6_adaptive_mutation --instances C101,R101 --num-runs 3
```

## 8) Đánh giá mức độ hoàn thiện ở góc nhìn nghiên cứu
Đánh giá theo code hiện có:

- Backend đã tương đối hoàn thiện cho mục tiêu nghiên cứu/KLTN:
  - Có API rõ contract,
  - Có bộ HGA đầy đủ toán tử,
  - Có hệ script thực nghiệm phong phú (benchmark, ablation, sensitivity, urgency),
  - Có pipeline tổng hợp kết quả và biểu đồ.
- Điểm còn thiếu chủ yếu là mở rộng và làm sâu thêm phần nghiên cứu:
  - API chính đã hỗ trợ chọn instance động, nhưng client (mobile/UI) vẫn cần cập nhật input để tận dụng,
  - Các phân tích tổng hợp (`analyze_results.py`) vẫn nên cập nhật thêm để bám sát naming/flow mới,
  - Có thể tăng chất lượng lời giải bằng các toán tử local search mạnh hơn.

Kết luận: với mục tiêu nghiên cứu thuật toán cá nhân hóa, dự án ở mức tốt và dùng được; phần cần nâng cấp tiếp theo nên tập trung vào độ nhất quán thực nghiệm và chất lượng heuristic.

## 9) Độ bám sát thuật toán gốc và "vi phạm"
### 9.1 Điểm bám sát
- Dùng khung GA cho TOPTW, route [depot...depot], OX1, tournament, repair, feasibility check theo time-window.
- Khởi tạo dựa trên hướng heuristic insertion + random (tham chiếu Botelho/Labadie trong comment).

### 9.2 Khác biệt so với bản gốc (chủ yếu là mở rộng, không phải lỗi)
- Score được cá nhân hóa theo interest weight (không còn fixed score thuần benchmark).
- Thêm budget constraint (TOPTW gốc thường không có chi phí tiền tệ).
- Thêm wait-time penalty và urgency heuristic.
- Thêm insertion mutation + diversity control + early stopping.

Kết luận:
- Không thấy dấu hiệu "vi phạm" theo nghĩa sai logic cốt lõi của TOPTW/GA.
- Đây là phiên bản mở rộng cho bài toán cá nhân hóa; khi so sánh benchmark cần nêu rõ khác biệt setting.

## 10) Các điểm cần chú ý trong code (theo hướng nghiên cứu)
1. Đã xử lý: `greedy_refill` urgency đã dùng mốc thời gian theo trạng thái route hiện tại (thay vì luôn neo ở `start_time_minutes`).
2. Đã xử lý: fairness benchmark giữa `exp1/exp3/exp4/exp5` đã đồng bộ (đều tắt `use_wait_penalty` khi so sánh theo BKS).
3. Đã xử lý: naming output trong nhóm script benchmark đã đồng bộ theo `instance_label.csv`.
4. Đã xử lý: `analyze_results.py` đã chuyển sang tự động đọc theo naming/flow hiện tại (exp1→exp6), giảm hardcode và giảm thao tác chỉnh tay khi tổng hợp.
5. API hiện đã nhận `instance_name` từ request; nếu mở rộng tiếp có thể thêm endpoint trả danh sách instance khả dụng để client không phải hardcode.
6. Đã xử lý: mutation có thể chạy chế độ Adaptive-Lite 2 tầng (progress + stagnation/diversity/insert-fail feedback) qua cờ `use_adaptive_mutation`.

## 11) Đề xuất cải tiến ưu tiên cho thuật toán
### Ưu tiên cao
- Bổ sung endpoint metadata nhẹ (vd: `/api/instances`) để client lấy danh sách instance hợp lệ trực tiếp từ backend.

### Ưu tiên trung bình
- Bổ sung local search mạnh hơn ngoài 2-opt/swap: relocate, or-opt, hoặc destroy-repair nhẹ.
- Tinh chỉnh dynamic penalty (đặc biệt budget/wait) theo giai đoạn tiến hóa để cân bằng exploration-exploitation.

### Ưu tiên nghiên cứu sâu
- Multi-objective optimization (score, wait, distance, cost) theo Pareto thay vì weighted-sum duy nhất.
- Nghiên cứu diversity metric giàu thông tin hơn set-ID route (ví dụ khoảng cách thứ tự/edge-based distance).
- Hyperparameter tuning có hệ thống hơn (Latin Hypercube/Bayesian optimization) trên nhiều instance và nhiều seed.

## 12) Kết luận cuối
- Backend hiện tại là một nền tảng nghiên cứu HGA khá đầy đủ, có khả năng sinh lộ trình cá nhân hóa và hỗ trợ thực nghiệm tương đối toàn diện.
- Không phát hiện vi phạm nghiêm trọng với cốt lõi thuật toán; phần lớn là mở rộng có chủ đích so với benchmark gốc.
- Để tăng giá trị học thuật, nên ưu tiên sửa điểm chưa nhất quán về heuristic thời gian, chuẩn hóa pipeline thí nghiệm, và nâng cấp các toán tử tìm kiếm cục bộ.

