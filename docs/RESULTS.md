# Kết quả — vòng 1 (chế độ không LLM)

Môi trường phát triển chỉ có CPU 2 nhân, nên mọi con số dưới đây là của **chế độ template + luật** (`llm.backend=none`).
Đây cũng là chế độ dự phòng khi hết giờ trong bản nộp có LLM. Chế độ có Qwen3-14B cần chạy trên GPU (README, mục A).

## 1. Train (3.000 câu, 30 shop) — chấm bằng luật chính thức (`scripts/evaluate.py`)

| Phiên bản | Đúng | Sai | Bỏ qua | PA |
|---|---:|---:|---:|---:|
| v2 — trước vòng sửa lỗi này | 1.988 | 315 | 697 | 0,558 |
| v3 — phân loại bảng theo tên sheet, bỏ dòng tổng đúng cách, lọc kênh/thanh toán, thuộc tính danh mục | 2.473 | 73 | 454 | 0,800 |
| v5 — quy đổi đơn vị theo chuỗi, linker theo dấu, chặn ràng buộc chưa xử lý, Tết theo lần nhắc đầu | 2.526 | 13 | 461 | 0,838 |
| **v6 — nhóm hàng/nhóm khách, bảng tên → mã, hạng khách** | **2.666** | **5** | **329** | **0,887** |
| v6 qua `run.py` + calibrator, chấm bằng `evaluate.py` | 2.667 | 3 | 330 | **0,888** |

**Cần đọc con số này đúng cách.** Đây là số *trong mẫu*: các lỗi được tìm bằng cách đọc câu sai trên train rồi sửa ở
tầng tổng quát (loader, vai trò cột, linker, bộ phân tích câu hỏi). Không có logic nào gắn với id câu hỏi hay id shop, nhưng
train vẫn là dữ liệu đã nhìn thấy. Trên shop mới, kỳ vọng thấp hơn. Hai tín hiệu cho thấy mức tổng quát hoá:

* Lỗi được sửa đều thuộc loại chung (xem mục 3), và mỗi lần sửa làm tăng số câu đúng ở nhiều shop cùng lúc.
* Trên public (shop khác, rulebook khác), hệ thống tự trả lời 63.5% số câu (mục 2), so với 89% trên
  train. Khoảng cách chủ yếu nằm ở các shop ngành mới (sửa xe, điện thoại, cơ sở dữ liệu nhiều bảng), nơi nhiều câu hỏi nằm
  ngoài các ý định mà template cài đặt.

Calibrator được fit với **cross-validation nhóm theo shop** (`scripts/fit_calibrator.py`). Trên train, độ chính xác của ứng
viên template là 0,998, nên calibrator gần như luôn trả lời khi template có ứng viên. Khi bật LLM, phải fit lại calibrator
trên một lần chạy có LLM, vì phân phối lỗi sẽ khác hẳn.

## 2. Public test (1.000 câu, 14 shop, chưa có đáp án)

`runs/public_submission.jsonl`, chạy trong khoảng 6 phút 40 giây (lần chạy có cache: 4 phút 10 giây) trên CPU (gồm nạp dữ liệu và dựng view, chưa có cache):

| Kết quả | Số câu |
|---|---:|
| Có đáp án (number, percent, date, string, list, boolean) | 611 |
| `unanswerable` (có bằng chứng dương) | 24 |
| `abstain` | 365 |

Nguyên nhân abstain: khoảng 235 câu không thuộc ý định nào template cài (công nợ, tồn kho, lô/hạn dùng, ngành sửa xe, điện
thoại, mẹ & bé…), đây là phần LLM xử lý. Khoảng 130 câu có ý định nhưng template tự rút lui (bộ lọc chưa hiểu, thực thể nhập
nhằng, kỳ ngoài dữ liệu một phần…). Định dạng đã được kiểm tra: đủ 1.000 dòng theo đúng thứ tự, `abstain` chỉ có `id`, `type`,
`reason` của `unanswerable` là chuỗi; mỗi câu có log ở `runs/public_logs/<id>.json`.

## 3. Các lỗi tổng quát đã sửa trong vòng này (tìm trên train)

| Lỗi | Hậu quả | Cách sửa (tổng quát) |
|---|---|---|
| Khách tên "Tống …" bị coi là dòng "Tổng" | mất đơn thật, doanh thu thấp ở nhiều shop | nhãn dòng tổng phải là ô chữ đầu tiên của dòng |
| Dòng "Cộng ngày 01/01" nằm ở cột thứ 10, ô `NaT` bị coi là chữ | dòng tổng lọt vào dữ liệu | duyệt mọi cột, coi `NaT` là ô trống |
| Tên file chung cho mọi sheet ("DuLieuBanHang") | mọi bảng bị coi là bảng bán hàng | phân loại theo tên sheet trước, tên file sau |
| Bảng quy đổi có cột "Đơn vị quy đổi" không được nhận | không quy đổi đơn vị, "bán chạy" sai | luật vai trò `big_unit`; hệ số dạng "15 vỉ" nhân chuỗi |
| Câu hỏi về hoạt chất / thuốc kê đơn trả doanh thu cả shop | sai hàng loạt ở nhà thuốc | module `attributes.py` + chặn ràng buộc chưa xử lý |
| "trả bằng chuyển khoản" liên kết với món "Trà" | doanh thu = 0 | dấu thanh là bằng chứng khi liên kết |
| "Acyclovir" bị sửa Telex thành "acyclovi", "họng" thành "không" | sai ý định | chỉ sửa Telex khi ra âm tiết hợp lệ; viết tắt chỉ bung khi gõ không dấu |
| "Tết năm nay so với Tết năm ngoái" lấy nhầm năm | `unanswerable` sai | năm theo lần nhắc Tết đầu tiên |
| "hạng" (thân thiết/sỉ) của khách bị hiểu là "hàng" | không lọc được nhóm khách | vai trò cột theo loại bảng |
| "nhân viên Bùi Dũng Lan" trong khi đơn ghi `NV003` | doanh thu cả shop | bảng tên → mã từ danh sách nhân viên/khách |

## 4. Bản bổ sung ReViSQL [R40] + Trace Inversion [R39]

| Kiểm tra | Kết quả |
|---|---|
| Train, chế độ không LLM, sau khi thêm đặc trưng mới và fit lại calibrator | 2.667 đúng, 4 sai, 329 bỏ qua (`eval_templates.py`); calibrator OOF @0,5 = 0,887. Không đổi so với trước: các phần mới chủ yếu tác động khi có LLM |
| Public, chế độ không LLM | 1.000/1.000 dòng **giống hệt** bài nộp trước |
| Test tự động | 25/25 test qua, gồm: requirement và kiểm tra chương trình, Algorithm 1 (giữ nhóm đạt; không nhóm nào đạt thì giữ tất cả), các module Trace Inversion, reward RLVR, và toàn pipeline với LLM giả lập |
| Smoke test với LLM thật (Qwen2.5-Coder-1.5B Q4 trên CPU, chỉ để kiểm tra đường chạy) | Pipeline chạy hết, file nộp đúng định dạng. Model 1,5B chép nguyên chữ `date_col` từ prompt vào SQL nên mọi ứng viên LLM lỗi, đáp án đến từ template (đúng cả 2 câu). Câu requirement đã được sửa để ghi thẳng `order_date` |
| Trace Inversion trên model thật, chương trình đúng vs sai cho "doanh thu tháng trước" | SE (từ nội dung, bỏ biểu thức thời gian): 1,00 vs 0,07; STRUCT bắt đúng kỳ sai của chương trình sai. TrInv-LLM của model 1,5B trả NO cho cả hai, nên biểu quyết đa số gắn cờ cả hai. Đây là lý do tín hiệu đi vào calibrator thay vì buộc abstain, và calibrator phải fit lại với Qwen3-14B |
| Reconciliation verdict trên model thật | Chương trình `SELECT COUNT(*) FROM v_orders` cho câu hỏi doanh thu nhận `VERDICT: NO` |

Hiệu quả của hai phần mới lên PA chỉ đo được trên GPU với Qwen3-14B (bảng B1–B5 trong DESIGN, mục 7).

## 5. Cách tái lập

```bash
python scripts/eval_templates.py --split path/to/train --out runs/train_offline_dump.jsonl   # bảng mục 1
python scripts/fit_calibrator.py runs/train_offline_dump.jsonl path/to/train/answers.jsonl \
       --out artifacts/calibrator_offline.pkl
python scripts/run.py --data path/to/public_test --out runs/public_submission.jsonl --logs runs/public_logs \
       --set llm.backend=none --set decision.calibrator_path=artifacts/calibrator_offline.pkl
```
Kết quả tất định: seed lấy từ nội dung câu hỏi, không dùng thời gian hay thứ tự chạy.
