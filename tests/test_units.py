from datetime import date

import pandas as pd

from datamate.formatter import answer_key, format_answer, rank_with_tiebreak
from datamate.loader import name_key, parse_date_cell
from datamate.lunar import LunarCalendar, lunar_to_solar_algo, solar_to_lunar_algo
from datamate.question import analyze
from datamate.readme import parse_readme
from datamate.rulebook import RuleBook, infer_from_text
from datamate.salesview import customer_key, norm_status
from datamate.scorer import is_correct, penalized_accuracy, score_one
from datamate.textnorm import fold, metric_norm, parse_number_vn, parse_qty_vn, qnorm
from datamate.timeresolver import TimeResolver

OFFICIAL_LIKE = """
| Đơn tính vào doanh thu | đơn **không huỷ** và **không trả hàng**. Đơn có trả hàng, **kể cả chỉ trả một phần**, là đơn trả hàng: bỏ **cả đơn** |
| Tuần | bắt đầu **thứ Hai**. **Tuần này**: từ thứ Hai của tuần chứa hôm nay đến hết hôm nay. **Tuần trước, tuần rồi**: **cả** tuần liền trước |
| Tháng này | từ ngày 1 đến **hết hôm nay**. So với tháng trước thì lấy **cùng số ngày** của tháng trước |
| N ngày gần nhất | N ngày liền nhau **kể cả hôm nay**, ngày cuối là hôm nay |
| Tết năm Y, dịp Tết | từ ngày dương của **23 tháng Chạp** (năm âm Y − 1) đến **hết** ngày dương của **mùng 10 tháng Giêng** |
| Bán chạy | theo **số lượng quy về đơn vị cơ bản**, không theo số dòng hay số tiền |
| Ca sáng, ca chiều | theo giờ tạo đơn: ca sáng 06:00–13:59, ca chiều 14:00–21:59 |
"""


# ------------------------------------------------------------ textnorm ---
def test_metric_norm_matches_official_rule():
    assert metric_norm("Sting dâu 330ml") == "sting dâu 330ml"          # diacritics kept
    assert metric_norm("  Cà-phê   SỮA (đá)! ") == "cà phê sữa đá"
    assert fold("Cà phê sữa đá") == "ca phe sua da"


def test_parse_numbers_and_telex():
    assert parse_number_vn("1.250.000đ") == 1250000
    assert parse_number_vn("45k") == 45000
    assert parse_number_vn("6,719.0000") == 6719
    assert parse_number_vn("-30.500đ") == -30500
    assert parse_qty_vn("1 hộp 30") == 1 and parse_qty_vn("5 viên") == 5
    assert qnorm("bữa qua toongr cộng bán dduwowcj bnhiu lon") == "hom qua tong cong ban duoc bao nhieu lon"
    assert qnorm("thangs 7/2025 dthu") == "thang 7/2025 doanh thu"


def test_dates_mixed_formats():
    assert parse_date_cell("1 thg 6 2025") == pd.Timestamp(2025, 6, 1)
    assert parse_date_cell("1/6", year_hint=2025) == pd.Timestamp(2025, 6, 1)
    assert parse_date_cell(45689) == pd.Timestamp(2025, 2, 1)
    assert parse_date_cell("28/02/2025") == pd.Timestamp(2025, 2, 28)
    assert name_key("NgayLap") == "ngay lap" and name_key("SLQuyDoi") == "sl quy doi"


# --------------------------------------------------------------- scorer ---
def test_scorer_official_rules_with_alias():
    g = {"type": "string", "answer": "Sting dâu 330ml", "alias": ["SP0042"]}
    assert is_correct({"type": "string", "answer": "STING DÂU 330ML"}, g)
    assert is_correct({"type": "string", "answer": "sp0042"}, g)
    assert not is_correct({"type": "string", "answer": "sting dau 330ml"}, g)     # diacritics matter
    assert not is_correct({"type": "number", "answer": 1}, g)
    assert is_correct({"type": "percent", "answer": -12.454}, {"type": "percent", "answer": -12.45})
    gl = {"type": "list", "answer": ["A", "B"], "alias": [["SP1"], []]}
    assert is_correct({"type": "list", "answer": ["sp1", "b"]}, gl)
    assert not is_correct({"type": "list", "answer": ["B", "A"]}, gl)
    t = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 10], ["Anh Tú", 5]]},
         "alias": [["KH01"], ["KH02"]]}
    assert is_correct({"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["KH02", 5], ["Chị Lan", 10]]}}, t)
    golds = {"a": {"type": "number", "answer": 5}, "b": {"type": "unanswerable", "answer": None},
             "c": {"type": "number", "answer": 1}}
    preds = {"a": {"type": "number", "answer": 5}, "b": {"type": "number", "answer": 0}}
    assert score_one(preds["b"], golds["b"]) == -1
    assert abs(penalized_accuracy(preds, golds)) < 1e-9


# ---------------------------------------------------------------- lunar ---
def test_lunar_known_dates():
    assert lunar_to_solar_algo(1, 1, 2026) == date(2026, 2, 17)
    assert lunar_to_solar_algo(1, 1, 2025) == date(2025, 1, 29)
    assert lunar_to_solar_algo(15, 8, 2025) == date(2025, 10, 6)
    assert solar_to_lunar_algo(date(2025, 7, 25)) == (1, 6, 2025, True)
    assert LunarCalendar(None).tet(2027) == date(2027, 2, 6)


# ---------------------------------------------------- rulebook + time ---
def test_rulebook_parsing_and_fingerprints():
    rb = RuleBook(OFFICIAL_LIKE, infer_from_text(OFFICIAL_LIKE))
    assert rb.has("valid_orders", "partial_return_whole_order", "bestseller_by_base_qty", "tet_window")
    assert not rb.has("cancel_rate_all_orders")
    s = infer_from_text(OFFICIAL_LIKE)
    assert s["time"]["tet_start_chap_day"] == 23 and s["time"]["tet_last_day"] == 10
    assert s["time"]["shift_morning"] == [360, 839]
    s2 = infer_from_text("| Tuần | bắt đầu **Chủ nhật** |\n| Bán chạy | theo **doanh thu** |")
    assert s2["time"]["week_start"] == "sunday" and s2["ranking"]["bestseller_metric"] == "revenue"


def test_time_resolver_official_conventions():
    rb = RuleBook(OFFICIAL_LIKE, infer_from_text(OFFICIAL_LIKE))
    tr = TimeResolver(rb)
    c = tr.resolve("sp bán dc nhiều nhất tuần rồi là gì", "2026-03-04")            # Wednesday
    assert (c.primary.start, c.primary.end) == (date(2026, 2, 23), date(2026, 3, 2))
    c = tr.resolve("doanh thu tháng này so với tháng trước tăng bao nhiêu %", "2026-03-31")
    assert (c.primary.start, c.primary.end) == (date(2026, 3, 1), date(2026, 4, 1))
    assert (c.comparison.start, c.comparison.end) == (date(2026, 2, 1), date(2026, 3, 1))   # same days, capped
    c = tr.resolve("thang 10/2025 dthu thay doi bao nhieu % so vs thang lien trc", "2026-05-16")
    assert (c.comparison.start, c.comparison.end) == (date(2025, 9, 1), date(2025, 10, 1))
    c = tr.resolve("e so don hang quy 4/2025 la bao nhiu", "2026-08-24")
    assert [(r.start, r.end) for r in c.ranges] == [(date(2025, 10, 1), date(2026, 1, 1))]
    c = tr.resolve("tinh tu ngay 1/1 den het hom nay, doanh thu nam nay", "2026-06-18")
    assert (c.primary.start, c.primary.end) == (date(2026, 1, 1), date(2026, 6, 19)) and c.weekday is None
    c = tr.resolve("Tổng doanh thu 7 ngày gần nhất", "2025-04-02")
    assert (c.primary.start, c.primary.end) == (date(2025, 3, 27), date(2025, 4, 3)) and c.weekday is None
    c = tr.resolve("doanh thu dịp Tết 2026", "2026-05-01")
    assert c.primary.start == date(2026, 2, 10) and c.primary.end == date(2026, 2, 27)
    c = tr.resolve("Hết hôm nay trong kho còn bao nhiêu hộp", "2026-07-31")
    assert c.snapshot == date(2026, 7, 31)


# ------------------------------------------------------- sales semantics ---
def test_status_and_customer_normalisation():
    assert norm_status("Đã hủy đơn") == "huy" and norm_status("hàng trả về") == "tra_hang"
    assert norm_status("đang ship") == "dang_giao" and norm_status("đã lấy") == "hoan_thanh"
    assert norm_status("completed") == "hoan_thanh" and norm_status("refunded") == "tra_hang"
    assert customer_key("Khách lẻ") is None and customer_key("kl") is None
    assert customer_key("Chị  LAN") == customer_key("chị lan")


def test_readme_dictionary():
    r = parse_readme("# x\n## Cột trong `ban_hang.csv`\n- **Thanh tien**: thành tiền sau giảm\n"
                     "## Quy ước\n- Dữ liệu có đến hết 24/08/2026.\n- **Quy đổi đơn vị**:\n  - Coca Cola 330ml: 1 thùng = 24 lon\n")
    assert r.desc_for(["ban_hang.csv"], "Thanh tien") == "thành tiền sau giảm"
    assert r.data_end == date(2026, 8, 24)
    assert r.conversions == [("Coca Cola 330ml", "thùng", 24.0, "lon")]


# ------------------------------------------------------- formatter/rank ---
def test_formatter_and_tiebreak():
    df = pd.DataFrame({"code": ["B", "A", "C"], "name": ["Bánh", "Áo", "Cơm"], "m": [5, 5, 1]})
    r, tie = rank_with_tiebreak(df, "m", "code", 1)
    assert list(r["code"]) == ["A"] and tie
    assert format_answer(pd.DataFrame({"v": [12.3456]}), "percent") == ("percent", 12.35)
    assert format_answer(pd.DataFrame({"v": [48250000.0]}), "number") == ("number", 48250000)
    assert format_answer(pd.DataFrame({"d": pd.to_datetime(["2025-09-14"])}), "date", month_answer=True) == ("date", "2025-09")
    assert answer_key("string", "Cà phê!") == answer_key("string", "cà  phê")


def test_question_intents():
    s = analyze("sp bán dc nhiều nhất tuần rồi là gì")
    assert s.intent == "top_items" and s.answer_type == "string" and s.metric == "quantity"
    assert analyze("thang 5/2025 hang nao ban chay nhat").answer_type == "string"
    assert analyze("e thang 6/2025 ti le don bi huy la bao nhieu %").intent == "cancel_rate"
    assert analyze("doanh thu trung binh tren mot don thang 11/2025").intent == "aov"
    assert analyze("dem so khach da mua ≥ 5 don trong thang 5/2026").threshold == 5
    assert analyze("Top 3 sản phẩm bán chạy nhất tháng trước").k == 3
    assert analyze("thang 10/2025 dthu thay doi bao nhieu % so vs thang lien trc").intent == "pct_change"


# ------------------------------------------------- robustness regressions ---
def test_text_normalisation_edge_cases():
    assert "acyclovir" in qnorm("bán ra bao nhiêu tuýp Acyclovir Stella kem 5%")      # not a Telex leftover
    assert "mat mui hong" in qnorm("nhóm mắt mũi họng bán được bao nhiêu tiền")        # 'họng' is not 'hông'
    assert qnorm("sp nay hong ban dc ha") == "san pham nay khong ban duoc ha"
    assert "don" in qnorm("dthu binh quan moi order")                                  # loanword kept, then mapped


def test_question_regressions():
    s = analyze("Tháng 6/2025 có mấy ngày doanh thu đạt 9.800.000 đồng trở lên?")
    assert s.intent == "n_days_threshold" and s.threshold == 9800000
    assert analyze("Tháng 2/2025 bán ra bao nhiêu tuýp Acyclovir Stella kem 5% tuýp 5g vậy?").intent == "qty"
    assert analyze("thang 4/2026 ban ra bn cai bon nuoc inox son ha 1000L ngang z").unit_word == "cai"
    s = analyze("Cho tôi doanh thu các đơn trả bằng ví điện tử và doanh thu các đơn trả bằng COD tháng 8/2025.")
    assert s.intent == "multi_filter" and s.multi == [("payment", "ewallet"), ("payment", "cod")]
    assert analyze("ê tỉ trọng của nhóm giảm đau hạ sốt trong doanh thu tháng 2/2026 là bao nhiêu %").intent == "share"
    assert analyze("e thang 10/2025 thu hoi dc bn tien cong no tu khach mua chiu").intent is None
    assert analyze("ai la nguoi ban co doanh thu cao nhat thang 4 nam 2026").intent == "top_staff"


def test_tet_year_follows_first_mention():
    rb = RuleBook(OFFICIAL_LIKE, infer_from_text(OFFICIAL_LIKE))
    c = TimeResolver(rb).resolve("doanh thu dịp Tết năm nay so với dịp Tết năm ngoái thay đổi bao nhiêu %", "2026-06-28")
    assert (c.primary.start, c.comparison.start) == (date(2026, 2, 10), date(2025, 1, 22))
    c = TimeResolver(rb).resolve("doanh thu tết năm ngoái", "2026-06-28")
    assert c.primary.start == date(2025, 1, 22) and c.comparison is None


def test_total_rows_and_surnames():
    from datamate.loader import drop_total_rows
    df = pd.DataFrame({"ma": ["D1", "D2", None, "D3"], "khach": ["Tống Nam Minh", "An", "Cộng ngày 01/01/2025", "Bình"],
                       "tien": [1.0, 2.0, 3.0, 4.0]})
    out = drop_total_rows(df, [])
    assert list(out["khach"]) == ["Tống Nam Minh", "An", "Bình"]


def test_chained_unit_conversion():
    from datamate.salesview import _factor_cell
    assert _factor_cell("15 vỉ") == (15.0, "vi") and _factor_cell(12) == (12.0, None) and _factor_cell("x") is None


# ------------------------------------------- ReViSQL requirements / reconciliation ---
def test_requirements_and_coverage():
    from datamate import requirements as R
    rb = RuleBook(OFFICIAL_LIKE + "\n| Đơn hợp lệ | đơn không huỷ và không trả hàng |",
                  infer_from_text(OFFICIAL_LIKE))
    q = "doanh thu tháng 5/2026 của các đơn trả bằng chuyển khoản là bao nhiêu"
    spec, tctx = analyze(q), TimeResolver(rb).resolve(q, "2026-07-01")
    reqs = R.build(spec, tctx, rb=rb)
    kinds = [r.kind for r in reqs]
    assert "time" in kinds and "filter" in kinds and "valid" in kinds
    good = ("SELECT SUM(revenue) FROM v_orders WHERE is_valid AND payment ILIKE '%chuyển khoản%' "
            "AND order_date >= DATE '2026-05-01' AND order_date < DATE '2026-06-01'")
    assert R.coverage(good, reqs)["failed"] == []
    no_filter = ("SELECT SUM(revenue) FROM v_orders WHERE is_valid AND order_date >= DATE '2026-05-01' "
                 "AND order_date < DATE '2026-06-01'")
    assert R.coverage("SELECT SUM(revenue) FROM v_orders WHERE is_valid AND payment = 'transfer' AND "
                      "strftime(order_date, '%Y-%m') = '2026-05'", reqs)["failed"] == []      # month by name
    assert [r.kind for r in reqs if r.rid in R.coverage(no_filter, reqs)["failed"]] == ["filter"]
    wrong_month = good.replace("2026-05-01", "2026-04-01").replace("2026-06-01", "2026-05-01")
    assert "time" in [r.kind for r in reqs if r.rid in R.coverage(wrong_month, reqs)["failed"]]
    assert "-- tháng 5" not in R._norm_code("SELECT 1 -- tháng 5")
    assert "R1" in R.prompt_block(reqs) and "CHECK R<i>" in R.prompt_block(reqs)


def test_reconciliation_keeps_only_covering_groups():
    from datamate import requirements as R
    from datamate.agent import Agent, Candidate
    from datamate.config import load_config
    rb = RuleBook(OFFICIAL_LIKE, infer_from_text(OFFICIAL_LIKE))
    q = "doanh thu tháng 5/2026 của các đơn trả bằng chuyển khoản là bao nhiêu"
    spec, tctx = analyze(q), TimeResolver(rb).resolve(q, "2026-07-01")
    reqs = R.build(spec, tctx, rb=rb)
    ag = Agent(load_config(overrides=["llm.backend=none"]), rb, None)
    base = "FROM v_orders WHERE is_valid AND order_date >= DATE '2026-05-01' AND order_date < DATE '2026-06-01'"
    a1 = Candidate("sql", ("number", 100), code=f"SELECT SUM(revenue) {base}")              # misses the filter
    a2 = Candidate("sql", ("number", 100), code=f"SELECT SUM(revenue) {base} AND 1=1")
    b = Candidate("sql", ("number", 40), code=f"SELECT SUM(revenue) {base} AND payment = 'transfer'")
    f = ag._reconcile([a1, a2, b], reqs, None, {"question": q, "as_of_date": "2026-07-01"}, spec, tctx, [], True)
    assert (a1.rc_pass, b.rc_pass) == (False, True) and f["rc_filtered"] == 1.0
    # Algorithm 1, lines 16-17: when no group passes, every group stays
    c1 = Candidate("sql", ("number", 1), code=f"SELECT 1 {base}")
    f = ag._reconcile([c1], reqs, None, {"question": q, "as_of_date": "2026-07-01"}, spec, tctx, [], True)
    assert c1.rc_pass is None and f["rc_pass_frac"] == 0.0


# --------------------------------------------------------------- trace inversion ---
def test_trace_inversion_modules():
    from datamate.inversion import (lexical_similarity, parse_reconstruction, strip_comments, struct_compare,
                                    template_inversion)
    from datamate import requirements as R
    q = "doanh thu tháng 5/2026 các đơn trả bằng chuyển khoản"
    assert lexical_similarity(q, "tháng 5/2026 doanh thu của đơn thanh toán chuyển khoản") > \
        lexical_similarity(q, "mặt hàng nào bán chạy nhất tuần trước") + 0.2
    rec = parse_reconstruction("QUESTION: Doanh thu tháng 5/2026 từ các đơn chuyển khoản?\n"
                               "PERIOD: 2026-05-01..2026-05-31\nCOMPARE: none\nFILTERS: transfer\nRESULT: number")
    assert rec["period"] == (date(2026, 5, 1), date(2026, 5, 31)) and rec["result"] == "number"
    rb = RuleBook(OFFICIAL_LIKE, infer_from_text(OFFICIAL_LIKE))
    spec, tctx = analyze(q), TimeResolver(rb).resolve(q, "2026-07-01")
    st = struct_compare(rec, spec, tctx, R.build(spec, tctx, rb=rb))
    assert st["aligned"], st
    rec_bad = dict(rec, period=(date(2026, 4, 1), date(2026, 4, 30)))
    assert not struct_compare(rec_bad, spec, tctx, R.build(spec, tctx, rb=rb))["aligned"]
    assert "tháng" not in strip_comments("SELECT 1 -- tháng 5\n# note")
    from datamate.inversion import SentenceEncoder
    se = SentenceEncoder(None)          # SE on content words: time is checked by STRUCT, not by SE
    q0 = "doanh thu thang truoc la bao nhieu?"
    assert se.similarity(q0, "Tính tổng doanh thu từ 1/2/2026 đến 28/2/2026?") > 0.8
    assert se.similarity(q0, "Tính số lượng đơn hàng trong tháng 2 năm 2025?") < 0.35
    p2 = parse_reconstruction("QUESTION: x\nPERIOD: none\nFILTERS: order_date >= DATE '2026-02-01' AND "
                              "order_date < DATE '2026-03-01'\nRESULT: number")["period"]
    assert p2 == (date(2026, 2, 1), date(2026, 3, 1))
    ok = template_inversion({"intent": "revenue", "period": "tháng 11/2025", "filters": [("item", "Nhà thuốc Lan")]},
                            "thang 11/2025 nha thuoc ban thuoc ke don dc bao nhieu tien")
    assert ok["tpl_grounded"] < 0.7                      # 'Lan' is not in the question: wrong question answered
