"""
选股筛选引擎

核心修复（v2）：
  1. RS Rating 改为 S&P 500 成分股百分位真实排名（O'Neil 原版）
  2. 九关顺序重排：高效过滤原则，先快速自动，再慢速手动
"""
import yfinance as yf
import numpy as np
import pandas as pd

# ── S&P 500 代表性宇宙（约 120 只，覆盖全部 GICS 板块）
# 足够计算准确百分位，避免下载 500 只造成超时
_SP500_UNIVERSE = [
    # 科技
    "AAPL","MSFT","NVDA","AVGO","ORCL","AMD","QCOM","TXN","AMAT","LRCX",
    "KLAC","MU","INTC","MRVL","ON","ADI","MCHP","NXPI","SWKS","QRVO",
    # 通信/互联网
    "GOOGL","META","NFLX","DIS","CHTR","VZ","T","TMUS","CMCSA",
    # 消费
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","COST","WMT","LOW",
    "BKNG","MAR","HLT","MGM","LVS","WYNN",
    # 医疗
    "LLY","UNH","JNJ","ABBV","MRK","BMY","AMGN","GILD","REGN","VRTX",
    "CVS","CI","HUM","MCK","ABT","MDT","SYK","BSX","EW","ISRG",
    # 金融
    "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","AXP","BLK",
    "SCHW","CME","ICE","SPGI","MCO","CB","PGR","AFL","MET","PRU",
    # 工业
    "CAT","DE","HON","GE","RTX","LMT","NOC","BA","UPS","FDX",
    "ETN","EMR","ROK","AME","GWW","FAST","ADP","PAYX",
    # 能源
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","PXD",
    # 原材料/公用
    "LIN","APD","DD","FCX","NEM","NUE","ALB","MP",
    "NEE","DUK","SO","D","AEP","EXC","SRE","PCG",
    # AI/光子/重点
    "AXTI","AAOI","COHR","LITE","VRT","EQIX","CEG","VST","CCJ",
]

# 缓存（进程内有效，避免重复下载）
_rs_cache: dict = {}

# ══════════════════════════════════════════════════════════════
# 选股配置常量（修改参数在此处）
# ══════════════════════════════════════════════════════════════

# ── O'Neil RS 评分权重（4段加权，近期权重最高）─────────────
ONEIL_WEIGHT_3MO            = 0.40     # 3个月收益权重（最近=最重要）
ONEIL_WEIGHT_6MO            = 0.20
ONEIL_WEIGHT_9MO            = 0.20
ONEIL_WEIGHT_12MO           = 0.20
ONEIL_3MO_DAYS              = 63       # 约3个月交易日数
ONEIL_6MO_DAYS              = 126
ONEIL_9MO_DAYS              = 189
ONEIL_12MO_DAYS             = 252

# ── RS 评级门限 ────────────────────────────────────────────
RS_PASS_THRESHOLD           = 85       # ≥85 = 强势
RS_WARN_THRESHOLD           = 70       # 70-84 = 勉强
RS_FAIL_THRESHOLD           = 70       # <70 = 弱势

# ── 流动性门限 ────────────────────────────────────────────
LIQUIDITY_VOL_MIN_SMALL     = 500_000  # 小账户：日均量≥50万
LIQUIDITY_VOL_MIN_LARGE     = 100_000  # 大账户：日均量≥10万
LIQUIDITY_VOL_EXCELLENT     = 2_000_000 # 流动性极佳：>200万

# ── 九关门限 ─────────────────────────────────────────────
GATE_INST_PASS              = 0.40     # 机构持仓≥40%
GATE_INST_WARN              = 0.15     # 机构持仓≥15%
GATE_REV_PASS               = 0.25     # 营收增速≥25%（O'Neil A标准）
GATE_REV_WARN               = 0.10     # 营收增速≥10%
GATE_DIST_DAYS_WARN         = 4        # ≥4个分发日 = 警告
GATE_RS_PASS                = 85       # RS≥85 = 强势
GATE_RS_WARN                = 70       # RS≥70 = 勉强
GATE_GAMMA_PASS             = 6        # Gamma挤压得分≥6
GATE_GAMMA_WARN             = 3        # Gamma挤压得分≥3

# ── 价格/ATR 最小值（防止计算无效信号）──────────────────────
MIN_CLOSE_PRICE             = 0.01     # 防止除以极小价格

# ══════════════════════════════════════════════════════════════


def _oneil_score(hist) -> float:
    """O'Neil 加权收益分：使用区间回报（非累积），避免近期涨幅重复叠加
    段1：最近3月（0-63天）权重40%
    段2：3-6月段（63-126天）权重20%
    段3：6-9月段（126-189天）权重20%
    段4：9-12月段（189-252天）权重20%
    """
    c = hist["Close"]
    n = len(c)
    def seg(start, end):
        if n < end:
            return 0.0
        base = float(c.iloc[-end])
        return float((c.iloc[-start] - base) / max(base, 0.01))
    return (ONEIL_WEIGHT_3MO  * seg(1,   ONEIL_3MO_DAYS)
          + ONEIL_WEIGHT_6MO  * seg(ONEIL_3MO_DAYS, ONEIL_6MO_DAYS)
          + ONEIL_WEIGHT_9MO  * seg(ONEIL_6MO_DAYS, ONEIL_9MO_DAYS)
          + ONEIL_WEIGHT_12MO * seg(ONEIL_9MO_DAYS, ONEIL_12MO_DAYS))


def _fetch_universe_scores() -> list:
    """批量拉取宇宙股票的 O'Neil 分，返回有序 score 列表（升序）"""
    scores = []
    data = yf.download(
        " ".join(_SP500_UNIVERSE),
        period="1y", interval="1d",
        group_by="ticker", auto_adjust=True,
        progress=False, threads=True,
    )
    for tk in _SP500_UNIVERSE:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                hist = data[tk].dropna() if tk in data.columns.get_level_values(0) else None
            else:
                hist = data  # 单 ticker 下载时列已是平铺结构
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue
            scores.append(_oneil_score(hist))
        except Exception:
            continue
    return sorted(scores)


def calculate_rs_rating(ticker: str) -> dict:
    """
    RS Rating（1-99）— O'Neil 真实百分位排名

    方法：
      1. 拉取 ~120 只 S&P 500 代表股的 O'Neil 加权收益分
      2. 计算 ticker 的分数在宇宙中的百分位
      3. 百分位直接映射到 1-99

    修复前问题：用 log 函数映射，且只和 SPY 比，不是真正的排名。
    """
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 63:
            return {"rs_rating": None, "error": "历史数据不足 3 个月"}

        stock_score = _oneil_score(hist)

        # 获取宇宙分数（有缓存则复用）
        import time
        cache_key = "universe"
        now_ts    = time.time()
        if cache_key not in _rs_cache or now_ts - _rs_cache[cache_key]["ts"] > 3600:
            universe_scores = _fetch_universe_scores()
            _rs_cache[cache_key] = {"scores": universe_scores, "ts": now_ts}
        else:
            universe_scores = _rs_cache[cache_key]["scores"]

        if not universe_scores:
            return {"rs_rating": None, "error": "宇宙数据获取失败"}

        # 百分位 → 1-99
        below = sum(1 for s in universe_scores if s < stock_score)
        percentile = below / len(universe_scores)
        rs_rating  = max(1, min(99, int(percentile * 98) + 1))

        def r(days):
            if len(hist) < days: return 0.0
            return float((hist["Close"].iloc[-1] - hist["Close"].iloc[-days])
                         / max(hist["Close"].iloc[-days], 0.01))

        # SPY 对比（仅参考，不再用于 RS 计算）
        spy_hist = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
        spy_3m = float((spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[-63])
                       / spy_hist["Close"].iloc[-63] * 100) if len(spy_hist) >= 63 else 0

        return {
            "rs_rating":         rs_rating,
            "oneil_score":       round(stock_score, 4),
            "universe_size":     len(universe_scores),
            "outperforming":     rs_rating >= 50,
            "grade": ("Elite"     if rs_rating >= 90 else
                      "Strong"    if rs_rating >= 80 else
                      "Good"      if rs_rating >= 70 else
                      "Average"   if rs_rating >= 50 else
                      "Weak"),
            "returns": {
                "stock_3m":  round(r(63)  * 100, 1),
                "stock_6m":  round(r(126) * 100, 1),
                "stock_12m": round(r(252) * 100, 1),
                "spy_3m":    round(spy_3m, 1),
            },
            "method": "S&P500 universe percentile rank (O'Neil真实百分位)",
        }
    except Exception as e:
        return {"rs_rating": None, "error": str(e)}


def liquidity_gate(info: dict, account_value: float = 100_000) -> dict:
    """
    流动性预过滤关卡（在九关之前调用）

    小账户必须通过流动性检查，避免滑点和买卖价差吞噬利润。
    日均成交量 < 500K 的股票对小账户来说交易成本过高。
    """
    avg_vol = (info.get("averageDailyVolume3Month") or
               info.get("averageVolume") or 0)
    avg_vol = int(avg_vol)

    if account_value < 25_000:
        min_vol = LIQUIDITY_VOL_MIN_SMALL
    else:
        min_vol = LIQUIDITY_VOL_MIN_LARGE

    if avg_vol >= LIQUIDITY_VOL_EXCELLENT:
        status = "pass"
        note   = f"日均成交量 {avg_vol/1e6:.1f}M，流动性极佳，买卖价差极低"
    elif avg_vol >= min_vol:
        status = "pass"
        note   = f"日均成交量 {avg_vol/1000:.0f}K，流动性达标"
    elif avg_vol >= LIQUIDITY_VOL_MIN_LARGE:
        status = "warn" if account_value >= 25_000 else "fail"
        note   = (f"日均成交量 {avg_vol/1000:.0f}K，偏低——"
                  f"{'小账户' if account_value < 25_000 else ''}建议使用限价单，滑点风险")
    else:
        status = "fail"
        note   = f"日均成交量 {avg_vol/1000:.0f}K，流动性严重不足，买卖困难"

    return _gate(0, "流动性预过滤", status, note)


def nine_gates_check(info: dict, rs_result: dict, market_state: dict,
                     gamma_score: int = 0,
                     account_value: float = 100_000) -> list:
    """
    九关筛选 v2 — 重排顺序：高效快速过滤原则

    顺序逻辑：
      先跑自动快速过滤（大盘/RS/机构）→ 通过后再做耗时手动分析
      任何 fail 理论上应终止后续分析（前端可实现"遇 fail 高亮警告"）

    v2.1 新增：
      Gate 0 流动性预过滤（account_value < $25k 时 min 500K 日均量）
    """
    gates = [liquidity_gate(info, account_value)]

    # ── 关卡1：大盘方向（自动，最快，全面否决）──────────────
    state = market_state.get("state", "E")
    dist  = market_state.get("distribution_days", 0)
    dist_warn = dist >= GATE_DIST_DAYS_WARN

    if state == "A" and not dist_warn:
        gates.append(_gate(1, "大盘方向", "pass",
            f"市场状态 A（上升趋势），分发日 {dist}/25，趋势健康"))
    elif state == "D" and not dist_warn:
        gates.append(_gate(1, "大盘方向", "warn",
            f"市场状态 D（极度恐慌，VIX>30），仓位减半，等待 FTD 底部确认后再做多"))
    elif state in ("A", "D") and dist_warn:
        gates.append(_gate(1, "大盘方向", "warn",
            f"市场状态 {state}，但已累计 {dist} 个分发日（≥4警惕机构出货）"))
    elif state == "B":
        gates.append(_gate(1, "大盘方向", "warn",
            f"横盘震荡（B），等待方向确认，仓位减半"))
    else:
        gates.append(_gate(1, "大盘方向", "fail",
            f"市场状态 {state}：{market_state.get('description', '')}，禁止做多"))

    # ── 关卡2：RS Rating（自动，核心过滤器）─────────────────
    rs = rs_result.get("rs_rating")
    grade = rs_result.get("grade", "")
    if rs is not None:
        if rs >= RS_PASS_THRESHOLD:
            gates.append(_gate(2, "相对强度 RS", "pass",
                f"RS={rs}（{grade}），跑赢 {rs}% 的 S&P500 成分股"))
        elif rs >= RS_WARN_THRESHOLD:
            gates.append(_gate(2, "相对强度 RS", "warn",
                f"RS={rs}（{grade}），尚可但未达 O'Neil ≥{RS_PASS_THRESHOLD} 标准"))
        else:
            gates.append(_gate(2, "相对强度 RS", "fail",
                f"RS={rs}（{grade}），跑输大多数股票，O'Neil 直接淘汰"))
    else:
        gates.append(_gate(2, "相对强度 RS", "manual",
            f"RS计算失败：{rs_result.get('error','')}，请手动核查"))

    # ── 关卡3：机构持仓趋势（半自动）────────────────────────
    inst = info.get("institutionPercentHeld") or info.get("institutional_ownership")
    if inst is not None:
        inst_pct = float(inst) * 100 if float(inst) <= 1 else float(inst)
        if inst_pct > GATE_INST_PASS * 100:
            gates.append(_gate(3, "机构持仓", "pass",
                f"机构持仓 {inst_pct:.1f}%，主力认可，查13F确认增减持趋势"))
        elif inst_pct > GATE_INST_WARN * 100:
            gates.append(_gate(3, "机构持仓", "warn",
                f"机构持仓 {inst_pct:.1f}%，偏低，需查13F确认是否在增持"))
        else:
            gates.append(_gate(3, "机构持仓", "fail",
                f"机构持仓 {inst_pct:.1f}%，机构未认可，慎入"))
    else:
        gates.append(_gate(3, "机构持仓", "manual",
            "无持仓数据，请查 /api/institutional-13f 获取季度增减持历史"))

    # ── 关卡4：供应链层位（半自动 + Serenity）────────────────
    gates.append(_gate(4, "供应链层位", "manual",
        "查供应链分析：该股处于哪一层？资金是否正在流向该层？"
        "（Serenity核心：资金在卡脖子层=最强信号）"))

    # ── 关卡5：能力圈（人工，投资前提）──────────────────────
    gates.append(_gate(5, "能力圈", "manual",
        "你能否解释清楚：这家公司靠什么赚钱？护城河是什么？"
        "主要竞争对手是谁？为什么这家比对手强？（不懂不碰）"))

    # ── 关卡6：散户密度（人工）───────────────────────────────
    gates.append(_gate(6, "散户密度", "manual",
        "散户是否大量扎堆？（Reddit/StockTwits 热度/期权散户OI占比）"
        "散户密集=等血洗后再进，散户冷淡=机构建仓期=好时机"))

    # ── 关卡7：从0到1（人工，Serenity核心）──────────────────
    gates.append(_gate(7, "从0到1", "manual",
        "这家公司是否处于指数级成长初期？"
        "营收加速（QoQ >20%）？TAM 还有 10x 空间？"
        "成熟公司靠分红，成长公司靠增速，策略完全不同。"))

    # ── 关卡8：财报验证（半自动）────────────────────────────
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        rg = float(rev_growth) * 100
        if rg > GATE_REV_PASS * 100:
            gates.append(_gate(8, "财报验证", "pass",
                f"营收 YoY +{rg:.0f}%，成长加速，投资论文有数据支撑"))
        elif rg > GATE_REV_WARN * 100:
            gates.append(_gate(8, "财报验证", "warn",
                f"营收 YoY +{rg:.0f}%，成长但未加速，注意趋势是否减速"))
        else:
            gates.append(_gate(8, "财报验证", "fail",
                f"营收 YoY {rg:.0f}%，成长停滞，重新审视投资论文"))
    else:
        gates.append(_gate(8, "财报验证", "manual",
            "请确认：上季度 EPS 和营收是否双双超预期？指引是否上调？"
            "超预期幅度 >5% 且指引上调 = 高质量财报"))

    # ── 关卡9：期权 Gamma 结构（自动）───────────────────────
    if gamma_score >= GATE_GAMMA_PASS:
        gates.append(_gate(9, "期权/Gamma结构", "pass",
            f"Gamma Squeeze 评分 {gamma_score}/10，做市商净Delta压力支撑上行"))
    elif gamma_score >= GATE_GAMMA_WARN:
        gates.append(_gate(9, "期权/Gamma结构", "warn",
            f"Gamma 评分 {gamma_score}/10，期权结构中性，无额外助力"))
    elif gamma_score > 0:
        gates.append(_gate(9, "期权/Gamma结构", "fail",
            f"Gamma 评分 {gamma_score}/10，Put 主导或结构看空"))
    else:
        gates.append(_gate(9, "期权/Gamma结构", "manual",
            "请查期权分析模块获取 Gamma Squeeze 评分"))

    return gates


def _gate(num, name, status, detail):
    return {"gate": num, "name": name, "status": status, "detail": detail}
