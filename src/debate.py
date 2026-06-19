"""
多方辩论引擎（借鉴 TauriCresearch/TradingAgents 架构思路）

原版TradingAgents：多个LLM智能体扮演分析师互相辩论（需付费API，非确定性）
本实现：基于定量数据的确定性规则辩论（免费，可重复，有数据支撑）

结构：
  看多分析师（Bull Analyst）— 寻找支持做多的证据
  看空分析师（Bear Analyst）— 寻找反对做多的证据
  风险评估官（Risk Officer）— 量化风险收益比
  最终裁决（Arbitrator）   — 综合评分 + 建议

Wall Street 原则：任何一笔交易在入场前必须能说清楚
  "为什么做多"和"为什么会错"，两者都要有答案。
"""

import yfinance as yf
import numpy as np


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def generate_trade_debate(ticker: str, direction: str = "LONG",
                          cold_result: dict = None,
                          account_value: float = 2000) -> dict:
    """
    生成做多/做空的多方辩论报告。

    参数：
      ticker       — 股票代码
      direction    — LONG / SHORT
      cold_result  — cold_decision() 的输出（如有，直接复用数据）
      account_value — 账户规模（影响风险容忍度判断）

    返回：
      bull_case    — 看多论据列表（含证据强度）
      bear_case    — 看空论据列表（风险/反驳点）
      risk_report  — 风险评估官报告
      verdict      — 辩论结论 + 置信度调整建议
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="1y", interval="1d")
        if hist.empty:
            return {"error": f"无法获取 {ticker} 数据"}
    except Exception as e:
        return {"error": str(e)}

    price    = float(hist["Close"].iloc[-1])
    close    = hist["Close"]
    volume   = hist["Volume"]

    # 预取一次 SPY，供 _bull_analyst / _risk_officer 共用，避免 56 只股票 × 2 次重复下载
    try:
        spy_shared = yf.Ticker("SPY").history(period="3mo")
    except Exception:
        spy_shared = None

    bull_args = _bull_analyst(info, hist, close, volume, price, direction, cold_result, spy_hist=spy_shared)
    bear_args = _bear_analyst(info, hist, close, volume, price, direction, cold_result)
    risk_rep  = _risk_officer(info, hist, close, price, account_value, cold_result, spy_hist=spy_shared)
    verdict   = _arbitrator(bull_args, bear_args, risk_rep, direction)

    return {
        "ticker":      ticker,
        "price":       round(price, 2),
        "direction":   direction,
        "bull_analyst": bull_args,
        "bear_analyst": bear_args,
        "risk_officer": risk_rep,
        "verdict":     verdict,
        "wall_street_rule": (
            "Wall Street铁律：入场前必须能清晰回答 "
            "'我为什么做多'和'我在什么情况下会错'。"
            "看空方没有好论据，才是真正高置信度的机会。"
        ),
    }


# ─────────────────────────────────────────────────────────────
# 看多分析师
# ─────────────────────────────────────────────────────────────

def _bull_analyst(info, hist, close, volume, price, direction, cold_result, spy_hist=None) -> dict:
    args      = []
    score     = 0
    max_score = 0

    def add(title, detail, pts, earned, evidence=None):
        args.append({
            "argument":  title,
            "detail":    detail,
            "strength":  "强" if earned >= pts * 0.8 else "中" if earned >= pts * 0.5 else "弱",
            "pts_max":   pts,
            "pts_earned": earned,
            "evidence":  evidence,
        })

    # ── 1. 趋势方向 ───────────────────────────────────────
    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    max_score += 20
    if direction == "LONG":
        if price > ma20 > ma50 and (ma200 is None or price > ma200):
            score += 20
            add("均线多头排列（完美趋势）",
                f"价格${price:.2f} > MA20({ma20:.2f}) > MA50({ma50:.2f})" +
                (f" > MA200({ma200:.2f})" if ma200 else ""),
                20, 20, "三均线顺序排列是趋势健康的最强信号")
        elif price > ma20 > ma50:
            score += 14
            add("均线多头排列", f"价格 > MA20 > MA50，趋势确认", 20, 14)
        elif price > ma20:
            score += 8
            add("价格站上MA20", f"短期趋势确认，MA50尚未对齐", 20, 8)
        else:
            add("趋势未确认", f"价格${price:.2f} < MA20({ma20:.2f})，做多趋势不支持", 20, 0)

    # ── 2. 相对强度 ───────────────────────────────────────
    max_score += 20
    try:
        if spy_hist is None:
            spy_hist = yf.Ticker("SPY").history(period="3mo")
        if len(close) >= 63 and len(spy_hist) >= 63:
            stk_ret = float((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63] * 100)
            spy_ret = float((spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[-63])
                            / spy_hist["Close"].iloc[-63] * 100)
            excess  = stk_ret - spy_ret
            if excess > 20:
                score += 20
                add("极强相对强度（RS）",
                    f"近3月跑赢SPY {excess:.1f}%（股票+{stk_ret:.1f}% vs SPY+{spy_ret:.1f}%）",
                    20, 20, "O'Neil: RS>85是最佳买入的必要条件")
            elif excess > 10:
                score += 14
                add("良好相对强度", f"近3月跑赢SPY {excess:.1f}%", 20, 14)
            elif excess > 0:
                score += 8
                add("轻微跑赢大盘", f"近3月跑赢SPY {excess:.1f}%，RS尚可", 20, 8)
            else:
                add("跑输大盘", f"近3月落后SPY {abs(excess):.1f}%，相对强度弱", 20, 0,
                    "O'Neil严格标准：RS<70的股票不值得做多")
    except Exception:
        add("RS数据获取失败", "无法计算相对强度", 20, 0)

    # ── 3. 成交量信号 ─────────────────────────────────────
    max_score += 15
    vol_ma20   = float(volume.rolling(20).mean().iloc[-1])
    vol_recent = float(volume.iloc[-1])
    vol_ratio  = vol_recent / vol_ma20 if vol_ma20 > 0 else 1.0
    price_chg  = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0

    if vol_ratio >= 1.5 and price_chg > 0:
        score += 15
        add("放量上涨（机构介入信号）",
            f"今日量比 {vol_ratio:.1f}x，上涨 {price_chg:.1f}%——大资金推动",
            15, 15, "Minervini: 放量突破是主力确认进场的最直接证据")
    elif vol_ratio >= 1.0 and price_chg > 0:
        score += 10
        add("量价配合", f"量比{vol_ratio:.1f}x，上涨{price_chg:.1f}%，量价健康", 15, 10)
    elif vol_ratio < 0.7:
        score += 5
        add("缩量整理（蓄力信号）",
            f"量比{vol_ratio:.1f}x，成交量萎缩——潜在突破蓄力",
            15, 5, "VCP理论：突破前必须有量能萎缩过程")
    else:
        add("量价信号中性", f"量比{vol_ratio:.1f}x，无明显放量突破", 15, 3)
        score += 3

    # ── 4. 基本面 ─────────────────────────────────────────
    max_score += 20
    rev_growth = info.get("revenueGrowth")
    eps_ttm    = info.get("trailingEps")
    pe         = info.get("trailingPE")
    gross_m    = info.get("grossMargins")
    fund_score = 0
    fund_details = []

    if rev_growth and float(rev_growth) > 0.25:
        fund_score += 8
        fund_details.append(f"营收YoY +{float(rev_growth)*100:.0f}%（CANSLIM标准：需>25%）")
    elif rev_growth and float(rev_growth) > 0.10:
        fund_score += 5
        fund_details.append(f"营收YoY +{float(rev_growth)*100:.0f}%，成长但未加速")
    elif rev_growth:
        fund_details.append(f"营收YoY {float(rev_growth)*100:.0f}%，成长停滞")

    if eps_ttm and float(eps_ttm) > 0:
        fund_score += 5
        fund_details.append(f"EPS={eps_ttm:.2f}（盈利公司）")
    elif eps_ttm:
        fund_details.append(f"EPS={eps_ttm:.2f}（亏损中，需确认成长路径）")

    if gross_m and float(gross_m) > 0.50:
        fund_score += 5
        fund_details.append(f"毛利率{float(gross_m)*100:.0f}%（优质护城河）")

    score += fund_score
    add("基本面质量", "；".join(fund_details) if fund_details else "无基本面数据",
        20, fund_score)

    # ── 5. 技术门限结构（突破位置 / RSI区间 / ATR止损）─────────
    # 使用 cold_result 的具体门输出，而非聚合分——避免"冷静模型说好 → 因此好"的循环推理
    max_score += 15
    pattern_score = 0
    pattern_details = []
    if cold_result:
        gates = cold_result.get("gates", {})

        # 价格位置：near_high 门——突破前沿是 VCP/杯柄的核心要素
        near_g = gates.get("near_high", {})
        if near_g.get("pass") is True:
            pattern_score += 6
            pattern_details.append(near_g.get("note", "近52周高，突破结构良好"))
        elif near_g.get("pass") == "warn":
            pattern_score += 3
            pattern_details.append(near_g.get("note", "接近突破区，位置边缘"))

        # RSI 区间：cold_model 独立计算，与 section 1 的均线信号正交
        rsi_g = gates.get("rsi", {})
        if rsi_g.get("pass") is True:
            pattern_score += 5
            pattern_details.append(rsi_g.get("note", "RSI区间合理"))
        elif rsi_g.get("pass") is False:
            pattern_details.append(f"⚠ {rsi_g.get('note', 'RSI不适合做多')}")

        # ATR止损结构：合理止损意味着入场点有可测量的支撑
        stop_g = gates.get("stop_distance", {})
        if stop_g.get("pass") is True:
            pattern_score += 4
            pattern_details.append(stop_g.get("note", "止损结构合理"))
        elif stop_g.get("pass") is False:
            pattern_details.append(f"⚠ {stop_g.get('note', '止损结构异常')}")

    pattern_detail = "；".join(pattern_details) if pattern_details else "需运行/scan获取技术门限数据"
    score += pattern_score
    add("技术门限结构（突破位置/RSI/ATR）", pattern_detail, 15, pattern_score)

    bull_score_pct = round(score / max_score * 100) if max_score > 0 else 0
    return {
        "analyst":    "看多分析师（Bull Analyst）",
        "arguments":  args,
        "total_score": score,
        "max_score":  max_score,
        "score_pct":  bull_score_pct,
        "conviction": ("极强" if bull_score_pct >= 80 else
                       "强"   if bull_score_pct >= 65 else
                       "中"   if bull_score_pct >= 45 else "弱"),
        "summary": f"看多得分 {bull_score_pct}%，置信度：{'需进一步确认' if bull_score_pct < 60 else '有足够论据支持'}",
    }


# ─────────────────────────────────────────────────────────────
# 看空分析师（魔鬼代言人）
# ─────────────────────────────────────────────────────────────

def _bear_analyst(info, hist, close, volume, price, direction, cold_result) -> dict:
    args           = []
    risk_pts       = 0
    risk_pts_high  = 0
    risk_pts_mid   = 0

    def add_risk(title, detail, severity, evidence=None):
        nonlocal risk_pts, risk_pts_high, risk_pts_mid
        sev_pts = {"高": 3, "中": 2, "低": 1}.get(severity, 1)
        risk_pts += sev_pts
        if severity == "高":
            risk_pts_high += sev_pts
        elif severity == "中":
            risk_pts_mid  += sev_pts
        args.append({
            "risk":     title,
            "detail":   detail,
            "severity": severity,
            "evidence": evidence,
        })

    # ── 1. 估值风险 ───────────────────────────────────────
    pe  = info.get("trailingPE")
    fpe = info.get("forwardPE")
    if pe and float(pe) > 100:
        add_risk("极度高估值", f"PE={float(pe):.0f}x，远超市场均值（S&P500约20x）",
                 "高", "高估值股票在市场下跌时跌幅往往2-3倍于指数")
    elif pe and float(pe) > 50:
        add_risk("估值偏高", f"PE={float(pe):.0f}x，需要持续高速增长支撑", "中")
    elif pe is None:
        add_risk("无法判断估值", "PE数据缺失，可能是亏损公司或数据不全", "低")

    # ── 2. 空头压力 ───────────────────────────────────────
    short_float = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")
    if short_float and float(short_float) > 0.15:
        add_risk("高空头比例", f"空头占流通股{float(short_float)*100:.1f}%，做空力量强",
                 "中", "高空头可能反成轧空（Short Squeeze），但也意味着市场存在强烈做空共识")
    if short_ratio and float(short_ratio) > 7:
        add_risk("高回补天数", f"空头回补需{float(short_ratio):.0f}天，流动性受限", "低")

    # ── 3. 价格位置风险 ───────────────────────────────────
    high_52w = info.get("fiftyTwoWeekHigh")
    low_52w  = info.get("fiftyTwoWeekLow")
    if high_52w and low_52w:
        pct_from_high = (price - float(high_52w)) / float(high_52w) * 100
        pct_from_low  = (price - float(low_52w))  / float(low_52w)  * 100
        if pct_from_high > -5 and direction == "LONG":
            add_risk("接近52周高点", f"距52周高${float(high_52w):.2f}仅{abs(pct_from_high):.1f}%",
                     "低", "突破新高需要放量确认，否则可能是假突破")
        if pct_from_low < 20 and direction == "LONG":
            add_risk("距52周低点较近", f"仅从低点反弹{pct_from_low:.0f}%，趋势尚未完全反转",
                     "中", "底部确认需要Follow-Through Day信号")

    # ── 4. 财报二元事件风险 ───────────────────────────────
    try:
        cal = yf.Ticker(info.get("symbol", "")).calendar if hasattr(info, "symbol") else None
    except Exception:
        cal = None
    # 简化：从cold_result获取财报门限
    if cold_result:
        eb = cold_result.get("gates", {}).get("earnings_blackout", {})
        if eb.get("pass") == "warn":
            add_risk("财报临近（7天内）", eb.get("note", ""), "高",
                     "财报是二元事件：超预期可涨20%，不及预期可跌15-30%")

    # ── 5. 宏观/大盘风险 ─────────────────────────────────
    if cold_result:
        market_state = cold_result.get("gates", {}).get("time_window", {})
        vix_gate = cold_result.get("gates", {}).get("vix", {})
        if vix_gate.get("pass") == "warn":
            add_risk("VIX偏高", vix_gate.get("note", ""), "中",
                     "VIX>28时市场恐慌情绪升温，持仓波动加剧")

    # ── 6. 流动性/滑点风险 ────────────────────────────────
    avg_vol = info.get("averageDailyVolume3Month", 0) or info.get("averageVolume", 0) or 0
    if avg_vol < 500_000:
        add_risk("流动性不足", f"日均成交量{avg_vol/1000:.0f}K，买卖价差大，滑点高",
                 "高", "小账户在低流动性股票中的实际成本可能比预期高0.5-1%")

    # ── 7. 行业周期风险 ───────────────────────────────────
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    if "Semiconductor" in industry or "semiconductor" in industry.lower():
        add_risk("半导体行业周期风险", "半导体具有强周期性，高点建仓易被套", "低",
                 "但AI驱动的需求可能延长本轮周期")

    bear_severity = ("极强" if risk_pts >= 12 else
                     "强"   if risk_pts >= 8  else
                     "中"   if risk_pts >= 5  else "弱")

    return {
        "analyst":   "看空分析师 · 魔鬼代言人（Bear/Devil's Advocate）",
        "risks":     args,
        "risk_pts":      risk_pts,
        "risk_pts_high": risk_pts_high,
        "risk_pts_mid":  risk_pts_mid,
        "severity":      bear_severity,
        "summary": (
            f"发现 {len(args)} 项风险因素，风险总分{risk_pts}。"
            f"{'风险较大，需要特别强的做多理由才能入场' if risk_pts >= 8 else '风险可控，但务必设好止损'}"
        ),
    }


# ─────────────────────────────────────────────────────────────
# 风险评估官
# ─────────────────────────────────────────────────────────────

def _risk_officer(info, hist, close, price, account_value, cold_result, spy_hist=None) -> dict:
    close_arr = close.values
    # 历史波动率（年化）
    if len(close_arr) >= 20:
        daily_rets = np.diff(close_arr) / close_arr[:-1]
        hist_vol   = float(np.std(daily_rets) * np.sqrt(252) * 100)
    else:
        hist_vol   = 30.0

    # Beta（简化：vs SPY 20日相关）
    try:
        if spy_hist is None:
            spy_hist = yf.Ticker("SPY").history(period="3mo")
        spy_close = spy_hist["Close"].values
        n         = min(len(close_arr), len(spy_close)) - 1
        stk_r     = np.diff(close_arr[-n-1:]) / close_arr[-n-1:-1]
        spy_r     = np.diff(spy_close[-n-1:]) / spy_close[-n-1:-1]
        cov       = float(np.cov(stk_r, spy_r)[0, 1])
        spy_var   = float(np.var(spy_r))
        beta      = cov / spy_var if spy_var > 0 else 1.0
    except Exception:
        beta = 1.0

    # 1% 市场下跌时的预期损失（Beta放大）
    market_down_1pct_loss = price * beta * 0.01

    # 账户风险指标
    entry_plan = cold_result.get("entry_plan") if cold_result else None
    if entry_plan:
        max_risk_usd = entry_plan.get("max_risk_usd", account_value * 0.01)
        pos_usd      = entry_plan.get("position_usd", account_value * 0.20)
        stop_pct     = entry_plan.get("max_risk_pct", 1.0)
    else:
        max_risk_usd = account_value * 0.03
        pos_usd      = account_value * 0.30
        stop_pct     = 3.0

    # Kelly 建议：优先从实测绩效数据读取，不足30笔时用保守默认值
    # 警告：0.53/2.0 来自个人历史交易记录，不代表当前股票的统计特性
    W, R = 0.53, 2.0
    kelly_source = "默认值（样本不足，仅供参考，非当前股票统计预期）"
    try:
        from .paper_trading import performance_report
        perf = performance_report("paper")
        kelly_data = perf.get("kelly", {})
        total_trades = perf.get("summary", {}).get("total_trades", 0)
        if total_trades >= 30:
            W = kelly_data.get("actual_win_rate", 53.0) / 100
            R = kelly_data.get("actual_rr_ratio", 2.0)
            kelly_source = f"实测数据（{total_trades}笔）"
    except Exception:
        pass
    full_kelly = W - (1 - W) / R
    half_kelly = max(0, full_kelly / 2)
    kelly_usd  = account_value * half_kelly

    kelly_warning = ""
    if full_kelly <= 0:
        kelly_warning = f"⚠️ Kelly为负({full_kelly*100:.1f}%)！W={W*100:.0f}%/R={R}组合期望值为负，不建议入场。"

    # 破产风险：Monte Carlo 模拟（MP1-1 修复）
    # 原 Gambler's Ruin 公式假设等额赌注（R=1），在 R≠1 时严重低估破产概率
    # 正确做法：模拟 N 局，统计账户跌破 0 的概率
    import random as _rand
    ror_approx = 0.0
    if max_risk_usd > 0 and account_value > 0:
        _SIMS = 2000  # 2000次模拟，精度±2%，速度<10ms
        _N_ROUNDS = max(int(account_value / max_risk_usd) * 3, 100)  # 至多3倍回本局数
        _ruins = 0
        for _ in range(_SIMS):
            balance = account_value
            for _r in range(_N_ROUNDS):
                if balance <= 0:
                    break
                # 每局：win→盈 R×risk；loss→亏 1×risk
                if _rand.random() < W:
                    balance += R * max_risk_usd
                else:
                    balance -= max_risk_usd
            if balance <= 0:
                _ruins += 1
        ror_approx = round(_ruins / _SIMS * 100, 2)

    return {
        "analyst":         "风险评估官（Risk Officer）",
        "hist_volatility": round(hist_vol, 1),
        "beta_vs_spy":     round(beta, 2),
        "loss_if_spy_down_1pct": round(market_down_1pct_loss, 2),
        "account_value":   account_value,
        "current_plan": {
            "max_risk_usd": round(max_risk_usd, 2),
            "position_usd": round(pos_usd, 2),
            "risk_pct_acct": round(stop_pct, 2),
        },
        "kelly_criterion": {
            "formula":    "Kelly% = W - (1-W)/R",
            "inputs":     f"W={W*100:.0f}%，R={R:.2f}（{kelly_source}）",
            "full_kelly": round(full_kelly * 100, 1),
            "half_kelly": round(half_kelly * 100, 1),
            "half_kelly_usd": round(kelly_usd, 2),
            "warning":    kelly_warning,
            "recommendation": (
                kelly_warning if kelly_warning else
                f"半Kelly建议仓位：${kelly_usd:.0f}（账户{half_kelly*100:.0f}%）。"
                "使用半Kelly以保护本金，等实际胜率统计后再调整。"
            ),
        },
        "risk_of_ruin_est": f"≈{ror_approx:.2f}%（每笔亏{stop_pct:.0f}%，连续亏损至0）",
        "summary": (
            f"年化波动率{hist_vol:.0f}%，Beta={beta:.2f}。"
            f"建议半Kelly仓位${kelly_usd:.0f}，"
            f"当前风险${max_risk_usd:.0f}（账户{stop_pct:.1f}%）"
        ),
    }


# ─────────────────────────────────────────────────────────────
# 最终裁决
# ─────────────────────────────────────────────────────────────

def _arbitrator(bull: dict, bear: dict, risk: dict, direction: str) -> dict:
    bull_pct  = bull.get("score_pct", 0)
    risk_pts  = bear.get("risk_pts", 0)

    # 区分个股特有风险（高权重）与行业背景风险（低权重），避免线性累积偏向 WAIT
    # ⚠️ 权重系数（high×5/mid×2）为经验值，无统计校准；仅供参考，勿视为精确置信度
    high_risk = bear.get("risk_pts_high", 0)
    mid_risk  = bear.get("risk_pts_mid",  0)
    low_risk  = risk_pts - high_risk - mid_risk
    penalty   = min(25, high_risk * 5 + mid_risk * 2 + max(0, low_risk) * 0)
    net_score = max(0, bull_pct - penalty)

    if net_score >= 70 and risk_pts <= 5:
        conclusion = "GO"
        reason     = f"看多论据充分（{bull_pct}%），风险可控（{risk_pts}pts），支持入场"
        confidence = "高"
    elif net_score >= 55 and risk_pts <= 8:
        conclusion = "CONDITIONAL"
        reason     = f"看多论据尚可（{bull_pct}%），但存在风险（{risk_pts}pts），建议缩小仓位50%"
        confidence = "中"
    elif net_score >= 40:
        conclusion = "WAIT"
        reason     = f"看多论据不足（净分{net_score}），等待更好信号再入场"
        confidence = "低"
    else:
        conclusion = "ABORT"
        reason     = f"看空风险过强（{risk_pts}pts），或看多论据薄弱（净分{net_score}），放弃"
        confidence = "极低"

    adjustments = []
    if risk_pts >= 8:
        adjustments.append("⚠️ 风险较高：建议将仓位缩小至计划的50%")
    if bull_pct >= 80 and risk_pts <= 4:
        adjustments.append("✅ 高置信度设置：可按计划满仓执行")
    if bear.get("severity") == "极强":
        adjustments.append("🚫 看空论据极强：即使做多也需严格止损，距入场价-3%即止")

    return {
        "conclusion":     conclusion,
        "net_score":      net_score,
        "bull_score_pct": bull_pct,
        "bear_risk_pts":  risk_pts,
        "confidence":     confidence,
        "reason":         reason,
        "position_adjustments": adjustments,
        "one_line": (
            f"【{conclusion}】{reason}" +
            (f" | {adjustments[0]}" if adjustments else "")
        ),
    }
