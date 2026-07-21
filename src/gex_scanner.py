"""
GEX（伽马敞口）收盘快照

基于 yfinance EOD 期权链 + Black-Scholes，计算做市商伽马敞口分布。
使用免费数据，为收盘日报提供期权结构参考层。

核心公式（dealer-centric）：
  Call GEX = +OI × Gamma × 100 × S²  （做市商被动对冲，买涨抑制波动）
  Put GEX  = -OI × Gamma × 100 × S²  （做市商被动对冲，买跌放大波动）
  净 GEX > 0 = 正伽马环境：做市商抑制波动，价格趋向 GEX King 区间震荡
  净 GEX < 0 = 负伽马环境：做市商放大波动，趋势延伸概率上升

局限（免费数据）：
  - 仅 EOD 快照，不含日内逐笔大单
  - yfinance 期权链更新频率约15分钟延迟
  - OI 为前一日结算数据，当日新开仓未反映
"""

import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz

ET = pytz.timezone("America/New_York")

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
# 2026-07-21：默认收盘快照改为只看大盘（SPX/SPY/QQQ），不再固定绑单只个股。
# 个股期权数据改用 /gex <TICKER> 按需查询（Telegram已支持，见telegram_bot.py）。
GEX_DEFAULT_TICKERS  = ["^SPX", "SPY", "QQQ"]
GEX_MAX_EXPIRIES     = 3       # 最多取3个到期日（近月伽马影响最大）
GEX_SPOT_BAND        = 0.07    # 只计算 spot ±7% 范围内的 strike
RF_RATE              = 0.045   # 近似无风险利率（美国短期国债）
MIN_OI               = 50      # 过滤流动性极差的 strike
# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# Black-Scholes Gamma（不依赖 scipy）
# ─────────────────────────────────────────────────────────────

def _phi(x: float) -> float:
    """标准正态 PDF"""
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def _gamma_bs(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Gamma = φ(d1) / (S × σ × √T)"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(_phi(d1) / (S * sigma * np.sqrt(T)))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# 单标的 GEX 计算
# ─────────────────────────────────────────────────────────────

def calc_gex(ticker: str) -> dict:
    """
    计算单个标的的 GEX 分布图。

    返回：
      spot, gex_king, gex_king_m, total_gex_m, gex_env,
      support, resistance, pc_ratio, pc_bias,
      gex_by_strike（近 spot ±7% 的 strike → GEX(M)）
    """
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if hist.empty:
            return {"ticker": ticker, "error": "无行情数据"}
        S = float(hist["Close"].iloc[-1])

        expiries = tk.options
        if not expiries:
            return {"ticker": ticker, "error": "无期权数据（该标的可能未上市期权）"}

        # ET-aware 当前时间，用于精确计算 T（服务器为 UTC，必须明确时区）
        now_et        = datetime.now(ET)
        gex_by_strike = {}
        total_call_oi = 0
        total_put_oi  = 0

        for exp in expiries[:GEX_MAX_EXPIRIES]:
            try:
                # 期权在美东时间 16:00 到期；用 total_seconds 保留小时精度
                exp_dt = ET.localize(datetime.strptime(exp, "%Y-%m-%d").replace(hour=16, minute=0))
                secs   = (exp_dt - now_et).total_seconds()
                if secs <= 0:
                    continue                     # 当日已过期，跳过，避免 T→0 导致 Gamma 爆炸
                T = max(secs / (365.25 * 86400), 1.0 / 365)

                chain = tk.option_chain(exp)

                for _, row in chain.calls.iterrows():
                    K  = float(row.get("strike") or 0)
                    oi = int(row.get("openInterest") or 0)
                    iv = float(row.get("impliedVolatility") or 0)
                    if oi < MIN_OI or iv <= 0 or K <= 0:
                        continue
                    if abs(K - S) / S > GEX_SPOT_BAND:
                        continue
                    g   = _gamma_bs(S, K, T, RF_RATE, iv)
                    gex = oi * g * 100 * S ** 2
                    gex_by_strike[K] = gex_by_strike.get(K, 0.0) + gex
                    total_call_oi   += oi

                for _, row in chain.puts.iterrows():
                    K  = float(row.get("strike") or 0)
                    oi = int(row.get("openInterest") or 0)
                    iv = float(row.get("impliedVolatility") or 0)
                    if oi < MIN_OI or iv <= 0 or K <= 0:
                        continue
                    if abs(K - S) / S > GEX_SPOT_BAND:
                        continue
                    g   = _gamma_bs(S, K, T, RF_RATE, iv)
                    gex = oi * g * 100 * S ** 2
                    gex_by_strike[K] = gex_by_strike.get(K, 0.0) - gex
                    total_put_oi    += oi

            except Exception:
                continue

        if not gex_by_strike:
            return {"ticker": ticker, "error": "GEX计算失败（近 spot 无有效期权 OI）"}

        total_gex = sum(gex_by_strike.values())

        # GEX King = 绝对值最大的 strike（做市商对冲最集中，价格磁吸效应最强）
        gex_king     = max(gex_by_strike, key=lambda k: abs(gex_by_strike[k]))
        gex_king_val = gex_by_strike[gex_king]

        # 支撑：spot 下方最近的正 GEX strike（做市商买涨托底）
        support_candidates = sorted(
            [k for k, v in gex_by_strike.items() if v > 0 and k < S],
            reverse=True
        )
        # 阻力：spot 上方最近的负 GEX strike（做市商卖跌施压）
        resistance_candidates = sorted(
            [k for k, v in gex_by_strike.items() if v < 0 and k > S]
        )

        support    = support_candidates[0]    if support_candidates    else None
        resistance = resistance_candidates[0] if resistance_candidates else None

        # Put/Call OI 比
        pc_ratio = round(total_put_oi / max(total_call_oi, 1), 2)
        if pc_ratio < 0.8:
            pc_bias = "偏多"
        elif pc_ratio > 1.2:
            pc_bias = "偏空"
        else:
            pc_bias = "中性"

        # 正负伽马翻转点（zero-gamma strike，近 spot 最近的 GEX=0 穿越）
        sorted_strikes = sorted(gex_by_strike.keys())
        flip_strike = None
        for i in range(len(sorted_strikes) - 1):
            k1, k2 = sorted_strikes[i], sorted_strikes[i + 1]
            v1, v2 = gex_by_strike[k1], gex_by_strike[k2]
            if v1 * v2 < 0:  # 符号变化
                flip_strike = round((k1 + k2) / 2, 2)
                break

        return {
            "ticker":        ticker,
            "spot":          round(S, 2),
            "gex_king":      gex_king,
            "gex_king_m":    round(gex_king_val / 1e6, 1),
            "total_gex_m":   round(total_gex / 1e6, 1),
            "gex_env":       "正伽马" if total_gex > 0 else "负伽马",
            "support":       support,
            "resistance":    resistance,
            "flip_strike":   flip_strike,
            "pc_ratio":      pc_ratio,
            "pc_bias":       pc_bias,
            "call_oi":       total_call_oi,
            "put_oi":        total_put_oi,
            "gex_by_strike": {
                k: round(v / 1e6, 2)
                for k, v in sorted(gex_by_strike.items())
            },
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ─────────────────────────────────────────────────────────────
# 批量快照（多标的）
# ─────────────────────────────────────────────────────────────

def gex_daily_snapshot(tickers: list = None) -> list:
    """批量计算 GEX 快照，返回结果列表。"""
    tickers = tickers or GEX_DEFAULT_TICKERS
    results = []
    for t in tickers:
        results.append(calc_gex(t))
    return results


# ─────────────────────────────────────────────────────────────
# 最高持仓量期权排行（Highest Open Interest Options，2026-07-21新增）
# ─────────────────────────────────────────────────────────────
# 跟上面的GEX计算是两回事：GEX是"算做市商对冲会怎么影响价格"，
# 这里只是把calls+puts持仓量(OI)原始数据合并、按OI从高到低排序，复刻
# 常见看盘软件"Highest Open Interest Options"图表的数据来源，不做
# Gamma/Black-Scholes计算，也不做spot±7%的范围过滤（要看全部行权价）。
# 默认（不指定到期日）合并本自然月内所有未到期的到期日，而不是只挑
# 单一到期日——同一行权价跨到期日的OI相加，给出本月整体持仓分布。
OI_RANK_TOP_N        = 20      # 排行榜显示条数
OI_RANK_MONTHS_AHEAD = 3        # 合并未来N个月内的到期日，排除更远期LEAPS


def top_open_interest(ticker: str, expiry: str = None, top_n: int = OI_RANK_TOP_N) -> dict:
    """
    期权持仓量(OI)排行。
    expiry未指定时（2026-07-21变更为"未来3个月"，此前是"本自然月"）：合并
    未来 OI_RANK_MONTHS_AHEAD 个月内所有尚未到期的到期日（周期权+月度期权）
    的calls+puts，同一行权价跨到期日OI相加。排除更远期的LEAPS——一只股票
    未到期到期日全量可能有15-30+个（远至2028年），LEAPS跟近月合并意义不大
    （近期投机 vs 长期布局性质不同），且逐个拉期权链耗时会明显变长。
    expiry指定时：仍可查任意单一到期日（用法不变，可查LEAPS）。
    """
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return {"ticker": ticker, "error": "无期权数据（该标的可能未上市期权）"}

        if expiry:
            if expiry not in expiries:
                return {"ticker": ticker,
                        "error": f"到期日{expiry}不存在，可选：{', '.join(expiries[:6])}"}
            candidates = [expiry]
        else:
            today  = datetime.now(ET).date()
            cutoff = today + timedelta(days=OI_RANK_MONTHS_AHEAD * 30)
            candidates = []
            for e in expiries:
                try:
                    d = datetime.strptime(e, "%Y-%m-%d").date()
                    if today <= d <= cutoff:
                        candidates.append(e)
                except Exception:
                    continue
            if not candidates:
                # 未来窗口内恰好无到期日（极少见），退回最近一个到期日，避免空结果
                candidates = expiries[:1]

        oi_by_key = {}    # (strike, type) -> 跨到期日累加OI
        oi_detail = {}    # (strike, type) -> {到期日: 该到期日的OI}，用于找主力到期日
        used_exps = []
        for exp in candidates:
            try:
                chain = tk.option_chain(exp)
                hit = False
                for _, row in chain.calls.iterrows():
                    oi = int(row.get("openInterest") or 0)
                    if oi > 0:
                        k = (float(row["strike"]), "C")
                        oi_by_key[k] = oi_by_key.get(k, 0) + oi
                        oi_detail.setdefault(k, {})[exp] = oi
                        hit = True
                for _, row in chain.puts.iterrows():
                    oi = int(row.get("openInterest") or 0)
                    if oi > 0:
                        k = (float(row["strike"]), "P")
                        oi_by_key[k] = oi_by_key.get(k, 0) + oi
                        oi_detail.setdefault(k, {})[exp] = oi
                        hit = True
                if hit:
                    used_exps.append(exp)
            except Exception:
                continue

        if not oi_by_key:
            return {"ticker": ticker, "error": "所选到期日均无有效OI数据"}

        # 每个行权价标注"主力到期日"（贡献OI最多的那个）+ 涉及的到期日总数，
        # 避免合并后看不出这堆持仓量到底集中在哪个到期日（2026-07-21用户反馈）
        rows = []
        for k, total in oi_by_key.items():
            detail = oi_detail[k]
            dominant_exp = max(detail, key=detail.get)
            rows.append({
                "strike": k[0], "type": k[1], "oi": total,
                "dominant_expiry": dominant_exp,
                "n_expiries": len(detail),
            })
        rows.sort(key=lambda r: r["oi"], reverse=True)

        if len(used_exps) > 1:
            expiry_label = f"{used_exps[0]}~{used_exps[-1]}（未来{OI_RANK_MONTHS_AHEAD}个月合并{len(used_exps)}档到期日，不含更远期LEAPS）"
        else:
            expiry_label = used_exps[0] if used_exps else candidates[0]

        return {"ticker": ticker, "expiry": expiry_label, "rows": rows[:top_n],
                "expiries_used": used_exps}

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def format_top_oi_telegram(result: dict) -> str:
    """把 top_open_interest() 的结果排成文本条形图，风格对齐常见看盘软件截图。"""
    if result.get("error"):
        return f"⚠️ {result.get('ticker','?')}：{result['error']}"
    rows = result.get("rows", [])
    if not rows:
        return f"{result.get('ticker','?')}：无持仓量数据"

    display_ticker = result["ticker"].lstrip("^")
    max_oi = max(r["oi"] for r in rows)
    bar_lines = []
    for r in rows:
        bar_len = max(1, round(r["oi"] / max_oi * 20))
        label = f"{r['strike']:.1f}{r['type']}"
        # 主力到期日（贡献OI最多的那个），MM/DD显示；若同一行权价跨多个
        # 到期日都有持仓，加"+N"提示还有N个其他到期日也贡献了OI
        dom = r.get("dominant_expiry")
        if dom:
            mm_dd = dom[5:].replace("-", "/")
            extra = f"+{r['n_expiries']-1}" if r.get("n_expiries", 1) > 1 else ""
            label = f"{label}({mm_dd}{extra})"
        bar_lines.append(f"{label:>14} {'█' * bar_len} {r['oi']:,}")

    call_n = sum(1 for r in rows if r["type"] == "C")
    put_n  = len(rows) - call_n
    return (
        f"📊 <b>{display_ticker} 最高持仓量期权</b>\n"
        f"到期日 {result['expiry']}，前{len(rows)}项（Call {call_n} / Put {put_n}）\n"
        f"<pre>{chr(10).join(bar_lines)}</pre>"
    )


# ─────────────────────────────────────────────────────────────
# Telegram 格式化
# ─────────────────────────────────────────────────────────────

def format_gex_telegram(results: list) -> str:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    lines = [f"📐 <b>收盘 GEX 快照</b>  {today}", ""]

    any_ok = False
    for r in results:
        if r.get("error"):
            lines.append(f"⚠️ {r.get('ticker','?')}：{r['error']}")
            continue
        any_ok = True

        env_icon = "🟢" if r["gex_env"] == "正伽马" else "🔴"
        env_hint = (
            "机构对冲会压制波动，股价容易横盘震荡"
            if r["gex_env"] == "正伽马"
            else "机构对冲会放大波动，容易出现大涨大跌"
        )
        display_ticker = r["ticker"].lstrip("^")  # ^SPX 等指数代码显示时去掉插入符，更好读
        lines.append(
            f"{env_icon} <b>{display_ticker}</b>  ${r['spot']}"
            f"  [{r['gex_env']}  净{r['total_gex_m']:+.0f}M]"
        )
        lines.append(f"  <i>（大白话：{env_hint}）</i>")
        lines.append(f"  GEX King：${r['gex_king']}（{r['gex_king_m']:+.0f}M，价格磁吸/翻转点）")
        lines.append("  <i>（大白话：这是期权持仓最集中的价位，股价容易被\"吸\"向这里）</i>")

        if r.get("resistance") and r["resistance"] != r["gex_king"]:
            lines.append(f"  上方压力：${r['resistance']}（负GEX，做市商卖出对冲）")
            lines.append("  <i>（大白话：股价涨到这附近可能遇到抛压，不容易再往上）</i>")
        if r.get("support") and r["support"] != r["gex_king"]:
            lines.append(f"  下方支撑：${r['support']}（正GEX，做市商买入托底）")
            lines.append("  <i>（大白话：股价跌到这附近可能有资金托底，不容易再往下）</i>")
        if r.get("flip_strike"):
            lines.append(f"  正负翻转：${r['flip_strike']}")
            lines.append("  <i>（大白话：越过这个价位，市场的\"脾气\"可能从稳变躁，或反过来）</i>")

        def _oi_str(n: int) -> str:
            return f"{n/1000:.1f}k" if n >= 1000 else str(n)
        lines.append(
            f"  P/C OI比：{r['pc_ratio']}"
            f"（Call {_oi_str(r['call_oi'])} / Put {_oi_str(r['put_oi'])}）→ {r['pc_bias']}"
        )
        lines.append("  <i>（大白话：看跌 vs 看涨的持仓比例，比例越高说明情绪越偏空）</i>")
        lines.append("")

    if any_ok:
        valid = [r for r in results if not r.get("error")]
        pos_n = sum(1 for r in valid if r["gex_env"] == "正伽马")
        neg_n = len(valid) - pos_n
        if pos_n >= neg_n:
            env_summary = "今日多数标的处于正伽马环境，大盘整体偏稳，更适合区间操作，追涨杀跌胜率较低"
        else:
            env_summary = "今日多数标的处于负伽马环境，大盘容易出现较大波动，追涨杀跌风险变大，仓位可适当谨慎"

        bull_n = sum(1 for r in valid if r["pc_bias"] == "偏多")
        bear_n = sum(1 for r in valid if r["pc_bias"] == "偏空")
        if bull_n > bear_n:
            sentiment_summary = "期权持仓情绪整体偏多"
        elif bear_n > bull_n:
            sentiment_summary = "期权持仓情绪整体偏空"
        else:
            sentiment_summary = "期权持仓情绪中性，多空分歧不大"

        lines.append("📝 <b>今日总结（大白话）</b>")
        lines.append(f"  {env_summary}")
        lines.append(f"  {sentiment_summary}，具体标的仍需结合各自支撑压力位判断")
        lines.append("")

        lines.append("─────────────────────────")
        lines.append("💡 <b>正伽马</b>：机构对冲抑制波动，价格趋向 GEX King 区间震荡")
        lines.append("   <b>负伽马</b>：机构对冲放大波动，突破后趋势可延伸")
        lines.append("   <b>GEX King</b>：最大对冲集中点，磁吸效应最强")
        lines.append("   ⚠️ 数据为 EOD 期权链，非实时逐笔大单")

    return "\n".join(lines)
