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
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
GEX_DEFAULT_TICKERS  = ["SPY", "QQQ", "NVDA"]
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

        now          = datetime.now()
        gex_by_strike = {}
        total_call_oi = 0
        total_put_oi  = 0

        for exp in expiries[:GEX_MAX_EXPIRIES]:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                T = max((exp_dt - now).days / 365.0, 1.0 / 365)

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
        lines.append(
            f"{env_icon} <b>{r['ticker']}</b>  ${r['spot']}"
            f"  [{r['gex_env']}  净{r['total_gex_m']:+.0f}M]"
        )
        lines.append(f"  GEX King：${r['gex_king']}（{r['gex_king_m']:+.0f}M，价格磁吸/翻转点）")

        if r.get("resistance") and r["resistance"] != r["gex_king"]:
            lines.append(f"  上方压力：${r['resistance']}（负GEX，做市商卖出对冲）")
        if r.get("support") and r["support"] != r["gex_king"]:
            lines.append(f"  下方支撑：${r['support']}（正GEX，做市商买入托底）")
        if r.get("flip_strike"):
            lines.append(f"  正负翻转：${r['flip_strike']}")

        lines.append(
            f"  P/C OI比：{r['pc_ratio']}"
            f"（Call {r['call_oi']//1000}k / Put {r['put_oi']//1000}k）→ {r['pc_bias']}"
        )
        lines.append("")

    if any_ok:
        lines.append("─────────────────────────")
        lines.append("💡 <b>正伽马</b>：做市商对冲抑制波动，价格趋向 GEX King 区间震荡")
        lines.append("   <b>负伽马</b>：做市商对冲放大波动，突破后趋势可延伸")
        lines.append("   <b>GEX King</b>：最大对冲集中点，磁吸效应最强")
        lines.append("   ⚠️ 数据为 EOD 期权链，非实时逐笔大单")

    return "\n".join(lines)
