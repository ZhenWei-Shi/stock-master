"""
财报质量分析器（Earnings Quality Analyzer）

对标标准：
  O'Neil CANSLIM —— C: 当季EPS >25% YoY；A: 年度EPS连续3年>25%增长
  Minervini ——     EPS/营收连续3季加速；超预期幅度>5%；指引上调
  学术PEAD研究 ——  超预期>5%的股票，财报后4-8周有统计显著的延续漂移

财报质量评分体系（0-100）：
  A: 80-100  — 机构级别财报，最高确信度入场
  B: 60-79   — 良好财报，可以入场
  C: 40-59   — 普通财报，谨慎
  D: 0-39    — 财报弱，避免

核心信号优先级（由高到低）：
  1. EPS + 营收 双双超预期 + 指引上调   ← 最强信号
  2. EPS 连续3季加速                    ← 机构首选
  3. EPS 超预期幅度 >10%                ← 可靠信号
  4. 营收单独超预期                      ← 中等信号
  5. EPS 超预期但营收未达                ← 质量低（成本压缩式增长）
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

ET = pytz.timezone("America/New_York")

# ══════════════════════════════════════════════════════════════
# 财报分析配置常量（修改参数在此处）
# 过拟合风险说明：PEAD/加速度分析仅基于6季历史，样本量有限
# ══════════════════════════════════════════════════════════════

# ── EPS 加速度 ────────────────────────────────────────────────
EPS_MIN_QUARTERS            = 2        # 最少N季数据
EPS_LOOKBACK                = 6        # 取最近N季分析
EPS_ONEIL_C_PCT             = 25.0     # O'Neil C标准：当季EPS YoY ≥25%
EPS_BEAT_MAJOR              = 10.0     # 超预期>10% = 重大
EPS_BEAT_NORMAL             = 5.0      # 超预期>5%  = 正常
EPS_ACCEL_MIN               = 2        # 连续N段加速才算强信号

# ── 营收分析 ─────────────────────────────────────────────────
REV_ONEIL_A_PCT             = 25.0     # O'Neil A标准：TTM营收YoY ≥25%
REV_LOOKBACK                = 5        # 最近N季

# ── PEAD ─────────────────────────────────────────────────────
PEAD_LOOKBACK               = 6        # 最近N次财报
PEAD_SURPRISE_THRESHOLD     = 5.0      # 超预期>5% 才计入PEAD
PEAD_DRIFT_DAYS             = 30       # 漂移窗口
PEAD_CONSISTENCY_MIN        = 0.60     # ≥60%继续率才加分

# ── 质量因子 ─────────────────────────────────────────────────
ROE_EXCELLENT               = 0.20     # ROE ≥20% = 优秀
ROE_GOOD                    = 0.15     # ROE ≥15% = 良好
GROSS_MARGIN_STRONG         = 0.60     # 毛利率 ≥60% = 强护城河
GROSS_MARGIN_GOOD           = 0.40     # 毛利率 ≥40%
GROSS_MARGIN_MOD            = 0.20     # 毛利率 ≥20%
FCF_HIGH_QUALITY            = 0.80     # FCF/净利润 ≥80% = 高质量
FCF_MED_QUALITY             = 0.50     # FCF/净利润 ≥50%
DE_LOW                      = 0.50     # D/E <0.5 = 低杠杆
DE_HIGH                     = 1.50     # D/E ≥1.5 = 高杠杆

# ── CANSLIM 综合评分 ─────────────────────────────────────────
CANSLIM_GRADE_A             = 80
CANSLIM_GRADE_B             = 60
CANSLIM_GRADE_C             = 40

# ── 市场宽度 ─────────────────────────────────────────────────
BREADTH_GOOD                = 70       # ≥70% 标的在均线上 = 健康
BREADTH_MODERATE            = 50       # ≥50% = 一般
BREADTH_WEAK                = 30       # <50% = 弱

# ── 仓位相关性 ────────────────────────────────────────────────
CORR_BLOCK                  = 0.85     # 相关性>0.85 = 拒绝开仓
CORR_WARN                   = 0.70     # 相关性>0.70 = 减半仓位
CORR_LOOKBACK_MONTHS        = 3        # 3个月日收益率相关性
CORR_MIN_BARS               = 15       # 计算相关性最少N个共同交易日

# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 1. EPS 加速度分析（核心：Minervini 三季连续加速）
# ─────────────────────────────────────────────────────────────

def analyze_eps_acceleration(ticker: str) -> dict:
    """
    分析 EPS 加速度趋势（最近4季）。

    加速判定：
      当季 EPS增速 > 上季 EPS增速 > 上上季 EPS增速 = 三段加速 → 最强
      两段加速 → 良好
      一段加速 → 中性
      减速     → 警告

    数据来源：yfinance earnings_history（季度实际EPS + 预期EPS）
    """
    try:
        tk = yf.Ticker(ticker)
        eh = tk.earnings_history

        if eh is None or eh.empty:
            return {"ok": False, "reason": "无季度财报历史数据"}

        # 按日期降序排列，取最近8季（YoY需i+4，8季可得4个增速、3段加速比较）
        eh = eh.sort_index(ascending=False).head(8)

        quarters = []
        for idx, row in eh.iterrows():
            eps_est = row.get("epsEstimate")
            eps_act = row.get("epsActual")
            if eps_est is None or eps_act is None:
                continue
            try:
                eps_est = float(eps_est)
                eps_act = float(eps_act)
                surprise_pct = (eps_act - eps_est) / abs(eps_est) * 100 if eps_est != 0 else 0
                quarters.append({
                    "date":         str(idx)[:10] if hasattr(idx, '__str__') else str(idx),
                    "eps_estimate": round(eps_est, 3),
                    "eps_actual":   round(eps_act, 3),
                    "surprise_pct": round(surprise_pct, 1),
                    "beat":         eps_act > eps_est,
                })
            except (TypeError, ValueError):
                continue

        if len(quarters) < 2:
            return {"ok": False, "reason": f"季度数据不足（仅{len(quarters)}季）"}

        # 计算 YoY EPS 增速（需要同比，至少4季数据）
        growth_rates = []
        for i in range(len(quarters)):
            if i + 4 < len(quarters):
                cur  = quarters[i]["eps_actual"]
                yago = quarters[i + 4]["eps_actual"]
                if yago != 0 and yago > 0:
                    gr = (cur - yago) / yago * 100
                    growth_rates.append(round(gr, 1))
                else:
                    growth_rates.append(None)
            else:
                growth_rates.append(None)

        # 加速度分析（最近3期有效增速的趋势）
        valid_rates = [(i, r) for i, r in enumerate(growth_rates) if r is not None][:3]

        acceleration_count = 0
        acceleration_detail = []
        for i in range(len(valid_rates) - 1):
            newer_idx, newer_rate = valid_rates[i]
            older_idx, older_rate = valid_rates[i + 1]
            if newer_rate > older_rate:
                acceleration_count += 1
                acceleration_detail.append(
                    f"Q{i+1}→Q{i+2}: {older_rate:.0f}% → {newer_rate:.0f}% ✅加速"
                )
            else:
                acceleration_detail.append(
                    f"Q{i+1}→Q{i+2}: {older_rate:.0f}% → {newer_rate:.0f}% ⚠️减速"
                )

        # 加速信号评级
        # FP1-4 修复：acceleration_count=2 表示"两段加速"（2次相邻季度比较各有加速）
        # O'Neil 真正"三段连续加速"需要4个增速数据点，产生3次相邻比较各自加速
        # 此处改为正确表述：count=2 → 两段加速，count>=3 → 三段或更多
        if acceleration_count >= 2 and valid_rates:
            latest_rate = valid_rates[0][1]
            accel_signal = (
                "🔥 三段或以上连续加速（CANSLIM最强信号）" if acceleration_count >= 3 and latest_rate > 25 else
                "✅ 两段连续加速" if acceleration_count == 2 else
                "✅ 加速中"
            )
            accel_grade = "A" if latest_rate > 50 else "B"
        elif acceleration_count == 1:
            accel_signal = "🟡 单季加速，持续观察"
            accel_grade = "C"
        else:
            accel_signal = "🔴 EPS增速减缓，谨慎"
            accel_grade = "D"

        # 最新一季超预期分析
        latest_q = quarters[0]
        surprise_analysis = (
            "💥 重大超预期" if latest_q["surprise_pct"] > 10 else
            "✅ 超预期"     if latest_q["surprise_pct"] > 5  else
            "🟡 轻微超预期" if latest_q["surprise_pct"] > 0  else
            "🔴 未达预期"   if latest_q["surprise_pct"] < 0  else "符合预期"
        )

        # O'Neil C 标准检查（当季 EPS YoY > 25%）
        latest_growth = valid_rates[0][1] if valid_rates else None
        oneil_c = (latest_growth is not None and latest_growth >= 25)

        return {
            "ok":                True,
            "ticker":            ticker,
            "quarters":          quarters[:4],
            "eps_growth_rates":  [(valid_rates[i][1] if i < len(valid_rates) else None) for i in range(3)],
            "acceleration_count": acceleration_count,
            "acceleration_detail": acceleration_detail,
            "acceleration_signal": accel_signal,
            "acceleration_grade":  accel_grade,
            "latest_surprise_pct": latest_q["surprise_pct"],
            "surprise_analysis":  surprise_analysis,
            "latest_eps_growth_pct": valid_rates[0][1] if valid_rates else None,
            "oneil_c_standard":  oneil_c,
            "oneil_c_note": (
                f"✅ 满足O'Neil C标准：当季EPS YoY {latest_growth:.0f}%≥25%"
                if oneil_c else
                f"❌ 未达O'Neil C标准（EPS YoY {latest_growth:.0f}%，需≥25%）"
                if latest_growth is not None else "数据不足"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 2. 营收加速度分析
# ─────────────────────────────────────────────────────────────

def analyze_revenue_acceleration(ticker: str) -> dict:
    """
    分析营收加速度（季度 YoY）。

    最强信号组合：
      营收加速 + EPS加速 + 超预期 = 机构最爱的"双击"形态
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info

        # 从 yfinance info 获取当前和历史营收数据
        rev_growth   = info.get("revenueGrowth")       # TTM YoY
        rev_quarterly = None

        try:
            qf = tk.quarterly_financials
            if qf is not None and not qf.empty:
                # 找营收行
                rev_row = None
                for possible in ["Total Revenue", "Revenue", "Net Revenue"]:
                    if possible in qf.index:
                        rev_row = qf.loc[possible]
                        break

                if rev_row is not None:
                    rev_quarterly = rev_row.sort_index(ascending=False)
        except Exception:
            pass

        revenue_accels = []
        if rev_quarterly is not None and len(rev_quarterly) >= 5:
            vals = rev_quarterly.values
            for i in range(min(3, len(vals) - 4)):
                cur  = float(vals[i])   if vals[i]   and not np.isnan(vals[i])   else None
                yago = float(vals[i+4]) if vals[i+4] and not np.isnan(vals[i+4]) else None
                if cur and yago and yago > 0:
                    gr = (cur - yago) / yago * 100
                    revenue_accels.append(round(gr, 1))

        # 计算加速方向
        rev_accel_count = 0
        rev_accel_detail = []
        for i in range(len(revenue_accels) - 1):
            if revenue_accels[i] > revenue_accels[i + 1]:
                rev_accel_count += 1
                rev_accel_detail.append(
                    f"Q{i+1}→Q{i+2}: {revenue_accels[i+1]:.0f}% → {revenue_accels[i]:.0f}% ✅"
                )
            else:
                rev_accel_detail.append(
                    f"Q{i+1}→Q{i+2}: {revenue_accels[i+1]:.0f}% → {revenue_accels[i]:.0f}% ⚠️减速"
                )

        # O'Neil A 标准：年度营收增长 >25%
        rev_ttm_pct = float(rev_growth) * 100 if rev_growth is not None else None
        oneil_a_rev = rev_ttm_pct is not None and rev_ttm_pct >= 25

        return {
            "ok":                  True,
            "ticker":              ticker,
            "rev_growth_ttm_pct":  round(rev_ttm_pct, 1) if rev_ttm_pct is not None else None,
            "quarterly_rev_growth": revenue_accels,
            "rev_accel_count":     rev_accel_count,
            "rev_accel_detail":    rev_accel_detail,
            "oneil_a_revenue":     oneil_a_rev,
            "oneil_note": (
                f"✅ 营收满足O'Neil A标准：TTM {rev_ttm_pct:.0f}%≥25%"
                if oneil_a_rev else
                f"营收TTM {rev_ttm_pct:.0f}%（O'Neil需≥25%）"
                if rev_ttm_pct is not None else "营收数据不足"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 3. 财报后漂移（PEAD：Post-Earnings Announcement Drift）
# ─────────────────────────────────────────────────────────────

def analyze_pead(ticker: str) -> dict:
    """
    财报后价格漂移分析（Post-Earnings Announcement Drift）。

    学术发现（Ball & Brown 1968，后续大量复制）：
      大幅超预期的股票在财报后4-8周会持续漂移上涨
      大幅未达预期的股票在财报后持续漂移下跌
      散户反应慢，机构逐渐建仓，造成延续性漂移

    使用方式：
      财报超预期 >5% + PEAD信号 = 可在财报后追入（不赌二元事件，赌PEAD）
    """
    try:
        tk  = yf.Ticker(ticker)
        eh  = tk.earnings_history
        prices = tk.history(period="3y", interval="1d")

        if eh is None or eh.empty or prices.empty:
            return {"ok": False, "reason": "数据不足"}

        prices.index = prices.index.tz_localize(None)

        pead_records = []
        for idx, row in eh.head(6).iterrows():
            try:
                eps_est = float(row.get("epsEstimate") or 0)
                eps_act = float(row.get("epsActual") or 0)
                if eps_est == 0:
                    continue
                surprise = (eps_act - eps_est) / abs(eps_est) * 100

                q_date = pd.Timestamp(str(idx)[:10])
                # 财报日后30天的价格区间
                after  = prices.loc[q_date: q_date + timedelta(days=35)]
                if len(after) < 5:
                    continue

                # FP0-2 修复：财报通常盘后发布，当日涨跌是"二元博弈"非漂移
                # PEAD 从 T+1 开盘价开始计量（Ball & Brown 1968 原始定义）
                # after.iloc[0] 是财报当日，after.iloc[1] 是 T+1（需至少2行）
                if len(after) < 2:
                    continue
                price_d0   = float(after["Open"].iloc[1])   # T+1 开盘（漂移起点）
                price_d30  = float(after["Close"].iloc[-1])
                drift_30d  = (price_d30 - price_d0) / price_d0 * 100

                # 前5天和后5天（从T+1开盘起计5天）
                price_d5   = float(after["Close"].iloc[min(5, len(after)-1)])
                initial_5d = (price_d5 - price_d0) / price_d0 * 100

                pead_records.append({
                    "date":         str(q_date.date()),
                    "surprise_pct": round(surprise, 1),
                    "initial_5d_pct": round(initial_5d, 1),
                    "drift_30d_pct":  round(drift_30d, 1),
                    "pead_type": (
                        "正向漂移" if surprise > 5 and drift_30d > 0 else
                        "负向漂移" if surprise < -5 and drift_30d < 0 else
                        "反转"    if surprise > 5 and drift_30d < 0 else
                        "中性"
                    ),
                })
            except Exception:
                continue

        if not pead_records:
            return {"ok": False, "reason": "无法计算PEAD（财报和价格数据匹配失败）"}

        # PEAD 一致性分析
        positive_pead = [r for r in pead_records if r["pead_type"] == "正向漂移"]
        pead_consistency = len(positive_pead) / len(pead_records) if pead_records else 0

        # 最近一季 PEAD 预测
        latest = pead_records[0]
        pead_signal = None
        if latest["surprise_pct"] > 5:
            if pead_consistency > 0.6:
                pead_signal = f"📈 PEAD做多：最近超预期{latest['surprise_pct']:.0f}%，历史{pead_consistency*100:.0f}%概率延续漂移"
            else:
                pead_signal = f"🟡 超预期{latest['surprise_pct']:.0f}%，但历史PEAD不稳定（{pead_consistency*100:.0f}%）"
        elif latest["surprise_pct"] < -5:
            pead_signal = f"📉 负向PEAD风险：未达预期{abs(latest['surprise_pct']):.0f}%，避免做多"

        avg_30d_drift = float(np.mean([r["drift_30d_pct"] for r in pead_records if r["surprise_pct"] > 5])) if positive_pead else 0

        return {
            "ok":              True,
            "ticker":          ticker,
            "pead_records":    pead_records,
            "pead_consistency": round(pead_consistency * 100, 1),
            "avg_30d_drift_when_beat": round(avg_30d_drift, 1),
            "latest_surprise_pct": latest["surprise_pct"],
            "pead_signal":     pead_signal,
            "note": (
                "PEAD效应在大型机构标的上较弱（机构反应快），"
                "在中小型成长股（市值10-100B）上效果最显著。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 4. 因子质量分析（ROE / ROIC / 毛利率）
# ─────────────────────────────────────────────────────────────

def analyze_quality_factors(ticker: str) -> dict:
    """
    质量因子分析。

    专业量化基金的质量因子包括：
      ROE  > 15%  — 资本效率高，每1美元权益产生的利润
      ROIC > 10%  — 投入资本回报，反映护城河深度
      毛利率趋势  — 扩张=定价权，收缩=竞争压力
      净利率      — 综合盈利能力

    巴菲特核心指标：ROE连续10年>15%，毛利率稳定>40% = 护城河
    Minervini核心：成长股可以暂时低ROE，但毛利率必须稳定或扩张
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        roe         = info.get("returnOnEquity")
        roe_pct     = float(roe) * 100 if roe is not None else None

        # 毛利率
        gross_margin = info.get("grossMargins")
        gm_pct       = float(gross_margin) * 100 if gross_margin is not None else None

        # 净利率
        profit_margin = info.get("profitMargins")
        pm_pct        = float(profit_margin) * 100 if profit_margin is not None else None

        # 营业利润率
        op_margin    = info.get("operatingMargins")
        om_pct       = float(op_margin) * 100 if op_margin is not None else None

        # 自由现金流 vs 净利润（FCF质量）
        fcf         = info.get("freeCashflow")
        net_income  = info.get("netIncomeToCommon")
        fcf_quality = None
        if fcf and net_income and net_income > 0:
            fcf_quality = float(fcf) / float(net_income)

        # 债务/权益比（杠杆风险）
        de_ratio    = info.get("debtToEquity")
        de          = float(de_ratio) / 100 if de_ratio is not None else None

        # 评分
        quality_score = 0
        quality_notes = []

        if roe_pct is not None:
            if roe_pct > 20:
                quality_score += 25
                quality_notes.append(f"ROE={roe_pct:.1f}%（>20%，优秀资本效率）")
            elif roe_pct > 15:
                quality_score += 15
                quality_notes.append(f"ROE={roe_pct:.1f}%（>15%，良好）")
            elif roe_pct > 0:
                quality_score += 5
                quality_notes.append(f"ROE={roe_pct:.1f}%（正值但偏低）")
            else:
                quality_notes.append(f"ROE={roe_pct:.1f}%（负值，亏损中）")

        if gm_pct is not None:
            if gm_pct > 60:
                quality_score += 25
                quality_notes.append(f"毛利率={gm_pct:.1f}%（>60%，强护城河）")
            elif gm_pct > 40:
                quality_score += 15
                quality_notes.append(f"毛利率={gm_pct:.1f}%（>40%，良好）")
            elif gm_pct > 20:
                quality_score += 8
                quality_notes.append(f"毛利率={gm_pct:.1f}%（20-40%，普通）")
            else:
                quality_notes.append(f"毛利率={gm_pct:.1f}%（<20%，竞争激烈或规模未达）")

        if fcf_quality is not None:
            if fcf_quality > 0.8:
                quality_score += 25
                quality_notes.append(f"FCF/净利={fcf_quality:.1f}x（现金流质量高）")
            elif fcf_quality > 0.5:
                quality_score += 15
                quality_notes.append(f"FCF/净利={fcf_quality:.1f}x（现金流质量中等）")
            elif fcf_quality < 0:
                quality_notes.append("FCF为负（烧钱阶段，需关注续航能力）")

        if de is not None:
            if de < 0.5:
                quality_score += 25
                quality_notes.append(f"债务/权益={de:.2f}x（低杠杆，财务稳健）")
            elif de < 1.5:
                quality_score += 10
                quality_notes.append(f"债务/权益={de:.2f}x（中等杠杆）")
            else:
                quality_notes.append(f"债务/权益={de:.2f}x（高杠杆，注意利率风险）")

        quality_score = min(100, quality_score)
        quality_grade = (
            "A" if quality_score >= 75 else
            "B" if quality_score >= 55 else
            "C" if quality_score >= 35 else "D"
        )

        return {
            "ok":              True,
            "ticker":          ticker,
            "roe_pct":         round(roe_pct, 1) if roe_pct is not None else None,
            "gross_margin_pct": round(gm_pct, 1)  if gm_pct  is not None else None,
            "profit_margin_pct": round(pm_pct, 1)  if pm_pct  is not None else None,
            "op_margin_pct":   round(om_pct, 1)   if om_pct  is not None else None,
            "fcf_quality_ratio": round(fcf_quality, 2) if fcf_quality is not None else None,
            "debt_equity":     round(de, 2)        if de      is not None else None,
            "quality_score":   quality_score,
            "quality_grade":   quality_grade,
            "quality_notes":   quality_notes,
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 5. 市场宽度（Market Breadth）
# ─────────────────────────────────────────────────────────────

def get_market_breadth() -> dict:
    """
    市场宽度指标：大盘健康度的X光片。

    个人可获取的免费指标：
      - 主要板块ETF vs 其MA50（多少板块在上升趋势）
      - 代表性股票池的MA200以上比例（约估市场宽度）
      - 新高/新低比值（实时市场内部强度）

    宽度指标的意义：
      宽度好 + 指数涨  = 健康牛市，放心做多
      宽度差 + 指数涨  = 少数股撑场，随时可能崩塌
      宽度好 + 指数跌  = 调整中的健康市场，即将见底
      宽度差 + 指数跌  = 熊市，空头当道，严格控制仓位
    """
    try:
        # 11个板块ETF
        sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
                        "XLI", "XLB", "XLRE", "XLU", "XLC"]
        # 代表性个股（用于估算MA200宽度）
        breadth_stocks = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
            "JPM", "JNJ", "XOM", "HD", "PG", "UNH", "V", "MA",
            "BAC", "AVGO", "LLY", "ABBV", "MRK", "CVX", "PFE",
            "COST", "DIS", "NFLX", "INTC", "AMD", "QCOM", "TXN",
        ]

        all_tickers = list(set(sector_etfs + breadth_stocks + ["SPY"]))
        raw = yf.download(all_tickers, period="1y", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)

        def _get_close(tk):
            try:
                if tk in raw.columns.get_level_values(0):
                    return raw[tk]["Close"].dropna()
            except Exception:
                pass
            return None

        # 板块宽度（多少板块在MA50上方）
        sectors_above_ma50 = 0
        sector_details = []
        for etf in sector_etfs:
            c = _get_close(etf)
            if c is None or len(c) < 50:
                continue
            ma50 = float(c.rolling(50).mean().iloc[-1])
            price = float(c.iloc[-1])
            above = price > ma50
            if above:
                sectors_above_ma50 += 1
            sector_details.append({
                "etf":    etf,
                "above_ma50": above,
                "pct_vs_ma50": round((price / ma50 - 1) * 100, 1),
            })

        # 个股宽度（估算MA200以上比例）
        above_ma200 = 0
        checked = 0
        for tk in breadth_stocks:
            c = _get_close(tk)
            if c is None or len(c) < 200:
                continue
            ma200 = float(c.rolling(200).mean().iloc[-1])
            price  = float(c.iloc[-1])
            checked += 1
            if price > ma200:
                above_ma200 += 1

        breadth_pct_ma200 = above_ma200 / checked * 100 if checked > 0 else None
        sectors_pct      = sectors_above_ma50 / len(sector_details) * 100 if sector_details else None

        # SPY 当前状态
        spy_close = _get_close("SPY")
        spy_ma50  = float(spy_close.rolling(50).mean().iloc[-1])  if spy_close is not None else None
        spy_ma200 = float(spy_close.rolling(200).mean().iloc[-1]) if spy_close is not None and len(spy_close) >= 200 else None
        spy_price = float(spy_close.iloc[-1])                     if spy_close is not None else None

        # 综合宽度信号
        breadth_score = 0
        if sectors_pct is not None:
            breadth_score += int(sectors_pct * 0.5)
        if breadth_pct_ma200 is not None:
            breadth_score += int(breadth_pct_ma200 * 0.5)

        breadth_signal = (
            "🟢 市场宽度健康（>70%），牛市基础扎实"  if breadth_score >= 70 else
            "🟡 市场宽度中性（50-70%），选股要严格"  if breadth_score >= 50 else
            "🔴 市场宽度差（<50%），少数龙头撑场，谨慎做多"
        )

        return {
            "ok":                    True,
            "sectors_above_ma50":    sectors_above_ma50,
            "total_sectors":         len(sector_details),
            "sectors_above_ma50_pct": round(sectors_pct, 1) if sectors_pct is not None else None,
            "stocks_above_ma200_pct": round(breadth_pct_ma200, 1) if breadth_pct_ma200 is not None else None,
            "breadth_score":         breadth_score,
            "breadth_signal":        breadth_signal,
            "sector_details":        sector_details,
            "spy": {
                "price":  round(spy_price, 2) if spy_price else None,
                "ma50":   round(spy_ma50, 2)  if spy_ma50  else None,
                "ma200":  round(spy_ma200, 2) if spy_ma200 else None,
                "above_ma50":  spy_price > spy_ma50  if spy_price and spy_ma50  else None,
                "above_ma200": spy_price > spy_ma200 if spy_price and spy_ma200 else None,
            },
            "interpretation": (
                "宽度>50%时，SPY的涨势是真实的，可以放心做多。"
                "宽度<30%时，即使指数在涨，底层在腐烂——这是熊市陷阱。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 6. 仓位相关性检查（Portfolio Correlation Gate）
# ─────────────────────────────────────────────────────────────

def check_position_correlation(new_ticker: str, existing_tickers: list) -> dict:
    """
    检查新仓位与现有持仓的相关性。

    原则：相关系数 > 0.7 = 高度相关 = 等同于加仓同一个方向
    后果：3只半导体股 = 1个集中赌注，不是3个独立持仓

    专业风险管理：
      - 相关>0.85：硬性拒绝（等于重复持仓）
      - 相关0.7-0.85：警告，仓位减半
      - 相关<0.7：可以正常开仓
    """
    if not existing_tickers:
        return {"ok": True, "can_add": True, "max_corr": 0,
                "note": "无现有持仓，可自由开仓"}

    try:
        all_tickers = list(set([new_ticker] + existing_tickers))
        raw = yf.download(all_tickers, period="3mo", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)

        # 提取收益率序列
        returns = {}
        for tk in all_tickers:
            try:
                if len(all_tickers) == 1:
                    c = raw["Close"]
                elif tk in raw.columns.get_level_values(0):
                    c = raw[tk]["Close"]
                else:
                    continue
                c = c.dropna()
                if len(c) >= 20:
                    returns[tk] = c.pct_change().dropna()
            except Exception:
                continue

        if new_ticker not in returns or not returns:
            return {"ok": True, "can_add": True, "max_corr": 0,
                    "note": "相关性数据不足，默认允许"}

        new_ret = returns[new_ticker]
        corr_results = []

        for existing in existing_tickers:
            if existing not in returns:
                continue
            ex_ret = returns[existing]
            # 对齐日期
            aligned = pd.concat([new_ret, ex_ret], axis=1).dropna()
            if len(aligned) < 15:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            corr_results.append({
                "existing_ticker": existing,
                "correlation":     round(corr, 3),
                "risk_level": (
                    "🔴 高度相关" if corr > 0.85 else
                    "🟡 中度相关" if corr > 0.70 else
                    "🟢 低相关"
                ),
            })

        if not corr_results:
            return {"ok": True, "can_add": True, "max_corr": 0,
                    "note": "无法计算相关性，默认允许"}

        max_corr = max(r["correlation"] for r in corr_results)
        most_correlated = max(corr_results, key=lambda r: r["correlation"])

        if max_corr > 0.85:
            action = "block"
            action_note = (
                f"❌ 禁止：{new_ticker}与{most_correlated['existing_ticker']}"
                f"相关性{max_corr:.2f}（>0.85），等同于重复持仓"
            )
        elif max_corr > 0.70:
            action = "warn"
            action_note = (
                f"⚠️ 警告：{new_ticker}与{most_correlated['existing_ticker']}"
                f"相关性{max_corr:.2f}（>0.70），建议仓位减半"
            )
        else:
            action = "allow"
            action_note = f"✅ {new_ticker}与现有持仓相关性低（最高{max_corr:.2f}），可正常开仓"

        return {
            "ok":           True,
            "new_ticker":   new_ticker,
            "can_add":      action != "block",
            "action":       action,
            "action_note":  action_note,
            "max_correlation": max_corr,
            "most_correlated": most_correlated,
            "all_correlations": corr_results,
            "position_sizing_note": (
                "半仓：相关性0.7-0.85时，将仓位减半以维持整体风险不变" if action == "warn" else ""
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e), "can_add": True,
                "note": f"相关性检查失败，默认允许：{e}"}


# ─────────────────────────────────────────────────────────────
# 7. 完整财报质量综合报告
# ─────────────────────────────────────────────────────────────

def full_earnings_analysis(ticker: str) -> dict:
    """
    一键生成完整财报质量报告。

    CANSLIM综合评分：
      C（当季EPS>25%）      25分
      A（年度增长>25%）     15分
      N（新品/新高/新管理） 人工
      S（供应/需求）        已在九关
      L（行业领头羊RS>85）  已在九关
      I（机构持仓）         已在九关
      M（大盘方向）         已在市场状态

    本模块负责 C + A + 财报质量 + PEAD + ROE/毛利率
    """
    eps_result  = analyze_eps_acceleration(ticker)
    rev_result  = analyze_revenue_acceleration(ticker)
    pead_result = analyze_pead(ticker)
    qual_result = analyze_quality_factors(ticker)

    # 综合评分
    score = 0
    grade_notes = []

    # EPS 加速度（最高35分）
    if eps_result.get("ok"):
        if eps_result.get("oneil_c_standard"):
            score += 20
            grade_notes.append("EPS满足O'Neil C标准（+20）")
        accel = eps_result.get("acceleration_count", 0)
        if accel >= 2:
            score += 15
            grade_notes.append(f"EPS三段加速（+15）")
        elif accel == 1:
            score += 8
            grade_notes.append(f"EPS单段加速（+8）")
        surprise = eps_result.get("latest_surprise_pct", 0)
        if surprise > 10:
            score += 10
            grade_notes.append(f"EPS超预期{surprise:.0f}%（+10）")
        elif surprise > 5:
            score += 5
            grade_notes.append(f"EPS超预期{surprise:.0f}%（+5）")

    # 营收加速（最高20分）
    if rev_result.get("ok"):
        if rev_result.get("oneil_a_revenue"):
            score += 10
            grade_notes.append("营收满足O'Neil A标准（+10）")
        if rev_result.get("rev_accel_count", 0) >= 2:
            score += 10
            grade_notes.append("营收加速（+10）")

    # PEAD 信号（最高15分）
    if pead_result.get("ok"):
        cons = pead_result.get("pead_consistency", 0)
        latest_surp = pead_result.get("latest_surprise_pct", 0)
        if latest_surp > 5 and cons > 60:
            score += 15
            grade_notes.append(f"PEAD历史延续率{cons:.0f}%（+15）")

    # 质量因子（最高30分）
    if qual_result.get("ok"):
        q = qual_result.get("quality_score", 0)
        q_pts = int(q * 0.3)
        score += q_pts
        if q_pts > 0:
            grade_notes.append(f"质量因子{qual_result.get('quality_grade','')}（+{q_pts}）")

    score = min(100, score)
    overall_grade = (
        "A" if score >= 80 else
        "B" if score >= 60 else
        "C" if score >= 40 else "D"
    )

    # 操作建议
    if score >= 80:
        action = "✅ 财报质量极高，技术信号一旦就位可果断入场"
    elif score >= 60:
        action = "🟡 财报质量良好，与技术信号结合使用"
    elif score >= 40:
        action = "⚠️ 财报质量一般，仓位不宜过重"
    else:
        action = "❌ 财报质量差，不建议做多，等待财报改善"

    return {
        "ok":            True,
        "ticker":        ticker,
        "canslim_score": score,
        "overall_grade": overall_grade,
        "action":        action,
        "grade_notes":   grade_notes,
        "eps_acceleration": eps_result,
        "revenue_acceleration": rev_result,
        "pead":          pead_result,
        "quality_factors": qual_result,
    }


# ─────────────────────────────────────────────────────────────
# CLI 运行
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    print(f"\n分析 {ticker} 财报质量...")
    result = full_earnings_analysis(ticker)

    print(f"\n{'='*55}")
    print(f"  {ticker} 财报质量报告")
    print(f"{'='*55}")
    print(f"  CANSLIM评分：{result['canslim_score']}/100  ({result['overall_grade']})")
    print(f"  {result['action']}")

    for note in result.get("grade_notes", []):
        print(f"    · {note}")

    eps = result.get("eps_acceleration", {})
    if eps.get("ok"):
        print(f"\n  EPS加速：{eps.get('acceleration_signal', '')}")
        print(f"  最新EPS超预期：{eps.get('latest_surprise_pct', 0):.1f}%")
        print(f"  O'Neil C标准：{'✅' if eps.get('oneil_c_standard') else '❌'}")

    qual = result.get("quality_factors", {})
    if qual.get("ok"):
        print(f"\n  质量因子：{qual.get('quality_grade', '')}（{qual.get('quality_score', 0)}/100）")
        if qual.get("gross_margin_pct"):
            print(f"  毛利率：{qual['gross_margin_pct']:.1f}%")
        if qual.get("roe_pct"):
            print(f"  ROE：{qual['roe_pct']:.1f}%")

    pead = result.get("pead", {})
    if pead.get("ok") and pead.get("pead_signal"):
        print(f"\n  PEAD信号：{pead['pead_signal']}")
    print(f"{'='*55}")
