"""
机构持仓 13F 分析模块

引用项目：
  edgartools ⭐3,000+ (dgunning/edgartools, MIT)
  — 直接读取 SEC EDGAR 13F 原始文件，追踪季度增减持

功能：
  - 当前主要机构持有者（yfinance）
  - 季度 QoQ 增减持追踪（edgartools）
  - Gate5 信号评估（机构认可 + 资金流向）
"""

import yfinance as yf
import numpy as np

try:
    import edgar as _edgar
    _edgar.set_identity("StockRadar research@stockradar.io")
    EDGAR_OK = True
except ImportError:
    EDGAR_OK = False


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def get_institutional_13f(ticker: str) -> dict:
    t    = yf.Ticker(ticker)
    info = t.info

    total_inst_pct    = float(info.get("institutionPercentHeld") or 0) * 100
    total_insider_pct = float(info.get("heldPercentInsiders")    or 0) * 100

    holders     = _parse_holders(t)
    edgar_data  = _enrich_with_edgar(ticker, holders[:3]) if EDGAR_OK else []
    flow        = _flow_summary(edgar_data)
    gate_status, gate_detail = _gate5_signal(total_inst_pct, holders, edgar_data)

    return {
        "ticker":                 ticker,
        "total_institutional_pct": round(total_inst_pct, 1),
        "total_insiders_pct":      round(total_insider_pct, 1),
        "holder_count":            len(holders),
        "holders":                 holders[:15],
        "edgar_history":           edgar_data,
        "flow_summary":            flow,
        "gate5_status":            gate_status,
        "gate5_detail":            gate_detail,
        "edgar_available":         EDGAR_OK,
        "source": ("edgartools ⭐3,000+ (dgunning/edgartools)" if EDGAR_OK
                   else "yfinance only — 安装 edgartools 获取季度历史"),
    }


# ─────────────────────────────────────────────────────────────
# 解析 yfinance 持有者数据
# ─────────────────────────────────────────────────────────────

def _parse_holders(t) -> list:
    holders = []

    for df_attr, h_type in [("institutional_holders", "institution"),
                             ("mutualfund_holders",    "mutual_fund")]:
        try:
            df = getattr(t, df_attr)
            if df is None or df.empty:
                continue
            limit = 15 if h_type == "institution" else 5
            for _, row in df.head(limit).iterrows():
                shares = row.get("Shares")
                value  = row.get("Value")
                pct    = row.get("% Out")
                holders.append({
                    "name":            str(row.get("Holder", "Unknown")),
                    "shares":          int(float(shares)) if _valid(shares) else None,
                    "value_usd":       int(float(value))  if _valid(value)  else None,
                    "pct_outstanding": round(float(pct) * 100, 3) if _valid(pct) else None,
                    "date_reported":   str(row.get("Date Reported", ""))[:10],
                    "type":            h_type,
                    "qoq_change":      None,
                    "trend":           "unknown",
                })
        except Exception:
            continue

    return holders


# ─────────────────────────────────────────────────────────────
# edgartools 季度追踪
# ─────────────────────────────────────────────────────────────

def _enrich_with_edgar(ticker: str, top_holders: list) -> list:
    """
    用 edgartools (⭐3,000+) 查各大持有者最近两个季度的13F
    匹配 ticker，计算 QoQ 持仓变化
    """
    results = []

    try:
        from edgar import Company

        for holder in top_holders:
            fund_name = holder.get("name", "")
            if not fund_name or fund_name == "Unknown":
                continue
            try:
                company = Company(fund_name)
                filings = company.get_filings(form="13F-HR").latest(2)
                if not filings:
                    continue

                quarters = []
                for filing in filings:
                    try:
                        obj       = filing.obj()
                        infotable = getattr(obj, "infotable", None)
                        if infotable is None or infotable.empty:
                            continue

                        # 定位列名
                        name_col   = next((c for c in infotable.columns
                                          if "issuer" in c.lower() or "name" in c.lower()), None)
                        shares_col = next((c for c in infotable.columns
                                          if "sshprnamt" in c.lower() or "shares" in c.lower()), None)
                        value_col  = next((c for c in infotable.columns
                                          if "value" in c.lower()), None)
                        if name_col is None:
                            continue

                        mask    = infotable[name_col].astype(str).str.contains(
                                      ticker.replace(".", r"\."), case=False, na=False, regex=True)
                        matched = infotable[mask]
                        if matched.empty:
                            continue

                        row = matched.iloc[0]
                        quarters.append({
                            "quarter": str(filing.filing_date)[:7],
                            "shares":  int(row[shares_col]) if shares_col and _valid(row[shares_col]) else None,
                            "value_k": int(row[value_col])  if value_col  and _valid(row[value_col])  else None,
                        })
                    except Exception:
                        continue

                if not quarters:
                    continue

                # QoQ 变化
                qoq_pct = None
                trend   = "unknown"
                if len(quarters) >= 2 and quarters[0].get("shares") and quarters[1].get("shares"):
                    new_sh = quarters[0]["shares"]
                    old_sh = quarters[1]["shares"]
                    if old_sh > 0:
                        qoq_pct = round((new_sh - old_sh) / old_sh * 100, 1)
                        trend   = ("增持" if qoq_pct > 5 else
                                   "减持" if qoq_pct < -5 else "持平")

                results.append({"fund": fund_name, "quarters": quarters,
                                 "qoq_change_pct": qoq_pct, "trend": trend})

                # 回写到 holder 对象
                for h in top_holders:
                    if h.get("name") == fund_name:
                        h["qoq_change"] = qoq_pct
                        h["trend"]      = trend

            except Exception:
                continue

    except Exception:
        pass

    return results


# ─────────────────────────────────────────────────────────────
# Gate5 信号 + 资金流向摘要
# ─────────────────────────────────────────────────────────────

def _gate5_signal(total_pct: float, holders: list, edgar: list):
    increasing = sum(1 for e in edgar if e.get("trend") == "增持")
    decreasing = sum(1 for e in edgar if e.get("trend") == "减持")

    if total_pct > 30:
        if increasing > decreasing:
            return ("pass",
                    f"机构持仓{total_pct:.1f}%，主力增持中（{increasing}家增/{decreasing}家减）")
        elif decreasing > increasing:
            return ("warn",
                    f"机构持仓{total_pct:.1f}%，但头部基金减持（{decreasing}家减持），需警惕")
        else:
            return ("pass",
                    f"机构持仓{total_pct:.1f}%（共{len(holders)}家机构），稳定持有")
    elif total_pct > 10:
        return ("warn", f"机构持仓{total_pct:.1f}%偏低，建议查13F确认趋势")
    else:
        return ("fail", f"机构持仓{total_pct:.1f}%，机构认可度不足")


def _flow_summary(edgar: list) -> dict:
    if not edgar:
        return {"status": "无季度历史数据（安装 edgartools 获取）"}
    increasing = [e["fund"] for e in edgar if e.get("trend") == "增持"]
    decreasing = [e["fund"] for e in edgar if e.get("trend") == "减持"]
    return {
        "increasing": increasing,
        "decreasing": decreasing,
        "net_trend":  ("净流入" if len(increasing) > len(decreasing) else
                       "净流出" if len(decreasing) > len(increasing) else "中性"),
    }


def _valid(v) -> bool:
    try:
        f = float(v)
        return f == f  # NaN → False
    except Exception:
        return False
