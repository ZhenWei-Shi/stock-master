"""
板块轮动追踪（Sector Rotation）

架构思路：
  资金流动是分层的：宏观 → 板块 → 行业/供应链层 → 个股
  当前 cold_model 直接从个股出发，缺少「板块背景」这一中间层。
  本模块填补这个空白：
    1. 追踪 11 大 GICS 板块 ETF + 4 个关键子行业 ETF
    2. 计算相对 SPY 的超额收益（1M / 3M），按加权综合热度排名
    3. 检测加速信号（1M RS > 3M RS）= 资金刚开始流入，而非已涨完
    4. 通过 ticker → yfinance sector/industry 映射确定个股所属板块
    5. 输出 check_sector_gate() 供 cold_model 前置判断

缓存策略：
  每 4 小时刷新一次（盘中行情变化，但板块轮动不是分钟级信号）
  强制刷新：force=True
"""
from __future__ import annotations

import os
import json
import functools
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pytz

ET     = pytz.timezone("America/New_York")
_DATA  = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE = os.path.join(_DATA, "sector_cache.json")

os.makedirs(_DATA, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 配置常量
# ══════════════════════════════════════════════════════════════
CACHE_TTL_HOURS     = 4       # 缓存有效期
SECTOR_TOP_N        = 3       # 前N名视为"热板块"
SECTOR_BOTTOM_N     = 3       # 后N名视为"冷板块"
RS_1M_WEIGHT        = 0.6     # 1个月超额收益权重
RS_3M_WEIGHT        = 0.4     # 3个月超额收益权重
ACCEL_BONUS         = 3       # 加速板块加分（1M RS > 3M RS）
# ══════════════════════════════════════════════════════════════

# 14 个追踪标的：11 大 GICS + 3 高优先子行业
SECTOR_ETFS = {
    # 11 大 GICS 板块
    "XLK":  {"name": "科技",       "gics": "Technology"},
    "XLF":  {"name": "金融",       "gics": "Financial Services"},
    "XLE":  {"name": "能源",       "gics": "Energy"},
    "XLV":  {"name": "医疗健康",   "gics": "Healthcare"},
    "XLI":  {"name": "工业",       "gics": "Industrials"},
    "XLC":  {"name": "通信服务",   "gics": "Communication Services"},
    "XLY":  {"name": "非必需消费", "gics": "Consumer Cyclical"},
    "XLP":  {"name": "必需消费",   "gics": "Consumer Defensive"},
    "XLB":  {"name": "材料",       "gics": "Basic Materials"},
    "XLRE": {"name": "房地产",     "gics": "Real Estate"},
    "XLU":  {"name": "公用事业",   "gics": "Utilities"},
    # 3 个高热度子行业 ETF（超越大板块分类）
    "SMH":  {"name": "半导体",     "gics": "_semi"},
    "XBI":  {"name": "生物医药",   "gics": "_biotech"},
    "ITA":  {"name": "国防航天",   "gics": "_defense"},
}

# yfinance sector/industry → 本系统 ETF 映射
# 优先用 industry 做精细匹配，其次用 sector 做粗粒度匹配
_INDUSTRY_MAP = {
    "Semiconductors":                  "SMH",
    "Semiconductor Equipment & Materials": "SMH",
    "Biotechnology":                   "XBI",
    "Drug Manufacturers - General":    "XLV",
    "Drug Manufacturers - Specialty & Generic": "XLV",
    "Medical Devices":                 "XLV",
    "Medical Instruments & Supplies":  "XLV",
    "Diagnostics & Research":          "XLV",
    "Aerospace & Defense":             "ITA",
    "Software - Infrastructure":       "XLK",
    "Software - Application":          "XLK",
    "Information Technology Services": "XLK",
    "Electronic Components":           "XLK",
    "Communication Equipment":         "XLK",
    "Internet Content & Information":  "XLC",
    "Telecom Services":                "XLC",
    "Oil & Gas E&P":                   "XLE",
    "Oil & Gas Midstream":             "XLE",
    "Oil & Gas Refining & Marketing":  "XLE",
    "Banks - Regional":                "XLF",
    "Banks - Diversified":             "XLF",
    "Insurance - Diversified":         "XLF",
    "Asset Management":                "XLF",
    "REIT - Diversified":              "XLRE",
    "REIT - Industrial":               "XLRE",
    "REIT - Office":                   "XLRE",
    "Utilities - Regulated Electric":  "XLU",
    "Utilities - Renewable":           "XLU",
    "Specialty Chemicals":             "XLB",
    "Gold":                            "XLB",
    "Copper":                          "XLB",
    "Auto Manufacturers":              "XLY",
    "Internet Retail":                 "XLY",
    "Specialty Retail":                "XLY",
    "Food Distribution":               "XLP",
    "Beverages - Non-Alcoholic":       "XLP",
    "Packaged Foods":                  "XLP",
    "Trucking":                        "XLI",
    "Farm & Heavy Construction Machinery": "XLI",
    "Electrical Equipment & Parts":    "XLI",
    "Medical Care Facilities":         "XLV",
    "Health Information Services":     "XLV",
    "Waste Management":                "XLI",
    "Scientific & Technical Instruments": "XLK",
    # CCJ/UEC 等铀矿公司行业标签为 Uranium，随核电板块涨跌，映射至 XLU
    "Uranium":                         "XLU",
}

_SECTOR_MAP = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Energy":                 "XLE",
    "Healthcare":             "XLV",
    "Industrials":            "XLI",
    "Communication Services": "XLC",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Basic Materials":        "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
}


# ─────────────────────────────────────────────────────────────
# 缓存 I/O
# ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not os.path.exists(_CACHE):
        return {}
    try:
        with open(_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(data: dict):
    try:
        with open(_CACHE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except OSError:
        pass


def _cache_stale(cache: dict) -> bool:
    ts = cache.get("updated_at")
    if not ts:
        return True
    try:
        # 保留时区信息做 aware-to-aware 比较，避免 UTC 服务器与 ET 缓存时间错位
        updated = datetime.fromisoformat(str(ts))
        now_et  = datetime.now(ET)
        return (now_et - updated).total_seconds() > CACHE_TTL_HOURS * 3600
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────
# 核心：获取板块排名
# ─────────────────────────────────────────────────────────────

def fetch_sector_rankings(force: bool = False) -> dict:
    """
    下载所有板块 ETF 行情，计算对 SPY 的超额收益，返回排名。
    结果缓存 CACHE_TTL_HOURS 小时。

    返回 dict：
      rankings: [{etf, name, rs_1m, rs_3m, heat, accel, rank}, ...]
      top3:     [etf, ...]  # 当前最强3板块
      bottom3:  [etf, ...]  # 当前最弱3板块
      hot_etf:  str         # 第1名
      updated_at: str
    """
    cache = _load_cache()
    if not force and not _cache_stale(cache) and cache.get("rankings"):
        return cache

    all_etfs = list(SECTOR_ETFS.keys()) + ["SPY"]
    try:
        raw = yf.download(
            " ".join(all_etfs),
            period="4mo", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception as e:
        return {"error": f"板块数据获取失败：{e}", "rankings": [], "top3": [], "bottom3": []}

    import pandas as _pd

    def get_close(tk: str):
        try:
            if isinstance(raw.columns, _pd.MultiIndex):
                if tk not in raw.columns.get_level_values(0):
                    return None
                col = raw[tk]["Close"].dropna()
            else:
                # 多 ticker 下载却返回扁平结构 = 数据异常，拒绝使用以免混淆
                return None
            return col if len(col) >= 10 else None
        except Exception:
            return None

    spy = get_close("SPY")
    if spy is None or len(spy) < 21:
        return {"error": "SPY 数据不足", "rankings": [], "top3": [], "bottom3": []}

    def excess_ret(etf_close, days: int) -> float | None:
        """
        计算 ETF 对 SPY 的超额收益（%）。
        使用日期对齐（index.intersection）避免停牌/节假日导致的数据错位。
        返回 None 表示数据不足，与真实超额为 0 的情况区分。
        """
        if etf_close is None:
            return None
        common_idx = etf_close.index.intersection(spy.index)
        if len(common_idx) < days:
            return None
        etf_a = etf_close.reindex(common_idx)
        spy_a = spy.reindex(common_idx)
        base_e = float(etf_a.iloc[-days])
        base_s = float(spy_a.iloc[-days])
        etf_r  = (float(etf_a.iloc[-1]) - base_e) / max(base_e, 0.01)
        spy_r  = (float(spy_a.iloc[-1]) - base_s) / max(base_s, 0.01)
        return round((etf_r - spy_r) * 100, 2)

    scores = []
    for etf, meta in SECTOR_ETFS.items():
        cl    = get_close(etf)
        rs_1m = excess_ret(cl, 21)
        rs_3m = excess_ret(cl, 63)
        # 只用有效数据计算热度，避免 0.0 fallback 污染排名
        if rs_1m is not None and rs_3m is not None:
            heat = RS_1M_WEIGHT * rs_1m + RS_3M_WEIGHT * rs_3m
        elif rs_1m is not None:
            heat = rs_1m   # 仅 1M 有数据
        else:
            heat = 0.0     # 完全无数据，排末尾
        rs_1m = rs_1m or 0.0
        rs_3m = rs_3m or 0.0
        accel = rs_1m > rs_3m  # 资金加速流入
        scores.append({
            "etf":   etf,
            "name":  meta["name"],
            "rs_1m": rs_1m,
            "rs_3m": rs_3m,
            "heat":  round(heat, 2),
            "accel": accel,
        })

    scores.sort(key=lambda x: x["heat"], reverse=True)
    for i, s in enumerate(scores):
        s["rank"] = i + 1

    top3    = [s["etf"] for s in scores[:SECTOR_TOP_N]]
    bottom3 = [s["etf"] for s in scores[-SECTOR_BOTTOM_N:]]

    result = {
        "rankings":   scores,
        "top3":       top3,
        "bottom3":    bottom3,
        "hot_etf":    scores[0]["etf"] if scores else None,
        "hot_name":   scores[0]["name"] if scores else None,
        "updated_at": str(datetime.now(ET)),
    }
    _save_cache(result)
    return result


# ─────────────────────────────────────────────────────────────
# Ticker → 所属板块 ETF
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
def get_sector_for_ticker(ticker: str) -> str | None:
    """
    通过 yfinance info 获取 ticker 所属板块 ETF。
    先尝试 industry（精细），再用 sector（粗粒度）。
    结果在进程生命周期内缓存，避免批量扫描时重复 HTTP 请求。
    """
    try:
        info     = yf.Ticker(ticker).info
        industry = info.get("industry", "")
        sector   = info.get("sector", "")
        etf = _INDUSTRY_MAP.get(industry) or _SECTOR_MAP.get(sector)
        return etf
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# 供 cold_model 调用的门关接口
# ─────────────────────────────────────────────────────────────

def check_sector_gate(ticker: str, sector_etf: str | None = None) -> dict:
    """
    返回板块背景门关结果（供 cold_model 的 gates 字典使用）。

    逻辑：
      当前板块排名前3  → pass，且若加速则 note 加标注
      排名中间（4-11） → pass（中性背景）
      排名后3          → warn（逆风操作，降低期望值）
      无法确定板块     → pass（不因信息缺失阻断交易）
    """
    rankings = fetch_sector_rankings()
    if rankings.get("error") or not rankings.get("rankings"):
        return {"pass": True, "note": "板块排名获取失败，跳过板块门", "rank": None}

    etf = sector_etf or get_sector_for_ticker(ticker)
    if not etf:
        return {"pass": True, "note": "无法确定所属板块，跳过板块门", "rank": None}

    score_map = {s["etf"]: s for s in rankings["rankings"]}
    meta      = score_map.get(etf)
    if not meta:
        return {"pass": True, "note": f"板块 {etf} 不在追踪列表，跳过板块门", "rank": None}

    rank      = meta["rank"]
    n_total   = len(rankings["rankings"])
    name      = meta["name"]
    heat      = meta["heat"]
    accel     = meta["accel"]
    accel_tag = "（资金加速流入↑）" if accel else ""

    if rank <= SECTOR_TOP_N:
        note = (f"板块【{name}】排名第{rank}/{n_total}，热度{heat:+.1f}%"
                + accel_tag + " — 顺势做多")
        return {"pass": True, "note": note, "rank": rank, "etf": etf,
                "accel": accel, "heat": heat}

    elif rank > n_total - SECTOR_BOTTOM_N:
        note = (f"板块【{name}】排名第{rank}/{n_total}（后{SECTOR_BOTTOM_N}名），"
                f"热度{heat:+.1f}% — 逆风区域，建议等板块轮动确认")
        return {"pass": "warn", "note": note, "rank": rank, "etf": etf,
                "accel": accel, "heat": heat}

    else:
        note = (f"板块【{name}】排名第{rank}/{n_total}，热度{heat:+.1f}%"
                + accel_tag + " — 背景中性")
        return {"pass": True, "note": note, "rank": rank, "etf": etf,
                "accel": accel, "heat": heat}


# ─────────────────────────────────────────────────────────────
# 完整报告（供 API / Telegram）
# ─────────────────────────────────────────────────────────────

def sector_rotation_report() -> dict:
    """生成完整板块轮动报告。"""
    data = fetch_sector_rankings()
    if data.get("error"):
        return data

    rankings = data.get("rankings", [])
    top3     = data.get("top3", [])
    bottom3  = data.get("bottom3", [])

    top_detail    = [r for r in rankings if r["etf"] in top3]
    bottom_detail = [r for r in rankings if r["etf"] in bottom3]
    accel_list    = [r for r in rankings if r["accel"] and r["rank"] <= 7]

    hot  = rankings[0] if rankings else {}
    note = _interpret_rotation(rankings)

    return {
        "updated_at":    data.get("updated_at"),
        "full_rankings": rankings,
        "top3_hot":      top_detail,
        "bottom3_cold":  bottom_detail,
        "accelerating":  accel_list,
        "hottest_sector": {
            "etf":  hot.get("etf"),
            "name": hot.get("name"),
            "heat": hot.get("heat"),
            "accel": hot.get("accel"),
        },
        "rotation_note": note,
    }


def _interpret_rotation(rankings: list) -> str:
    """根据排名格局生成人话解读。"""
    if not rankings:
        return "数据不足，无法解读。"

    top = rankings[:3]
    top_names  = "、".join(r["name"] for r in top)
    accel_top  = [r for r in top if r["accel"]]
    hot_name   = rankings[0]["name"]
    hot_heat   = rankings[0]["heat"]

    # 全面熊市：最强板块超额收益也为负，禁止误判为"资金均匀"
    if all(r["heat"] < 0 for r in rankings):
        return (f"全部板块超额收益均为负值，市场处于全面承压状态。"
                f"最强板块【{hot_name}】超额{hot_heat:+.1f}%，仍跑输 SPY。"
                f"建议大幅减仓或空仓观望，等待至少一个板块转正后再介入。")

    # 判断是否有强主线
    if hot_heat > 5:
        if accel_top:
            msg = (f"当前资金集中在【{hot_name}】板块（超额{hot_heat:+.1f}%且加速），"
                   f"前三强为{top_names}。这是典型的主线驱动行情，优先沿主线做多，避开垫底板块。")
        else:
            msg = (f"【{hot_name}】领涨（超额{hot_heat:+.1f}%）但动量开始减弱（1M<3M），"
                   f"留意板块切换信号——看加速度板块（1M RS > 3M RS 的层）是否出现新主线。")
    elif abs(hot_heat) < 2:
        msg = "各板块超额收益差距极小，市场没有明显主线，适合缩减仓位等待方向明朗。"
    else:
        msg = f"前三强：{top_names}，资金较均匀，没有压倒性主线。以趋势最强且加速的板块为主攻方向。"

    return msg


# ─────────────────────────────────────────────────────────────
# 板块 → 代表性个股映射
# 原则：每板块选 6 只高流动性、高 Beta 的摆动交易优先股
#       防御型板块（XLP/XLRE/XLU）收益相对低，不作为主攻方向
# ─────────────────────────────────────────────────────────────
SECTOR_TICKERS = {
    "XLK":  ["NVDA", "AMD", "AVGO", "MRVL", "ORCL", "MSFT"],
    "XLF":  ["GS",   "JPM", "V",    "PYPL", "SQ",   "MS"],
    "XLE":  ["OXY",  "DVN", "EOG",  "MPC",  "XOM",  "VLO"],
    "XLV":  ["LLY",  "ISRG","DXCM", "UNH",  "HUM",  "REGN"],
    "XLI":  ["CAT",  "DE",  "GE",   "ROK",  "EMR",  "PWR"],
    "XLC":  ["META", "NFLX","GOOGL", "DIS",  "TTWO", "EA"],
    "XLY":  ["TSLA", "BKNG","HD",    "NKE",  "LULU", "UBER"],
    "XLP":  ["COST", "WMT", "PG",    "KO",   "PEP",  "MDLZ"],  # 防御，热时才看
    "XLB":  ["FCX",  "NEM", "CF",    "MP",   "ALB",  "LIN"],
    "XLRE": ["AMT",  "EQIX","PLD",   "WELL", "DLR",  "O"],    # 防御
    "XLU":  ["CEG",  "NEE", "VST",   "EXC",  "CCJ",  "DUK"],  # 核电/公用事业（NNE流动性不足，换 EXC）
    # 子行业 ETF
    "SMH":  ["NVDA", "AMD", "AVGO",  "AMAT", "LRCX", "KLAC"],  # AAOI/AXTI 流动性太低，换高流动性设备股
    "XBI":  ["MRNA", "REGN","BIIB",  "BMRN", "NTLA", "CRSP"],
    "ITA":  ["LMT",  "RTX", "NOC",   "RKLB", "KTOS", "BWXT"],
}

# 纯防御板块：即使热度排名高，也不主动推荐（除非是地缘/危机行情）
_DEFENSIVE_ETFS = {"XLP", "XLRE"}


def get_hot_tickers(top_n_sectors: int = 3,
                    per_sector: int = 5,
                    skip_defensive: bool = True) -> list[dict]:
    """
    返回当前最热板块的代表性个股列表（用于动态 watchlist）。

    逻辑：
      1. 获取板块排名（带缓存）
      2. 取前 top_n_sectors 名（默认3个）
      3. 可选跳过防御性板块（XLP/XLRE）
      4. 每个板块取 per_sector 只代表股（按 SECTOR_TICKERS 顺序）
      5. 去重，返回有序列表

    返回 [(ticker, sector_etf, sector_name, rank), ...]
    """
    data = fetch_sector_rankings()
    if data.get("error") or not data.get("rankings"):
        return []

    rankings = data["rankings"]
    result   = []
    seen     = set()
    selected_sectors = []

    for r in rankings:
        if len(selected_sectors) >= top_n_sectors:
            break
        etf = r["etf"]
        if skip_defensive and etf in _DEFENSIVE_ETFS:
            continue
        selected_sectors.append(r)

    for r in selected_sectors:
        etf   = r["etf"]
        tlist = SECTOR_TICKERS.get(etf, [])
        added = 0
        for tk in tlist:
            if tk not in seen and added < per_sector:
                result.append({
                    "ticker":       tk,
                    "sector_etf":   etf,
                    "sector_name":  r["name"],
                    "sector_rank":  r["rank"],
                    "sector_heat":  r["heat"],
                    "accel":        r["accel"],
                })
                seen.add(tk)
                added += 1

    return result


def build_dynamic_watchlist(core: list[str] | None = None,
                             max_total: int = 20,
                             top_n_sectors: int = 3,
                             per_sector: int = 5) -> dict:
    """
    合并核心持续关注股 + 当前热板块代表股，构建动态 watchlist。

    core: 用户固定自选股（优先级最高，排在前面）
    返回：{
        "tickers":  [完整列表，max 20只],
        "core":     [用户固定的],
        "sector_add": [板块轮动动态追加的],
        "sectors_used": [{etf, name, rank, heat, accel}],
    }
    """
    core = [t.upper() for t in (core or [])]
    hot  = get_hot_tickers(top_n_sectors=top_n_sectors, per_sector=per_sector)

    sector_add   = []
    sectors_used = {}
    for item in hot:
        tk  = item["ticker"]
        etf = item["sector_etf"]
        if tk not in core:
            sector_add.append(tk)
        if etf not in sectors_used:
            sectors_used[etf] = {
                "etf":  etf,
                "name": item["sector_name"],
                "rank": item["sector_rank"],
                "heat": item["sector_heat"],
                "accel": item["accel"],
            }

    combined = core + sector_add
    combined = list(dict.fromkeys(combined))  # 保序去重
    combined = combined[:max_total]

    return {
        "tickers":      combined,
        "core":         core,
        "sector_add":   sector_add[:max_total - len(core)],
        "sectors_used": list(sectors_used.values()),
        "total":        len(combined),
    }


def format_telegram_report() -> str:
    """生成适合 Telegram 发送的板块轮动报告。"""
    rpt = sector_rotation_report()
    if rpt.get("error"):
        return f"📊 板块轮动：获取失败（{rpt['error']}）"

    lines = ["📊 <b>板块轮动快报</b>\n"]
    lines.append(f"更新时间：{str(rpt.get('updated_at',''))[:16]}\n")

    top3_hot = rpt.get("top3_hot", [])
    if top3_hot:
        lines.append("🔥 <b>当前热门板块（前3）</b>")
        for r in top3_hot:
            accel = "↑加速" if r["accel"] else ""
            lines.append(f"  [{r['rank']}] {r['name']}（{r['etf']}）"
                         f"  1M:{r['rs_1m']:+.1f}%  3M:{r['rs_3m']:+.1f}%  {accel}")

    bottom3_cold = rpt.get("bottom3_cold", [])
    if bottom3_cold:
        lines.append("\n❄️ <b>资金流出板块（后3）</b>")
        for r in bottom3_cold:
            lines.append(f"  [{r['rank']}] {r['name']}（{r['etf']}）"
                         f"  1M:{r['rs_1m']:+.1f}%  3M:{r['rs_3m']:+.1f}%")

    if rpt.get("accelerating"):
        accel_names = "、".join(r["name"] for r in rpt["accelerating"])
        lines.append(f"\n⚡ <b>加速流入</b>（1M RS > 3M RS）：{accel_names}")

    # 14板块完整排名（紧凑格式）
    full = rpt.get("full_rankings", [])
    if full:
        lines.append(f"\n📋 <b>完整排名（{len(full)}板块）</b>  1M超额 vs SPY")
        for r in full:
            accel_mark = "↑" if r.get("accel") else " "
            lines.append(
                f"  [{r['rank']:>2}] {r['name']:<10}  {r['rs_1m']:+.1f}%{accel_mark}"
            )

    lines.append(f"\n💡 {rpt.get('rotation_note', '')}")
    return "\n".join(lines)
