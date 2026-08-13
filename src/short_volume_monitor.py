"""
空头成交量参考信号（FINRA Daily Short Sale Volume）

背景：2026-08-13 讨论"资金流向"数据来源时发现，现有 sector_rotation.py/
market_breadth.py 都是用相对强弱（RS）价格代理"资金去向"，不是真实成交数据。
FINRA 每日发布官方空头成交量文件（做空标记成交量/总成交量），是监管层面的
真实交易数据，免费、历史可回溯（实测至少到2020年），比价格代理更接近"真实"，
但仍不是完整的"资金净流入/流出"（FINRA不发布，需付费数据源）。

⚠️ 关键局限：FINRA"空头成交量"≠"看空押注"。这个统计口径包含大量做市商/
期权对冲盘的合规性做空（例如做市商卖出Call后要做空正股对冲delta），流动性好、
期权活跃的股票空头占比常年就有50%+的基线，绝对值本身没有意义。必须跟该标的
自己的历史基线比较（z-score式偏离），不能套用固定阈值——这是此前
social_sentiment.py（StockTwits）"80%看多阈值对谁都成立"被否决的同一个教训，
这次提前避开。

集成方式：
  - scheduler.py 09:00晨报调用 run_short_volume_monitor()，增量刷新
    data/short_volume_history.json（每ticker滚动保留 HISTORY_DAYS 个交易日）
  - cold_model.py 新增 short_volume gate，只读快照，pass 恒为 True——
    2026-08-13 用户明确决定：先展示，不参与打分/否决，等 CONDITIONAL
    仓位修复（PR#3）的模拟盘样本验证完再决定是否正式接成gate
  - Telegram /shortvol TICKER 按需查询

数据源本身是T+1（FINRA当晚发布当天数据，早盘读到的是上一交易日收盘后的统计），
不是实时数据，展示文案需明确标注这一点，不能让用户误以为是盘中实时资金流。
"""
from __future__ import annotations

import os
import json
import statistics
from datetime import datetime, timedelta, date

import requests
import pytz

ET_TZ = pytz.timezone("America/New_York")

_DATA         = os.path.join(os.path.dirname(__file__), "..", "data")
_HISTORY_FILE = os.path.join(_DATA, "short_volume_history.json")

HISTORY_DAYS      = 60   # 每ticker滚动保留的交易日基线窗口
MIN_BASELINE_DAYS = 10   # 少于这个天数不判断是否异常，只展示原始数据
BACKFILL_CALENDAR_TRIES = 90   # 首次回填时最多往回尝试的自然日数（覆盖60个交易日+节假日余量）
REFRESH_CALENDAR_TRIES  = 5    # 日常增量刷新时最多往回找几天（跨长周末场景）

_FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"


def _fetch_finra_file(d: date) -> dict[str, tuple[float, float]] | None:
    """
    下载单日FINRA全市场空头成交量文件，解析成 {ticker: (short_volume, total_volume)}。
    非交易日（周末/假日）文件不存在，返回 None（正常情况，非错误，调用方不应告警）。
    """
    url = _FINRA_URL.format(date=d.strftime("%Y%m%d"))
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        result: dict[str, tuple[float, float]] = {}
        for line in r.text.splitlines()[1:]:   # 跳过表头 Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
            parts = line.split("|")
            if len(parts) < 5:
                continue
            symbol, short_vol, total_vol = parts[1], parts[2], parts[4]
            try:
                result[symbol] = (float(short_vol), float(total_vol))
            except ValueError:
                continue
        return result
    except Exception:
        return None


def _load_history() -> dict:
    if not os.path.exists(_HISTORY_FILE):
        return {}
    try:
        with open(_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_history(history: dict):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _HISTORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _HISTORY_FILE)


def _fill_history_backward(history: dict, watchlist_set: set[str],
                            target_days: int, max_calendar_tries: int) -> tuple[dict, int]:
    """
    从昨天开始往回逐个自然日尝试拉FINRA文件，直到凑够 target_days 个交易日
    或达到 max_calendar_tries 次尝试上限。一天一份全市场文件，一次下载
    覆盖所有watchlist股票，不是逐ticker请求。返回 (更新后的history, 实际填了几天)。
    """
    today = datetime.now(ET_TZ).date()
    d = today - timedelta(days=1)
    filled = 0
    tried = 0
    while filled < target_days and tried < max_calendar_tries:
        tried += 1
        data = _fetch_finra_file(d)
        if data is not None:
            date_str = d.strftime("%Y-%m-%d")
            for ticker in watchlist_set:
                row = data.get(ticker)
                if row and row[1] > 0:
                    ratio = round(row[0] / row[1] * 100, 2)
                    history.setdefault(ticker, {})[date_str] = ratio
            filled += 1
        d -= timedelta(days=1)

    for ticker in list(history.keys()):
        trimmed = dict(sorted(history[ticker].items())[-HISTORY_DAYS:])
        history[ticker] = trimmed

    return history, filled


def backfill_history(watchlist: list[str], days: int = HISTORY_DAYS) -> dict:
    """
    一次性回填 watchlist 中每只股票近 `days` 个交易日的空头成交比例，用于
    首次建立基线。FINRA数据可回溯多年，不需要"日拱一卒"逐天积累历史。
    """
    watchlist_set = {t.upper() for t in watchlist}
    history = _load_history()
    history, filled = _fill_history_backward(
        history, watchlist_set, target_days=days, max_calendar_tries=BACKFILL_CALENDAR_TRIES)
    _save_history(history)
    return {"ok": True, "days_filled": filled, "tickers": sorted(watchlist_set)}


def refresh_latest(watchlist: list[str]) -> dict:
    """
    每日增量刷新：拉最近一份可用的FINRA文件（通常是昨天的），追加进历史。
    供 scheduler.py 09:00晨报调用。
    """
    watchlist_set = {t.upper() for t in watchlist}
    history = _load_history()
    history, filled = _fill_history_backward(
        history, watchlist_set, target_days=1, max_calendar_tries=REFRESH_CALENDAR_TRIES)
    if filled == 0:
        return {"ok": False, "note": f"近{REFRESH_CALENDAR_TRIES}天内未取到可用FINRA文件"}
    _save_history(history)
    return {"ok": True, "tickers": sorted(watchlist_set)}


def _calc_signal(ratios: list[float]) -> dict:
    """
    纯函数：给定某ticker按日期排序的历史空头占比序列（含最新一天），
    算最新一天相对此前基线的偏离程度。独立抽出，便于pytest直接测试。
    """
    if len(ratios) < 2:
        return {"flag": "insufficient", "note": "数据不足，暂无法判断"}

    latest = ratios[-1]
    baseline = ratios[:-1]

    if len(baseline) < MIN_BASELINE_DAYS:
        mean_b = statistics.mean(baseline)
        return {
            "flag": "insufficient",
            "ratio": round(latest, 2),
            "baseline_mean": round(mean_b, 2),
            "baseline_days": len(baseline),
            "note": (f"空头成交占比{latest:.1f}%（近{len(baseline)}天均值{mean_b:.1f}%，"
                     f"样本不足{MIN_BASELINE_DAYS}天，暂不判断是否异常）"),
        }

    mean_b = statistics.mean(baseline)
    std_b  = statistics.pstdev(baseline)
    if std_b < 1e-6:
        # 历史基线从未变化过：任何偏离都算异常，不能按“除以0”强行判正常
        z = 0.0 if abs(latest - mean_b) < 1e-6 else (999.0 if latest > mean_b else -999.0)
    else:
        z = (latest - mean_b) / std_b

    if abs(z) >= 2:
        flag, degree = ("extreme_high", "明显偏高") if z > 0 else ("extreme_low", "明显偏低")
    elif abs(z) >= 1:
        flag, degree = ("elevated_high", "略偏高") if z > 0 else ("elevated_low", "略偏低")
    else:
        flag, degree = "normal", "正常范围"

    note = (f"空头成交占比{latest:.1f}%，较自身{len(baseline)}日均值"
            f"{mean_b:.1f}%（标准差{std_b:.1f}%）{degree}（偏离{z:+.1f}个标准差）")
    if flag != "normal":
        note += "——该比例含做市商合规对冲盘，不等于看空押注，仅供参考"

    return {
        "flag": flag,
        "ratio": round(latest, 2),
        "baseline_mean": round(mean_b, 2),
        "baseline_std": round(std_b, 2),
        "baseline_days": len(baseline),
        "z_score": round(z, 2),
        "note": note,
    }


def check_ticker_short_volume(ticker: str) -> dict:
    """
    供 cold_model.py 高频调用，只读快照不发HTTP请求。
    始终 pass=True——2026-08-13用户决定先展示不参与打分/否决，等
    CONDITIONAL仓位修复的模拟盘样本验证完再决定是否正式接成gate。
    """
    history = _load_history()
    entries = history.get(ticker.upper())
    if not entries:
        return {"pass": True, "note": "空头成交量数据尚未建立基线（T+1官方数据，明天起会有）"}

    ratios = [v for _, v in sorted(entries.items())]
    signal = _calc_signal(ratios)
    extra = {k: v for k, v in signal.items() if k != "note"}
    return {"pass": True, "note": signal["note"], **extra}


def format_short_volume_telegram(ticker: str) -> str:
    """/shortvol TICKER 按需查询的Telegram格式化输出。"""
    result = check_ticker_short_volume(ticker.upper())
    icon = {
        "insufficient":  "⚪",
        "normal":        "🟢",
        "elevated_high": "🟡",
        "elevated_low":  "🟡",
        "extreme_high":  "🔴",
        "extreme_low":   "🔴",
    }.get(result.get("flag"), "⚪")

    lines = [
        f"{icon} <b>{ticker.upper()} 空头成交量</b>（FINRA官方数据，T+1，非实时）",
        result["note"],
    ]
    if result.get("flag") not in (None, "insufficient"):
        lines.append(
            "<i>（大白话：这个比例统计的是带做空标记的成交占比，"
            "很大一部分来自做市商对冲盘的合规操作，不是直接的\"看空押注\"，"
            "只有明显偏离这只股票自己的历史习惯时才值得多留意一眼）</i>"
        )
    return "\n".join(lines)


def run_short_volume_monitor(watchlist: list[str] | None = None) -> dict:
    """
    scheduler.py 09:00晨报调用：给基线不足的ticker先自动回填，再做当日增量刷新。
    """
    wl = watchlist or []
    if not wl:
        return {"ok": True, "note": "watchlist为空，跳过"}

    watchlist_set = {t.upper() for t in wl}
    history = _load_history()
    need_backfill = [t for t in watchlist_set if len(history.get(t, {})) < MIN_BASELINE_DAYS]
    if need_backfill:
        backfill_history(need_backfill)

    return refresh_latest(list(watchlist_set))
