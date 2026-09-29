"""
基于CBOE公开期权链的GEX计算（2026-09-29新增，供假突破前向样本记录使用）

与gex_scanner.calc_gex（yfinance期权链+自算Black-Scholes）相比：
  - 数据源：CBOE延迟报价接口一次返回整条期权链，每个合约自带IV/gamma/OI，
    不用自己反推IV（yfinance的IV偶尔是0.00001这类坏值，2026-09-29 ASTS实测遇到）
  - 伽马翻转点：在现价±20%的一组假设价位上重算全链净伽马，取正负交界处
    （业内常用的"gamma profile"做法）；calc_gex只看相邻两个行权价的符号变化，较粗

思路参考（重写实现，未复制代码）：
  - itsfabtrading/Gex-Multi（Apache-2.0）：call wall=净伽马最大行权价、put wall=最小
  - GMestreM/gex_data（MIT）：CBOE接口用法与gamma profile求翻转点

口径与gex_scanner一致（dealer-centric）：做市商视为多Call、空Put，
Call GEX为正、Put GEX为负，单位为"股价每变动1%对应的美元伽马敞口"。
"""
import re
from datetime import datetime

import numpy as np
import pytz
import requests

ET = pytz.timezone("America/New_York")
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-master personal research)"}

MAX_DTE         = 45      # 只看45天内到期的合约（近月伽马影响最大）
STRIKE_BAND     = 0.20    # 墙/分布只看现价±20%的行权价
PROFILE_BAND    = 0.20    # 翻转点扫描范围：现价±20%
PROFILE_STEPS   = 81
MIN_OI          = 1

_OCC = re.compile(r"^(?P<root>[A-Z.]+?)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ(symbol: str):
    """OCC期权代码 → (到期日, 'C'/'P', 行权价)。例：ASTS261002C00035000 → (2026-10-02, C, 35.0)"""
    m = _OCC.match(symbol)
    if not m:
        return None
    exp = datetime.strptime(m["yymmdd"], "%y%m%d").date()
    return exp, m["cp"], int(m["strike"]) / 1000


def fetch_cboe_chain(symbol: str, timeout: int = 20) -> dict:
    r = requests.get(CBOE_URL.format(symbol=symbol.upper()), headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]


def _bs_gamma(S, K, iv, T):
    """向量化Black-Scholes gamma（r=q=0，只需正态密度，不依赖scipy）。"""
    S = np.asarray(S, dtype=float)[:, None]
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sd ** 2) / sd
    return np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * sd)


def compute_gex(data: dict, asof=None) -> dict:
    """
    由CBOE返回的data计算GEX结构（纯函数，便于测试）。
    返回字段与gex_scanner.calc_gex兼容（gex_env/total_gex_m/gex_king/flip_strike/
    gex_by_strike/pc_ratio），另加call_wall/put_wall/iv30。
    """
    spot = float(data.get("current_price") or data.get("close") or 0)
    if spot <= 0:
        return {"error": "CBOE无现价"}
    asof = asof or datetime.now(ET).date()

    rows = []
    for o in data.get("options", []):
        p = parse_occ(o.get("option", ""))
        if not p:
            continue
        exp, cp, strike = p
        dte = (exp - asof).days
        oi = float(o.get("open_interest") or 0)
        if dte < 0 or dte > MAX_DTE or oi < MIN_OI:
            continue
        rows.append((strike, cp, oi, float(o.get("gamma") or 0), float(o.get("iv") or 0), max(dte, 1) / 365))
    if not rows:
        return {"error": f"{MAX_DTE}天内无有效OI"}

    strikes = np.array([r[0] for r in rows])
    sign = np.array([1.0 if r[1] == "C" else -1.0 for r in rows])
    oi = np.array([r[2] for r in rows])
    gamma = np.array([r[3] for r in rows])
    iv = np.array([r[4] for r in rows])
    T = np.array([r[5] for r in rows])

    # 分行权价GEX：用CBOE给的gamma（当前价位下）
    gex = sign * oi * gamma * 100 * spot ** 2 * 0.01
    by_strike = {}
    for k, g in zip(strikes, gex):
        if abs(k / spot - 1) <= STRIKE_BAND:
            by_strike[float(k)] = by_strike.get(float(k), 0.0) + float(g)
    if not by_strike:
        return {"error": "现价附近无行权价"}

    call_wall = max(by_strike, key=by_strike.get)
    put_wall = min(by_strike, key=by_strike.get)
    total = float(gex.sum())

    # 伽马翻转点：在一组假设价位上用BS重算全链净伽马，找符号变化处线性插值
    flip = None
    ok = iv > 0
    if ok.any():
        levels = np.linspace(spot * (1 - PROFILE_BAND), spot * (1 + PROFILE_BAND), PROFILE_STEPS)
        g = _bs_gamma(levels, strikes[ok], iv[ok], T[ok])
        profile = (g * (sign[ok] * oi[ok] * 100)).sum(axis=1) * levels ** 2 * 0.01
        cross = np.where(np.diff(np.sign(profile)) != 0)[0]
        if len(cross):
            # 多个交界时取离现价最近的一个
            i = cross[np.argmin(np.abs(levels[cross] - spot))]
            x0, x1, y0, y1 = levels[i], levels[i + 1], profile[i], profile[i + 1]
            flip = round(float(x0 - y0 * (x1 - x0) / (y1 - y0)), 2)

    call_oi = float(oi[sign > 0].sum())
    put_oi = float(oi[sign < 0].sum())
    return {
        "source": "cboe",
        "spot": round(spot, 2),
        "gex_env": "正伽马" if total > 0 else "负伽马",
        "total_gex_m": round(total / 1e6, 1),
        "gex_king": call_wall,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "flip_strike": flip,
        "pc_ratio": round(put_oi / call_oi, 2) if call_oi else None,
        "iv30": data.get("iv30"),
        "gex_by_strike": {k: round(v / 1e6, 2) for k, v in sorted(by_strike.items())},
    }


def calc_gex_cboe(symbol: str) -> dict:
    try:
        return compute_gex(fetch_cboe_chain(symbol))
    except Exception as e:
        return {"error": f"CBOE取数失败：{e}"}
