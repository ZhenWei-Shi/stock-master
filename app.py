from flask import Flask, render_template, jsonify, request
import traceback
import yfinance as yf
import json
import numpy as np

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from src.fetcher import get_stock_history, get_stock_info, get_options_chain, get_market_snapshot
from src.market import detect_market_state
from src.options import calculate_max_pain, analyze_gamma_squeeze, calculate_dealer_delta
from src.screener import calculate_rs_rating, nine_gates_check
from src.supply_chain import get_chain_names, get_chain, detect_capital_flow
from src.technical import get_indicators, get_extended_indicators
from src.short_term import analyze as short_term_analyze
from src.scraper import get_retail_density, get_institutional_analysis, get_earnings_analysis
from src.options_sources import aggregate_options
from src.options_analytics import get_iv_analytics, get_vix_term_structure
from src.fundamentals import (get_short_interest, get_analyst_ratings, get_news_sentiment,
                               get_insider_trades, get_earnings_reaction,
                               get_fear_greed, get_sector_rotation)
from src.intraday import get_intraday_data, get_premarket_data, run_quant_model
from src.backtest_engine import run_backtest
from src.institutional_13f import get_institutional_13f
from src.quant_strategy import run_all_strategies, strategy_scalping
from src.cold_model import cold_decision, scan_tickers
from src.pattern import analyze_patterns
from src.fundamentals import get_earnings_quality
from src.pdt_guard import check_pdt_risk, swing_trade_plan
from src.small_account import assess_small_account, check_stock_suitability, position_size_small
from src.debate import generate_trade_debate
from src.paper_trading import (init_account, open_position, close_position,
                                mark_to_market, performance_report,
                                list_positions, compare_paper_vs_real,
                                reset_circuit_breaker)
from src.trading_agent import run_scan, run_monitor, daily_report
from src.market import detect_follow_through_day
from src.backtest import run_backtest as run_historical_backtest, print_report as bt_print_report
from src.smart_money import (full_smart_money_scan, detect_unusual_options,
                              calculate_gex, detect_short_squeeze,
                              smart_money_flow, institutional_momentum)
from src.macro_filter import full_macro_report, get_economic_calendar, get_etf_signals
from src.earnings_analyzer import (full_earnings_analysis, get_market_breadth,
                                    check_position_correlation, analyze_eps_acceleration,
                                    analyze_pead, analyze_quality_factors)
from src.paper_trading import update_trailing_stop

app = Flask(__name__)
# Flask 2.3+ 弃用 json_encoder，改用 json.provider.DefaultJSONProvider
try:
    from flask.json.provider import DefaultJSONProvider
    class _FlaskProvider(DefaultJSONProvider):
        def default(self, o):
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
            if isinstance(o, (np.bool_,)):    return bool(o)
            if isinstance(o, np.ndarray):     return o.tolist()
            return super().default(o)
    app.json_provider_class = _FlaskProvider
    app.json = _FlaskProvider(app)
except ImportError:
    app.json_encoder = _NpEncoder  # Flask < 2.3 fallback


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/market")
def api_market():
    try:
        snap = get_market_snapshot()
        state = detect_market_state(snap["spy_hist"], snap["vix"])
        state["qqq_price"] = snap["qqq_price"]
        state["spy_price"] = snap.get("spy_price") or snap.get("current_price")
        return jsonify({"ok": True, "data": state})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/analyze")
def api_analyze():
    ticker = request.args.get("ticker", "").upper().strip()
    expiry = request.args.get("expiry")

    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})

    try:
        # 大盘状态
        snap = get_market_snapshot()
        market = detect_market_state(snap["spy_hist"], snap["vix"])

        # 股票基本信息 + 历史价格
        info = get_stock_info(ticker)
        hist = get_stock_history(ticker, "1y")
        if hist.empty:
            return jsonify({"ok": False, "error": f"找不到 {ticker}，请检查代码"})

        current_price = float(hist["Close"].iloc[-1])
        info["current_price"] = current_price

        # K线图（最近90天，精简传输）
        price_chart = [
            {"d": idx.strftime("%m/%d"), "c": round(float(row["Close"]), 2),
             "v": int(row["Volume"])}
            for idx, row in hist.tail(90).iterrows()
        ]

        # RS Rating
        rs = calculate_rs_rating(ticker)

        # 期权
        calls, puts, used_expiry, expirations = get_options_chain(ticker, expiry)
        max_pain = {}
        gamma = {"signal": False, "score": 0, "max_score": 10, "signals": [], "oi_chart": []}

        if calls is not None and puts is not None:
            max_pain = calculate_max_pain(calls, puts, current_price)
            gamma = analyze_gamma_squeeze(calls, puts, current_price)

        # 九关筛选
        gates = nine_gates_check(info, rs, market)
        _auto_gate9(gates, gamma)

        # ATR(14)：用 shift(1) 对齐前收盘价，避免 np.roll 首行脏值
        _c = hist["Close"]
        _hl = (hist["High"] - hist["Low"]).values[1:]
        _hc = (hist["High"] - _c.shift(1)).abs().values[1:]
        _lc = (hist["Low"]  - _c.shift(1)).abs().values[1:]
        _tr = np.maximum(_hl, np.maximum(_hc, _lc))
        _atr = float(np.mean(_tr[-14:]))

        # 综合建议
        recommendation = _recommend(market, rs, gamma, gates, current_price, _atr)

        return jsonify({
            "ok": True,
            "ticker": ticker,
            "info": info,
            "market": market,
            "rs": rs,
            "options": {
                "expiry": used_expiry,
                "expirations": expirations[:12],
                "max_pain": max_pain,
                "gamma": gamma,
            },
            "gates": gates,
            "price_chart": price_chart,
            "recommendation": recommendation,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


def _recommend(market, rs, gamma, gates, price, atr=None):
    state = market.get("state", "E")
    rs_val = rs.get("rs_rating") or 0
    gamma_signal = gamma.get("signal", False)
    auto_fails = sum(1 for g in gates if g["status"] == "fail")

    bull = 0
    if state == "A": bull += 2
    if state == "D": bull += 4
    if rs_val >= 85: bull += 2
    if gamma_signal: bull += 2

    bear = 0
    if state == "C": bear += 3
    if rs_val < 50: bear += 1
    if auto_fails > 0: bear += auto_fails * 2

    # ATR 止损/目标（ATR×1.5 止损，ATR×3.0 目标）；无 ATR 时退回固定百分比
    stop_note = ""
    if atr and atr > 0:
        stop  = round(price - atr * 1.5, 2)
        tgt1  = round(price + atr * 3.0, 2)
        tgt2  = round(price + atr * 4.5, 2)
        stop_note = f"（ATR×1.5=${atr*1.5:.2f}）"
    else:
        stop  = round(price * 0.90, 2)
        tgt1  = round(price * 1.25, 2)
        tgt2  = round(price * 1.40, 2)

    if state == "E":
        return _rec("空仓观望", "gray",
            "方向不明，既然涨跌都有可能且无法判断，不如空仓。",
            None, None, "0%", "无")

    if bear > bull or auto_fails >= 2:
        return _rec("不建议入场", "red",
            f"{auto_fails}个关卡未通过或市场偏空，等待信号改善。",
            None, None, "0%", "Bear Put Spread（如强烈看空）")

    if bull >= 6:
        return _rec("积极做多", "green",
            f"大盘{state} + RS={rs_val} + {'Gamma信号确认' if gamma_signal else '期权结构偏多'}，综合信号强。",
            stop, tgt1,
            "正股 20-30%（凡人修仙中等仓位）",
            "买入ATM Call" if gamma_signal else "Bull Call Spread",
            stop_note)

    return _rec("小仓试探", "yellow",
        "信号部分确认，轻仓试探，等待财报或技术突破进一步验证。",
        stop, tgt1,
        "正股 5-10%（轻仓）",
        "3% 仓位 Bull Call Spread，控制风险",
        stop_note)


def _rec(action, color, reason, stop, target, size, options, stop_note=""):
    return {
        "action": action, "color": color, "reason": reason,
        "stop_loss": stop, "stop_note": stop_note,
        "target": target,
        "position_size": size, "options_strategy": options,
    }


@app.route("/api/nine-gates")
def api_nine_gates():
    """九关筛选深度分析：关卡4/5/8 使用真实爬虫数据"""
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        snap   = get_market_snapshot()
        market = detect_market_state(snap["spy_hist"], snap["vix"])
        info   = get_stock_info(ticker)
        rs     = calculate_rs_rating(ticker)

        # 并行抓取三个关卡数据（顺序执行，各自有超时保护）
        retail    = get_retail_density(ticker)          # 关卡4
        inst      = get_institutional_analysis(ticker)  # 关卡5
        earnings  = get_earnings_analysis(ticker)       # 关卡8

        gates = _build_enhanced_gates(info, rs, market, retail, inst, earnings)

        return jsonify({
            "ok": True, "ticker": ticker,
            "gates": gates,
            "retail": retail,
            "institutional": inst,
            "earnings": earnings,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/options-full")
def api_options_full():
    """多源期权数据聚合 + 异常活动检测"""
    ticker = request.args.get("ticker", "").upper().strip()
    expiry = request.args.get("expiry")
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        hist = get_stock_history(ticker, "5d")
        if hist.empty:
            return jsonify({"ok": False, "error": f"找不到 {ticker}"})
        current_price = float(hist["Close"].iloc[-1])

        calls, puts, used_expiry, expirations = get_options_chain(ticker, expiry)
        agg = aggregate_options(ticker, calls, puts, current_price, expiry)

        # Max Pain + Gamma（使用最佳数据源）
        best_calls = agg.get("best_calls")
        best_puts  = agg.get("best_puts")
        from src.options import calculate_max_pain, analyze_gamma_squeeze
        max_pain = calculate_max_pain(best_calls, best_puts, current_price) if best_calls is not None else {}
        gamma    = analyze_gamma_squeeze(best_calls, best_puts, current_price) if best_calls is not None else {}

        # 转换 sources 为可序列化格式
        sources_info = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, type(None))}
                        for k, v in agg["sources"].items()}

        return jsonify({
            "ok": True,
            "ticker": ticker,
            "current_price": current_price,
            "expiry": used_expiry,
            "expirations": expirations[:12],
            "sources": sources_info,
            "max_pain": max_pain,
            "gamma": gamma,
            "unusual_activity": agg["unusual_activity"],
            "cboe": agg["cboe"],
            "oi_chart": gamma.get("oi_chart", []),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


def _auto_gate9(gates, gamma):
    """关卡9：期权OI方向，用 Gamma Squeeze 评分自动判断"""
    score = gamma.get("score", 0)
    pc    = gamma.get("pc_ratio", 1.0) or 1.0
    for g in gates:
        if g["gate"] != 9:
            continue
        if score >= 6:
            g["status"] = "pass"
            g["detail"] = f"Gamma评分{score}/10，期权OI强力支撑做多方向"
        elif score >= 3:
            g["status"] = "warn"
            g["detail"] = f"Gamma评分{score}/10，OI结构中性，方向待确认"
        elif pc > 1.3:
            g["status"] = "fail"
            g["detail"] = f"Gamma评分{score}/10，P/C比{pc:.2f}，Put堆积，做多需谨慎"
        else:
            g["status"] = "warn"
            g["detail"] = f"Gamma评分{score}/10，期权数据量不足或结构平淡"


def _build_enhanced_gates(info, rs_result, market_state, retail, inst, earnings):
    """九关筛选：4/5/8 使用真实数据，其余保持不变"""
    gates = nine_gates_check(info, rs_result, market_state)
    for g in gates:
        if g["gate"] == 4:
            g["status"] = retail.get("gate_status", "manual")
            g["detail"] = retail.get("gate_detail", "数据获取中")
            score = retail.get("avg_score")
            if score is not None:
                g["detail"] += f"（热度综合分 {score}/100）"
        elif g["gate"] == 5:
            g["status"] = inst.get("gate_status", "manual")
            g["detail"] = inst.get("gate_detail", "数据获取中")
        elif g["gate"] == 8:
            g["status"] = earnings.get("gate_status", "manual")
            g["detail"] = earnings.get("gate_detail", "数据获取中")
    return gates


@app.route("/api/watchlist")
def api_watchlist():
    """
    自选股批量快速分析
    参数：tickers=NVDA,AAPL,TSLA（逗号分隔）
    每只股票返回：价格、涨跌幅、RS、短线方向、下次财报、九关快速通过数
    """
    raw = request.args.get("tickers", "").upper().strip()
    if not raw:
        return jsonify({"ok": False, "error": "请提供 tickers 参数"})

    tickers = [t.strip() for t in raw.split(",") if t.strip()][:20]  # 最多20只

    try:
        snap   = get_market_snapshot()
        market = detect_market_state(snap["spy_hist"], snap["vix"])
        vix    = float(snap["vix"])
    except Exception:
        market = {"state": "E", "description": "无法获取大盘数据"}
        vix    = 0.0

    results = []
    for ticker in tickers:
        item = {"ticker": ticker}
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty:
                item["error"] = "无数据"
                results.append(item)
                continue

            price   = float(hist["Close"].iloc[-1])
            prev    = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            chg_pct = round((price - prev) / prev * 100, 2) if prev else 0

            # RS Rating（快速版，用3月数据）
            try:
                rs_data = calculate_rs_rating(ticker)
                rs_val  = rs_data.get("rs_rating")
            except Exception:
                rs_val = None

            # 短线方向（快速，用60天数据）
            direction = "NEUTRAL"
            try:
                hist60 = t.history(period="3mo")
                if not hist60.empty:
                    from src.technical import get_indicators
                    from src.short_term import analyze as st_analyze
                    ind = get_indicators(hist60)
                    st  = st_analyze(ind, market, ticker)
                    direction = st.get("direction", "NEUTRAL")
            except Exception:
                pass

            # 下次财报日
            next_earnings = None
            try:
                cal = t.calendar
                if cal is not None:
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if ed:
                            next_earnings = str(ed[0])[:10] if hasattr(ed, '__iter__') and not isinstance(ed, str) else str(ed)[:10]
                    elif hasattr(cal, 'columns') and "Earnings Date" in cal.columns:
                        next_earnings = str(cal["Earnings Date"].iloc[0])[:10]
            except Exception:
                pass

            # 九关快速通过计数（排除手动关卡 4/5/6/7，计所有自动评估关卡）
            try:
                info  = get_stock_info(ticker)
                gates = nine_gates_check(info, rs_data if rs_val else {}, market)
                auto_gates = [g for g in gates if g["status"] != "manual"]
                pass_count = sum(1 for g in auto_gates if g["status"] == "pass")
                fail_count = sum(1 for g in auto_gates if g["status"] in ("fail", "warn"))
            except Exception:
                pass_count = fail_count = None

            item.update({
                "price":        round(price, 2),
                "chg_pct":      chg_pct,
                "rs":           rs_val,
                "direction":    direction,
                "next_earnings": next_earnings,
                "pass_count":   pass_count,
                "fail_count":   fail_count,
                "vix":          round(vix, 1),
            })
        except Exception as e:
            item["error"] = str(e)

        results.append(item)

    return jsonify({"ok": True, "market": market, "data": results})


@app.route("/api/earnings-calendar")
def api_earnings_calendar():
    """
    财报日历：给定多只股票，按财报日期排序返回
    """
    raw = request.args.get("tickers", "").upper().strip()
    if not raw:
        return jsonify({"ok": False, "error": "请提供 tickers 参数"})

    tickers = [t.strip() for t in raw.split(",") if t.strip()][:20]
    events  = []

    failed = []
    for ticker in tickers:
        try:
            t   = yf.Ticker(ticker)
            cal = t.calendar

            dates = []
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    dates = list(ed) if hasattr(ed, '__iter__') and not isinstance(ed, str) else [ed]
            elif cal is not None and hasattr(cal, 'columns') and "Earnings Date" in cal.columns:
                dates = cal["Earnings Date"].tolist()

            if not dates:
                failed.append(ticker)
            for d in dates[:2]:
                date_str = str(d)[:10]
                events.append({"ticker": ticker, "date": date_str})
        except Exception:
            failed.append(ticker)

    events.sort(key=lambda x: x["date"])
    result = {"ok": True, "events": events}
    if failed:
        result["warning"] = f"以下 {len(failed)} 只股票未获取到财报日期：{', '.join(failed)}"
    return jsonify(result)


@app.route("/api/iv-analytics")
def api_iv_analytics():
    """IV 溢价、预期移动幅度、期权斜率"""
    ticker = request.args.get("ticker", "").upper().strip()
    expiry = request.args.get("expiry")
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        hist = get_stock_history(ticker, "1y")
        if hist.empty:
            return jsonify({"ok": False, "error": f"找不到 {ticker}"})
        price  = float(hist["Close"].iloc[-1])
        calls, puts, used_expiry, _ = get_options_chain(ticker, expiry)
        data   = get_iv_analytics(calls, puts, price, used_expiry, hist)
        return jsonify({"ok": True, "ticker": ticker, "price": price,
                        "expiry": used_expiry, **data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/vix-term")
def api_vix_term():
    """VIX 期限结构"""
    try:
        return jsonify({"ok": True, **get_vix_term_structure()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/fundamentals")
def api_fundamentals():
    """基本面雷达：空头兴趣 + 分析师 + 新闻 + 内部人 + 财报反应"""
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        info   = get_stock_info(ticker)
        short  = get_short_interest(info)
        analyst = get_analyst_ratings(ticker)
        news   = get_news_sentiment(ticker)
        insider = get_insider_trades(ticker)
        react  = get_earnings_reaction(ticker)
        return jsonify({
            "ok": True, "ticker": ticker,
            "short_interest":    short,
            "analyst":           analyst,
            "news":              news,
            "insider":           insider,
            "earnings_reaction": react,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/market-sentiment")
def api_market_sentiment():
    """宏观情绪：CNN恐贪指数 + VIX期限结构 + 板块轮动"""
    try:
        fg      = get_fear_greed()
        vix     = get_vix_term_structure()
        sectors = get_sector_rotation()
        return jsonify({"ok": True, "fear_greed": fg, "vix_term": vix, "sectors": sectors})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/top-oi")
def api_top_oi():
    """
    跨所有到期日聚合 OI，取 Top N 排行（复刻 Barchart Highest OI 图）
    参数：ticker, top=25, expiry=all|具体日期
    """
    ticker = request.args.get("ticker", "").upper().strip()
    top_n  = min(int(request.args.get("top", 25)), 100)
    expiry_filter = request.args.get("expiry", "all")

    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})

    try:
        t = yf.Ticker(ticker)
        all_exps = t.options
        if not all_exps:
            return jsonify({"ok": False, "error": f"{ticker} 无期权数据"})

        # 决定要抓哪些到期日
        if expiry_filter == "all":
            exps_to_fetch = list(all_exps)
        else:
            exps_to_fetch = [expiry_filter] if expiry_filter in all_exps else [all_exps[0]]

        rows = []
        for exp in exps_to_fetch:
            try:
                chain = t.option_chain(exp)
                for side, df in [("C", chain.calls), ("P", chain.puts)]:
                    if df is None or df.empty:
                        continue
                    for _, r in df.iterrows():
                        oi = int(r.get("openInterest") or 0)
                        if oi == 0:
                            continue
                        strike = float(r.get("strike") or 0)
                        iv     = float(r.get("impliedVolatility") or 0)
                        vol    = int(r.get("volume") or 0)
                        label  = f"{ticker} {_fmt_exp(exp)} {strike:.1f}{side}"
                        rows.append({
                            "label":  label,
                            "ticker": ticker,
                            "expiry": exp,
                            "strike": strike,
                            "side":   side,
                            "oi":     oi,
                            "volume": vol,
                            "iv":     round(iv * 100, 1),
                            "itm":    bool(r.get("inTheMoney", False)),
                        })
            except Exception:
                continue

        if not rows:
            return jsonify({"ok": False, "error": "无有效 OI 数据"})

        rows.sort(key=lambda x: x["oi"], reverse=True)
        top = rows[:top_n]

        # 汇总统计
        total_call_oi = sum(r["oi"] for r in rows if r["side"] == "C")
        total_put_oi  = sum(r["oi"] for r in rows if r["side"] == "P")
        pc_ratio      = round(total_put_oi / total_call_oi, 3) if total_call_oi else None

        return jsonify({
            "ok": True,
            "ticker": ticker,
            "expirations": list(all_exps),
            "expiry_filter": expiry_filter,
            "top": top,
            "stats": {
                "total_call_oi": total_call_oi,
                "total_put_oi":  total_put_oi,
                "pc_ratio":      pc_ratio,
                "total_contracts": len(rows),
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


def _fmt_exp(exp_str):
    """2026-06-18 → 06/18/26"""
    try:
        parts = exp_str.split("-")
        return f"{parts[1]}/{parts[2]}/{parts[0][2:]}"
    except Exception:
        return exp_str


@app.route("/api/short-term")
def api_short_term():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        snap = get_market_snapshot()
        market = detect_market_state(snap["spy_hist"], snap["vix"])

        hist = get_stock_history(ticker, "1y")
        if hist.empty:
            return jsonify({"ok": False, "error": f"找不到 {ticker}"})

        indicators = get_indicators(hist)
        result = short_term_analyze(indicators, market, ticker)

        return jsonify({"ok": True, "ticker": ticker, "market": market,
                        "indicators": indicators, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/chains")
def api_chains():
    return jsonify({"ok": True, "data": get_chain_names()})


@app.route("/api/chain/<chain_id>")
def api_chain(chain_id):
    chain = get_chain(chain_id)
    if not chain:
        return jsonify({"ok": False, "error": "未知供应链"})

    try:
        # 为每个 ticker 获取当前价格和 RS Rating
        result_layers = []
        for layer in chain["layers"]:
            tickers_data = []
            for ticker in layer["tickers"]:
                try:
                    hist = yf.Ticker(ticker).history(period="1y")
                    if hist.empty:
                        continue
                    price = round(float(hist["Close"].iloc[-1]), 2)

                    # 简化RS：仅用近3月 vs SPY
                    spy = yf.Ticker("SPY").history(period="3mo")
                    stock3m = hist["Close"].tail(63)
                    ret_stock = float((stock3m.iloc[-1] - stock3m.iloc[0]) / stock3m.iloc[0] * 100) if len(stock3m) >= 2 else 0
                    spy3m = spy["Close"].tail(63)
                    ret_spy = float((spy3m.iloc[-1] - spy3m.iloc[0]) / spy3m.iloc[0] * 100) if len(spy3m) >= 2 else 0
                    outperform = ret_stock - ret_spy

                    tickers_data.append({
                        "ticker": ticker,
                        "price": price,
                        "ret_3m": round(ret_stock, 1),
                        "ret_spy": round(ret_spy, 1),
                        "outperform": round(outperform, 1),
                        "momentum": "hot" if outperform > 10 else ("warm" if outperform > 0 else "cold"),
                    })
                except Exception:
                    tickers_data.append({"ticker": ticker, "price": None, "momentum": "unknown"})

            result_layers.append({**layer, "tickers_data": tickers_data})

        # 检测机构资金层位：3个月超额收益最高的层
        layer_scores = []
        for l in result_layers:
            scores = [t.get("outperform", 0) for t in l["tickers_data"] if t.get("outperform") is not None]
            avg = sum(scores) / len(scores) if scores else -999
            layer_scores.append((l["layer"], avg))

        hot_layer = max(layer_scores, key=lambda x: x[1])[0] if layer_scores else None
        # 下一棒：热层的上一层（层号-1）
        next_layer = hot_layer - 1 if hot_layer and hot_layer > 1 else None

        return jsonify({
            "ok": True,
            "chain_id": chain_id,
            "name": chain["name"],
            "desc": chain["desc"],
            "layers": result_layers,
            "hot_layer": hot_layer,
            "next_layer": next_layer,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/intraday")
def api_intraday():
    """日内K线 + VWAP + 成交量速率"""
    ticker   = request.args.get("ticker", "").upper().strip()
    interval = request.args.get("interval", "5m")
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        data = get_intraday_data(ticker, interval)
        if "error" in data:
            return jsonify({"ok": False, "error": data["error"]})
        pre = get_premarket_data(ticker)
        return jsonify({"ok": True, **data, "premarket": pre})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/quant-signal")
def api_quant_signal():
    """冷静量化模型：五维评分 + 过滤器 + 交易参数"""
    ticker    = request.args.get("ticker", "").upper().strip()
    portfolio = float(request.args.get("portfolio", 100_000))
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = run_quant_model(ticker, portfolio_size=portfolio)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/backtest")
def api_backtest():
    """历史回测：VWAP动量策略（MA20 + RSI + ATR止损）"""
    ticker = request.args.get("ticker", "").upper().strip()
    period = request.args.get("period", "1y")
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = run_backtest(ticker, period)
        if "error" in result:
            return jsonify({"ok": False, "error": result["error"]})
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/institutional-13f")
def api_institutional_13f():
    """
    机构持仓13F深度分析
    引用：edgartools ⭐3,000+ (dgunning/edgartools) — SEC EDGAR 季度增减持追踪
    """
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        data = get_institutional_13f(ticker)
        return jsonify({"ok": True, **data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/quant-strategies")
def api_quant_strategies():
    """
    量化短线策略综合分析 — 5大策略 + 仓位建议
    引用：pandas-ta ⭐5,299 / quantstats ⭐6,500 / vectorbt ⭐7,900
    """
    ticker    = request.args.get("ticker", "").upper().strip()
    portfolio = float(request.args.get("portfolio", 100_000))
    peers_raw = request.args.get("peers", "")
    peers     = [p.strip().upper() for p in peers_raw.split(",") if p.strip()]

    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = run_all_strategies(ticker, portfolio=portfolio,
                                    compare_tickers=peers if peers else None)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/extended-indicators")
def api_extended_indicators():
    """
    扩展技术指标：Stochastic, OBV, Williams%R, Ichimoku, CCI
    引用：pandas-ta ⭐5,299
    """
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        hist = get_stock_history(ticker, "6mo")
        if hist.empty:
            return jsonify({"ok": False, "error": f"找不到 {ticker}"})
        data = get_extended_indicators(hist)
        return jsonify({"ok": True, "ticker": ticker, **data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/patterns")
def api_patterns():
    """VCP + 杯柄 + 平台整理形态识别（Minervini + O'Neil）"""
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = analyze_patterns(ticker)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/earnings-quality")
def api_earnings_quality():
    """财报质量深度评分（EPS超预期幅度/营收加速/利润率/指引/连续超预期）"""
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = get_earnings_quality(ticker)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/scalping")
def api_scalping():
    """
    剥头皮策略信号（三重时间框架：15m+5m+1m）
    注意：yfinance 数据有15分钟延迟，实盘请接入 Alpaca WebSocket
    """
    ticker    = request.args.get("ticker", "").upper().strip()
    portfolio = float(request.args.get("portfolio", 100_000))
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = strategy_scalping(ticker, portfolio)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/capital-flow")
def api_capital_flow():
    """
    Serenity 核心：供应链资金层位流向探测
    检测各层 RS 热度，输出资金接力方向 + 下一层预测
    """
    chain_id = request.args.get("chain", "ai_photonics")
    try:
        result = detect_capital_flow(chain_id)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/sector-rotation")
def api_sector_rotation():
    """
    板块轮动仪表盘：11大GICS板块 + 半导体/生物医药/国防 子行业
    计算各板块相对SPY超额收益（1M/3M），输出当前主线、冷门板块、加速信号
    ?force=1 强制刷新缓存（默认4小时缓存）
    """
    from src.sector_rotation import sector_rotation_report
    force = request.args.get("force", "0") == "1"
    try:
        if force:
            from src.sector_rotation import fetch_sector_rankings
            fetch_sector_rankings(force=True)
        result = sector_rotation_report()
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/dealer-delta")
def api_dealer_delta():
    """
    做市商净 Delta 暴露分析
    dealer_net_delta < 0 → 负Gamma区 → 价格上涨时做市商买入 → 加速（Gamma Squeeze 燃料）
    dealer_net_delta > 0 → 正Gamma区 → 价格上涨时做市商卖出 → 压制（Pin Risk 区间）
    """
    ticker = request.args.get("ticker", "").upper().strip()
    expiry = request.args.get("expiry")
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        calls, puts, _, _ = get_options_chain(ticker, expiry)
        if calls is None or puts is None or (calls.empty and puts.empty):
            return jsonify({"ok": False, "error": "无期权数据"})
        hist = yf.Ticker(ticker).history(period="2d")
        if hist.empty:
            return jsonify({"ok": False, "error": "无法获取当前价格"})
        current_price = float(hist["Close"].iloc[-1])
        result = calculate_dealer_delta(calls, puts, current_price)
        return jsonify({"ok": True, "ticker": ticker, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/cold-decision")
def api_cold_decision():
    """
    冷静决策引擎 — 一票否决制，全绿才入场
    返回：verdict(GO/WAIT/ABORT), score(0-100), 每关状态, 操作计划

    小账户新增参数：
      account_type   — margin 或 cash（默认 margin）
      day_trades_used — 本周已用日内交易次数（默认0）
      is_intraday    — true/false（默认 true）
    """
    ticker    = request.args.get("ticker", "").upper().strip()
    direction = request.args.get("direction", "LONG").upper()
    portfolio = float(request.args.get("portfolio", 100_000))
    acct_type  = request.args.get("account_type", "margin").lower()
    dt_used    = int(request.args.get("day_trades_used", 0))
    intraday   = request.args.get("is_intraday", "true").lower() != "false"
    aggressive = request.args.get("aggressive", "false").lower() == "true"

    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"
    # portfolio < $5k 自动激进（在 cold_decision 内部也有此逻辑）
    if portfolio <= 5_000:
        aggressive = True
    try:
        result = cold_decision(ticker, portfolio=portfolio, direction=direction,
                               account_type=acct_type, day_trades_used=dt_used,
                               is_intraday=intraday, aggressive_mode=aggressive)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/scan")
def api_scan():
    """
    批量冷静模型扫描 — 按分数排序，筛出最优入场时机
    参数：tickers=NVDA,AAPL,TSLA&direction=LONG&portfolio=100000
    """
    raw       = request.args.get("tickers", "")
    tickers   = [t.strip().upper() for t in raw.split(",") if t.strip()]
    direction = request.args.get("direction", "LONG").upper()
    portfolio = float(request.args.get("portfolio", 100_000))

    if not tickers:
        return jsonify({"ok": False, "error": "请提供 tickers 参数，如 ?tickers=NVDA,AAPL,TSLA"})
    try:
        result = scan_tickers(tickers, portfolio=portfolio, direction=direction)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/pdt-check")
def api_pdt_check():
    """
    PDT 规则检查（Pattern Day Trader）

    参数：
      account   — 账户净值（美元，默认10000）
      type      — margin 或 cash（默认 margin）
      used      — 本周已用日内交易次数（默认0）

    返回：
      status（SAFE/WARNING/DANGER/EXEMPT）
      day_trades_left — 剩余次数
      strategy_mode  — 推荐交易模式
      recommendations — 针对性操作建议

    示例：
      /api/pdt-check?account=8000&type=margin&used=2
    """
    account  = float(request.args.get("account", 10_000))
    acct_type = request.args.get("type", "margin").lower()
    used     = int(request.args.get("used", 0))
    try:
        result = check_pdt_risk(account, acct_type, used)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/swing-plan")
def api_swing_plan():
    """
    摆动交易计划生成器（PDT安全，适合小账户）

    参数：
      ticker  — 股票代码
      account — 账户净值（默认10000）
      stop    — 止损价（可选，不填则用ATR估算）
      target  — 目标价（可选，不填则用2:1风险收益比估算）
      days    — 计划持有天数（默认5）

    返回：
      仓位大小、风险收益比、退出规则、税务提示
    """
    ticker  = request.args.get("ticker", "").upper().strip()
    account = float(request.args.get("account", 10_000))
    stop    = request.args.get("stop")
    target  = request.args.get("target")
    days    = int(request.args.get("days", 5))

    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="30d")
        if hist.empty:
            return jsonify({"ok": False, "error": f"找不到 {ticker}"})

        price = float(hist["Close"].iloc[-1])

        if len(hist) >= 14:
            import numpy as np_
            tr_list = []
            for i in range(1, len(hist)):
                h = float(hist["High"].iloc[i])
                l = float(hist["Low"].iloc[i])
                cp = float(hist["Close"].iloc[i-1])
                tr_list.append(max(h-l, abs(h-cp), abs(l-cp)))
            atr = float(np_.mean(tr_list[-14:]))
        else:
            atr = price * 0.02

        stop_price   = float(stop)   if stop   else round(price - atr * 1.5, 2)
        target_price = float(target) if target else round(price + atr * 3.0, 2)

        result = swing_trade_plan(ticker, price, stop_price, target_price, account, days)
        result["current_price"] = round(price, 2)
        result["atr_14d"]       = round(atr, 2)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/account-check")
def api_account_check():
    """
    小账户综合评估

    参数：
      account      — 账户净值（默认10000）
      ticker       — 可选，检查某只股票是否适合此账户操作
      contribution — 每月定投金额（可选，用于成长路径计算）

    返回：
      账户阶段、可用策略列表、期权适合性、成长路径、强制规则
    """
    account      = float(request.args.get("account", 10_000))
    ticker       = request.args.get("ticker", "").upper().strip() or None
    contribution = float(request.args.get("contribution", 0))
    try:
        result = assess_small_account(account, ticker, contribution)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/stock-suitability")
def api_stock_suitability():
    """
    股票适合性检查（流动性 + 可负担性 + 波动性）

    参数：
      ticker  — 股票代码
      account — 账户净值（默认10000）

    返回：
      整体评估、流动性评级、买卖价差、可买股数、波动性
    """
    ticker  = request.args.get("ticker", "").upper().strip()
    account = float(request.args.get("account", 10_000))
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = check_stock_suitability(ticker, account)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/position-size")
def api_position_size():
    """
    小账户仓位计算（1%风险规则 + 小账户约束）

    参数：
      account  — 账户净值（默认10000）
      price    — 股票当前价格
      stop     — 止损价格
      risk_pct — 每笔风险占比（默认1%）

    返回：
      建议股数、仓位金额、最大风险、约束原因
    """
    account  = float(request.args.get("account", 10_000))
    price    = float(request.args.get("price", 0))
    stop     = float(request.args.get("stop",  0))
    risk_pct = float(request.args.get("risk_pct", 1.0)) / 100.0
    if price <= 0 or stop <= 0:
        return jsonify({"ok": False, "error": "请提供有效的 price 和 stop 参数"})
    try:
        stop_dist = abs(price - stop)
        result    = position_size_small(account, price, stop_dist, risk_pct)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── 辩论引擎 ──────────────────────────────────────────────────

@app.route("/api/debate")
def api_debate():
    """
    多方辩论报告（Bull vs Bear · Devil's Advocate）
    借鉴 TauriCresearch/TradingAgents 架构，确定性实现无API费用

    参数：ticker, direction=LONG, account=2000
    返回：看多论据、看空风险、Kelly仓位建议、辩论结论
    """
    ticker    = request.args.get("ticker", "").upper().strip()
    direction = request.args.get("direction", "LONG").upper()
    account   = float(request.args.get("account", 2000))
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        result = generate_trade_debate(ticker, direction, account_value=account)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/debate-with-cold")
def api_debate_with_cold():
    """
    冷静决策 + 辩论引擎 一体化（推荐入口）
    先跑 cold_decision，再用结果喂给 debate，输出完整决策包
    """
    ticker    = request.args.get("ticker", "").upper().strip()
    account   = float(request.args.get("account", 2000))
    direction = request.args.get("direction", "LONG").upper()
    if not ticker:
        return jsonify({"ok": False, "error": "请输入股票代码"})
    try:
        cold   = cold_decision(ticker, portfolio=account, direction=direction,
                               is_intraday=False, aggressive_mode=(account <= 5000))
        debate = generate_trade_debate(ticker, direction, cold, account)
        return jsonify({"ok": True, "cold_decision": cold, "debate": debate})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── 模拟交易 & Agent ───────────────────────────────────────────

@app.route("/api/paper/init", methods=["POST"])
def api_paper_init():
    """初始化模拟盘账户 POST {account: 2000, mode: "paper", label: "激进账户"}"""
    body    = request.get_json(silent=True) or {}
    try:
        account = float(body.get("account") or 2000)
        mode    = body.get("mode", "paper")
        label   = body.get("label", "默认账户")
        return jsonify(init_account(account, mode, label))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/open", methods=["POST"])
def api_paper_open():
    """
    开仓 POST {ticker, shares, entry_price, stop_loss, target, strategy, mode}
    """
    body = request.get_json(silent=True) or {}
    try:
        r = open_position(
            ticker      = body.get("ticker", "").upper(),
            shares      = int(body.get("shares", 1)),
            entry_price = float(body.get("entry_price", 0)),
            stop_loss   = float(body.get("stop_loss", 0)),
            target      = float(body.get("target", 0)),
            strategy    = body.get("strategy", "手动"),
            mode        = body.get("mode", "paper"),
        )
        return jsonify(r)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/close", methods=["POST"])
def api_paper_close():
    """平仓 POST {trade_id, exit_price, exit_reason, mode}"""
    body = request.get_json(silent=True) or {}
    try:
        r = close_position(
            trade_id   = body.get("trade_id", ""),
            exit_price = float(body.get("exit_price", 0)),
            exit_reason= body.get("exit_reason", "手动平仓"),
            mode       = body.get("mode", "paper"),
        )
        return jsonify(r)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/positions")
def api_paper_positions():
    """列出所有持仓 ?mode=paper"""
    mode = request.args.get("mode", "paper")
    try:
        return jsonify(list_positions(mode))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/mtm")
def api_paper_mtm():
    """盯市（Mark to Market）— 获取当前市值和止损/目标警报"""
    mode = request.args.get("mode", "paper")
    try:
        return jsonify(mark_to_market(mode))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/performance")
def api_paper_performance():
    """
    完整绩效报告：Sharpe/Sortino/MaxDD/Kelly/胜率
    ?mode=paper
    """
    mode = request.args.get("mode", "paper")
    try:
        return jsonify(performance_report(mode))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/compare")
def api_paper_compare():
    """模拟盘 vs 真实盘对比报告"""
    try:
        return jsonify(compare_paper_vs_real())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/reset-breaker", methods=["POST"])
def api_paper_reset_breaker():
    """重置熔断器（需要复盘后手动解除）POST {mode, confirm: true}"""
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(reset_circuit_breaker(body.get("mode","paper"), body.get("confirm",False)))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/agent/scan")
def api_agent_scan():
    """
    自动化Agent扫描（每日10AM ET运行）
    ?watchlist=NVDA,AAPL,TSLA&account=2000&auto_paper=true
    """
    raw       = request.args.get("watchlist", "NVDA,AAPL,TSLA,AMD,META")
    tickers   = [t.strip().upper() for t in raw.split(",") if t.strip()]
    account   = float(request.args.get("account", 2000))
    auto_p    = request.args.get("auto_paper", "false").lower() != "false"
    mode      = request.args.get("mode", "paper")
    direction = request.args.get("direction", "LONG").upper()
    try:
        result = run_scan(tickers, account, direction, auto_p, mode)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/agent/monitor")
def api_agent_monitor():
    """Agent监控：检查止损/目标，自动平仓 ?mode=paper&auto_stop=true"""
    mode      = request.args.get("mode", "paper")
    auto_stop = request.args.get("auto_stop", "true").lower() != "false"
    try:
        result = run_monitor(mode, auto_stop)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/agent/report")
def api_agent_report():
    """每日交易报告 ?mode=paper"""
    mode = request.args.get("mode", "paper")
    try:
        result = daily_report(mode)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/earnings/<ticker>")
def api_earnings(ticker):
    """完整财报质量报告（CANSLIM评分 + EPS加速 + PEAD + 质量因子）"""
    try:
        result = full_earnings_analysis(ticker.upper())
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/earnings/<ticker>/eps")
def api_eps_acceleration(ticker):
    """EPS加速度分析（季度超预期 + YoY加速趋势）"""
    try:
        return app.response_class(
            response=json.dumps(analyze_eps_acceleration(ticker.upper()),
                                ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/earnings/<ticker>/pead")
def api_pead(ticker):
    """财报后漂移分析（Post-Earnings Announcement Drift）"""
    try:
        return app.response_class(
            response=json.dumps(analyze_pead(ticker.upper()),
                                ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/earnings/<ticker>/quality")
def api_earnings_ticker_quality(ticker):
    """质量因子分析（ROE / 毛利率 / FCF质量 / 债务）"""
    try:
        return app.response_class(
            response=json.dumps(analyze_quality_factors(ticker.upper()),
                                ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/market/breadth")
def api_market_breadth():
    """市场宽度（板块MA50以上比例 + 个股MA200以上比例）"""
    try:
        result = get_market_breadth()
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/portfolio/correlation")
def api_correlation():
    """
    仓位相关性检查
    ?new=NVDA&existing=AMD,AVGO,INTC
    """
    new_ticker   = request.args.get("new", "").upper()
    existing_raw = request.args.get("existing", "")
    existing     = [t.strip().upper() for t in existing_raw.split(",") if t.strip()]
    if not new_ticker:
        return jsonify({"ok": False, "error": "缺少 ?new= 参数"})
    try:
        result = check_position_correlation(new_ticker, existing)
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/paper/trailing-stop", methods=["POST"])
def api_trailing_stop():
    """更新追踪止损 {trade_id, current_price, trail_pct, mode}"""
    d = request.get_json(force=True) or {}
    try:
        result = update_trailing_stop(
            trade_id      = d.get("trade_id", ""),
            current_price = float(d.get("current_price", 0)),
            trail_pct     = float(d.get("trail_pct", 8.0)),
            mode          = d.get("mode", "paper"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/macro")
def api_macro():
    """宏观环境完整报告（经济日历 + 主题 + 传导链 + 板块建议）"""
    tickers_raw = request.args.get("tickers", "")
    watchlist   = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    try:
        result = full_macro_report(watchlist or None)
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/macro/calendar")
def api_macro_calendar():
    """经济事件日历（FOMC/CPI/非农）"""
    try:
        return jsonify(get_economic_calendar())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/macro/etf-signals")
def api_macro_etf():
    """ETF 实时宏观主题推断"""
    try:
        return app.response_class(
            response=json.dumps(get_etf_signals(), ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/smart-money/<ticker>")
def api_smart_money(ticker):
    """机构资金完整追踪报告（UOA + GEX + 空头挤压 + 智能资金流 + 13F）"""
    try:
        result = full_smart_money_scan(ticker.upper())
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/smart-money/<ticker>/uoa")
def api_uoa(ticker):
    """异常期权活动（Unusual Options Activity）"""
    try:
        return app.response_class(
            response=json.dumps(detect_unusual_options(ticker.upper()),
                                ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/smart-money/<ticker>/gex")
def api_gex(ticker):
    """做市商 Gamma 敞口（Gamma Exposure）"""
    try:
        return app.response_class(
            response=json.dumps(calculate_gex(ticker.upper()),
                                ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/smart-money/<ticker>/squeeze")
def api_squeeze(ticker):
    """空头挤压探测器"""
    try:
        return app.response_class(
            response=json.dumps(detect_short_squeeze(ticker.upper()),
                                ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/smart-money/<ticker>/flow")
def api_smf(ticker):
    """智能资金流向（盘中机构 vs 散户）"""
    try:
        return app.response_class(
            response=json.dumps(smart_money_flow(ticker.upper()),
                                ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/backtest/historical", methods=["GET", "POST"])
def api_backtest_historical():
    """
    历史回测（多标的逐日切片，零未来数据泄露）
    参数：tickers, start, end, account, risk, rr, verbose
    示例：GET /api/backtest/historical?tickers=NVDA,AMD&start=2023-01-01&end=2024-12-31&account=2000
    """
    if request.method == "POST":
        d = request.get_json(force=True) or {}
    else:
        d = request.args

    tickers_raw = d.get("tickers", "NVDA,AMD,TSLA")
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    try:
        result = run_historical_backtest(
            tickers    = tickers,
            start_date = d.get("start", "2023-01-01"),
            end_date   = d.get("end",   "2024-12-31"),
            account    = float(d.get("account", 2000)),
            risk_pct   = float(d.get("risk", 3.0)),
            target_rr  = float(d.get("rr", 2.0)),
            verbose    = str(d.get("verbose", "false")).lower() == "true",
        )
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, cls=_NpEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/market/ftd")
def api_market_ftd():
    """
    O'Neil Follow-Through Day 底部确认检测
    是否所有反弹都是 Dead Cat Bounce？FTD 是最重要的底部确认信号。
    """
    try:
        snap     = get_market_snapshot()
        ftd      = detect_follow_through_day(snap["spy_hist"])
        return jsonify({"ok": True, **ftd})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    import os as _os
    _debug = _os.getenv("FLASK_DEBUG", "0") == "1"
    print("StockRadar 启动中... http://127.0.0.1:5000")
    app.run(debug=_debug, port=5000, use_reloader=False)
