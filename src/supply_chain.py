"""
供应链七层定位数据库 + 资金流向探测（Serenity方法论核心）

v2 新增：
  detect_capital_flow(chain_id) — 计算各层 RS 强度，输出资金接力方向
  Serenity 霍尔木兹原则：资金必须流过的"卡脖子层"才是最强机会

资金接力路径（正常周期）：
  触发层涨 → 基础设施层 → 卡脖子层（爆发力最强）→ 应用层 → 饱和撤离
"""
import yfinance as yf
import numpy as np

CHAINS = {
    "ai_photonics": {
        "name": "AI光学互联供应链",
        "desc": "从InP基板到超级算力数据中心的完整光子学供应链（Serenity重点赛道）",
        "layers": [
            {
                "layer": 7,
                "name": "终端算力平台",
                "desc": "超大规模云计算买家，最终需求方",
                "tickers": ["MSFT", "GOOGL", "AMZN", "META"],
            },
            {
                "layer": 6,
                "name": "数据中心基建",
                "desc": "机房、电力、散热基础设施",
                "tickers": ["VRT", "EQIX", "DLR", "NXT"],
            },
            {
                "layer": 5,
                "name": "光学收发器模组",
                "desc": "800G/1.6T光收发器，当前机构资金主战场",
                "tickers": ["AAOI", "COHR", "LITE", "VIAV", "FNSR"],
            },
            {
                "layer": 4,
                "name": "连续波激光源",
                "desc": "硅光子片内激光源，下一个卡脖子",
                "tickers": ["SIVE", "POET"],
            },
            {
                "layer": 3,
                "name": "InP磷化铟基板",
                "desc": "激光源不可或缺的III-V族材料基板，全球供应商极少",
                "tickers": ["AXTI"],
            },
            {
                "layer": 2,
                "name": "前驱材料/特种气体",
                "desc": "半导体级磷化氢、砷化氢等原材料",
                "tickers": ["APD", "LIN"],
            },
            {
                "layer": 1,
                "name": "稀有矿产原材料",
                "desc": "铟、镓、砷等关键稀有金属",
                "tickers": ["MP", "LTHM"],
            },
        ]
    },
    "ai_chips": {
        "name": "AI芯片供应链",
        "desc": "从GPU设计到算力芯片的完整半导体供应链",
        "layers": [
            {
                "layer": 7,
                "name": "AI模型/应用层",
                "desc": "消耗算力的AI原生公司",
                "tickers": ["MSFT", "GOOGL", "META", "AMZN"],
            },
            {
                "layer": 6,
                "name": "GPU/加速器设计",
                "desc": "核心算力芯片设计，当前行业最热层",
                "tickers": ["NVDA", "AMD", "INTC", "MRVL"],
            },
            {
                "layer": 5,
                "name": "晶圆代工",
                "desc": "先进制程制造，2nm/3nm节点",
                "tickers": ["TSM", "INTC", "GFS"],
            },
            {
                "layer": 4,
                "name": "半导体设备",
                "desc": "光刻、刻蚀、沉积设备",
                "tickers": ["AMAT", "LRCX", "KLAC", "ASML"],
            },
            {
                "layer": 3,
                "name": "半导体材料",
                "desc": "光刻胶、CMP材料、特种气体",
                "tickers": ["ENTG", "CEVA", "UCTT"],
            },
            {
                "layer": 2,
                "name": "硅片/基底材料",
                "desc": "300mm硅片及特种衬底",
                "tickers": ["SUMCO", "SK", "AXTI"],
            },
            {
                "layer": 1,
                "name": "多晶硅/矿产",
                "desc": "高纯多晶硅及稀土原材料",
                "tickers": ["MP", "LTHM", "DQ"],
            },
        ]
    },
    "ai_power": {
        "name": "AI数据中心电力供应链",
        "desc": "支撑GPU集群运行的电力基础设施供应链",
        "layers": [
            {
                "layer": 7,
                "name": "超大规模数据中心",
                "desc": "最终电力消耗方",
                "tickers": ["MSFT", "GOOGL", "AMZN", "META"],
            },
            {
                "layer": 6,
                "name": "数据中心运营商",
                "desc": "托管数据中心，转售电力和冷却",
                "tickers": ["EQIX", "DLR", "CONE"],
            },
            {
                "layer": 5,
                "name": "电力基础设施",
                "desc": "变压器、配电、UPS系统",
                "tickers": ["VRT", "ETN", "HUBB"],
            },
            {
                "layer": 4,
                "name": "冷却/散热系统",
                "desc": "液冷、浸没式冷却解决方案",
                "tickers": ["GTLS", "IIVI", "LIQT"],
            },
            {
                "layer": 3,
                "name": "电网/电力供应",
                "desc": "为数据中心供电的公用事业",
                "tickers": ["NEE", "AEP", "VST"],
            },
            {
                "layer": 2,
                "name": "核能/清洁电力",
                "desc": "24/7稳定清洁电力，核电复兴",
                "tickers": ["CEG", "NNE", "SMR"],
            },
            {
                "layer": 1,
                "name": "铀矿/燃料",
                "desc": "核电燃料原材料",
                "tickers": ["CCJ", "UEC", "DNN"],
            },
        ]
    },

    # ── 新增：GLP-1 / 减肥药革命供应链 ──────────────────────
    "glp1": {
        "name": "GLP-1减肥药供应链",
        "desc": "从药物研发到患者给药的完整GLP-1产业链（年市场规模1000亿+）",
        "layers": [
            {
                "layer": 7,
                "name": "医保/PBM支付方",
                "desc": "最终埋单的保险公司和药品福利管理商",
                "tickers": ["CVS", "CI", "UNH"],
            },
            {
                "layer": 6,
                "name": "医院/零售药房分销",
                "desc": "GLP-1药品的分发渠道",
                "tickers": ["MCK", "ABC", "CAH"],
            },
            {
                "layer": 5,
                "name": "GLP-1原研药企",
                "desc": "当前行业最热卡脖子层：Ozempic/Wegovy/Zepbound垄断",
                "tickers": ["LLY", "NVO"],
            },
            {
                "layer": 4,
                "name": "给药装置/并发症管理",
                "desc": "自动注射笔、血糖仪、心血管监测（GLP-1患者需要）",
                "tickers": ["DXCM", "PODD", "NTRA", "ITGR"],
            },
            {
                "layer": 3,
                "name": "合同研究/CDMO生产",
                "desc": "委托生产GLP-1多肽活性成分（产能严重不足）",
                "tickers": ["CTLT", "MEDP", "WCG"],
            },
            {
                "layer": 2,
                "name": "生物工具/API化学合成",
                "desc": "多肽合成试剂、色谱纯化设备、酶工程",
                "tickers": ["TMO", "BRKR", "AMGN"],
            },
            {
                "layer": 1,
                "name": "基因/蛋白质工具",
                "desc": "靶点发现、基因组学、蛋白质结构研究工具",
                "tickers": ["ILMN", "A", "REGN"],
            },
        ]
    },

    # ── 新增：国防航天供应链 ─────────────────────────────────
    "defense": {
        "name": "国防航天供应链",
        "desc": "从稀有金属到武器系统的完整国防产业链（地缘政治驱动）",
        "layers": [
            {
                "layer": 7,
                "name": "政府/北约盟国（终端买家）",
                "desc": "国防预算无弹性需求，地缘风险升温时无上限",
                "tickers": ["LMT", "RTX", "NOC"],
            },
            {
                "layer": 6,
                "name": "系统集成主承包商",
                "desc": "F-35/导弹/舰船整机集成商",
                "tickers": ["LMT", "RTX", "NOC", "GD", "BA"],
            },
            {
                "layer": 5,
                "name": "太空/无人机/新域作战",
                "desc": "商业航天、无人作战、超音速——国防新前沿",
                "tickers": ["RKLB", "KTOS", "JOBY", "ACHR"],
            },
            {
                "layer": 4,
                "name": "情报/网络安全/C4ISR",
                "desc": "指控通信情报监视侦察系统（信息化战争核心）",
                "tickers": ["BAH", "CACI", "SAIC", "LDOS"],
            },
            {
                "layer": 3,
                "name": "推进系统/精密结构",
                "desc": "火箭发动机、反应堆推进（潜艇）、精密机加工",
                "tickers": ["BWXT", "HII", "TDY", "AXON"],
            },
            {
                "layer": 2,
                "name": "先进材料/复合材料",
                "desc": "碳纤维复合材料、钛合金结构件",
                "tickers": ["HXL", "CRS", "TXT"],
            },
            {
                "layer": 1,
                "name": "稀土/战略原材料",
                "desc": "磁铁稀土（制导系统）、锂（电池）、铀（核推进）",
                "tickers": ["MP", "UUUU", "ALB"],
            },
        ]
    },

    # ── 新增：机器人自动化供应链 ─────────────────────────────
    "robotics": {
        "name": "机器人自动化供应链",
        "desc": "从稀土磁铁到手术机器人/工业自动化的完整产业链",
        "layers": [
            {
                "layer": 7,
                "name": "终端用户（汽车/物流/医院）",
                "desc": "最终部署自动化的行业，需求端",
                "tickers": ["AMZN", "F", "GM", "HCA"],
            },
            {
                "layer": 6,
                "name": "系统集成/工业软件",
                "desc": "数字孪生、仿真、PLC编程软件",
                "tickers": ["PTC", "ANSS", "ROK", "EMR"],
            },
            {
                "layer": 5,
                "name": "机器人本体/末端执行",
                "desc": "手术机器人、协作机器人、测试设备",
                "tickers": ["ISRG", "TER", "BRKS", "MBOT"],
            },
            {
                "layer": 4,
                "name": "计算机视觉/传感器",
                "desc": "机器视觉、激光雷达、工业相机——机器人的「眼睛」",
                "tickers": ["CGNX", "KEYS", "MKSI", "AMBA"],
            },
            {
                "layer": 3,
                "name": "运动控制/伺服驱动",
                "desc": "精密伺服电机、编码器、运动控制器——机器人的「肌肉」",
                "tickers": ["NOVT", "RBC", "AEIS", "AMETEK"],
            },
            {
                "layer": 2,
                "name": "功率半导体/驱动IC",
                "desc": "SiC/GaN功率器件、马达驱动芯片",
                "tickers": ["ON", "MPWR", "WOLF", "AEHR"],
            },
            {
                "layer": 1,
                "name": "稀土磁铁/精密材料",
                "desc": "钕铁硼永磁体（伺服电机核心）、精密轴承钢",
                "tickers": ["MP", "UUUU", "ARNC"],
            },
        ]
    }
}


def get_chain_names():
    return {k: v["name"] for k, v in CHAINS.items()}


def get_chain(chain_id: str) -> dict:
    return CHAINS.get(chain_id)


# ─────────────────────────────────────────────────────────────
# Serenity 核心：资金层位流向探测
# ─────────────────────────────────────────────────────────────

def detect_capital_flow(chain_id: str) -> dict:
    """
    Serenity 方法论核心功能：
      计算供应链各层代表 ticker 的 RS 强度（1月/3月超额收益），
      检测资金当前流向哪一层，预测下一层接棒方向。

    输出：
      layer_strength   — 各层当前 RS 热度
      hot_layer        — 当前最热层（资金聚集）
      next_likely_layer — 根据接力规律预测下一层
      flow_signal      — 综合流向信号
      serenity_note    — Serenity 风格解读
    """
    chain = CHAINS.get(chain_id)
    if not chain:
        return {"error": f"未知供应链 ID：{chain_id}"}

    layers = chain["layers"]

    # ── 批量拉取所有 ticker 的行情（含 SPY 基准，避免额外单独请求）────
    all_tickers = list({tk for layer in layers for tk in layer["tickers"]} | {"SPY"})
    try:
        raw = yf.download(
            " ".join(all_tickers),
            period="6mo", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception as e:
        return {"error": f"数据获取失败：{e}"}

    def get_close(tk: str):
        try:
            if len(all_tickers) == 1:
                return raw["Close"].dropna()
            return raw[tk]["Close"].dropna() if tk in raw.columns.get_level_values(0) else None
        except Exception:
            return None

    # SPY 基准
    spy_close = get_close("SPY")
    if spy_close is None:
        try:
            spy_close = yf.Ticker("SPY").history(period="6mo")["Close"].dropna()
        except Exception:
            spy_close = None

    def excess_return(tk_close, days: int) -> float:
        """ticker 超额收益 vs SPY"""
        if tk_close is None or len(tk_close) < days:
            return float("nan")
        tk_ret = (tk_close.iloc[-1] - tk_close.iloc[-days]) / tk_close.iloc[-days]
        if spy_close is not None and len(spy_close) >= days:
            spy_ret = (spy_close.iloc[-1] - spy_close.iloc[-days]) / spy_close.iloc[-days]
            return (tk_ret - spy_ret) * 100
        return tk_ret * 100

    # ── 逐层计算平均超额收益 ──────────────────────────────────
    layer_results = []
    for layer in sorted(layers, key=lambda x: x["layer"]):
        layer_num  = layer["layer"]
        layer_name = layer["name"]
        tickers    = layer["tickers"]

        scores_1m, scores_3m = [], []
        ticker_details = []
        for tk in tickers:
            cl = get_close(tk)
            if cl is None:
                try:
                    cl = yf.Ticker(tk).history(period="6mo")["Close"].dropna()
                except Exception:
                    cl = None
            ex_1m = excess_return(cl, 21)
            ex_3m = excess_return(cl, 63)
            if not np.isnan(ex_1m):
                scores_1m.append(ex_1m)
            if not np.isnan(ex_3m):
                scores_3m.append(ex_3m)
            ticker_details.append({
                "ticker":     tk,
                "excess_1m":  round(ex_1m, 1) if not np.isnan(ex_1m) else None,
                "excess_3m":  round(ex_3m, 1) if not np.isnan(ex_3m) else None,
            })

        avg_1m = float(np.mean(scores_1m)) if scores_1m else float("nan")
        avg_3m = float(np.mean(scores_3m)) if scores_3m else float("nan")

        # 综合热度分（1月权重 0.6，3月权重 0.4）
        if not np.isnan(avg_1m) and not np.isnan(avg_3m):
            heat = avg_1m * 0.6 + avg_3m * 0.4
        elif not np.isnan(avg_1m):
            heat = avg_1m
        elif not np.isnan(avg_3m):
            heat = avg_3m
        else:
            heat = float("nan")

        layer_results.append({
            "layer":       layer_num,
            "name":        layer_name,
            "desc":        layer.get("desc", ""),
            "tickers":     tickers,
            "avg_excess_1m": round(avg_1m, 1) if not np.isnan(avg_1m) else None,
            "avg_excess_3m": round(avg_3m, 1) if not np.isnan(avg_3m) else None,
            "heat_score":   round(heat, 1)     if not np.isnan(heat)  else None,
            "ticker_details": ticker_details,
        })

    # ── 判断热点层 + 资金流向 ─────────────────────────────────
    valid = [lr for lr in layer_results if lr["heat_score"] is not None]
    if not valid:
        return {"chain": chain_id, "error": "数据不足，无法判断资金流向"}

    # 按热度排序
    ranked = sorted(valid, key=lambda x: x["heat_score"], reverse=True)
    hot_layer    = ranked[0]
    hot_layer_n  = hot_layer["layer"]

    # 预测下一层（Serenity 接力方向：从高层号向低层号流动）
    # 资金从终端(7)→基建(6)→卡脖子层(3-5)→材料层(1-2)
    # 当前热点层的下一层 = 层号-1（向上游流）
    next_layer_n    = hot_layer_n - 1
    next_layer_info = next(
        (lr for lr in layer_results if lr["layer"] == next_layer_n), None
    )

    # 动量变化（1月 vs 3月：1月 > 3月 = 加速流入）
    momentum_accel = None
    if hot_layer["avg_excess_1m"] is not None and hot_layer["avg_excess_3m"] is not None:
        momentum_accel = hot_layer["avg_excess_1m"] > hot_layer["avg_excess_3m"]

    # Serenity 风格解读
    serenity_note = _serenity_interpret(
        hot_layer, next_layer_info, momentum_accel, chain["name"]
    )

    # 流向信号
    if hot_layer_n in (3, 4, 5) and momentum_accel:
        flow_signal = "CHOKEPOINT_ACTIVE"
        signal_desc = "卡脖子层正在爆发，资金加速流入，是最强做多信号"
    elif hot_layer_n in (6, 7) and momentum_accel:
        flow_signal = "INFRASTRUCTURE_HOT"
        signal_desc = "资金仍在基础设施层，尚未传导到卡脖子，提前布局卡脖子层"
    elif hot_layer_n in (1, 2):
        flow_signal = "UPSTREAM_SPILLOVER"
        signal_desc = "资金已溢出至原材料层，接力进入尾声，卡脖子层可能已见顶"
    elif not momentum_accel and hot_layer_n in (3, 4, 5):
        flow_signal = "CHOKEPOINT_FADING"
        signal_desc = "卡脖子层热度开始减弱，资金可能准备向应用层轮动"
    else:
        flow_signal = "MIXED"
        signal_desc = "资金流向混乱，暂无明确接力信号"

    return {
        "chain_id":          chain_id,
        "chain_name":        chain["name"],
        "layer_strength":    layer_results,
        "ranked_by_heat":    [{"layer": r["layer"], "name": r["name"],
                               "heat": r["heat_score"]} for r in ranked],
        "hot_layer":         hot_layer,
        "next_likely_layer": next_layer_info,
        "momentum_accel":    momentum_accel,
        "flow_signal":       flow_signal,
        "signal_desc":       signal_desc,
        "serenity_note":     serenity_note,
    }


def _serenity_interpret(hot: dict, nxt, accel, chain_name: str) -> str:
    """用 Serenity 霍尔木兹海峡原则给出操作解读"""
    hot_n = hot["layer"]
    hot_name = hot["name"]
    heat = hot["heat_score"]

    if hot_n in (3, 4, 5):
        chokepoint_status = "⚡ 卡脖子层爆发"
        if accel:
            detail = f"【{chain_name}】资金正在加速涌入【{hot_name}】（卡脖子层{hot_n}），超额收益{heat:+.1f}%。这是霍尔木兹时刻——下游买家必须经过这里，供不应求。现在是最强做多窗口。"
        else:
            detail = f"【{chain_name}】资金在【{hot_name}】（卡脖子层{hot_n}），热度{heat:+.1f}%但开始减弱。关注是否准备向上游或应用层轮动。"
    elif hot_n in (6, 7):
        chokepoint_status = "🏗️ 基础设施先行"
        if nxt:
            detail = f"【{chain_name}】资金目前在【{hot_name}】（基础设施层{hot_n}），超额{heat:+.1f}%。根据行业接力规律，资金接下来大概率流向【{nxt['name']}】（第{nxt['layer']}层）。现在是提前布局卡脖子层的窗口期。"
        else:
            detail = f"【{chain_name}】基础设施层领跑，卡脖子层布局窗口即将打开。"
    elif hot_n in (1, 2):
        chokepoint_status = "⚠️ 接力尾声"
        detail = f"【{chain_name}】资金已溢出至原材料层【{hot_name}】（第{hot_n}层），超额{heat:+.1f}%。这通常是行业景气接力的尾声阶段，卡脖子层可能已见顶，需谨慎追高。"
    else:
        chokepoint_status = "❓ 信号不明"
        detail = f"【{chain_name}】热点层：{hot_name}（第{hot_n}层），资金流向尚不明确。"

    return f"{chokepoint_status}\n{detail}"
