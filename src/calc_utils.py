"""通用计算工具函数"""


def calc_surprise_pct(eps_act: float, eps_est: float):
    """百分比超预期 = (实际-预期)/|预期|×100；eps_est=0 时返回 None。"""
    if eps_est == 0:
        return None
    return round((eps_act - eps_est) / abs(eps_est) * 100, 1)
