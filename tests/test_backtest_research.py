"""backtest_research.py 的统计辅助函数测试（不联网）。"""
import math

import numpy as np
import pandas as pd

from src.backtest_research import _mean_se, compare_to_random


def test_mean_se():
    m, se, n = _mean_se([1.0, 2.0, 3.0, np.nan])
    assert (m, n) == (2.0, 3) and math.isclose(se, 1 / math.sqrt(3))


def test_compare_to_random_empty():
    assert compare_to_random({}, pd.DataFrame()) == {"n": 0}
