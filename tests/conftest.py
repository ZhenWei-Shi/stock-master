import os
import sys

# 将项目根目录加入 Python 路径，使 import src.xxx 可用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
