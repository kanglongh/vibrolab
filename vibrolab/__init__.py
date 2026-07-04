"""
vibrolab — 工业振动信号故障诊断工具箱

Modules
-------
    io           数据加载 + 滑窗切分
    features     120 维 CFD 特征 + 漂移度量工具
    paths        项目路径管理
"""
from . import io, features, paths

__version__ = "0.1.0"
__all__ = ["io", "features", "paths"]
