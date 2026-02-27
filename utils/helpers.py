"""
工具函数模块
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


def save_json(data: Dict[str, Any], filepath: str):
    """保存数据为 JSON"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=_json_serializer)


def load_json(filepath: str) -> Dict[str, Any]:
    """从 JSON 加载数据"""
    with open(filepath, 'r') as f:
        return json.load(f)


def _json_serializer(obj):
    """JSON 序列化辅助函数"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def batch_iterator(items: List[Any], batch_size: int):
    """批量迭代器"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
