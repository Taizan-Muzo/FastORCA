"""
External Bridge Module
用于集成外部量子化学分析工具（critic2, HORTON, Psi4 等）
"""

from .bridge_context import BridgeContext
from .executor import ExternalExecutor
from .adapters.base_adapter import ExternalAdapter, ExternalResult, InputBundle, RunResult

__all__ = [
    "BridgeContext",
    "ExternalExecutor", 
    "ExternalAdapter",
    "ExternalResult",
    "InputBundle",
    "RunResult",
]
