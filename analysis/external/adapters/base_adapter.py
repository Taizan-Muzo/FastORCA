"""
External Adapter Base Class
外部工具适配器基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from ..bridge_context import BridgeContext
from ..executor import ExternalExecutor, ExecutionResult


@dataclass
class InputBundle:
    """输入文件包"""
    input_file: Path
    output_file: Path
    working_directory: Path
    auxiliary_files: List[Path] = field(default_factory=list)
    command_args: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    """运行结果"""
    execution_result: ExecutionResult
    output_files: List[Path] = field(default_factory=list)


@dataclass
class ParsedResult:
    """解析结果"""
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExternalResult:
    """完整外部结果"""
    success: bool
    execution_status: str  # "success", "failed", "timeout", "skipped"
    failure_reason: Optional[str]
    
    # Bridge 执行信息
    bridge_input_file: Optional[str] = None
    bridge_output_file: Optional[str] = None
    bridge_execution_time_seconds: Optional[float] = None
    bridge_tool_version: Optional[str] = None
    
    # 解析后的特征
    features: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success_result(
        cls,
        features: Dict[str, Any],
        input_file: str,
        output_file: str,
        execution_time: float,
        tool_version: Optional[str] = None,
    ) -> "ExternalResult":
        """创建成功结果"""
        return cls(
            success=True,
            execution_status="success",
            failure_reason=None,
            bridge_input_file=input_file,
            bridge_output_file=output_file,
            bridge_execution_time_seconds=execution_time,
            bridge_tool_version=tool_version,
            features=features,
        )
    
    @classmethod
    def failed_result(cls, reason: str) -> "ExternalResult":
        """创建失败结果"""
        return cls(
            success=False,
            execution_status="failed",
            failure_reason=reason,
            features={},
        )
    
    @classmethod
    def timeout_result(cls, timeout_seconds: float) -> "ExternalResult":
        """创建超时结果"""
        return cls(
            success=False,
            execution_status="timeout",
            failure_reason=f"external_tool_timeout: {timeout_seconds}s",
            features={},
        )


class ExternalAdapter(ABC):
    """
    外部工具适配器基类
    
    所有外部工具适配器必须继承此类，并实现三个抽象方法
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.executor: Optional[ExternalExecutor] = None
    
    @abstractmethod
    def prepare_inputs(self, context: BridgeContext) -> InputBundle:
        """
        准备外部工具输入
        
        生成输入文件、确定工作目录、设置命令行参数
        
        Args:
            context: 桥接上下文
            
        Returns:
            InputBundle 包含所有输入信息
            
        Raises:
            ValueError: context 验证失败
            IOError: 文件写入失败
        """
        pass
    
    @abstractmethod
    def run_external(self, input_bundle: InputBundle) -> RunResult:
        """
        执行外部工具
        
        使用 executor 运行外部命令，捕获输出
        
        Args:
            input_bundle: 输入文件包
            
        Returns:
            RunResult 包含执行结果
            
        Raises:
            FileNotFoundError: 可执行文件不存在
            subprocess.TimeoutExpired: 执行超时
        """
        pass
    
    @abstractmethod
    def parse_outputs(self, run_result: RunResult) -> ParsedResult:
        """
        解析外部工具输出
        
        从输出文件中提取特征和元数据
        
        Args:
            run_result: 运行结果
            
        Returns:
            ParsedResult 包含解析后的特征
            
        Raises:
            ValueError: 输出格式无法解析
            FileNotFoundError: 输出文件不存在
        """
        pass
    
    def execute(self, context: BridgeContext) -> ExternalResult:
        """
        完整执行流程
        
        按顺序执行 prepare -> run -> parse
        提供统一的错误处理和 soft-fail 机制
        
        Args:
            context: 桥接上下文
            
        Returns:
            ExternalResult（成功或失败）
        """
        # 验证 context
        valid, error_msg = context.validate()
        if not valid:
            logger.error(f"Bridge context validation failed: {error_msg}")
            return ExternalResult.failed_result(f"bridge_metadata_inconsistent: {error_msg}")
        
        try:
            # Step 1: 准备输入
            logger.info(f"[{context.molecule_id}] Preparing {self.__class__.__name__} inputs...")
            input_bundle = self.prepare_inputs(context)
            
            # Step 2: 执行外部工具
            logger.info(f"[{context.molecule_id}] Running external tool...")
            run_result = self.run_external(input_bundle)
            
            if run_result.execution_result.timed_out:
                logger.error(f"[{context.molecule_id}] External tool timed out")
                return ExternalResult.timeout_result(self.executor.timeout if self.executor else 0)
            
            if run_result.execution_result.returncode != 0:
                logger.error(f"[{context.molecule_id}] External tool failed with code {run_result.execution_result.returncode}")
                logger.error(f"stderr: {run_result.execution_result.stderr[:500]}")
                return ExternalResult.failed_result("external_execution_failed: non-zero exit code")
            
            # Step 3: 解析输出
            logger.info(f"[{context.molecule_id}] Parsing outputs...")
            parsed = self.parse_outputs(run_result)
            
            # 构建成功结果
            return ExternalResult.success_result(
                features=parsed.features,
                input_file=str(input_bundle.input_file),
                output_file=str(input_bundle.output_file),
                execution_time=run_result.execution_result.execution_time_seconds,
                tool_version=parsed.metadata.get("tool_version"),
            )
            
        except FileNotFoundError as e:
            logger.error(f"[{context.molecule_id}] External tool not found: {e}")
            return ExternalResult.failed_result("external_tool_not_found")
        
        except subprocess.TimeoutExpired:
            logger.error(f"[{context.molecule_id}] External tool timed out")
            return ExternalResult.timeout_result(self.executor.timeout if self.executor else 0)
        
        except Exception as e:
            logger.error(f"[{context.molecule_id}] External execution error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ExternalResult.failed_result(f"external_execution_failed: {e}")
    
    def _init_executor(self, executable: str, timeout: Optional[int] = None, cwd: Optional[str] = None):
        """初始化执行器"""
        self.executor = ExternalExecutor(
            executable=executable,
            timeout_seconds=timeout or self.config.get("timeout_seconds", 300),
            cwd=cwd,
        )
