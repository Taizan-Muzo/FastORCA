"""
External Executor
统一的 subprocess 执行器，用于运行外部工具
"""

import subprocess
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class ExecutionResult:
    """执行结果"""
    returncode: int
    stdout: str
    stderr: str
    execution_time_seconds: float
    command: str
    timed_out: bool = False


class ExternalExecutor:
    """
    外部工具执行器
    
    统一的 subprocess 封装，提供：
    - timeout 控制
    - stdout/stderr 捕获
    - 工作目录管理
    - 环境变量设置
    """
    
    DEFAULT_TIMEOUT = 300  # 5 分钟
    
    def __init__(
        self,
        executable: str,
        timeout_seconds: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.executable = executable
        self.timeout = timeout_seconds or self.DEFAULT_TIMEOUT
        self.env = env
        self.cwd = cwd
    
    def check_executable(self) -> bool:
        """检查可执行文件是否存在"""
        import shutil
        return shutil.which(self.executable) is not None
    
    def execute(
        self,
        args: List[str],
        input_text: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        """
        执行外部命令
        
        Args:
            args: 命令行参数（不包括 executable）
            input_text: 通过 stdin 传递的输入
            cwd: 工作目录（覆盖默认）
            
        Returns:
            ExecutionResult
            
        Raises:
            FileNotFoundError: 可执行文件不存在
            subprocess.TimeoutExpired: 执行超时
        """
        cmd = [self.executable] + args
        cmd_str = " ".join(shlex.quote(str(a)) for a in cmd)
        
        work_dir = cwd or self.cwd
        
        logger.debug(f"Executing: {cmd_str}")
        logger.debug(f"Working directory: {work_dir}")
        logger.debug(f"Timeout: {self.timeout}s")
        
        import time
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=work_dir,
                env=self.env,
            )
            
            elapsed = time.time() - start_time
            
            logger.debug(f"Command completed in {elapsed:.3f}s, returncode={result.returncode}")
            
            if result.stderr:
                logger.debug(f"stderr: {result.stderr[:500]}")
            
            return ExecutionResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_seconds=elapsed,
                command=cmd_str,
                timed_out=False,
            )
            
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            logger.error(f"Command timed out after {elapsed:.3f}s")
            
            return ExecutionResult(
                returncode=-1,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
                execution_time_seconds=elapsed,
                command=cmd_str,
                timed_out=True,
            )
    
    def execute_script(
        self,
        script_path: Path,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        """
        执行脚本文件
        
        Args:
            script_path: 脚本文件路径
            args: 附加参数
            cwd: 工作目录
        """
        all_args = [str(script_path)] + (args or [])
        return self.execute(all_args, cwd=cwd)
