"""
异步任务队列模块
支持 multiprocessing.Queue 和 Redis 两种后端
"""

import json
import pickle
import time
from typing import Any, Dict, Optional, Union
from multiprocessing import Queue as MPQueue, Manager
from contextlib import contextmanager

from loguru import logger

# 可选的 Redis 支持
try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, only multiprocessing backend supported")


# 全局 Manager 实例（用于共享队列）
_manager = None

def get_manager():
    """获取全局 Manager 实例"""
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager


class TaskQueue:
    """
    异步任务队列
    
    支持两种后端：
    - multiprocessing: 单机多核，适合单机部署（使用 Manager().Queue() 实现共享）
    - redis: 分布式，适合多机集群
    """
    
    def __init__(
        self,
        backend: str = "mp",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        queue_name: str = "qcgem_tasks",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        mp_queue: Optional[MPQueue] = None,  # 允许传入共享队列
    ):
        """
        初始化任务队列
        
        Args:
            backend: 后端类型 ("mp" 或 "redis")
            redis_host: Redis 主机地址
            redis_port: Redis 端口
            redis_db: Redis 数据库编号
            redis_password: Redis 密码
            queue_name: 队列名称
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.backend = backend.lower()
        self.queue_name = queue_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if self.backend == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis backend requires 'redis' package")
            self._init_redis(redis_host, redis_port, redis_db, redis_password)
        elif self.backend == "mp":
            self._init_mp(mp_queue)  # 传入共享队列
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        logger.info(f"TaskQueue initialized with {backend} backend")
    
    def _init_redis(
        self,
        host: str,
        port: int,
        db: int,
        password: Optional[str],
    ):
        """初始化 Redis 连接"""
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # 我们使用 pickle 序列化
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
        
        # 测试连接
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _init_mp(self, mp_queue: Optional[MPQueue] = None):
        """初始化 multiprocessing 队列（使用 Manager 实现共享）"""
        if mp_queue is not None:
            # 使用传入的共享队列
            self.mp_queue = mp_queue
            self._owns_queue = False
            logger.info("Using shared multiprocessing queue")
        else:
            # 创建新的共享队列
            manager = get_manager()
            self.mp_queue = manager.Queue()
            self._owns_queue = True
            logger.info("Created shared multiprocessing queue via Manager")
    
    def put(self, task: Dict[str, Any], block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        将任务放入队列
        
        Args:
            task: 任务字典，格式：{"molecule_id": str, "fchk_path": str, "metadata": dict}
            block: 是否阻塞等待
            timeout: 超时时间（秒）
            
        Returns:
            是否成功放入队列
        """
        # 验证任务格式（允许 poison pill）
        if not isinstance(task, dict):
            raise ValueError("Task must be a dictionary")
        if "molecule_id" not in task and "__poison_pill__" not in task:
            raise ValueError("Task must contain 'molecule_id'")
        
        for attempt in range(self.max_retries):
            try:
                if self.backend == "redis":
                    # 使用 pickle 序列化
                    serialized = pickle.dumps(task)
                    self.redis_client.lpush(self.queue_name, serialized)
                else:
                    self.mp_queue.put(task, block=block, timeout=timeout)
                
                logger.debug(f"Task {task.get('molecule_id', 'poison_pill')} queued")
                return True
                
            except Exception as e:
                logger.warning(f"Put attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to put task after {self.max_retries} attempts")
                    raise
        
        return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        从队列获取任务
        
        Args:
            block: 是否阻塞等待
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            任务字典，或 None（如果超时）
        """
        for attempt in range(self.max_retries):
            try:
                if self.backend == "redis":
                    # 使用 brpop 阻塞获取
                    result = self.redis_client.brpop(self.queue_name, timeout=timeout or 0)
                    if result is None:
                        return None
                    # result 是 (queue_name, data) 元组
                    _, serialized = result
                    task = pickle.loads(serialized)
                else:
                    try:
                        task = self.mp_queue.get(block=block, timeout=timeout)
                    except:
                        return None
                
                logger.debug(f"Got task {task.get('molecule_id', 'poison_pill')}")
                return task
                
            except Exception as e:
                logger.warning(f"Get attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to get task after {self.max_retries} attempts")
                    raise
        
        return None
    
    def task_done(self):
        """
        标记任务完成
        
        对于 multiprocessing 队列，调用 task_done()
        对于 Redis，不需要显式标记（brpop 已经移除元素）
        """
        if self.backend == "mp":
            try:
                self.mp_queue.task_done()
            except:
                pass
    
    def qsize(self) -> int:
        """获取队列大小"""
        try:
            if self.backend == "redis":
                return self.redis_client.llen(self.queue_name)
            else:
                return self.mp_queue.qsize()
        except:
            return 0
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self.qsize() == 0
    
    def clear(self):
        """清空队列"""
        try:
            if self.backend == "redis":
                self.redis_client.delete(self.queue_name)
            else:
                while not self.mp_queue.empty():
                    try:
                        self.mp_queue.get_nowait()
                    except:
                        break
            logger.info("Queue cleared")
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
    
    def close(self):
        """关闭队列连接"""
        try:
            if self.backend == "redis":
                self.redis_client.close()
            # Note: Manager().Queue() doesn't have close() method
            logger.info("Queue closed")
        except Exception as e:
            logger.error(f"Error closing queue: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
