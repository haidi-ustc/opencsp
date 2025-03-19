# opencsp/utils/logger.py
import logging
import os
import sys
from typing import Optional, Dict, Any

class Logger:
    """
    日志工具类，用于记录系统日志
    """
    
    def __init__(self, name: str = 'opencsp', level: int = logging.INFO, 
                 log_file: Optional[str] = None, console: bool = True):
        """
        初始化日志工具
        
        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径
            console: 是否输出到控制台
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 清除已有的处理器
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加文件处理器
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        # 添加控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
    def debug(self, message: str) -> None:
        """记录调试信息"""
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """记录一般信息"""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """记录警告信息"""
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """记录错误信息"""
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """记录严重错误信息"""
        self.logger.critical(message)
        
    def log_dict(self, data: Dict[str, Any], level: int = logging.INFO) -> None:
        """
        记录字典数据
        
        Args:
            data: 要记录的字典数据
            level: 日志级别
        """
        import json
        message = json.dumps(data, indent=2)
        self.logger.log(level, f"Data: \n{message}")
        
    def log_optimizer_state(self, optimizer_state: Dict[str, Any]) -> None:
        """
        记录优化器状态
        
        Args:
            optimizer_state: 优化器状态字典
        """
        iteration = optimizer_state.get('iteration', optimizer_state.get('generation', 0))
        best_fitness = optimizer_state.get('best_fitness')
        best_energy = optimizer_state.get('best_energy')
        evaluations = optimizer_state.get('evaluations')
        
        self.info(f"Optimization step {iteration}: " +
                 f"Best fitness = {best_fitness:.6f}, " +
                 f"Best energy = {best_energy:.6f}, " +
                 f"Evaluations = {evaluations}")
