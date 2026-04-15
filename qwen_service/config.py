# 配置加载模块
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "service_config.json"):
        """初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        base_dir = Path(__file__).resolve().parent
        self.config_path = str((base_dir / config_path).resolve()) if not os.path.isabs(config_path) else config_path
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置项合法性"""
        required_keys = [
            "python_bin", "model_path", "served_model_name", 
            "host", "port", "api_key", "pid_file", "log_file"
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置项缺失: {key}")
        
        # 检查模型路径是否存在
        model_path = self.config["model_path"]
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")
        
        # 检查python解释器是否存在
        python_bin = self.config["python_bin"]
        if not os.path.exists(python_bin):
            raise ValueError(f"Python解释器不存在: {python_bin}")
        
        # 检查端口是否合法
        port = self.config["port"]
        if not (1 <= port <= 65535):
            raise ValueError(f"端口号不合法: {port}")
        
        # 检查日志目录是否可创建
        log_file = self.config["log_file"]
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"无法创建日志目录: {log_dir}, 错误: {e}")
        
        # 检查PID文件目录是否有效
        pid_file = self.config["pid_file"]
        pid_dir = os.path.dirname(pid_file)
        if pid_dir and not os.path.exists(pid_dir):
            try:
                os.makedirs(pid_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"无法创建PID文件目录: {pid_dir}, 错误: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置项名称
            default: 默认值
            
        Returns:
            配置项值
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """支持字典方式访问"""
        return self.config[key]
    
    def __getattr__(self, key: str) -> Any:
        """支持属性方式访问"""
        if key in self.config:
            return self.config[key]
        raise AttributeError(f"'Config' object has no attribute '{key}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self.config.copy()


# 全局配置实例
_config_instance: Optional[Config] = None


def get_config(config_path: str = "service_config.json") -> Config:
    """获取全局配置实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: str = "service_config.json") -> Config:
    """重新加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config实例
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
