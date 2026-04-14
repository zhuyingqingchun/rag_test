# 公共工具模块
import os
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from config import Config, get_config


def log(message: str, level: str = "INFO") -> None:
    """打印统一格式的日志
    
    Args:
        message: 日志消息
        level: 日志级别
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def check_port(port: int, host: str = "127.0.0.1") -> bool:
    """检查端口是否被占用
    
    Args:
        port: 端口号
        host: 主机地址
        
    Returns:
        True表示端口被占用，False表示端口可用
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return False
    except OSError:
        return True


def check_process_exists(pid: int) -> bool:
    """检查进程是否存在
    
    Args:
        pid: 进程ID
        
    Returns:
        True表示进程存在，False表示进程不存在
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def create_directory(dir_path: str) -> bool:
    """创建目录
    
    Args:
        dir_path: 目录路径
        
    Returns:
        True表示创建成功，False表示创建失败
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        log(f"创建目录失败: {dir_path}, 错误: {e}", "ERROR")
        return False


def write_pid_file(pid: int, pid_file: str) -> bool:
    """写入PID文件
    
    Args:
        pid: 进程ID
        pid_file: PID文件路径
        
    Returns:
        True表示写入成功，False表示写入失败
    """
    try:
        with open(pid_file, 'w') as f:
            f.write(str(pid))
        return True
    except OSError as e:
        log(f"写入PID文件失败: {pid_file}, 错误: {e}", "ERROR")
        return False


def read_pid_file(pid_file: str) -> Optional[int]:
    """读取PID文件
    
    Args:
        pid_file: PID文件路径
        
    Returns:
        进程ID，如果文件不存在或读取失败返回None
    """
    try:
        with open(pid_file, 'r') as f:
            return int(f.read().strip())
    except (OSError, ValueError) as e:
        log(f"读取PID文件失败: {pid_file}, 错误: {e}", "ERROR")
        return None


def delete_pid_file(pid_file: str) -> bool:
    """删除PID文件
    
    Args:
        pid_file: PID文件路径
        
    Returns:
        True表示删除成功，False表示删除失败
    """
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
        return True
    except OSError as e:
        log(f"删除PID文件失败: {pid_file}, 错误: {e}", "ERROR")
        return False


def generate_start_command(config: Config) -> list:
    """生成启动命令行
    
    Args:
        config: 配置对象
        
    Returns:
        启动命令列表
    """
    cmd = [
        config.python_bin,
        "-m", "vllm.entrypoints.openai.api_server",
        "--host", config.host,
        "--port", str(config.port),
        "--model", config.model_path,
        "--served-model-name", config.served_model_name,
        "--dtype", config.dtype,
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
        "--tensor-parallel-size", str(config.tensor_parallel_size),
    ]
    
    if config.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    return cmd


def health_check(base_url: str, timeout: int = 5) -> bool:
    """发起健康检查请求
    
    Args:
        base_url: 基础URL (应包含 /v1 路径)
        timeout: 超时时间（秒）
        
    Returns:
        True表示健康，False表示不健康
    """
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(f"{base_url}/models", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_for_server(base_url: str, max_retries: int = 30, retry_interval: int = 2) -> bool:
    """等待服务器启动完成
    
    Args:
        base_url: 基础URL
        max_retries: 最大重试次数
        retry_interval: 重试间隔（秒）
        
    Returns:
        True表示服务器启动成功，False表示启动失败
    """
    for i in range(max_retries):
        if health_check(base_url):
            log(f"服务器启动成功 (尝试 {i + 1}/{max_retries})")
            return True
        log(f"等待服务器启动中... (尝试 {i + 1}/{max_retries})")
        time.sleep(retry_interval)
    
    log("服务器启动超时", "ERROR")
    return False


def stop_process_gracefully(pid: int, timeout: int = 10) -> bool:
    """优雅地停止进程
    
    Args:
        pid: 进程ID
        timeout: 等待超时时间（秒）
        
    Returns:
        True表示成功停止，False表示停止失败
    """
    if not check_process_exists(pid):
        log(f"进程 {pid} 不存在")
        return True
    
    log(f"正在优雅地停止进程 {pid}...")
    
    # 发送 SIGTERM 信号
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        log(f"进程 {pid} 已经不存在")
        return True
    
    # 等待进程退出
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not check_process_exists(pid):
            log(f"进程 {pid} 已成功停止")
            return True
        time.sleep(0.5)
    
    # 如果超时仍未退出，强制终止
    log(f"进程 {pid} 未在超时时间内退出，强制终止...")
    try:
        os.kill(pid, signal.SIGKILL)
        log(f"进程 {pid} 已强制终止")
        return True
    except ProcessLookupError:
        log(f"进程 {pid} 已经不存在")
        return True
