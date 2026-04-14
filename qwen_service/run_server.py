#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本
用于启动Qwen模型服务
"""

import os
import sys
import subprocess

from config import get_config
from utils import (
    check_port,
    check_process_exists,
    create_directory,
    generate_start_command,
    log,
    read_pid_file,
    write_pid_file,
)


def check_dependencies() -> bool:
    """检查依赖项"""
    try:
        import vllm
        log("vllm 已安装")
    except ImportError:
        log("vllm 未安装，请运行: pip install vllm", "ERROR")
        return False
    
    try:
        import requests
        log("requests 已安装")
    except ImportError:
        log("requests 未安装，请运行: pip install requests", "ERROR")
        return False
    
    return True


def check_model_path(config) -> bool:
    """检查模型路径"""
    if not os.path.exists(config.model_path):
        log(f"模型路径不存在: {config.model_path}", "ERROR")
        return False
    
    # 检查关键文件
    required_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
    for file in required_files:
        file_path = os.path.join(config.model_path, file)
        if not os.path.exists(file_path):
            log(f"关键文件不存在: {file_path}", "ERROR")
            return False
    
    log(f"模型路径检查通过: {config.model_path}")
    return True


def check_service_running(config) -> bool:
    """检查服务是否已在运行"""
    pid_file = config.pid_file
    
    if os.path.exists(pid_file):
        pid = read_pid_file(pid_file)
        if pid is not None and check_process_exists(pid):
            log(f"服务已在运行 (PID: {pid})", "ERROR")
            return True
    
    return False


def start_service(config) -> subprocess.Popen:
    """启动服务
    
    Args:
        config: 配置对象
        
    Returns:
        进程对象
    """
    # 生成启动命令
    cmd = generate_start_command(config)
    
    log(f"启动命令: {' '.join(cmd)}")
    
    # 创建日志目录
    create_directory(os.path.dirname(config.log_file))
    
    # 打开日志文件
    log_file = open(config.log_file, 'w', encoding='utf-8')
    
    # 启动进程
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )
    
    return process


def wait_for_startup(config, process, timeout=60) -> bool:
    """等待服务启动
    
    Args:
        config: 配置对象
        process: 进程对象
        timeout: 超时时间（秒）
        
    Returns:
        True表示启动成功，False表示启动失败
    """
    import time
    
    base_url = f"http://{config.host}:{config.port}/v1"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 检查进程是否还在运行
        if process.poll() is not None:
            log("服务进程已退出", "ERROR")
            return False
        
        # 检查端口是否已监听
        if not check_port(config.port, config.host):
            log("端口已监听，等待接口就绪...")
            
            # 检查接口是否可用
            import requests
            try:
                response = requests.get(f"{base_url}/v1/models", timeout=5)
                if response.status_code == 200:
                    log("服务启动成功")
                    return True
            except requests.RequestException:
                pass
        
        time.sleep(2)
        log(f"等待服务启动中... ({int(time.time() - start_time)}s)")
    
    log("服务启动超时", "ERROR")
    return False


def main():
    """主函数"""
    try:
        config = get_config()
        
        log("=" * 60)
        log("Qwen 模型服务启动")
        log("=" * 60)
        
        # 检查依赖
        if not check_dependencies():
            return 1
        
        # 检查模型路径
        if not check_model_path(config):
            return 1
        
        # 检查服务是否已在运行
        if check_service_running(config):
            return 1
        
        # 检查端口是否被占用
        if check_port(config.port, config.host):
            log(f"端口 {config.port} 已被占用", "ERROR")
            return 1
        
        # 启动服务
        log("正在启动服务...")
        process = start_service(config)
        
        # 写入PID文件
        write_pid_file(process.pid, config.pid_file)
        log(f"PID 已写入: {config.pid_file} (PID: {process.pid})")
        
        # 等待服务启动
        if not wait_for_startup(config, process):
            log("服务启动失败", "ERROR")
            # 清理PID文件
            delete_pid_file(config.pid_file)
            return 1
        
        # 启动成功
        base_url = f"http://{config.host}:{config.port}/v1"
        log("=" * 60)
        log("服务启动成功!")
        log(f"访问地址: {base_url}")
        log(f"模型名称: {config.served_model_name}")
        log(f"日志文件: {config.log_file}")
        log(f"PID文件: {config.pid_file}")
        log("=" * 60)
        
        return 0
        
    except Exception as e:
        log(f"启动服务时发生错误: {e}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
