#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态检查脚本
用于检查Qwen模型服务的运行状态
"""

import os
import sys

from config import get_config
from utils import (
    check_port,
    check_process_exists,
    delete_pid_file,
    health_check,
    log,
    read_pid_file,
)


def check_service_status() -> str:
    """检查服务状态
    
    Returns:
        服务状态字符串
    """
    config = get_config()
    
    pid_file = config.pid_file
    host = config.host
    port = config.port
    base_url = f"http://{host}:{port}/v1"
    
    # 检查PID文件是否存在
    if not os.path.exists(pid_file):
        return "未运行"
    
    # 读取PID
    pid = read_pid_file(pid_file)
    if pid is None:
        log("无法读取PID文件", "WARNING")
        return "未运行"
    
    # 检查进程是否存在
    process_exists = check_process_exists(pid)
    
    # 检查端口是否监听
    port_listening = check_port(port, host)
    
    # 检查OpenAI接口是否可访问
    interface_available = health_check(base_url)
    
    # 打印服务状态
    print("=" * 60)
    print("Qwen 模型服务状态")
    print("=" * 60)
    print(f"PID: {pid}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Base URL: {base_url}")
    print(f"Model Name: {config.served_model_name}")
    print(f"Log File: {config.log_file}")
    print("-" * 60)
    
    if not process_exists:
        print("状态: 进程不存在")
        print("建议: PID文件存在但进程已退出，建议清理PID文件")
        delete_pid_file(pid_file)
        return "未运行"
    
    if port_listening:
        print("状态: 端口被占用")
    else:
        print("状态: 端口未监听")
    
    if interface_available:
        print("状态: 接口正常")
        print("服务状态: 正常运行")
        print("=" * 60)
        return "正常运行"
    else:
        print("状态: 接口异常")
        print("服务状态: 进程存在但接口异常")
        print("=" * 60)
        return "进程存在但接口异常"


def main():
    """主函数"""
    try:
        status = check_service_status()
        print(f"\n最终状态: {status}")
        return 0 if status == "正常运行" else 1
    except Exception as e:
        log(f"检查状态时发生错误: {e}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
