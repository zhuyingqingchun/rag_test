#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
停止脚本
用于安全停止Qwen模型服务
"""

import os
import sys

from config import get_config
from utils import (
    check_process_exists,
    delete_pid_file,
    log,
    read_pid_file,
    stop_process_gracefully,
)


def stop_service(config):
    """停止服务
    
    Args:
        config: 配置对象
    """
    pid_file = config.pid_file
    
    log("=" * 60)
    log("Qwen 模型服务停止")
    log("=" * 60)
    
    # 检查PID文件是否存在
    if not os.path.exists(pid_file):
        log("PID文件不存在，服务可能未运行或已停止")
        return 0
    
    # 读取PID
    pid = read_pid_file(pid_file)
    if pid is None:
        log("无法读取PID文件", "WARNING")
        delete_pid_file(pid_file)
        return 1
    
    log(f"找到服务 PID: {pid}")
    
    # 检查进程是否存在
    if not check_process_exists(pid):
        log(f"进程 {pid} 不存在，可能是已退出但PID文件未清理")
        delete_pid_file(pid_file)
        return 0
    
    log(f"进程 {pid} 正在运行，正在停止...")
    
    # 优雅地停止进程
    success = stop_process_gracefully(pid)
    
    if success:
        log("服务已停止")
        
        # 删除PID文件
        if delete_pid_file(pid_file):
            log("PID文件已清理")
        else:
            log("PID文件清理失败", "WARNING")
        
        log("=" * 60)
        return 0
    else:
        log("服务停止失败", "ERROR")
        log("=" * 60)
        return 1


def main():
    """主函数"""
    try:
        config = get_config()
        return stop_service(config)
    except Exception as e:
        log(f"停止服务时发生错误: {e}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
