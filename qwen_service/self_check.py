#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自检脚本
用于部署前检查系统环境和配置
"""

import os
import sys

try:
    import requests
except ImportError:
    print("错误: requests 未安装，请运行: pip install requests")
    sys.exit(1)

try:
    import vllm
except ImportError:
    print("错误: vllm 未安装，请运行: pip install vllm")
    sys.exit(1)

from config import get_config
from utils import check_port, health_check, log


def check_python_interpreter(config):
    """检查 Python 解释器"""
    print("-" * 60)
    print("1. Python 解释器检查")
    print("-" * 60)
    
    python_bin = config.python_bin
    if not os.path.exists(python_bin):
        log(f"Python解释器不存在: {python_bin}", "ERROR")
        return False
    
    log(f"Python解释器存在: {python_bin}")
    
    try:
        import subprocess
        result = subprocess.run(
            [python_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        log(f"Python版本: {result.stdout.strip()}")
    except Exception as e:
        log(f"检查Python版本失败: {e}", "WARNING")
    
    return True


def check_vllm(config):
    """检查 vllm 安装"""
    print("\n" + "-" * 60)
    print("2. vllm 模块检查")
    print("-" * 60)
    
    try:
        import vllm
        log(f"vllm 已安装 (版本: {vllm.__version__})")
        return True
    except ImportError as e:
        log(f"vllm 未安装或无法导入: {e}", "ERROR")
        return False


def check_model_path(config):
    """检查模型路径"""
    print("\n" + "-" * 60)
    print("3. 模型路径检查")
    print("-" * 60)
    
    model_path = config.model_path
    if not os.path.exists(model_path):
        log(f"模型路径不存在: {model_path}", "ERROR")
        return False
    
    log(f"模型路径存在: {model_path}")
    
    required_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
    all_exist = True
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            log(f"  ✓ {file}")
        else:
            log(f"  ✗ {file} 缺失", "WARNING")
            all_exist = False
    
    return all_exist


def check_port_status(config):
    """检查端口状态"""
    print("\n" + "-" * 60)
    print("4. 端口检查")
    print("-" * 60)
    
    port = config.port
    host = config.host
    
    if check_port(port, host):
        log(f"端口 {port} 已被占用")
        return True
    else:
        log(f"端口 {port} 可用")
        return True


def check_proxy_settings():
    """检查代理设置"""
    print("\n" + "-" * 60)
    print("5. 代理环境变量检查")
    print("-" * 60)
    
    proxy_vars = ["http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
    has_proxy = False
    
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            has_proxy = True
            log(f"发现代理变量 {var}: {value}")
    
    if has_proxy:
        log("注意: 系统设置了代理环境变量，但本地回环地址请求会被忽略代理", "WARNING")
    
    return True


def check_service_health(config):
    """检查服务健康状态"""
    print("\n" + "-" * 60)
    print("6. 服务健康检查")
    print("-" * 60)
    
    base_url = f"http://{config.host}:{config.port}/v1"
    
    if health_check(base_url):
        log(f"服务健康检查通过: {base_url}")
        return True
    else:
        log(f"服务健康检查失败: {base_url}", "WARNING")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Qwen 服务系统自检")
    print("=" * 60)
    
    try:
        config = get_config()
    except Exception as e:
        log(f"无法加载配置: {e}", "ERROR")
        return 1
    
    results = []
    
    results.append(("Python解释器", check_python_interpreter(config)))
    results.append(("vllm模块", check_vllm(config)))
    results.append(("模型路径", check_model_path(config)))
    results.append(("端口状态", check_port_status(config)))
    results.append(("代理设置", check_proxy_settings()))
    results.append(("服务健康", check_service_health(config)))
    
    print("\n" + "=" * 60)
    print("自检结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        log("所有检查通过！系统已准备就绪。")
        return 0
    else:
        log("部分检查失败，请根据上述提示进行修复。", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
