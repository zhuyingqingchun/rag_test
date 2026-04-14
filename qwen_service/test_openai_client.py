#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本
用于测试Qwen模型服务是否可调用
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("openai 未安装，请运行: pip install openai")
    sys.exit(1)

from config import get_config
from utils import health_check, log


def test_service(config):
    """测试服务
    
    Args:
        config: 配置对象
    """
    base_url = f"http://{config.host}:{config.port}/v1"
    api_key = config.api_key
    model_name = config.served_model_name
    
    log("=" * 60)
    log("Qwen 模型服务测试")
    log("=" * 60)
    
    # 检查服务是否可用
    if not health_check(base_url):
        log("服务不可用，请先启动服务", "ERROR")
        return 1
    
    log(f"服务可用: {base_url}")
    
    # 创建客户端
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # 测试请求
    prompts = [
        "请用一句话介绍你自己。",
        "写一个三点列表，说明 RAG 系统的核心组成。"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        log(f"\n测试请求 {i}:")
        log(f"Prompt: {prompt}")
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            # 打印响应
            content = response.choices[0].message.content
            log(f"Response: {content}")
            
            # 打印使用情况
            if response.usage:
                log(f"Token 使用: {response.usage}")
            
            log(f"测试请求 {i} 成功")
            
        except Exception as e:
            log(f"测试请求 {i} 失败: {e}", "ERROR")
            return 1
    
    log("\n" + "=" * 60)
    log("所有测试通过!")
    log("=" * 60)
    
    return 0


def main():
    """主函数"""
    try:
        config = get_config()
        return test_service(config)
    except Exception as e:
        log(f"测试服务时发生错误: {e}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
