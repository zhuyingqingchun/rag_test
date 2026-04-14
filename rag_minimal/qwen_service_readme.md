# Qwen 模型服务系统

## 系统概述

Qwen 模型服务系统是一个基于 vLLM 的 OpenAI 兼容 API 服务，提供高效的 Qwen 模型推理能力。

## 目录结构

```
qwen_service/
├── run_server.py        # 启动服务脚本
├── stop_server.py       # 停止服务脚本
├── status_server.py     # 状态检查脚本
├── test_openai_client.py # OpenAI 客户端测试
├── self_check.py        # 自检脚本
├── config.py            # 配置管理模块
├── utils.py             # 工具函数模块
├── service_config.json  # 服务配置文件
├── README.md            # 本文档
└── logs/                # 日志目录
```

## 安装依赖

```bash
conda activate swdtorch12
pip install vllm openai requests
```

## 配置说明

编辑 `service_config.json`：

```json
{
  "python_bin": "/home/a/miniconda3/envs/swdtorch12/bin/python",
  "model_path": "/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
  "served_model_name": "next80b_fp8",
  "host": "127.0.0.1",
  "port": 8000,
  "api_key": "EMPTY",
  "pid_file": "./qwen_service.pid",
  "log_file": "./logs/qwen_service.log",
  "gpu_memory_utilization": 0.90,
  "tensor_parallel_size": 1,
  "max_model_len": 8192,
  "dtype": "auto",
  "trust_remote_code": true
}
```

## 快速开始

### 1. 自检

```bash
python self_check.py
```

### 2. 启动服务

```bash
python run_server.py
```

### 3. 检查状态

```bash
python status_server.py
```

### 4. 测试 API

```bash
python test_openai_client.py
```

### 5. 停止服务

```bash
python stop_server.py
```

## API 端点

- `GET /v1/models` - 列出可用模型
- `POST /v1/chat/completions` - 聊天补全
- `GET /health` - 健康检查
- `GET /v1/models/{model_id}` - 获取模型信息

## 常见问题

### 1. 代理问题

如果系统设置了代理环境变量，可能导致本地回环地址请求失败。解决方案：

```bash
unset http_proxy https_proxy ALL_PROXY all_proxy
```

或在代码中设置：

```python
session = requests.Session()
session.trust_env = False
```

### 2. 端口被占用

检查端口占用：

```bash
lsof -i :8000
```

或修改配置文件中的 port。

### 3. 模型加载失败

确保模型路径正确且包含必要文件：
- config.json
- tokenizer_config.json
- generation_config.json

## 技术栈

- **vLLM**：高性能 LLM 推理引擎
- **OpenAI 兼容 API**：标准 API 接口
- **Python 3.11+**：核心实现语言
