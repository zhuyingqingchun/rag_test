# Qwen 模型服务化工具

基于 Python 核心脚本的 Qwen 模型服务化方案，用于将本地模型启动为 OpenAI 兼容接口服务。

## 1. 项目简介

这是一个以 Python 脚本为核心的 Qwen 模型服务化工具，具有以下特点：

- **纯 Python 实现**：所有核心功能由 Python 脚本实现，不依赖 `conda activate`
- **OpenAI 兼容接口**：提供标准的 OpenAI API 兼容接口
- **完整的生命周期管理**：支持启动、停止、状态检查和接口测试
- **配置化管理**：通过 JSON 配置文件统一管理服务参数
- **清晰的日志输出**：提供详细的启动、运行和错误日志

## 2. 目录结构

```
qwen_service/
├── config.py                 # 配置加载模块
├── service_config.json       # 配置文件
├── utils.py                  # 公共工具模块
├── run_server.py             # 启动脚本
├── stop_server.py            # 停止脚本
├── status_server.py          # 状态检查脚本
├── test_openai_client.py     # 测试脚本
├── README.md                 # 本文档
└── logs/                     # 日志目录
    └── qwen_service.log      # 服务日志
```

### 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 配置加载模块，负责读取和验证配置文件 |
| `service_config.json` | 配置文件，包含模型路径、端口、GPU等参数 |
| `utils.py` | 公共工具模块，提供PID管理、端口检查、健康检查等工具函数 |
| `run_server.py` | 启动脚本，负责启动模型服务 |
| `stop_server.py` | 停止脚本，负责优雅地停止服务 |
| `status_server.py` | 状态检查脚本，检查服务运行状态 |
| `test_openai_client.py` | 测试脚本，验证服务是否可调用 |
| `logs/` | 日志目录，存储服务运行日志 |

## 3. 前置依赖

### Python 版本
- Python 3.10+

### 必要的 Python 包
```bash
pip install vllm openai requests psutil
```

### 系统要求
- Linux 或 macOS
- 足够的 GPU 显存（根据模型大小）
- 至少 4GB 可用内存

## 4. 配置说明

配置文件 `service_config.json` 包含以下字段：

| 字段 | 说明 | 示例值 |
|------|------|--------|
| `python_bin` | Python 解释器路径 | `/home/a/miniconda3/envs/swdtorch12/bin/python` |
| `model_path` | 模型路径 | `/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` |
| `served_model_name` | 服务模型名称 | `next80b_fp8` |
| `host` | 服务主机地址 | `127.0.0.1` |
| `port` | 服务端口号 | `8000` |
| `api_key` | API 密钥 | `EMPTY` |
| `pid_file` | PID 文件路径 | `./qwen_service.pid` |
| `log_file` | 日志文件路径 | `./logs/qwen_service.log` |
| `gpu_memory_utilization` | GPU 内存利用率 | `0.90` |
| `tensor_parallel_size` | 张量并行大小 | `1` |
| `max_model_len` | 最大模型长度 | `8192` |
| `dtype` | 数据类型 | `auto` |
| `trust_remote_code` | 是否信任远程代码 | `true` |

## 5. 使用步骤

### 5.1 修改配置文件

首先编辑 `service_config.json` 文件，确保以下配置正确：

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

### 5.2 启动服务

```bash
cd qwen_service
python run_server.py
```

成功启动后，会显示类似以下信息：

```
============================================================
Qwen 模型服务启动
============================================================
[2026-04-14 12:00:00] [INFO] vllm 已安装
[2026-04-14 12:00:00] [INFO] requests 已安装
[2026-04-14 12:00:00] [INFO] 模型路径检查通过: /mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
[2026-04-14 12:00:00] [INFO] 正在启动服务...
[2026-04-14 12:00:00] [INFO] PID 已写入: ./qwen_service.pid (PID: 12345)
[2026-04-14 12:00:00] [INFO] 等待服务启动中... (10s)
[2026-04-14 12:00:00] [INFO] 服务启动成功
============================================================
服务启动成功!
访问地址: http://127.0.0.1:8000/v1
模型名称: next80b_fp8
日志文件: ./logs/qwen_service.log
PID文件: ./qwen_service.pid
============================================================
```

### 5.3 查询状态

```bash
python status_server.py
```

输出示例：

```
============================================================
Qwen 模型服务状态
============================================================
PID: 12345
Host: 127.0.0.1
Port: 8000
Base URL: http://127.0.0.1:8000/v1
Model Name: next80b_fp8
Log File: ./logs/qwen_service.log
------------------------------------------------------------
状态: 进程存在
状态: 端口被占用
状态: 接口正常
服务状态: 正常运行
============================================================
```

### 5.4 测试接口

```bash
python test_openai_client.py
```

输出示例：

```
============================================================
Qwen 模型服务测试
============================================================
[2026-04-14 12:00:00] [INFO] 服务可用: http://127.0.0.1:8000/v1

测试请求 1:
Prompt: 请用一句话介绍你自己。
Response: 我是一个AI助手，由Qwen开发，旨在回答问题、提供帮助和与用户进行对话。
Token 使用: CompletionUsage(completion_tokens=25, prompt_tokens=12, total_tokens=37)
测试请求 1 成功

测试请求 2:
Prompt: 写一个三点列表，说明 RAG 系统的核心组成。
Response: 1. **检索（Retrieval）**：从知识库中检索相关文档或信息。
2. **增强（Augmentation）**：将检索到的信息与用户查询结合，生成更丰富的上下文。
3. **生成（Generation）**：基于增强后的上下文，生成高质量的回答或内容。

测试请求 2 成功

============================================================
所有测试通过!
============================================================
```

### 5.5 停止服务

```bash
python stop_server.py
```

输出示例：

```
============================================================
Qwen 模型服务停止
============================================================
[2026-04-14 12:00:00] [INFO] 找到服务 PID: 12345
[2026-04-14 12:00:00] [INFO] 进程 12345 正在运行，正在停止...
[2026-04-14 12:00:00] [INFO] 正在优雅地停止进程 12345...
[2026-04-14 12:00:00] [INFO] 进程 12345 已成功停止
[2026-04-14 12:00:00] [INFO] 服务已停止
[2026-04-14 12:00:00] [INFO] PID文件已清理
============================================================
```

## 6. 常见问题

### 6.1 为什么不用 conda activate？

答：因为不同环境下 shell 初始化状态不一致，容易出现 `CondaError: Run 'conda init' before 'conda activate'`，因此采用"直接指定 Python 解释器"的方案更稳定。

### 6.2 服务启动失败怎么办？

答：请按以下步骤排查：

1. 查看日志文件：`cat logs/qwen_service.log`
2. 检查配置文件：`service_config.json`
3. 检查模型路径是否存在
4. 检查端口是否被占用
5. 检查 Python 解释器路径是否正确

### 6.3 PID 文件残留怎么办？

答：PID 文件残留通常表示服务异常退出。可以：

1. 运行 `python stop_server.py` 尝试清理
2. 或手动删除 PID 文件：`rm qwen_service.pid`

### 6.4 如何查看服务日志？

答：日志文件位置在配置文件中指定，默认为 `./logs/qwen_service.log`。

```bash
tail -f logs/qwen_service.log
```

### 6.5 如何修改服务端口？

答：编辑 `service_config.json` 文件，修改 `port` 字段：

```json
{
  "port": 8001
}
```

然后重启服务。

### 6.6 如何修改模型？

答：编辑 `service_config.json` 文件，修改 `model_path` 和 `served_model_name` 字段：

```json
{
  "model_path": "/path/to/new/model",
  "served_model_name": "new_model_name"
}
```

然后重启服务。

## 7. 技术细节

### 7.1 启动流程

1. 读取配置文件
2. 检查依赖和模型路径
3. 检查端口和进程状态
4. 生成启动命令
5. 启动 vLLM 服务进程
6. 写入 PID 文件
7. 等待服务就绪
8. 输出访问地址

### 7.2 停止流程

1. 读取 PID 文件
2. 检查进程是否存在
3. 发送 SIGTERM 信号
4. 等待进程退出
5. 如超时则发送 SIGKILL
6. 删除 PID 文件

### 7.3 健康检查

通过请求 `/v1/models` 接口检查服务是否可用。

## 8. 注意事项

- 启动服务前确保 GPU 显存充足
- 服务运行时不要手动删除 PID 文件
- 停止服务时使用 `stop_server.py` 而不是直接 kill
- 日志文件会持续增长，建议定期清理
- 修改配置后需要重启服务才能生效

## 9. 扩展功能

后续可以考虑添加：

- 自动重启功能
- 多模型注册功能
- Web 管理页面
- Benchmark 脚本
- 并发压测脚本
- 健康巡检脚本

## 10. 许可证

本项目采用 MIT 许可证。
