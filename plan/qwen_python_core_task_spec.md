# Qwen 模型服务化任务说明（Python 核心脚本版）

## 1. 任务背景

当前已经完成以下工作：

- 模型已成功下载。
- 模型路径为：`/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- 缓存路径为：`/mnt/PRO6000_disk/modelscope_cache`
- 关键文件检查通过：
  - `config.json`
  - `tokenizer_config.json`
  - `generation_config.json`
- 可用模型别名：
  - `next80b_fp8 -> Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
  - `qwen235b_fp8 -> Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`

但当前环境存在一个问题：

```text
CondaError: Run 'conda init' before 'conda activate'
```

这说明不能依赖 `bash + conda activate` 作为主流程，因为不同机器、不同用户 shell、不同终端初始化状态下，`conda activate` 很容易失败。

因此，本任务要求：

## 2. 总体目标

构建一套**以 Python 脚本为核心**的 Qwen 模型服务化方案，使模型能够在加载后持续提供 API 服务，后续测试脚本可以通过 OpenAI 兼容接口进行调用，而不需要每次重新手工加载模型。

这里的“核心脚本是 Python 脚本”含义如下：

1. 服务启动逻辑必须由 Python 脚本负责。
2. 服务停止逻辑必须由 Python 脚本负责。
3. 服务状态检查逻辑必须由 Python 脚本负责。
4. 接口测试逻辑必须由 Python 脚本负责。
5. 配置读取逻辑必须由 Python 脚本负责。
6. 即使保留 shell 脚本，也只能是薄封装，不能承载核心业务逻辑。
7. 在最终交付中，**用户直接执行 Python 脚本就可以完成主要操作**。

---

## 3. 目标交付物

需要生成一个完整的工程目录，至少包含如下文件：

```text
qwen_service/
├── config.py
├── service_config.json
├── run_server.py
├── stop_server.py
├── status_server.py
├── test_openai_client.py
├── utils.py
├── README.md
└── logs/
```

如果实现时需要补充其他 Python 文件，也可以增加，但核心入口必须是以下四个：

- `run_server.py`
- `stop_server.py`
- `status_server.py`
- `test_openai_client.py`

---

## 4. 功能要求

### 4.1 启动脚本：`run_server.py`

该脚本负责启动模型服务，是整个系统最重要的入口之一。

#### 必须实现的功能

1. 从配置文件读取模型路径、服务端口、host、GPU 配置、日志路径、PID 文件路径等参数。
2. 检查模型目录是否存在。
3. 检查关键文件是否存在。
4. 检查端口是否已被占用。
5. 检查是否已有同名服务正在运行。
6. 以 Python 方式启动推理服务，而不是依赖 `conda activate`。
7. 将服务进程 PID 写入文件。
8. 将标准输出和标准错误写入日志文件。
9. 启动后输出访问地址，例如：
   - `http://127.0.0.1:8000/v1`
10. 如果启动失败，要明确打印失败原因。

#### 强制要求

- 不要在核心逻辑里使用 `source ~/.bashrc && conda activate ...` 这类写法。
- 应使用**指定 Python 解释器路径**的方式启动服务。
- 例如配置中可以写：
  - `/home/xxx/miniconda3/envs/swdtorch12/bin/python`
- 或者使用当前解释器 `sys.executable`，但必须可配置。

#### 推荐实现方式

可以通过 `subprocess.Popen(...)` 调用类似下面的命令：

```python
[python_bin, "-m", "vllm.entrypoints.openai.api_server", ...]
```

或者调用你当前已经验证可运行的服务入口，只要满足 OpenAI 兼容接口即可。

#### 推荐支持的参数

- `--host`
- `--port`
- `--model`
- `--served-model-name`
- `--dtype`
- `--gpu-memory-utilization`
- `--max-model-len`
- `--tensor-parallel-size`
- `--trust-remote-code`

参数名称允许根据真实服务框架调整，但整体能力要保留。

---

### 4.2 停止脚本：`stop_server.py`

该脚本负责安全停止服务。

#### 必须实现的功能

1. 读取 PID 文件。
2. 判断 PID 是否仍然存在。
3. 如果进程存在，则优先尝试正常终止。
4. 如果若干秒后仍未退出，再进行强制终止。
5. 删除 PID 文件。
6. 输出明确日志，例如：
   - 找到进程
   - 正在停止
   - 已停止
   - PID 文件已清理

#### 注意事项

- 不要简单粗暴地只执行一次 kill。
- 要考虑 PID 文件存在但进程已不存在的情况。
- 要考虑服务已经退出但 PID 文件残留的情况。

---

### 4.3 状态检查脚本：`status_server.py`

该脚本负责检查服务是否健康运行。

#### 必须实现的功能

1. 检查 PID 文件是否存在。
2. 检查对应进程是否存在。
3. 检查端口是否已监听。
4. 检查 OpenAI 兼容接口是否可访问。
5. 打印当前服务状态：
   - 未运行
   - 进程存在但接口异常
   - 正常运行
6. 打印服务关键信息：
   - PID
   - host
   - port
   - base_url
   - model_name
   - 日志文件位置

#### 推荐增加的健康检查

可以请求：

- `/v1/models`

如果返回成功，则视为服务可用。

---

### 4.4 测试脚本：`test_openai_client.py`

该脚本用于验证服务是否真的可调用。

#### 必须实现的功能

1. 使用 OpenAI Python SDK 或兼容方式发起请求。
2. 从配置文件读取 `base_url` 和 `api_key`。
3. 支持一个最小测试请求。
4. 支持打印完整响应。
5. 支持设置模型名。
6. 支持设置简单 prompt。

#### 推荐测试内容

例如发送：

```text
请用一句话介绍你自己。
```

或者：

```text
写一个三点列表，说明 RAG 系统的核心组成。
```

#### 目标

运行该脚本后，应能证明：

- 服务已经真正启动
- OpenAI 接口可访问
- 模型可以返回结果

---

### 4.5 配置文件：`service_config.json`

必须使用配置文件统一管理参数，不能把所有路径写死在代码里。

#### 至少要包含以下字段

```json
{
  "python_bin": "/path/to/python",
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

字段可以扩展，但不能少于这类核心配置。

---

### 4.6 配置加载模块：`config.py`

该模块负责：

1. 读取 JSON 配置文件。
2. 补充默认值。
3. 校验配置项合法性。
4. 统一向其他 Python 脚本提供配置访问接口。

#### 例如应校验

- `model_path` 是否存在
- `python_bin` 是否存在
- `port` 是否为合法端口
- `log_file` 的目录是否可创建
- `pid_file` 的目录是否有效

---

### 4.7 公共工具模块：`utils.py`

建议把下面这些通用能力封装到工具模块中：

- 检查端口是否占用
- 检查进程是否存在
- 创建目录
- 写入 PID 文件
- 读取 PID 文件
- 删除 PID 文件
- 发起健康检查 HTTP 请求
- 打印统一格式日志
- 生成启动命令行

这样可以减少多个脚本之间的重复代码。

---

## 5. 环境与实现约束

### 5.1 不允许把 conda activate 作为核心依赖

由于当前已经明确报错：

```text
CondaError: Run 'conda init' before 'conda activate'
```

因此实现时必须遵守：

- 不依赖 shell 初始化
- 不依赖 `.bashrc`
- 不依赖 `conda init`
- 不要求用户先进入某种交互 shell

正确方式是：

- 直接指定 Python 解释器路径
- 或明确指定虚拟环境中的 Python 可执行文件

---

### 5.2 优先保证“可运行”而不是“花哨”

低级 AI 在生成代码时，重点不是把代码写得很复杂，而是保证：

1. 启动成功
2. 状态可查
3. 停止可靠
4. 测试可通
5. 日志清楚

---

### 5.3 所有主要命令都必须能单独运行

目标是让用户可以直接执行：

```bash
python run_server.py
python status_server.py
python test_openai_client.py
python stop_server.py
```

即可完成服务的启动、检查、测试和停止。

---

## 6. README 要求

必须生成一个 `README.md`，内容尽量详细，至少包括：

### 6.1 项目简介

说明这是一个基于 Python 核心脚本的 Qwen 模型服务化工具，用于将本地模型启动为 OpenAI 兼容接口服务。

### 6.2 目录结构说明

逐个解释每个文件的用途。

### 6.3 前置依赖

例如：

- Python 版本
- `pip install vllm openai requests psutil`
- 其他必要依赖

### 6.4 配置方法

解释 `service_config.json` 每个字段的意义。

### 6.5 使用步骤

至少包括：

1. 修改配置文件
2. 启动服务
3. 查询状态
4. 测试接口
5. 停止服务

### 6.6 常见问题

必须明确解释：

#### 为什么不用 conda activate？

答：因为不同环境下 shell 初始化状态不一致，容易出现 `CondaError: Run 'conda init' before 'conda activate'`，因此采用“直接指定 Python 解释器”的方案更稳定。

#### 如果服务启动失败怎么办？

答：查看日志文件和状态检查结果。

#### 如果 PID 文件残留怎么办？

答：`stop_server.py` 或 `status_server.py` 需要能够识别并清理无效 PID 文件。

---

## 7. 验收标准

生成的项目必须满足以下验收条件：

### 7.1 基本功能验收

- [ ] `python run_server.py` 能启动服务
- [ ] `python status_server.py` 能看到服务状态
- [ ] `python test_openai_client.py` 能成功请求模型
- [ ] `python stop_server.py` 能停止服务

### 7.2 工程质量验收

- [ ] 配置独立管理
- [ ] 日志输出清晰
- [ ] PID 管理完整
- [ ] 代码结构清楚
- [ ] 不依赖 `conda activate`

### 7.3 错误处理验收

- [ ] 模型路径不存在时给出明确报错
- [ ] 端口冲突时给出明确报错
- [ ] 依赖缺失时给出明确报错
- [ ] 服务未启动时测试脚本能提示用户先启动服务

---

## 8. 编码风格要求

要求低级 AI 生成的代码尽量符合以下规范：

1. 使用 Python 3.10+ 语法。
2. 适当添加类型标注。
3. 关键函数写清晰注释。
4. 异常处理不要省略。
5. 尽量不要把大量逻辑全部塞进 `main()`。
6. 代码要可读、可维护。
7. 日志打印尽量统一风格。

---

## 9. 推荐实现思路

为了帮助低级 AI 更稳定地产出代码，建议按照如下步骤实现：

### 第一步：先写配置加载模块

完成：

- `config.py`
- `service_config.json`

### 第二步：再写工具模块

完成：

- `utils.py`

### 第三步：实现状态检查逻辑

完成：

- `status_server.py`

因为状态检查比较基础，适合作为其他脚本复用。

### 第四步：实现启动逻辑

完成：

- `run_server.py`

### 第五步：实现停止逻辑

完成：

- `stop_server.py`

### 第六步：实现客户端测试逻辑

完成：

- `test_openai_client.py`

### 第七步：补 README

完成：

- `README.md`

---

## 10. 最终给低级 AI 的任务指令（可直接复制）

下面这段话是可以直接发给低级 AI 的：

---

请帮我生成一个完整的 Python 工程，用于将本地 Qwen 模型启动为 OpenAI 兼容接口服务。

已知条件如下：

- 模型路径：`/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- 当前不希望依赖 `conda activate`
- 当前环境出现过错误：`CondaError: Run 'conda init' before 'conda activate'`
- 因此整个系统必须以 **Python 脚本作为核心入口**

请生成以下文件：

- `config.py`
- `service_config.json`
- `utils.py`
- `run_server.py`
- `stop_server.py`
- `status_server.py`
- `test_openai_client.py`
- `README.md`

要求如下：

1. 所有核心操作都必须通过 Python 脚本完成。
2. 启动脚本必须读取配置文件，并通过指定 Python 解释器的方式启动 vLLM OpenAI API 服务。
3. 停止脚本必须支持读取 PID 并优雅停止进程。
4. 状态脚本必须检查 PID、端口和 `/v1/models` 接口健康状态。
5. 测试脚本必须使用 OpenAI SDK 发起一次最小请求。
6. 代码要有异常处理、日志输出、配置校验。
7. 不要依赖 `conda init`、`.bashrc`、`conda activate`。
8. 最终应支持直接运行：
   - `python run_server.py`
   - `python status_server.py`
   - `python test_openai_client.py`
   - `python stop_server.py`

请直接输出完整代码内容，不要只给思路。

---

## 11. 补充说明

如果后续还需要继续扩展，可以在这个 Python 核心脚本工程基础上继续加入：

- 自动重启功能
- 模型切换功能
- 多模型注册功能
- Web 管理页面
- Benchmark 脚本
- 并发压测脚本
- 健康巡检脚本

但当前第一阶段目标不要扩得太大，优先把“单模型、单服务、稳定启动、稳定调用”做好。

