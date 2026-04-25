# Harness Guideline

从零手写一个 AI Agent Harness 的完整学习记录。

## 什么是 Agent Harness？

Agent Harness 是包裹在 LLM 外面的执行外壳，负责：

- 组装 Prompt / messages 数组
- 调用 LLM API
- 注册并分发工具（Tool）
- 驱动 Agent Loop（while / for 循环）
- 管理停止条件、Token 预算、错误恢复

本仓库通过 11 个渐进式练习 + 2 个完整版本，把上面每一块拆开讲清楚、跑明白。

---

## 文件结构

```
.
├── learn-harness.md              # 原始任务书（TRACE 格式）
├── agent-harness-learning.md     # 知识点文档（17 个知识点，五大模块）
├── full-conversation.md          # 完整学习对话记录
└── harness-practice/
    ├── step1_bare_call.py        # 练习 A-1：裸 API 调用
    ├── step2_see_tool_call.py    # 练习 A-2：注册工具，观察 tool_calls 结构
    ├── step3_full_loop.py        # 练习 A-3：完整 Agent Loop
    ├── step4_three_tools_plan.py # 练习 B-1：三工具声明
    ├── step5_dispatcher.py       # 练习 B-2：Dispatcher + 错误回传
    ├── step6_error_recovery.py   # 练习 B-3：错误恢复演示
    ├── step7_token_monitor.py    # 练习 C-1：Token 监控
    ├── step8_hitl_complete.py    # 练习 C-2：Human-in-the-loop
    ├── step9_checkpoint.py       # 练习 D-1：检查点 & 崩溃恢复
    ├── step10_sandbox.py         # 练习 D-2：沙箱路径隔离
    ├── step11_eval.py            # 练习 D-3：LLM-as-Judge Eval
    ├── exercise_e_annotation.md  # 练习 E：Claude Code 源码五大组件标注
    ├── harness_v1.py             # 合体版（A-D 所有组件）
    └── harness_v2.py             # 升级版（CC 源码启发，并行 + 压缩 + Tool dataclass）
```

---

## 学习路径

### 练习 A：基础 Agent Loop

| 文件 | 核心概念 |
|------|---------|
| `step1_bare_call.py` | `finish_reason="stop"`，`tool_calls=None` |
| `step2_see_tool_call.py` | `finish_reason="tool_calls"`，tool 结构 |
| `step3_full_loop.py` | messages 数组累积，4 条消息完成一次工具调用 |

### 练习 B：三个真实工具

| 文件 | 核心概念 |
|------|---------|
| `step4_three_tools_plan.py` | read_file / write_file / run_shell 声明 |
| `step5_dispatcher.py` | 单函数路由，try/except 错误回传 |
| `step6_error_recovery.py` | 模型收到错误 → 自动换方案 |

### 练习 C：Token 预算 & 人工确认

| 文件 | 核心概念 |
|------|---------|
| `step7_token_monitor.py` | 每轮打印 token，超限演示 |
| `step8_hitl_complete.py` | 危险操作前 `input()` 暂停确认，`READ_CHAR_LIMIT` 截断 |

### 练习 D：生产级功能

| 文件 | 核心概念 |
|------|---------|
| `step9_checkpoint.py` | JSON 序列化，`os.replace()` 原子写，崩溃恢复 |
| `step10_sandbox.py` | `os.realpath()` 路径穿越防护 |
| `step11_eval.py` | LLM-as-Judge，YES/NO 裁判 |

### 练习 E：Claude Code 源码分析

`exercise_e_annotation.md` — 从 Claude Code 二进制（239MB ELF）中提取的关键 JS 代码，
标注五大组件并与 harness_v1.py 对照：

| 组件 | Claude Code | harness_v1.py |
|------|------------|---------------|
| Prompt 构造 | `DE6` 硬编码 summarization prompt | `system` 参数 + 截断 |
| LLM 调用 | `beta.messages.stream()` | `chat.completions.create()` |
| 工具注册 | `{name, run}` 合一对象 | TOOLS schema + dispatch() 分离 |
| 工具执行 | `Promise.all()` 并行 | 串行 `for tc in tool_calls` |
| 循环控制 | ES6 Generator `async*` | `for turn in range(MAX_TURNS)` |

---

## 两个完整版本

### harness_v1.py（317 行）

A-D 所有组件合并，可直接运行：

```bash
MOONSHOT_API_KEY=your_key python3 harness-practice/harness_v1.py "列出工作区所有 Python 文件并统计行数"
```

包含：Agent Loop · 三工具 Dispatcher · 错误回传 · 大文件截断 · Token 监控 · HITL · 检查点 · 沙箱隔离

### harness_v2.py（407 行）

根据 Claude Code 源码分析升级的版本：

```bash
MOONSHOT_API_KEY=your_key python3 harness-practice/harness_v2.py "你的任务"
```

新增：
- **`Tool` dataclass**：schema + run 函数合一，有可选 `parse()` 钩子
- **并行工具执行**：`ThreadPoolExecutor`，多工具同时跑
- **自动上下文压缩**：token 超阈值时 LLM 自动摘要整个历史，镜像 CC 的 `DE6` compaction

---

## 核心知识点速查

**停止条件**
```python
# finish_reason == "stop"  → 没有工具调用，任务完成
# finish_reason == "tool_calls" → 需要执行工具，继续循环
if finish == "stop":
    break
```

**错误回传（永远不 raise）**
```python
try:
    result = do_tool(args)
except Exception as e:
    result = f"错误：{type(e).__name__}: {str(e)}"
messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
```

**路径穿越防护**
```python
abs_path = os.path.realpath(os.path.join(WORKSPACE, user_input))
if not abs_path.startswith(os.path.realpath(WORKSPACE)):
    raise PermissionError("路径穿越被拦截")
```

**检查点原子写**
```python
tmp = CHECKPOINT_FILE + ".tmp"
with open(tmp, "w") as f:
    json.dump(state, f)
os.replace(tmp, CHECKPOINT_FILE)  # 原子替换，崩溃不会写坏文件
```

---

## 依赖

```bash
pip install openai
```

使用 OpenAI 兼容接口（示例用 Kimi / Moonshot），替换 `base_url` 和 `MODEL` 即可对接任意兼容 API。
