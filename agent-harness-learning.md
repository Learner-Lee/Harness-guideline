# AI Agent Harness 学习路线

---

## 0. 它是什么 & 和 Ralph Loop / Claude Code / Cursor 的关系

**白话版（120 字）**

LLM（如 Claude）是大脑，只负责"想"——输入文字，输出文字。**Harness** 是给大脑装上的身体：它负责组装提示词、调用 LLM、把工具结果塞回去、判断什么时候停。**Agent = 大脑 + 身体 + 目标**，三者缺一不可。

- **Ralph Loop**：`while true; do claude "xxx"; done` —— 最极简的 Harness 子集，只有 shell + 循环，没有工具和状态管理。
- **Claude Code**：一个完整的 Harness 实现，内置工具（读写文件、跑 shell）、上下文管理、人机交互。
- **Cursor**：在 IDE 层面的 Harness，把编辑器操作封装成工具。
- **LangChain / LangGraph**：Framework（抽象库），提供可复用积木，Harness 可以用它来搭，但不等于 Harness。

> **一句话记住**：Harness = 让 LLM 能真正"干活"的执行壳；Framework = 帮你搭 Harness 的积木库。

---

## 1. 基础必学模块 —— 手搓一个最小可跑的 Harness

**学完这节你能做到什么**：从零写出一个能真实运行的 Agent，让 Claude 调用你定义的工具完成一个小任务。

---

### 1.1 Harness 五大组件概览

**🔧 这是什么**
每个 Harness 无论多复杂，都由五个组件构成：①Prompt 构造、②LLM 调用、③工具注册、④工具执行、⑤循环与停止。

**✅ 理解五大组件，你会得到**
- 看任何现成 Harness（Claude Code、Aider、SWE-agent）的源码时，能立刻找到对应代码在哪里。
- 调试时知道问题出在哪一环，不会一头雾水。

**⚠️ 缺少某个组件，会发生什么**
- 没有工具执行：Claude 会"说"它调用了工具，但什么都没发生，任务永远完不成。
- 没有循环控制：每次只能一问一答，无法完成多步任务。
- 没有停止条件：循环永不退出，token 账单爆炸。

**🧪 最小可跑示例（概念演示，仅结构，细节在后续小节展开）**

```python
import anthropic

# ① Prompt 构造
messages = [{"role": "user", "content": "帮我查一下今天是星期几，然后告诉我。"}]
system = "你是一个助手，有需要时使用工具。"

# ③ 工具注册
tools = [{"name": "get_weekday", "description": "返回今天是星期几",
          "input_schema": {"type": "object", "properties": {}}}]

client = anthropic.Anthropic()

# ⑤ 循环与停止
for _ in range(10):
    # ② LLM 调用
    resp = client.messages.create(model="claude-sonnet-4-6",
        max_tokens=1024, system=system, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn":
        break
    # ④ 工具执行（下一节展开）
    results = []
    for block in resp.content:
        if block.type == "tool_use":
            import datetime
            result = datetime.datetime.now().strftime("%A")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
    if results:
        messages.append({"role": "user", "content": results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 1.2 最简 Agent Loop 结构

**🔧 这是什么**
一个 `while` 循环：调 LLM → 如果模型要调工具就执行 → 把结果塞回消息 → 再调 LLM → 直到模型说"我说完了"。这个循环就是整个 Harness 的心跳。

**✅ 有了 Agent Loop，你会得到**
- 模型可以把一个任务拆成多步自动完成，不需要你每步手动触发。
- 一次用户请求可以触发 N 次工具调用，最终给出完整答案。

**⚠️ 改动它，会发生什么**
- 不设最大轮数（`max_turns`）→ 遇到工具永远出错时，会死循环直到 API 超时或账单爆炸。
- 只运行一次就退出 → 工具调用的结果永远送不回给模型，它看不到自己"干了什么"。
- 用 `for` 而不是 `while` 且上限太小 → 复杂任务可能在完成前被截断，模型输出不完整。

**🧪 最小可跑示例**

```python
import anthropic, datetime

client = anthropic.Anthropic()
tools = [{"name": "today", "description": "返回今天的日期",
          "input_schema": {"type": "object", "properties": {}}}]
messages = [{"role": "user", "content": "今天几号？用工具查。"}]

max_turns = 10  # 防死循环的安全阀
for turn in range(max_turns):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512,
        tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason == "end_turn":
        print("完成，共", turn + 1, "轮")
        break

    tool_results = []
    for block in resp.content:
        if block.type == "tool_use":
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": datetime.date.today().isoformat()
            })
    messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 1.3 tool_use / tool_result 消息协议

**🔧 这是什么**
`tool_use`（工具调用请求）：模型在回复里用这个结构告诉你"我要调哪个工具、传什么参数"。`tool_result`（工具执行结果）：你执行完工具后，用这个结构把结果塞回给模型。两者通过 `tool_use_id` 配对——这个 id 是模型分配的唯一标识符，你必须原样回传。

**✅ 正确实现这个协议，你会得到**
- 模型能精确知道它的每个工具调用得到了什么结果，判断下一步该做什么。
- 支持一轮里模型同时调多个工具（并行调用），你全部执行完再一起回传。

**⚠️ 协议错误，会发生什么**
- `tool_result` 里的 `tool_use_id` 和 `tool_use` 的 `id` 对不上 → API 报错，请求直接失败。
- 模型在一轮里调了 3 个工具，你只回传了 1 个结果 → 模型会因为信息不完整而困惑，通常会重复调用漏掉的工具。
- 把 `tool_result` 放进 `assistant` 消息而不是 `user` 消息 → API 拒绝，消息格式不合法。

**🧪 最小可跑示例（展示完整协议结构）**

```python
import anthropic, json

client = anthropic.Anthropic()
tools = [
    {"name": "get_price", "description": "查询某商品价格",
     "input_schema": {"type": "object",
                      "properties": {"item": {"type": "string", "description": "商品名"}},
                      "required": ["item"]}}
]
messages = [{"role": "user", "content": "苹果和香蕉各多少钱？同时查。"}]

resp = client.messages.create(
    model="claude-sonnet-4-6", max_tokens=1024, tools=tools, messages=messages)
messages.append({"role": "assistant", "content": resp.content})

# 模型可能并行调用多个工具，全部处理
tool_results = []
for block in resp.content:
    if block.type == "tool_use":
        prices = {"苹果": "3元/斤", "香蕉": "2元/斤"}
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,           # 必须和 tool_use 的 id 完全一致
            "content": prices.get(block.input.get("item"), "未知")
        })

messages.append({"role": "user", "content": tool_results})  # 必须放 user 角色
resp2 = client.messages.create(
    model="claude-sonnet-4-6", max_tokens=512, tools=tools, messages=messages)
print(resp2.content[0].text)
```

---

### 1.4 messages 数组如何累积和传回

**🔧 这是什么**
`messages` 是一个列表，记录整个对话历史：每轮你把用户输入、模型回复、工具结果全追加进去，下次调用时完整传给 API。模型没有"记忆"，它的所有"上下文"都来自这个数组——你传多少，它就看到多少。

**✅ 正确维护 messages，你会得到**
- 模型能"记住"前几轮发生了什么，不会重复调已经完成的工具。
- 多步任务的中间状态（如"第 1 步已完成"）被自动保留。

**⚠️ 改动它，会发生什么**
- 每轮都重置 `messages` 只保留最新 → 模型失忆，不知道自己之前做了什么，会重头开始。
- 把 `resp.content`（一个列表）直接 append 为整体 → 格式错误，应该 append `{"role": "assistant", "content": resp.content}`。
- messages 越来越长不做任何清理 → 超出 context window 后 API 报错（200K token 上限）。

**🧪 最小可跑示例（展示正确累积方式）**

```python
import anthropic, datetime

client = anthropic.Anthropic()
tools = [{"name": "clock", "description": "返回当前时间",
          "input_schema": {"type": "object", "properties": {}}}]

messages = []  # 初始为空

# 第一轮：用户提问
messages.append({"role": "user", "content": "现在几点？"})

for _ in range(5):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, tools=tools, messages=messages)
    # 关键：把模型回复追加进去
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason == "end_turn":
        break

    tool_results = []
    for block in resp.content:
        if block.type == "tool_use":
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": datetime.datetime.now().strftime("%H:%M:%S")
            })
    # 关键：把工具结果也追加进去（role 必须是 user）
    messages.append({"role": "user", "content": tool_results})

print(f"对话共 {len(messages)} 条消息")
print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 1.5 停止条件：stop_reason 的四种取值

**🔧 这是什么**
`stop_reason`（停止原因）：每次模型返回时，都会带一个字段告诉你它为什么停下来。你的 Harness 必须根据这个字段决定下一步动作。四种取值：`end_turn`（正常完成）、`tool_use`（要调工具，继续循环）、`max_tokens`（输出 token 到上限了）、`stop_sequence`（触发了你设的停止词）。

**✅ 正确处理 stop_reason，你会得到**
- 模型正常完成时自动退出，不多跑一轮浪费 token。
- `max_tokens` 时能发现模型回答被截断，可以提示用户或增大 `max_tokens`。

**⚠️ 只处理 end_turn，会发生什么**
- `stop_reason == "max_tokens"` 时不处理 → 模型回答被截断，用户收到不完整输出，但你的 Harness 当作"正常完成"对待。
- `stop_reason == "tool_use"` 时误当 end_turn 退出 → 工具结果永远没送回去，模型的工具调用白费。

**🧪 最小可跑示例（完整 stop_reason 处理）**

```python
import anthropic, datetime

client = anthropic.Anthropic()
tools = [{"name": "now", "description": "返回当前时间戳",
          "input_schema": {"type": "object", "properties": {}}}]
messages = [{"role": "user", "content": "当前时间戳是多少？"}]

for _ in range(10):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=256, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason == "end_turn":
        print("[正常完成]")
        break
    elif resp.stop_reason == "max_tokens":
        print("[警告] 回复被截断，请增大 max_tokens")
        break
    elif resp.stop_reason == "tool_use":
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(datetime.datetime.now().timestamp())
                })
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"[未知停止原因] {resp.stop_reason}")
        break

print(next((b.text for b in resp.content if hasattr(b, "text")), "（无文本输出）"))
```

---

## 2. 进阶模块 —— 让 Harness 能干真活

**学完这节你能做到什么**：为你的 Harness 加上多个真实工具、错误处理、token 管理、安全确认和日志，让它能完成有实际价值的任务。

---

### 2.1 多工具注册与调度

**🔧 这是什么**
在 `tools` 列表里注册多个工具，用一个分发函数（dispatcher）根据 `block.name` 路由到对应的执行函数。模型自己决定什么时候调哪个工具，你只负责注册和执行。

**✅ 有了多工具，你会得到**
- Agent 能"读文件 → 分析内容 → 写结果"这种多步骤、跨工具的真实工作流。
- 新增工具只需在注册列表和 dispatcher 里各加一行，不动核心循环。

**⚠️ 常见错误**
- 工具 `description` 写得含糊（如"处理数据"）→ 模型不知道该什么时候调它，调用率极低。
- `input_schema` 里的字段名和 dispatcher 里读取的 key 不一致 → `KeyError`，工具执行崩溃。
- 注册了工具但 dispatcher 没有对应分支 → 工具调用静默失败，返回空字符串，模型困惑。

**🧪 最小可跑示例**

```python
import anthropic, os, subprocess

client = anthropic.Anthropic()

tools = [
    {"name": "read_file", "description": "读取本地文件内容，返回文本",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "文件绝对路径"}}, "required": ["path"]}},
    {"name": "write_file", "description": "把文本写入本地文件",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "run_shell", "description": "执行 shell 命令，返回 stdout",
     "input_schema": {"type": "object", "properties": {
         "cmd": {"type": "string", "description": "shell 命令"}}, "required": ["cmd"]}},
]

def dispatch(name, args):
    if name == "read_file":
        with open(args["path"]) as f: return f.read()
    elif name == "write_file":
        with open(args["path"], "w") as f: f.write(args["content"]); return "写入成功"
    elif name == "run_shell":
        result = subprocess.run(args["cmd"], shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout or result.stderr
    return f"未知工具: {name}"

messages = [{"role": "user", "content": "用 shell 查一下当前目录有哪些 .py 文件，把结果写进 /tmp/pyfiles.txt，再读出来告诉我。"}]

for _ in range(10):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1024, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break
    tool_results = [{"type": "tool_result", "tool_use_id": b.id, "content": dispatch(b.name, b.input)}
                    for b in resp.content if b.type == "tool_use"]
    if tool_results: messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 2.2 错误处理：工具报错了怎么塞回给模型

**🔧 这是什么**
工具执行时可能抛异常（文件不存在、命令失败、网络超时）。正确做法是 catch 异常，把错误信息作为 `tool_result` 的 `content` 塞回给模型（同时设 `is_error: true`），让模型自己决定怎么应对。

**✅ 正确处理工具错误，你会得到**
- 模型收到错误信息后，通常会自动尝试修正参数或换一个思路，而不是直接失败。
- 错误被记录在对话历史里，方便事后调试：能看到模型当时看到了什么。

**⚠️ 不处理错误，会发生什么**
- 直接把异常往外抛 → 整个 Harness 崩溃，任务彻底中断，用户看到裸 traceback。
- 捕获异常但返回空字符串 → 模型以为工具"成功但没有输出"，会基于错误假设继续，产生胡说八道的结果。
- 错误信息太长（如完整 traceback）→ 浪费大量 token，核心错误信息淹没在细节里。

**🧪 最小可跑示例**

```python
import anthropic

client = anthropic.Anthropic()
tools = [{"name": "read_file", "description": "读取文件内容",
          "input_schema": {"type": "object",
                           "properties": {"path": {"type": "string"}}, "required": ["path"]}}]

def safe_read_file(path):
    try:
        with open(path) as f: return f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"
    except PermissionError:
        return f"错误：没有读取 {path} 的权限"
    except Exception as e:
        return f"错误：{type(e).__name__}: {str(e)[:100]}"  # 截断，避免 token 爆炸

messages = [{"role": "user", "content": "读取 /tmp/nonexistent.txt 的内容。"}]

for _ in range(5):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break
    tool_results = []
    for b in resp.content:
        if b.type == "tool_use":
            result = safe_read_file(b.input["path"])
            is_error = result.startswith("错误：")
            tool_results.append({
                "type": "tool_result", "tool_use_id": b.id,
                "content": result, "is_error": is_error  # 告诉模型这是错误结果
            })
    if tool_results: messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 2.3 Token 预算与上下文压缩

**🔧 这是什么**
每次调用 API，`messages` 数组里所有内容都算 input token。Claude 的上下文窗口是 200K token，超了就报错。Token 预算管理 = 监控当前 token 数 + 在快满之前压缩历史（要么总结旧对话，要么截断最早的轮次）。

**✅ 做了 token 管理，你会得到**
- Agent 能跑非常长的任务而不崩溃，不会跑到一半因超出上下文而报错。
- token 消耗可控，避免把整个代码库每轮都塞进去导致账单爆炸。

**⚠️ 不做 token 管理，会发生什么**
- messages 无限增长 → 任务跑到后期必然命中 200K 上限，API 返回 `context_length_exceeded` 错误。
- 简单截断最早的消息 → 可能删掉关键的系统状态，模型突然"失忆"，做出矛盾的决策。
- 把压缩摘要放进 `user` 消息而不是 `system` → 每轮都重新输入摘要，没有节省 token。

**🧪 最小可跑示例（估算 token + 触发压缩）**

```python
import anthropic

client = anthropic.Anthropic()

def estimate_tokens(messages):
    # 粗略估算：每 4 个字符约 1 token
    text = str(messages)
    return len(text) // 4

def compress_history(messages, client, keep_last=4):
    """保留最近 keep_last 轮，其余压缩成摘要"""
    if len(messages) <= keep_last:
        return messages
    old = messages[:-keep_last]
    old_text = "\n".join(
        f"{m['role']}: {m['content'] if isinstance(m['content'], str) else '[tool content]'}"
        for m in old)
    summary_resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=300,
        messages=[{"role": "user", "content": f"请用 2-3 句话总结以下对话：\n{old_text[:2000]}"}])
    summary = summary_resp.content[0].text
    return [{"role": "user", "content": f"[历史摘要] {summary}"}] + messages[-keep_last:]

messages = [{"role": "user", "content": "用数字 1 到 20 逐步计数，每次用工具记录一个数字。"}]
TOKEN_LIMIT = 50000  # 设一个保守上限，在快满之前压缩

for _ in range(30):
    if estimate_tokens(messages) > TOKEN_LIMIT:
        print("[压缩历史]")
        messages = compress_history(messages, client)
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=256, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break

print(f"最终 messages 条数: {len(messages)}")
```

---

### 2.4 系统提示词 vs 用户提示词的分工

**🔧 这是什么**
`system` 提示词：在整个对话里始终有效的"角色设定 + 规则 + 工具使用说明"，不随对话轮次变化。`user` 提示词：每轮的具体任务指令。两者分开写，模型区分得更清楚，遵守率更高。

**✅ 正确分工，你会得到**
- 系统规则（如"不要删除文件"）每轮都自动生效，不需要在每个 user 消息里重复。
- 用户消息更简洁，模型理解任务更准确，不会把规则当任务执行。

**⚠️ 分工混乱，会发生什么**
- 把所有规则都塞进第一条 `user` 消息 → 规则被当成对话内容，后续轮次模型可能不再遵守。
- `system` 里写太多细节（如整个代码库）→ 每轮都计入 input token，成本极高；且 system 不支持工具结果，只适合静态内容。
- `system` 和 `user` 里都写了同一条规则但措辞不同 → 模型可能困惑，产生不一致行为。

**🧪 最小可跑示例**

```python
import anthropic

client = anthropic.Anthropic()

# system：角色 + 规则（静态，整个任务期间不变）
system = """你是一个代码审查助手。
规则：
1. 发现问题时，先描述问题，再给出修复建议。
2. 不要直接修改代码，只给建议。
3. 每次回复不超过 200 字。"""

tools = [{"name": "read_file", "description": "读取代码文件",
          "input_schema": {"type": "object",
                           "properties": {"path": {"type": "string"}}, "required": ["path"]}}]

# user：具体任务（每轮可以变化）
messages = [{"role": "user", "content": "请审查 /tmp/demo.py 这个文件。"}]

import os
os.makedirs("/tmp", exist_ok=True)
with open("/tmp/demo.py", "w") as f:
    f.write("def add(a,b):\n    return a+b\nx=add(1,'2')\nprint(x)\n")

for _ in range(5):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512,
        system=system, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break
    tool_results = []
    for b in resp.content:
        if b.type == "tool_use" and b.name == "read_file":
            try:
                with open(b.input["path"]) as f: content = f.read()
            except Exception as e: content = str(e)
            tool_results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
    if tool_results: messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 2.5 可观测性：日志 / trace / 每轮 token 消耗记录

**🔧 这是什么**
在每次 LLM 调用前后记录：调用了什么工具、用了多少 token、耗时多少秒、模型回了什么。这些信息就是你调试 Agent 跑偏的唯一线索。

**✅ 加了可观测性，你会得到**
- Agent 跑偏时，能回溯"第 3 轮它调了什么工具、看到了什么结果、然后做了什么错误决策"。
- token 消耗按轮记录后，能发现哪些工具结果最"贵"，针对性优化。
- 异常 token 消耗（如突然 10x）立刻可见，不等账单出来才发现。

**⚠️ 不做可观测性，会发生什么**
- Agent 输出错误结果，你完全不知道是哪一步出的问题，只能重跑调试，浪费 token。
- 生产中某轮工具调用卡住了，你没有日志，无法判断是超时、报错还是死循环。

**🧪 最小可跑示例**

```python
import anthropic, time, datetime

client = anthropic.Anthropic()
tools = [{"name": "get_info", "description": "返回一句话信息",
          "input_schema": {"type": "object", "properties": {}}}]
messages = [{"role": "user", "content": "用工具查一条信息，然后总结给我。"}]

total_input = total_output = 0

for turn in range(10):
    t0 = time.time()
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, tools=tools, messages=messages)
    elapsed = time.time() - t0

    # 记录每轮 token 消耗
    inp = resp.usage.input_tokens
    out = resp.usage.output_tokens
    total_input += inp; total_output += out
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] 轮{turn+1} stop={resp.stop_reason} in={inp} out={out} 耗时={elapsed:.1f}s")

    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break

    tool_results = []
    for b in resp.content:
        if b.type == "tool_use":
            print(f"  → 调用工具: {b.name}({b.input})")
            tool_results.append({"type": "tool_result", "tool_use_id": b.id,
                                  "content": "这是一条测试信息"})
    if tool_results: messages.append({"role": "user", "content": tool_results})

print(f"\n总 token: input={total_input} output={total_output} 合计={total_input+total_output}")
print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 2.6 人在回路（human-in-the-loop）

**🔧 这是什么**
在 Agent 即将执行"危险操作"（删文件、写数据库、发邮件）前，暂停循环，把操作描述打印给用户，等用户输入 `y` 确认后再执行。"危险"由你定义，通常是不可逆的操作。

**✅ 加了人在回路，你会得到**
- 避免 Agent 因误解指令而自动执行破坏性操作（比如理解错了路径删了重要文件）。
- 用户对 Agent 行为有感知和控制，不是完全黑盒，信任度大幅提升。

**⚠️ 不加人在回路，会发生什么**
- Agent 把"把旧文件备份一下"理解成"把旧文件重命名然后删掉"并自动执行 → 数据丢失。
- 模型提示词注入攻击（外部文件里藏了"执行 rm -rf ~"的指令）→ 没有确认直接执行，后果严重。
- 对低风险操作也确认 → 用户体验变差，Agent 失去自主性价值。

**🧪 最小可跑示例**

```python
import anthropic, subprocess

client = anthropic.Anthropic()
DANGEROUS_TOOLS = {"delete_file", "run_shell"}  # 需要人工确认的工具

tools = [
    {"name": "read_file", "description": "读取文件（安全操作）",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "run_shell", "description": "执行 shell 命令（危险操作，需要确认）",
     "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
]

def dispatch_with_hitl(name, args):
    if name in DANGEROUS_TOOLS:
        print(f"\n⚠️  Agent 要执行危险操作: {name}({args})")
        confirm = input("确认执行？[y/N] ").strip().lower()
        if confirm != "y":
            return "用户拒绝了此操作"
    if name == "read_file":
        try:
            with open(args["path"]) as f: return f.read()
        except Exception as e: return str(e)
    elif name == "run_shell":
        r = subprocess.run(args["cmd"], shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout or r.stderr
    return "未知工具"

messages = [{"role": "user", "content": "列出 /tmp 目录下的文件，然后删除其中的临时文件。"}]

for _ in range(10):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break
    tool_results = [{"type": "tool_result", "tool_use_id": b.id,
                     "content": dispatch_with_hitl(b.name, b.input)}
                    for b in resp.content if b.type == "tool_use"]
    if tool_results: messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

## 3. Master 模块 —— 生产级 Harness 的工程细节

**学完这节你能做到什么**：设计一个能跑几小时、支持多 Agent 协作、崩溃可恢复、有安全防护的生产级 Harness。

---

### 3.1 Sub-agent / 多 agent 协作

**🔧 这是什么**
主 Agent（Orchestrator）把大任务拆成子任务，每个子任务启动一个独立的子 Agent（Worker）去执行，Worker 完成后把结果汇报回主 Agent。主 Agent 只负责规划和整合，不直接执行细节。

**✅ 用了多 agent 协作，你会得到**
- 并行执行多个子任务，总耗时大幅缩短（如同时搜索 5 个关键词）。
- 每个 Worker 的 context window 只包含自己那份子任务，不被整体上下文拖累。
- 主 Agent 和 Worker 可以使用不同模型（主用 Opus，Worker 用 Haiku），节省成本。

**⚠️ 多 agent 的常见陷阱**
- Worker 之间有隐式依赖（B 依赖 A 的结果）但并行启动 → B 拿到空结果，静默失败，主 Agent 不知道。
- 主 Agent 的 system prompt 里没有清楚定义 Worker 的输入/输出格式 → Worker 回来的结果格式各异，主 Agent 无法解析。
- 递归调用（Worker 又启动了 Worker）没有深度限制 → 指数爆炸，API 调用和费用失控。

**🧪 最小可跑示例（主 Agent 派发两个并行子任务）**

```python
import anthropic, concurrent.futures

client = anthropic.Anthropic()

def run_worker(task_description):
    """Worker Agent：接收子任务描述，返回结果文本"""
    resp = client.messages.create(
        model="claude-haiku-4-5",  # Worker 用便宜模型
        max_tokens=300,
        messages=[{"role": "user", "content": task_description}])
    return resp.content[0].text

# 主 Agent：规划任务
orchestrator_resp = client.messages.create(
    model="claude-sonnet-4-6",  # 主 Agent 用强模型
    max_tokens=512,
    messages=[{"role": "user",
               "content": "我需要了解 Python 和 JavaScript 各自的优缺点。请分别总结，每种语言一段话。"}])

# 模拟主 Agent 把任务拆成两个子任务并行执行
subtasks = [
    "用两句话总结 Python 语言的主要优点和缺点",
    "用两句话总结 JavaScript 语言的主要优点和缺点"
]

print("主 Agent 启动 2 个 Worker 并行执行...")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(run_worker, task): task for task in subtasks}
    results = {}
    for future in concurrent.futures.as_completed(futures):
        task = futures[future]
        results[task] = future.result()
        print(f"Worker 完成: {task[:20]}...")

# 主 Agent 整合结果
combined = "\n\n".join(f"子任务：{k}\n结果：{v}" for k, v in results.items())
final_resp = client.messages.create(
    model="claude-sonnet-4-6", max_tokens=512,
    messages=[{"role": "user",
               "content": f"以下是两个子任务的结果，请整合成一段对比分析：\n\n{combined}"}])
print("\n最终整合结果：")
print(final_resp.content[0].text)
```

---

### 3.2 检查点与恢复（崩了能接着跑）

**🔧 这是什么**
每完成若干轮工具调用，把当前的 `messages` 数组序列化保存到磁盘（检查点文件）。如果进程崩溃或被中断，下次启动时先检查有没有检查点文件，有就从那里恢复继续跑，而不是从头开始。

**✅ 有了检查点，你会得到**
- 跑了 2 小时的任务因网络抖动崩溃，重启后从第 1h55min 继续，不用重头来。
- 跨多次会话完成一个长任务（今天跑到一半，明天继续）。

**⚠️ 检查点的陷阱**
- 只保存 messages 不保存工具的外部状态（如已写入文件的内容）→ 恢复后 messages 里记录的"文件已写入"和磁盘实际状态不一致，模型基于错误假设继续操作。
- 每轮都保存（频率过高）→ I/O 开销大，适合每 5-10 轮保存一次。
- 检查点文件不加版本号 → 新旧版本代码都读同一个文件，格式不兼容时静默损坏。

**🧪 最小可跑示例**

```python
import anthropic, json, os, datetime

client = anthropic.Anthropic()
CHECKPOINT_FILE = "/tmp/agent_checkpoint.json"
SAVE_EVERY = 3  # 每 3 轮保存一次

def save_checkpoint(messages, turn):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"messages": [
            {"role": m["role"],
             "content": m["content"] if isinstance(m["content"], str)
                        else [{"type": b.type, "text": getattr(b, "text", "")}
                              if hasattr(b, "type") else b for b in m["content"]]}
            for m in messages], "turn": turn}, f, ensure_ascii=False, indent=2)
    print(f"  [检查点已保存，第 {turn} 轮]")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        print(f"[恢复检查点，从第 {data['turn']} 轮继续]")
        return data["messages"], data["turn"]
    return None, 0

tools = [{"name": "timestamp", "description": "返回当前时间戳",
          "input_schema": {"type": "object", "properties": {}}}]

messages, start_turn = load_checkpoint()
if messages is None:
    messages = [{"role": "user", "content": "每隔一步用工具记录时间戳，共记录 5 次，最后汇总。"}]

for turn in range(start_turn, start_turn + 20):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn":
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
        print("[任务完成，检查点已清除]")
        break
    tool_results = [{"type": "tool_result", "tool_use_id": b.id,
                     "content": datetime.datetime.now().isoformat()}
                    for b in resp.content if b.type == "tool_use"]
    if tool_results: messages.append({"role": "user", "content": tool_results})
    if (turn + 1) % SAVE_EVERY == 0:
        save_checkpoint(messages, turn + 1)

print(next((b.text for b in resp.content if hasattr(b, "text")), ""))
```

---

### 3.3 评估（eval）：怎么衡量 Harness 是变强还是变弱

**🔧 这是什么**
为你的 Harness 建立一套评估流程：准备若干"测试用例"（输入任务 + 预期输出），每次改动 Harness 后跑一遍，用打分函数（LLM-as-judge 或精确匹配）量化改动前后的效果差异。

**✅ 有了 eval，你会得到**
- 改了 system prompt 后，立刻知道是"进步了"还是"退步了"，而不是靠感觉。
- 能发现"新功能在 A 场景下好了，但把 B 场景搞坏了"的回归问题。

**⚠️ 没有 eval，会发生什么**
- 每次改 prompt 都是盲猜，可能越改越差。
- 生产上线后才发现某类任务成功率从 80% 跌到 40%，用户已经受害。
- 用少数几个"手感好"的例子测试 → 样本偏差，遮蔽了真实问题。

**🧪 最小可跑示例（LLM-as-judge 评估器）**

```python
import anthropic

client = anthropic.Anthropic()

# 测试用例：输入任务 + 期望答案关键词
TEST_CASES = [
    {"task": "2+2 等于多少？", "expected_keywords": ["4"]},
    {"task": "Python 创建列表的语法是什么？", "expected_keywords": ["[]", "list()"]},
    {"task": "什么是 HTTP 状态码 404？", "expected_keywords": ["找不到", "Not Found", "not found"]},
]

def run_agent(task):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=256,
        messages=[{"role": "user", "content": task}])
    return resp.content[0].text

def judge(task, response, expected_keywords):
    """用 LLM 判断回答是否正确"""
    judge_resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=50,
        messages=[{"role": "user", "content":
                   f"任务：{task}\n回答：{response}\n"
                   f"期望包含关键词：{expected_keywords}\n"
                   f"回答是否正确？只回复 YES 或 NO。"}])
    return "YES" in judge_resp.content[0].text.upper()

passed = total = 0
for case in TEST_CASES:
    response = run_agent(case["task"])
    ok = judge(case["task"], response, case["expected_keywords"])
    status = "✅" if ok else "❌"
    print(f"{status} 任务: {case['task'][:30]}...")
    if ok: passed += 1
    total += 1

print(f"\n通过率: {passed}/{total} = {passed/total*100:.0f}%")
```

---

### 3.4 沙箱与安全：文件系统隔离、命令白名单、网络限制

**🔧 这是什么**
给 Agent 的工具执行加防护层：①只允许操作指定目录（文件隔离）；②只允许运行白名单里的命令（命令过滤）；③网络访问只走代理或白名单域名。防止提示词注入或模型判断失误导致的破坏性操作。

**✅ 加了沙箱，你会得到**
- 即使 Agent 被注入了恶意指令（"删除所有文件"），也因为命令不在白名单而被拦截。
- 文件操作被限制在 `/tmp/agent_workspace` 里，不会意外修改系统文件。

**⚠️ 不做沙箱，会发生什么**
- 提示词注入（外部文件里藏了 `rm -rf ~`）→ Agent 执行 shell 工具时直接跑了。
- Agent 误解任务，把"清理缓存"理解成"删除所有 .pyc 文件"，递归删了项目里的编译文件。
- 白名单过于严格（连 `ls` 都不允许）→ Agent 完全无法工作，白名单要从实际使用中迭代。

**🧪 最小可跑示例（路径隔离 + 命令白名单）**

```python
import anthropic, subprocess, os

client = anthropic.Anthropic()
WORKSPACE = "/tmp/agent_workspace"  # Agent 只能操作这个目录
os.makedirs(WORKSPACE, exist_ok=True)

ALLOWED_COMMANDS = {"ls", "cat", "echo", "pwd", "find", "wc"}  # 白名单

def safe_run_shell(cmd):
    cmd_name = cmd.strip().split()[0]
    if cmd_name not in ALLOWED_COMMANDS:
        return f"[拒绝] 命令 '{cmd_name}' 不在白名单中，允许的命令: {ALLOWED_COMMANDS}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            timeout=5, cwd=WORKSPACE)  # cwd 限制工作目录
    return result.stdout or result.stderr

def safe_write_file(path, content):
    abs_path = os.path.abspath(os.path.join(WORKSPACE, path))
    if not abs_path.startswith(WORKSPACE):  # 防止路径穿越攻击（如 ../../etc/passwd）
        return "[拒绝] 不允许写入工作目录之外的文件"
    with open(abs_path, "w") as f: f.write(content)
    return f"写入成功: {abs_path}"

tools = [
    {"name": "run_shell", "description": f"在 {WORKSPACE} 目录执行 shell 命令（仅白名单命令）",
     "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
    {"name": "write_file", "description": f"在 {WORKSPACE} 目录写入文件",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "相对路径"},
         "content": {"type": "string"}}, "required": ["path", "content"]}},
]

messages = [{"role": "user", "content": "创建一个 hello.txt 文件，写入 hello world，然后用命令查看它。"}]

for _ in range(10):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break
    tool_results = []
    for b in resp.content:
        if b.type == "tool_use":
            if b.name == "run_shell": result = safe_run_shell(b.input["cmd"])
            elif b.name == "write_file": result = safe_write_file(b.input["path"], b.input["content"])
            else: result = "未知工具"
            tool_results.append({"type": "tool_result", "tool_use_id": b.id, "content": result})
    if tool_results: messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

### 3.5 模型编排：便宜模型做初筛，贵模型做关键决策

**🔧 这是什么**
在一个 Harness 里混用不同能力/价格的模型：用 Haiku（速度快、价格低）做初步判断、格式化、简单工具调用；用 Sonnet/Opus（能力强、价格高）做需要推理的关键决策。

**✅ 模型编排，你会得到**
- 在相同效果下，成本降低 50-80%（大量简单工具调用不再烧贵模型）。
- Haiku 响应速度更快，用户等待时间短；Opus 只在真正需要时出现。

**⚠️ 编排的陷阱**
- 把需要复杂推理的步骤分配给 Haiku → 结果质量下降，省了工具调用费但增加了重试次数，反而更贵。
- Haiku 的输出作为 Opus 的输入，但没有格式约束 → Haiku 输出格式多变，Opus 解析失败。
- 两个模型的 context 不同步 → Opus 做决策时不知道 Haiku 刚才发现了什么。

**🧪 最小可跑示例（Haiku 做工具调用，Sonnet 做最终总结）**

```python
import anthropic, datetime

client = anthropic.Anthropic()

tools = [
    {"name": "get_date", "description": "返回今天的日期",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_weather_mock", "description": "返回模拟天气信息",
     "input_schema": {"type": "object", "properties": {
         "city": {"type": "string"}}, "required": ["city"]}},
]

def dispatch(name, args):
    if name == "get_date": return datetime.date.today().isoformat()
    if name == "get_weather_mock": return f"{args['city']}：晴，25°C，适合外出"
    return "未知工具"

messages = [{"role": "user", "content": "告诉我今天日期和北京的天气，然后给我一句出行建议。"}]

# 阶段一：用 Haiku 完成工具调用（收集数据）
print("[Haiku 阶段：执行工具调用]")
for _ in range(5):
    resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=512, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break
    tool_results = [{"type": "tool_result", "tool_use_id": b.id, "content": dispatch(b.name, b.input)}
                    for b in resp.content if b.type == "tool_use"]
    if tool_results:
        print(f"  工具结果: {[r['content'] for r in tool_results]}")
        messages.append({"role": "user", "content": tool_results})

# 阶段二：用 Sonnet 做最终推理和总结
print("[Sonnet 阶段：生成最终建议]")
messages.append({"role": "user", "content": "基于以上信息，用一句话给出今天的出行建议。"})
final = client.messages.create(model="claude-sonnet-4-6", max_tokens=256, messages=messages)
print("最终建议:", final.content[0].text)
```

---

### 3.6 和 MCP（Model Context Protocol）的关系与对接方式

**🔧 这是什么**
MCP（Model Context Protocol）是 Anthropic 提出的一个开放标准：把工具、数据源、提示词模板统一用一种协议暴露出来，让任何 Harness 都能"即插即用"地接入这些能力，而不用每个 Harness 自己重复造轮子。你的 Harness 是 MCP 的"客户端"，各种工具服务是"MCP Server"。

**✅ 接入 MCP，你会得到**
- 直接复用社区现成的 MCP Server（文件系统、数据库、浏览器控制），不用自己写工具。
- 你写的工具也能被 Claude Code、其他 Harness 直接用，代码复用率极高。
- 工具的更新和 Harness 解耦，MCP Server 升级不影响你的 Harness 代码。

**⚠️ MCP 的注意事项**
- MCP Server 是独立进程，通过 stdio 或 HTTP 通信，你的 Harness 需要先启动 Server 再调用 API。
- MCP 工具描述由 Server 动态暴露，不能在代码里硬编码；每次获取工具列表要调一次 `list_tools`。
- 自建 MCP Server 安全性由你负责：不要暴露危险操作而缺少认证。

**🧪 最小可跑示例（手动模拟 MCP 工具注册与调用的概念）**

```python
import anthropic, json

client = anthropic.Anthropic()

# 模拟 MCP Server 动态暴露的工具列表（真实场景里通过 MCP SDK 获取）
MCP_TOOLS_FROM_SERVER = [
    {"name": "mcp__filesystem__read_file",
     "description": "通过 MCP 文件系统服务读取文件",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "mcp__filesystem__list_dir",
     "description": "通过 MCP 文件系统服务列出目录",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]

def call_mcp_tool(tool_name, args):
    """真实场景：这里调用 MCP SDK 的 call_tool 方法"""
    import os
    if tool_name == "mcp__filesystem__read_file":
        try:
            with open(args["path"]) as f: return f.read()
        except Exception as e: return str(e)
    elif tool_name == "mcp__filesystem__list_dir":
        try: return "\n".join(os.listdir(args["path"]))
        except Exception as e: return str(e)
    return f"MCP 工具 {tool_name} 未实现"

messages = [{"role": "user", "content": "列出 /tmp 目录的内容。"}]

for _ in range(5):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512,
        tools=MCP_TOOLS_FROM_SERVER, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn": break
    tool_results = [{"type": "tool_result", "tool_use_id": b.id,
                     "content": call_mcp_tool(b.name, b.input)}
                    for b in resp.content if b.type == "tool_use"]
    if tool_results: messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in resp.content if hasattr(b, "text")))
```

---

## 4. 什么时候【不该】自己造 Harness

> **判断原则**：如果现成工具能满足 80% 的需求，用现成的；只在现成工具无法满足你的核心场景时，才值得自己造。

| 场景 | 推荐 | 理由 |
|------|------|------|
| 编写/调试代码、探索代码库 | 直接用 **Claude Code** | 已内置文件、shell、搜索工具，harness 已经写好了 |
| 需要复杂的多步 Agent 工作流 | 用 **LangGraph** 或 **CrewAI** | 状态机、多 agent 协调已有完善抽象 |
| RAG（检索增强生成） | 用 **LlamaIndex** | 文档切分、向量检索、重排序已经有成熟实现 |
| 快速原型，不需要自定义执行逻辑 | 用 **Claude API + 直接对话** | 不需要循环，一次调用就够 |
| 需要极致自定义（自定义上下文策略/安全策略/评估） | **自己造** | 现成框架无法满足特殊需求 |
| 学习目的，理解原理 | **自己造最小版本** | 亲手搓一遍是最好的学习方式 |

**三个问号帮你决策**：
1. 你的工具调用逻辑是否有特殊业务规则，现有框架无法表达？→ 否则用框架。
2. 你的上下文管理/安全策略是否极度特殊？→ 否则用框架。
3. 这个 Harness 是你的核心竞争力，还是只是基础设施？→ 基础设施尽量用现成的。

---

## 5. 动手路线图

**推荐练习顺序（从这里开始）**

```
练习 A【基础，1-2 小时】
  目标：第一个真正跑起来的 Agent
  内容：实现 1.2 节的 Agent Loop + 1.3 节的 tool_use 协议
  验收：Claude 能调你定义的工具，打印出正确结果

  ↓

练习 B【进阶，2-3 小时】
  目标：能干真活的多工具 Agent
  内容：实现 2.1 读文件+写文件+shell 三个工具 + 2.2 错误处理 + 2.5 日志
  验收：让 Agent 读一个文件，分析内容，把结果写入另一个文件，全程有日志

  ↓

练习 C【进阶，1-2 小时】
  目标：不会 token 爆炸的 Agent
  内容：实现 2.3 token 预算 + 2.6 human-in-the-loop
  验收：让 Agent 处理一个大文件，token 接近上限时自动压缩；危险操作前暂停确认

  ↓

练习 D【Master，3-5 小时】
  目标：生产级 Harness
  内容：实现 3.2 检查点 + 3.4 沙箱安全 + 3.3 一个简单 eval 套件
  验收：Harness 崩溃后能恢复；文件操作被限制在 workspace；eval 能量化改动效果

  ↓

练习 E【选做，Master】
  目标：看懂一个真实 Harness 的源码
  内容：把 Claude Code 的核心循环（tool dispatch → context 管理）找出来，
        对照这份文档的五大组件一一标注
  验收：能用自己的话解释 Claude Code 的 agent loop 在哪里、做了什么
```

**三个你最担心的问题，现在有答案了**

| 担心 | 具体手段 |
|------|----------|
| Token 烧太快 | 2.3 节的 token 预算监控 + compress_history；2.5 节的每轮 token 日志；3.5 节的 Haiku/Opus 分层 |
| 工具调用死循环 | 所有循环都设 `max_turns=10`（1.2节）；stop_reason 完整处理（1.5节）；工具错误一定要回传（2.2节） |
| 不知道怎么调试跑偏 | 2.5 节每轮打印工具调用和 token 消耗；检查点（3.2节）让你可以重放任何一轮的 messages |

---

*文档生成基于 Anthropic 官方资料：[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) / [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) / [How tool use works](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/how-tool-use-works)*
