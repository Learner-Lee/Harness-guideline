# Prompt: 帮我系统学习 AI Agent Harness（给 Claude Code 的任务书）

## T — Task（任务）
你是我的**技术导师 + 陪练**。我要学习 **AI Agent Harness**（给 LLM 搭的外层执行框架：负责 prompt 组装、工具调用、循环控制、状态管理、评估与停止）。请为我产出一份**实践导向**的学习文档，并在我准备好之后，**带我从零手搓一个最小 Harness**，再逐步加料到进阶和 Master 级。

交付物分两阶段：
1. **阶段一（文档）**：生成 `agent-harness-learning.md`，按三层模块组织。
2. **阶段二（陪练）**：文档写完后暂停，问我："要从哪个模块开始动手？" 然后带我在本机真实跑一遍。

## R — Request（具体要求）

### 文档结构（严格遵守）
```
# AI Agent Harness 学习路线
## 0. 它是什么 & 和 Ralph Loop / Claude Code / Cursor 的关系（150 字以内白话）
## 1. 基础必学模块 —— 手搓一个最小可跑的 Harness
## 2. 进阶模块 —— 让 Harness 能干真活
## 3. Master 模块 —— 生产级 Harness 的工程细节
## 4. 什么时候【不该】自己造 Harness（直接用现成的就行）
## 5. 动手路线图（从哪个练习开始）
```

### 三层模块必须覆盖的知识点（最少这些，可多不可少）

**基础必学（目标：能跑起来）**
- Harness 的五大组件：Prompt 构造 / LLM 调用 / 工具注册 / 工具执行 / 循环与停止
- 一个最简 agent loop 的结构（while + tool_use 判断）
- `tool_use` / `tool_result` 消息协议是怎么回事
- 消息历史（messages 数组）如何累积和传回
- 最简单的停止条件（stop_reason == "end_turn"）

**进阶（目标：能干真活）**
- 多工具注册与调度（读文件 / 写文件 / 跑 shell）
- 错误处理：工具报错了怎么塞回给模型
- Token 预算与上下文压缩（summarize / truncate 策略）
- 系统提示词 vs 用户提示词的分工
- 可观测性：日志、trace、每轮 token 消耗记录
- 人在回路（human-in-the-loop）：危险操作前暂停确认

**Master（目标：生产可用）**
- Sub-agent / 多 agent 协作（主 agent 派发子任务）
- 检查点与恢复（崩了能接着跑）
- 评估（eval）：怎么衡量你的 Harness 是变强还是变弱
- 沙箱与安全：文件系统隔离、命令白名单、网络限制
- 模型编排：便宜模型做初筛，贵模型做关键决策
- 和 MCP（Model Context Protocol）的关系与对接方式

### 每个知识点必须包含这 4 块（缺一不可）
- **🔧 这是什么**：一句话。
- **✅ 加上它/这么做，你会得到**：具体、可观察的效果。例："Agent 不再每轮都把整个代码库塞进去，token 消耗直接砍半"。
- **⚠️ 改动它，会发生什么**：2-3 个常见修改和后果。例："把工具错误直接抛出而不塞回模型 → Agent 不知道自己失败了，下一轮会重复同样的错"。
- **🧪 最小可跑示例**：Python 代码（用 `anthropic` 官方 SDK），**15-40 行，能直接复制运行**，不要伪代码、不要 `# ... 省略`。

### 风格要求
- 新手友好。出现术语（tool_use、stop_reason、context window 等）时**第一次出现必须一句话解释**。
- 多用「如果你…就会…」句式。
- 每个模块开头写一句"**学完这节你能做到什么**"。
- 每个知识点 1 屏以内。
- 代码示例统一用 Anthropic Python SDK 和 Claude 模型，便于我后面接 Claude Code 练习。

## A — Action（执行步骤）

按顺序，不要跳步：

1. **先搜一手资料**：用 WebSearch 检索以下关键词，读完再动笔：
   - "agent harness" anthropic OR cookbook
   - "tool use loop" site:docs.anthropic.com
   - "SWE-agent" harness architecture
   - "building effective agents" Anthropic（这是 Anthropic 官方一篇关键博客）
   
   如果关键一手资料搜不到，**停下来告诉我**，不要编造。
2. **列大纲**：把三层模块的知识点清单先发给我，等我说 "OK 继续" 再展开写。
3. **写文档**：按上面结构和四块格式生成 `agent-harness-learning.md`。
4. **自检**：逐条对照 Request 打勾，缺了就补。尤其检查每个代码示例**是不是真能跑**（import 完整、没有省略号）。
5. **暂停提问**：问我从哪个模块开始动手。
6. **陪练**：根据我选的模块，创建 `harness-practice/` 目录，带我真实跑一遍。
   - 每一步先解释原理 → 再写代码 → 再运行 → 让我确认现象。
   - 每加一个新组件，都要让我先预测 "加了这个会发生什么"，再验证。

## C — Context（上下文）

- 我是**新手**，刚接触 Claude Code。Python 基础会一点（能看懂函数和字典），Shell 会基本命令。
- 我的最终目标：能看懂主流 Harness（Claude Code、Aider、SWE-agent）的源码结构，并能自己搭一个小的给特定任务用。
- 边界澄清（文档里必须讲清楚，别让我混淆）：
  - **LLM ≠ Agent**：LLM 是大脑，Harness 是身体，Agent = 大脑 + 身体 + 目标。
  - **Harness ≠ Framework**：LangChain 是框架，Claude Code 是 harness。Harness 更偏"执行壳"，Framework 更偏"抽象库"。
  - **Ralph Loop 是 Harness 的一个极简子集**（shell + while + CLI），不是 Harness 的全部。
- 我担心的事：token 烧太快、工具调用死循环、不知道怎么调试 Agent 跑偏。文档里请针对这三点给出具体手段。
- 环境：macOS / Linux，Python 3.10+，会用 `pip install anthropic`。
- 我有 Anthropic API Key，可以真实调用 Claude。

## E — Example（样板，请照此风格写每个知识点）

> ### 1.3 Tool Use 循环 —— Harness 的心跳
> **🔧 这是什么**  
> 一个 while 循环：拿模型的回复 → 如果模型说要调工具就去调 → 把工具结果塞回消息历史 → 再问模型 → 直到模型说"我说完了"（`stop_reason == "end_turn"`）。
>
> **✅ 这么做，你会得到**  
> - 模型能"动手"而不只是"动嘴"：它想读文件，你就帮它读，读完把内容给它看。
> - 一次对话可以完成多步任务，而不是一问一答结束。
>
> **⚠️ 改动它，会发生什么**  
> - 忘记把 `tool_result` 加回 `messages` → 模型下一轮看不到工具结果，会反复调同一个工具。
> - 不设最大轮数上限 → 模型和工具之间可能死循环，token 账单爆炸。
> - 把多个 `tool_use` 块只处理第一个 → 模型并行调了 3 个工具，你只回了 1 个，它会困惑。
>
> **🧪 最小可跑示例**
> ```python
> import anthropic
> 
> client = anthropic.Anthropic()
> tools = [{
>     "name": "get_time",
>     "description": "返回当前时间字符串",
>     "input_schema": {"type": "object", "properties": {}}
> }]
> 
> def run_tool(name, args):
>     if name == "get_time":
>         import datetime; return datetime.datetime.now().isoformat()
> 
> messages = [{"role": "user", "content": "现在几点？用工具查。"}]
> for _ in range(5):  # 最大 5 轮，防死循环
>     resp = client.messages.create(
>         model="claude-sonnet-4-5", max_tokens=1024,
>         tools=tools, messages=messages)
>     messages.append({"role": "assistant", "content": resp.content})
>     if resp.stop_reason == "end_turn": break
>     tool_results = []
>     for block in resp.content:
>         if block.type == "tool_use":
>             tool_results.append({
>                 "type": "tool_result", "tool_use_id": block.id,
>                 "content": run_tool(block.name, block.input)})
>     messages.append({"role": "user", "content": tool_results})
> 
> print(resp.content[-1].text)
> ```

---

**请现在开始执行 Action 第 1 步（搜一手资料）。搜完先把找到的关键来源列给我看，再进入第 2 步。**