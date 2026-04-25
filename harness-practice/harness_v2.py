"""
harness_v2.py — 对照 Claude Code 源码升级的版本

v1 → v2 三处核心升级（来自练习 E 的发现）：
  ① 工具对象合一   Tool(schema, run) 把 schema 和执行函数绑在一起，不再分离
  ② 并行工具执行   ThreadPoolExecutor 并行跑多个 tool_use，对应 CC 的 Promise.all
  ③ 自动上下文压缩  token 超阈值时调 LLM 生成摘要替换历史，对应 CC 的 compaction
  ④ 停止条件改写   build_tool_result() 返回 None → 没有 tool_use → 退出，更底层

用法：
  MOONSHOT_API_KEY=xxx python3 harness_v2.py "你的任务"
  MOONSHOT_API_KEY=xxx python3 harness_v2.py "任务" --fresh
"""

import os, sys, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any
from openai import OpenAI

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
MODEL              = "moonshot-v1-8k"
MAX_ITERATIONS     = 20
COMPACTION_TOKENS  = 5000          # 对应 CC 的 ME6=1e5，按 8k 窗口等比缩小
CHECKPOINT_FILE    = "/tmp/harness_v2_ckpt.json"
WORKSPACE          = "/tmp/harness_v2_workspace"
SHELL_WHITELIST    = {"ls", "cat", "echo", "find", "wc", "head",
                      "tail", "grep", "sort", "uniq", "pwd"}
DANGEROUS_TOOLS    = {"write_file"}
DANGEROUS_SHELL    = {"rm", "rmdir", "dd", "mkfs", "chmod",
                      "chown", "mv", ">", ">>", "tee"}

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

# ══════════════════════════════════════════════════════════
#  ① 工具对象合一：Tool dataclass
#  CC：工具对象同时含 schema 和 run() 方法
#  v1：TOOLS 列表（schema）和 dispatch()（执行）完全分离
# ══════════════════════════════════════════════════════════
class ToolError(Exception):
    """对应 CC 源码的 gZH（ToolError 类）：工具可以抛这个来返回结构化错误"""
    def __init__(self, content: str):
        super().__init__(content)
        self.content = content

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], str]
    parse: Callable[[dict], dict] = None   # 可选的输入预处理钩子（对应 CC 的 parse）

    def schema(self) -> dict:
        """返回 API 所需的工具声明格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

# ══════════════════════════════════════════════════════════
#  沙箱 & HITL（从 v1 继承，逻辑不变）
# ══════════════════════════════════════════════════════════
def safe_path(relative: str) -> str:
    abs_path = os.path.realpath(os.path.join(WORKSPACE, relative))
    workspace_real = os.path.realpath(WORKSPACE)
    if abs_path != workspace_real and not abs_path.startswith(workspace_real + os.sep):
        raise PermissionError(f"路径穿越被拦截：{relative!r} → {abs_path}")
    return abs_path

def is_dangerous_shell(cmd: str) -> bool:
    return bool(set(cmd.replace(";", " ").replace("&&", " ").split()) & DANGEROUS_SHELL)

def ask_confirm(tool_name: str, args: dict) -> bool:
    print(f"\n⚠️  危险操作需要确认")
    print(f"   工具：{tool_name}")
    print(f"   参数：{json.dumps(args, ensure_ascii=False, indent=4)}")
    try:
        answer = input("   确认执行？[y/N] ").strip().lower()
    except EOFError:
        answer = "y"
    return answer == "y"

# ══════════════════════════════════════════════════════════
#  工具实现函数
# ══════════════════════════════════════════════════════════
READ_CHAR_LIMIT = 4000

def _run_read_file(args: dict) -> str:
    path = safe_path(args["path"])
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > READ_CHAR_LIMIT:
        return content[:READ_CHAR_LIMIT] + f"\n[截断：文件共 {len(content)} 字符，只返回前 {READ_CHAR_LIMIT}]"
    return content

def _run_write_file(args: dict) -> str:
    if not ask_confirm("write_file", args):
        return "用户拒绝了此操作，请换方案。"
    path = safe_path(args["path"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(args["content"])
    return f"已写入：{path}（{len(args['content'])} 字符）"

def _run_shell(args: dict) -> str:
    cmd = args["cmd"]
    cmd_name = cmd.strip().split()[0]
    if is_dangerous_shell(cmd):
        if not ask_confirm("run_shell", args):
            return "用户拒绝了此操作，请换方案。"
    elif cmd_name not in SHELL_WHITELIST:
        return f"[拒绝] '{cmd_name}' 不在白名单中。允许：{sorted(SHELL_WHITELIST)}"
    r = subprocess.run(cmd, shell=True, capture_output=True,
                       text=True, timeout=15, cwd=WORKSPACE)
    output = r.stdout or r.stderr or "（无输出）"
    if len(output) > READ_CHAR_LIMIT:
        output = output[:READ_CHAR_LIMIT] + f"\n[输出截断，共 {len(output)} 字符]"
    return output

# ══════════════════════════════════════════════════════════
#  工具注册表（schema + run 合一）
# ══════════════════════════════════════════════════════════
TOOLS: list[Tool] = [
    Tool(
        name="read_file",
        description=f"读取工作区文件内容，超过 {READ_CHAR_LIMIT} 字符自动截断。",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string", "description": "相对于工作区的路径"}},
                    "required": ["path"]},
        run=_run_read_file,
    ),
    Tool(
        name="write_file",
        description="把文本写入工作区文件（覆盖写，危险操作，需确认）。",
        parameters={"type": "object",
                    "properties": {"path":    {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["path", "content"]},
        run=_run_write_file,
    ),
    Tool(
        name="run_shell",
        description=f"在工作区执行 shell 命令。白名单：{sorted(SHELL_WHITELIST)}。",
        parameters={"type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]},
        run=_run_shell,
    ),
]

TOOL_MAP = {t.name: t for t in TOOLS}   # name → Tool，对应 CC 的 H.tools.find(...)
SCHEMAS  = [t.schema() for t in TOOLS]  # 传给 API 的纯 schema 列表

# ══════════════════════════════════════════════════════════
#  ② 并行工具执行：build_tool_result()
#  CC：Promise.all(q.map(async (_) => { ... }))
#  v1：串行 for tc in msg.tool_calls
# ══════════════════════════════════════════════════════════
def _execute_single(tc_dict: dict) -> dict:
    """执行单个工具调用，返回 tool_result 消息块。
    tc_dict 格式：{"id": ..., "function": {"name": ..., "arguments": "..."}}
    """
    name = tc_dict["function"]["name"]
    raw_args = tc_dict["function"]["arguments"]
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    tool = TOOL_MAP.get(name)

    if tool is None:
        return {
            "type": "tool_result",
            "tool_call_id": tc_dict["id"],
            "content": f"Error: Tool '{name}' not found",
            "is_error": True,
        }
    try:
        parsed_args = tool.parse(args) if tool.parse else args
        result = tool.run(parsed_args)
        return {"type": "tool_result", "tool_call_id": tc_dict["id"], "content": result}
    except ToolError as e:
        return {"type": "tool_result", "tool_call_id": tc_dict["id"],
                "content": e.content, "is_error": True}
    except Exception as e:
        return {"type": "tool_result", "tool_call_id": tc_dict["id"],
                "content": f"Error: {type(e).__name__}: {str(e)[:150]}", "is_error": True}

def build_tool_result(last_message: dict) -> dict | None:
    """
    对应 CC 的 tG4()：
    - 从 assistant 消息里提取所有 tool_use 块
    - 并行执行（ThreadPoolExecutor 对应 Promise.all）
    - 返回 {role:"user", content:[tool_result, ...]}
    - 没有 tool_use 块时返回 None（这是循环停止的信号）
    """
    if not last_message or last_message.get("role") != "assistant":
        return None

    # tool_calls 存为 OpenAI 格式的列表 [{id, function:{name, arguments}}]
    tool_calls = last_message.get("tool_calls") or []
    if not tool_calls:
        return None

    # 并行执行所有工具
    results = [None] * len(tool_calls)
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        future_to_idx = {executor.submit(_execute_single, tc): i
                         for i, tc in enumerate(tool_calls)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # OpenAI 格式：每个工具结果是独立的 role="tool" 消息
    return [
        {"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]}
        for r in results
    ]

# ══════════════════════════════════════════════════════════
#  ③ 自动上下文压缩（Compaction）
#  CC：token 超 ME6(1e5) 时，调 LLM 生成摘要替换整个 messages 历史
#  v1：只做字符截断
# ══════════════════════════════════════════════════════════

# 对应 CC 的 DE6：硬编码在 Harness 里的 summarization prompt
COMPACTION_PROMPT = """你正在协助完成一项任务，但尚未结束。
请写一份续接摘要，让你（或你的另一个实例）能在新的对话窗口里高效接续工作。

摘要须包含以下五部分：
1. 任务概述：用户的核心请求和成功标准
2. 当前状态：已完成的工作、已创建/修改的文件（含路径）
3. 重要发现：遇到的技术约束、已做的决策、失败的尝试及原因
4. 下一步：完成任务所需的具体操作，按优先级排列
5. 需保留的上下文：用户偏好、领域细节、对用户的承诺

简洁但完整——宁可多写，也不要让后续实例重复已做过的工作。
用 <summary></summary> 标签包裹你的摘要。"""

def maybe_compact(messages: list, last_input_tokens: int) -> tuple[list, bool]:
    """
    如果 token 超过阈值，调 LLM 压缩历史并替换 messages。
    返回 (新messages, 是否触发了压缩)
    """
    if last_input_tokens < COMPACTION_TOKENS:
        return messages, False

    log(f"🗜️  触发上下文压缩（input tokens={last_input_tokens} ≥ {COMPACTION_TOKENS}）")

    # 把当前历史 + 压缩指令发给 LLM
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=800,
        messages=messages + [{"role": "user", "content": COMPACTION_PROMPT}],
    )
    summary_text = resp.choices[0].message.content

    # 提取 <summary> 标签内容，没有标签就用整段回复
    if "<summary>" in summary_text and "</summary>" in summary_text:
        start = summary_text.index("<summary>") + len("<summary>")
        end   = summary_text.index("</summary>")
        summary_text = summary_text[start:end].strip()

    # 用单条摘要消息替换整个历史（对应 CC：params.messages = [{role:"user", content:z.content}]）
    new_messages = [{"role": "user", "content": f"[对话历史摘要]\n{summary_text}"}]
    log(f"   压缩完成：{len(messages)} 条消息 → 1 条摘要（{len(summary_text)} 字符）")
    return new_messages, True

# ══════════════════════════════════════════════════════════
#  检查点（从 v1 继承，逻辑不变）
# ══════════════════════════════════════════════════════════
def save_checkpoint(messages: list, turn: int):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"turn": turn, "messages": messages}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)
    log(f"💾 检查点已保存（第 {turn} 轮，{len(messages)} 条）")

def load_checkpoint() -> tuple[list | None, int]:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                data = json.load(f)
            log(f"🔄 从第 {data['turn']} 轮恢复（{len(data['messages'])} 条消息）")
            return data["messages"], data["turn"]
        except Exception as e:
            log(f"⚠️  检查点损坏，忽略：{e}")
    return None, 0

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ══════════════════════════════════════════════════════════
#  主循环
#  ⑤ 停止条件：build_tool_result() 返回 None（没有 tool_use 块）
#     对应 CC：tG4() 返回 null → break
# ══════════════════════════════════════════════════════════
def run(task: str, system: str = None, fresh: bool = False):
    os.makedirs(WORKSPACE, exist_ok=True)

    if fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("🗑️  已清除旧检查点")

    messages, start_turn = load_checkpoint()
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": task})
        log("🆕 全新开始")
    else:
        log(f"   继续执行（从第 {start_turn + 1} 轮）")

    log(f"📋 任务：{task[:80]}{'...' if len(task) > 80 else ''}")
    print()

    total_in = total_out = 0

    for turn in range(start_turn, start_turn + MAX_ITERATIONS):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            tools=SCHEMAS,
            messages=messages,
        )
        elapsed = time.time() - t0

        inp  = resp.usage.prompt_tokens
        out  = resp.usage.completion_tokens
        total_in  += inp
        total_out += out
        finish = resp.choices[0].finish_reason
        msg    = resp.choices[0].message

        log(f"[轮 {turn}] finish={finish}  in={inp}  out={out}  {elapsed:.1f}s")

        # ── 把 assistant 消息追加到 messages（保持 OpenAI 格式）──────
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            # 序列化 SDK 对象为纯字典，checkpoint 和 build_tool_result 都能用
            assistant_msg["tool_calls"] = [
                {"id": tc.id,
                 "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # ── ③ 自动压缩（token 超阈值时）────────────────────────────
        messages, compacted = maybe_compact(messages, inp)

        # ── ④ 停止条件：build_tool_result() 返回 None ──────────────
        # 对应 CC：tG4() 返回 null → 没有 tool_use 块 → break
        tool_result_msgs = build_tool_result(messages[-1])

        if tool_result_msgs is None:
            final_text = msg.content or ""
            print()
            log("✅ 任务完成！")
            print("─" * 60)
            print(final_text)
            print("─" * 60)
            log(f"总消耗：input={total_in}  output={total_out}  合计={total_in+total_out}")
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                log("🗑️  检查点已清除")
            return final_text

        # ── 打印并行工具调用情况 ─────────────────────────────────────
        for m in tool_result_msgs:
            log(f"  ✅ tool_call_id={m['tool_call_id'][:20]}  "
                f"result={str(m['content'])[:60]}")

        # ── 把工具结果追加进 messages（每个工具一条，检查点在此之后）
        messages.extend(tool_result_msgs)

        if (turn + 1) % 3 == 0:
            save_checkpoint(messages, turn)

    log("⚠️  达到最大轮数，任务未完成")
    return None

# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 harness_v2.py '任务描述' [--fresh]")
        sys.exit(1)

    task = sys.argv[1]
    system = (
        f"你是一个高效的本地文件助手。工作区目录是 {WORKSPACE}/。"
        "所有文件操作在这个目录内进行。遇到问题先用工具查清楚再行动。"
    )
    run(task, system=system, fresh="--fresh" in sys.argv)
