"""
harness_v1.py — 练习 A-D 所有组件的合体版
组件清单：
  ✅ Agent Loop        (while + finish_reason)
  ✅ 三工具 dispatcher  (read_file / write_file / run_shell)
  ✅ 错误回传          (异常不抛出，塞回给模型)
  ✅ 大文件截断        (READ_CHAR_LIMIT)
  ✅ Token 监控        (每轮打印，接近上限时警告)
  ✅ Human-in-the-loop (危险操作前暂停确认)
  ✅ 检查点 & 恢复     (原子写入，工具结果追加后再保存)
  ✅ 沙箱路径隔离      (workspace + 白名单命令)

用法：
  MOONSHOT_API_KEY=xxx python3 harness_v1.py "你的任务"
"""

import os, sys, json, subprocess, time
from datetime import datetime
from openai import OpenAI

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
MODEL           = "moonshot-v1-8k"
MAX_TURNS       = 20
TOKEN_WARN_AT   = 5000          # 超过这个数就警告
READ_CHAR_LIMIT = 4000          # read_file 最多返回多少字符
SAVE_EVERY      = 3             # 每 N 轮保存一次检查点
CHECKPOINT_FILE = "/tmp/harness_v1_ckpt.json"
WORKSPACE       = "/tmp/harness_workspace"
SHELL_WHITELIST = {"ls", "cat", "echo", "find", "wc", "head",
                   "tail", "grep", "sort", "uniq", "pwd"}
DANGEROUS_TOOLS = {"write_file"}
DANGEROUS_SHELL = {"rm", "rmdir", "dd", "mkfs", "chmod", "chown",
                   "mv", "cp", ">", ">>", "tee"}

# ══════════════════════════════════════════════════════════
#  CLIENT
# ══════════════════════════════════════════════════════════
client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

# ══════════════════════════════════════════════════════════
#  TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": (
            f"读取工作区文件的文本内容。"
            f"超过 {READ_CHAR_LIMIT} 字符自动截断并提示。"
            f"只能访问 {WORKSPACE}/ 内的文件。"
        ),
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string",
                                    "description": "相对于工作区的路径，如 report.txt"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": (
            f"把文本写入工作区文件（覆盖写）。"
            f"只能写入 {WORKSPACE}/ 内的文件。危险操作，会提示用户确认。"
        ),
        "parameters": {"type": "object",
                       "properties": {
                           "path":    {"type": "string", "description": "相对于工作区的路径"},
                           "content": {"type": "string", "description": "要写入的文本"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": (
            f"在工作区执行 shell 命令。"
            f"白名单命令：{sorted(SHELL_WHITELIST)}。"
            f"危险命令（{sorted(DANGEROUS_SHELL)}）会提示用户确认。"
        ),
        "parameters": {"type": "object",
                       "properties": {
                           "cmd": {"type": "string", "description": "要执行的 shell 命令"}},
                       "required": ["cmd"]}}},
]

# ══════════════════════════════════════════════════════════
#  SANDBOX
# ══════════════════════════════════════════════════════════
def safe_path(relative):
    """解析路径并验证在 WORKSPACE 内，防止路径穿越。"""
    abs_path = os.path.realpath(os.path.join(WORKSPACE, relative))
    if not abs_path.startswith(os.path.realpath(WORKSPACE) + os.sep) and \
       abs_path != os.path.realpath(WORKSPACE):
        raise PermissionError(f"路径穿越被拦截：{relative!r} → {abs_path}")
    return abs_path

def is_dangerous_shell(cmd):
    tokens = set(cmd.replace(";", " ").replace("&&", " ").split())
    return bool(tokens & DANGEROUS_SHELL)

# ══════════════════════════════════════════════════════════
#  HUMAN-IN-THE-LOOP
# ══════════════════════════════════════════════════════════
def ask_confirm(tool_name, args):
    print(f"\n⚠️  危险操作需要确认")
    print(f"   工具：{tool_name}")
    print(f"   参数：{json.dumps(args, ensure_ascii=False, indent=4)}")
    try:
        answer = input("   确认执行？[y/N] ").strip().lower()
    except EOFError:
        answer = "y"   # 非交互模式默认同意（测试/CI 场景）
    return answer == "y"

# ══════════════════════════════════════════════════════════
#  DISPATCHER
# ══════════════════════════════════════════════════════════
def dispatch(name, args):
    """执行工具：含沙箱验证、HITL、错误回传。"""

    # 判断是否需要人工确认
    needs_confirm = (name in DANGEROUS_TOOLS) or \
                    (name == "run_shell" and is_dangerous_shell(args.get("cmd", "")))
    if needs_confirm:
        if not ask_confirm(name, args):
            return "用户拒绝了此操作，请换一个方案。"

    try:
        if name == "read_file":
            path = safe_path(args["path"])
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > READ_CHAR_LIMIT:
                note = f"\n\n[已截断：文件共 {len(content)} 字符，只返回前 {READ_CHAR_LIMIT} 字符]"
                return content[:READ_CHAR_LIMIT] + note
            return content

        elif name == "write_file":
            path = safe_path(args["path"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"已写入：{path}（{len(args['content'])} 字符）"

        elif name == "run_shell":
            cmd_name = args["cmd"].strip().split()[0]
            if cmd_name not in SHELL_WHITELIST and not is_dangerous_shell(args["cmd"]):
                return f"[拒绝] '{cmd_name}' 不在白名单中。允许：{sorted(SHELL_WHITELIST)}"
            result = subprocess.run(
                args["cmd"], shell=True, capture_output=True,
                text=True, timeout=15, cwd=WORKSPACE)
            output = result.stdout or result.stderr or "（命令执行完毕，无输出）"
            # shell 输出也截断，防止暴增 token
            if len(output) > READ_CHAR_LIMIT:
                output = output[:READ_CHAR_LIMIT] + f"\n[输出已截断，共 {len(output)} 字符]"
            return output

        else:
            return f"错误：未知工具 '{name}'"

    except PermissionError as e:
        return "🚫 安全拦截：" + str(e)
    except FileNotFoundError:
        return "错误：文件不存在 → " + args.get("path", args.get("cmd", "?"))
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（>15秒）"
    except Exception as e:
        return f"错误：{type(e).__name__}: {str(e)[:150]}"

# ══════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════
def tc_to_dict(tc):
    if isinstance(tc, dict):
        return tc
    return {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name,
                         "arguments": tc.function.arguments}}

def serialize_messages(messages):
    rows = []
    for m in messages:
        e = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            e["tool_calls"] = [tc_to_dict(tc) for tc in m["tool_calls"]]
        if "tool_call_id" in m:
            e["tool_call_id"] = m["tool_call_id"]
        rows.append(e)
    return rows

def save_checkpoint(messages, turn):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"turn": turn, "messages": serialize_messages(messages)},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)   # 原子替换，防止写到一半崩溃
    log(f"💾 检查点已保存（第 {turn} 轮，{len(messages)} 条消息）")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                data = json.load(f)
            log(f"🔄 发现检查点，从第 {data['turn']} 轮恢复（{len(data['messages'])} 条消息）")
            return data["messages"], data["turn"]
        except Exception as e:
            log(f"⚠️ 检查点损坏，忽略：{e}")
    return None, 0

# ══════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ══════════════════════════════════════════════════════════
#  AGENT LOOP
# ══════════════════════════════════════════════════════════
def run(task, system=None, fresh=False):
    os.makedirs(WORKSPACE, exist_ok=True)

    if fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("🗑️ 已清除旧检查点")

    messages, start_turn = load_checkpoint()
    if messages is None:
        messages = []
        if system:
            # system 消息作为 messages[0]（Kimi 兼容写法）
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": task})
        log("🆕 全新开始")
    else:
        log(f"   继续执行（从第 {start_turn + 1} 轮）")

    log(f"📋 任务：{task[:80]}{'...' if len(task) > 80 else ''}")
    print()

    total_in = total_out = 0

    for turn in range(start_turn, start_turn + MAX_TURNS):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        elapsed = time.time() - t0

        inp = resp.usage.prompt_tokens
        out = resp.usage.completion_tokens
        total_in += inp
        total_out += out
        finish = resp.choices[0].finish_reason
        msg    = resp.choices[0].message

        # Token 监控
        warn = "  ⚠️ 接近上限！" if inp > TOKEN_WARN_AT else ""
        log(f"[轮 {turn}] finish={finish}  in={inp}{warn}  out={out}  {elapsed:.1f}s")

        messages.append({
            "role":       "assistant",
            "content":    msg.content,
            "tool_calls": msg.tool_calls,
        })

        # 正常完成
        if finish == "stop":
            print()
            log("✅ 任务完成！")
            print("─" * 60)
            print(msg.content)
            print("─" * 60)
            log(f"总消耗：input={total_in}  output={total_out}  合计={total_in+total_out}")
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                log("🗑️ 检查点已清除")
            return msg.content

        # 执行工具
        if finish == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                d    = tc_to_dict(tc)
                args = json.loads(d["function"]["arguments"])
                log(f"  → {d['function']['name']}({json.dumps(args, ensure_ascii=False)[:60]})")
                result = dispatch(d["function"]["name"], args)
                log(f"  ← {result[:80]}{'...' if len(result) > 80 else ''}")
                messages.append({
                    "role":        "tool",
                    "tool_call_id": d["id"],
                    "content":     result,
                })

        # 检查点（工具结果追加完之后）
        if (turn + 1) % SAVE_EVERY == 0:
            save_checkpoint(messages, turn)

    log("⚠️ 达到最大轮数，任务未完成")
    return None

# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 harness_v1.py '任务描述'")
        sys.exit(1)

    task = sys.argv[1]
    system = (
        "你是一个高效的本地文件助手。"
        "工作区目录是 /tmp/harness_workspace/。"
        "所有文件操作都在这个目录内进行。"
        "遇到问题时，先用工具查清楚再行动，不要猜测。"
    )
    run(task, system=system, fresh="--fresh" in sys.argv)
