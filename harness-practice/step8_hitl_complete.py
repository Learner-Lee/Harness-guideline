"""
练习 C - Step 2：Token 截断 + Human-in-the-loop 完整版
这是练习 B dispatcher 的生产级升级
"""
import os, json, subprocess, time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

# ── 配置 ────────────────────────────────────────────────────
TOKEN_WARN_AT   = 4000          # 超过这个数就警告
READ_CHAR_LIMIT = 3000          # read_file 最多返回多少字符（≈750 tokens）

# 危险工具/命令的判断规则
DANGEROUS_TOOLS = {"write_file"}                        # 整个工具都需要确认
DANGEROUS_SHELL_KEYWORDS = {"rm", "rmdir", "mkfs",      # shell 命令里含这些词需要确认
                             "dd", "chmod", ">", "mv"}

# ── 工具定义 ────────────────────────────────────────────────
tools = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": f"读取本地文件，超过 {READ_CHAR_LIMIT} 字符自动截断",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "把文本写入本地文件（覆盖写，危险操作，会提示用户确认）",
        "parameters": {"type": "object",
                       "properties": {"path":    {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "执行 shell 命令，危险命令（rm/dd 等）会提示用户确认",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]

# ── Human-in-the-loop 确认 ───────────────────────────────────
def ask_confirm(tool_name, args):
    """返回 True 表示用户确认，False 表示拒绝"""
    print(f"\n⚠️  【需要确认】Agent 要执行危险操作：")
    print(f"   工具：{tool_name}")
    print(f"   参数：{json.dumps(args, ensure_ascii=False)}")
    answer = input("   确认执行？[y/N] ").strip().lower()
    return answer == "y"

def is_dangerous_shell(cmd):
    tokens = cmd.replace(";", " ").replace("&&", " ").split()
    return any(t in DANGEROUS_SHELL_KEYWORDS for t in tokens)

# ── Dispatcher（带截断 + HITL）──────────────────────────────
def dispatch(name, args):
    # 判断是否需要人工确认
    needs_confirm = False
    if name in DANGEROUS_TOOLS:
        needs_confirm = True
    elif name == "run_shell" and is_dangerous_shell(args.get("cmd", "")):
        needs_confirm = True

    if needs_confirm:
        confirmed = ask_confirm(name, args)
        if not confirmed:
            return "用户拒绝了此操作，请换一个方案。"

    try:
        if name == "read_file":
            with open(args["path"], "r", encoding="utf-8") as f:
                content = f.read()
            # 超过限制时截断
            if len(content) > READ_CHAR_LIMIT:
                truncated = content[:READ_CHAR_LIMIT]
                note = f"\n\n[注意：文件共 {len(content)} 字符，已截断至前 {READ_CHAR_LIMIT} 字符]"
                return truncated + note
            return content

        elif name == "write_file":
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(args["content"])
            return "已写入 " + args["path"]

        elif name == "run_shell":
            r = subprocess.run(args["cmd"], shell=True,
                               capture_output=True, text=True, timeout=10)
            return r.stdout or r.stderr or "（无输出）"

    except FileNotFoundError:
        return "错误：文件不存在 → " + args.get("path", "?")
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（>10秒）"
    except Exception as e:
        return "错误：" + type(e).__name__ + ": " + str(e)[:120]

# ── Agent Loop ───────────────────────────────────────────────
task = (
    "读取 /tmp/bigfile.txt 的内容，"
    "统计有多少行，然后把统计结果写入 /tmp/summary.txt，"
    "最后告诉我结果。"
)
messages = [{"role": "user", "content": task}]

print("=== 完整 Harness（Token 监控 + HITL）===")
print(f"任务：{task}\n")

total_in = total_out = 0

for turn in range(10):
    t0 = time.time()
    resp = client.chat.completions.create(
        model="moonshot-v1-8k",
        max_tokens=512,
        tools=tools,
        messages=messages,
    )
    elapsed = time.time() - t0
    inp = resp.usage.prompt_tokens
    out = resp.usage.completion_tokens
    total_in += inp; total_out += out
    finish = resp.choices[0].finish_reason
    msg = resp.choices[0].message

    # Token 状态
    warn = " ⚠️ 接近上限！" if inp > TOKEN_WARN_AT else ""
    print(f"[轮 {turn+1}] finish={finish}  input={inp}{warn}  output={out}  耗时={elapsed:.1f}s")

    messages.append({
        "role": "assistant",
        "content": msg.content,
        "tool_calls": msg.tool_calls,
    })

    if finish == "stop":
        print(f"\n✅ 完成！\n{msg.content}")
        break

    for tc in (msg.tool_calls or []):
        args = json.loads(tc.function.arguments)
        result = dispatch(tc.function.name, args)
        chars = len(result)
        print(f"  → {tc.function.name}()  结果 {chars} 字符")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

print(f"\n总消耗：input={total_in}  output={total_out}  合计={total_in+total_out}")
print(f"messages 共 {len(messages)} 条")
