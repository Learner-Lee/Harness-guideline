"""
练习 D - Step 2：沙箱路径隔离
所有文件操作限制在 WORKSPACE，Shell 命令限白名单
"""
import os, json, subprocess
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

WORKSPACE     = "/tmp/agent_workspace"
SHELL_WHITELIST = {"ls", "cat", "echo", "find", "wc", "pwd", "head", "tail"}
os.makedirs(WORKSPACE, exist_ok=True)

tools = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": f"读取 {WORKSPACE}/ 内的文件（只能访问工作区）",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string",
                                               "description": "相对于工作区的路径"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": f"写入 {WORKSPACE}/ 内的文件（只能访问工作区）",
        "parameters": {"type": "object",
                       "properties": {"path":    {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": f"在 {WORKSPACE} 内执行白名单命令：{SHELL_WHITELIST}",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]

# ── 沙箱核心：路径验证 ───────────────────────────────────────
def safe_path(relative_path):
    """
    把相对路径解析成绝对路径，并验证在 WORKSPACE 内。
    返回安全的绝对路径，或在路径穿越时抛 PermissionError。
    """
    abs_path = os.path.realpath(os.path.join(WORKSPACE, relative_path))
    if not abs_path.startswith(os.path.realpath(WORKSPACE)):
        raise PermissionError(f"路径穿越攻击被拦截：{relative_path} → {abs_path}")
    return abs_path

def dispatch(name, args):
    try:
        if name == "read_file":
            path = safe_path(args["path"])
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        elif name == "write_file":
            path = safe_path(args["path"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(args["content"])
            return "已写入 " + path

        elif name == "run_shell":
            cmd_name = args["cmd"].strip().split()[0]
            if cmd_name not in SHELL_WHITELIST:
                return f"[拒绝] '{cmd_name}' 不在白名单中，允许：{SHELL_WHITELIST}"
            r = subprocess.run(args["cmd"], shell=True, capture_output=True,
                               text=True, timeout=5, cwd=WORKSPACE)
            return r.stdout or r.stderr or "（无输出）"

    except PermissionError as e:
        return "🚫 安全拦截：" + str(e)
    except FileNotFoundError:
        return "错误：文件不存在 → " + args.get("path", "?")
    except Exception as e:
        return "错误：" + type(e).__name__ + ": " + str(e)[:120]

# ── 测试 1：正常任务 ─────────────────────────────────────────
print("=== 测试 1：正常任务（工作区内操作）===\n")
messages = [{"role": "user",
             "content": "创建 hello.txt 写入 'hello sandbox'，再用 ls 列出工作区文件，读出 hello.txt 告诉我内容。"}]

for _ in range(8):
    resp = client.chat.completions.create(
        model="moonshot-v1-8k", max_tokens=512, tools=tools, messages=messages)
    finish = resp.choices[0].finish_reason
    msg = resp.choices[0].message
    messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
    if finish == "stop":
        print("✅ " + msg.content + "\n")
        break
    for tc in (msg.tool_calls or []):
        args = json.loads(tc.function.arguments)
        result = dispatch(tc.function.name, args)
        print(f"  {tc.function.name}({args})  →  {result[:60]}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# ── 测试 2：路径穿越攻击 ─────────────────────────────────────
print("=== 测试 2：路径穿越攻击模拟 ===\n")
attack_cases = [
    ("../../etc/passwd",         "read_file"),
    ("../../../root/.ssh/id_rsa","read_file"),
    ("subdir/../../etc/hosts",   "write_file"),
]
for path, tool in attack_cases:
    args = {"path": path, "content": "pwned"} if tool == "write_file" else {"path": path}
    result = dispatch(tool, args)
    print(f"  路径: {path}")
    print(f"  结果: {result}\n")
