"""
练习 B - Step 2：实现 dispatcher + 错误处理 + 完整循环
核心：所有工具执行都经过 dispatch()，统一处理错误和日志
"""
import os, json, subprocess, time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

# ── 工具定义（和 Step 1 一样）──────────────────────────────
tools = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "读取本地文件的文本内容并返回",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "把文本内容写入本地文件（覆盖写）",
        "parameters": {"type": "object",
                       "properties": {"path":    {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "执行一条 shell 命令，返回 stdout 输出",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]

# ── Dispatcher：工具路由 + 错误处理 ────────────────────────
def dispatch(name, args):
    """执行工具，出错时返回错误描述字符串（不抛异常）"""
    try:
        if name == "read_file":
            with open(args["path"], "r", encoding="utf-8") as f:
                return f.read()

        elif name == "write_file":
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"已写入 {args['path']}"

        elif name == "run_shell":
            result = subprocess.run(
                args["cmd"], shell=True,
                capture_output=True, text=True, timeout=10
            )
            # stdout 优先，没有则返回 stderr
            return result.stdout or result.stderr or "（命令执行完毕，无输出）"

        else:
            return f"错误：未知工具 '{name}'"

    except FileNotFoundError:
        return f"错误：文件不存在 → {args.get('path', '?')}"
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（>10秒）"
    except Exception as e:
        return f"错误：{type(e).__name__}: {str(e)[:120]}"

# ── Agent Loop ──────────────────────────────────────────────
messages = [{
    "role": "user",
    "content": (
        "用 shell 命令列出 /tmp 目录下的文件，"
        "把结果保存到 /tmp/filelist.txt，"
        "再读出来告诉我有哪些文件。"
    )
}]

print("=== Agent 开始执行任务 ===\n")
total_tokens = 0

for turn in range(10):
    t0 = time.time()
    resp = client.chat.completions.create(
        model="moonshot-v1-8k",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )
    elapsed = time.time() - t0
    tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
    total_tokens += tokens
    finish = resp.choices[0].finish_reason
    msg = resp.choices[0].message

    print(f"[轮 {turn+1}] finish={finish}  tokens={tokens}  耗时={elapsed:.1f}s")

    # 把模型回复追加进 messages
    messages.append({
        "role": "assistant",
        "content": msg.content,
        "tool_calls": msg.tool_calls,
    })

    # 正常完成
    if finish == "stop":
        print(f"\n✅ 任务完成！\n{msg.content}")
        break

    # 执行工具
    if finish == "tool_calls" and msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  → {tc.function.name}({args})")

            result = dispatch(tc.function.name, args)
            is_error = result.startswith("错误：")
            print(f"  ← {'❌' if is_error else '✅'} {result[:80]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

print(f"\n总 token 消耗：{total_tokens}，messages 共 {len(messages)} 条")
