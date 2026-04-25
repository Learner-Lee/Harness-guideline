"""
练习 C - Step 1：Token 预算监控
每轮打印 token 消耗，累计超过预算时发出警告
"""
import os, json, subprocess, time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

# moonshot-v1-8k 的上下文窗口是 8000 token，留 20% 作为安全缓冲
TOKEN_BUDGET = 6000
TOKEN_WARN_AT = 4500   # 超过这个数就警告

tools = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "读取本地文件的文本内容并返回",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "执行一条 shell 命令，返回 stdout 输出",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]

def dispatch(name, args):
    try:
        if name == "read_file":
            with open(args["path"], "r", encoding="utf-8") as f:
                return f.read()
        elif name == "run_shell":
            r = subprocess.run(args["cmd"], shell=True, capture_output=True, text=True, timeout=10)
            return r.stdout or r.stderr or "（无输出）"
    except FileNotFoundError:
        return "错误：文件不存在 → " + args.get("path", "?")
    except Exception as e:
        return "错误：" + type(e).__name__ + ": " + str(e)[:120]

def token_bar(used, budget):
    """用字符画出 token 使用进度条"""
    pct = used / budget
    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {used}/{budget} ({pct*100:.0f}%)"

messages = [{
    "role": "user",
    "content": "读取 /tmp/bigfile.txt，统计它有多少行，然后告诉我前 3 行的内容。"
}]

print("=== Token 预算监控演示 ===")
print(f"预算上限：{TOKEN_BUDGET} tokens，{TOKEN_WARN_AT} tokens 时警告\n")

total_input = total_output = 0

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
    total_input += inp
    total_output += out
    total = total_input + total_output
    finish = resp.choices[0].finish_reason
    msg = resp.choices[0].message

    # 每轮打印 token 情况
    print(f"[轮 {turn+1}] {token_bar(inp, TOKEN_BUDGET)}  本轮 in={inp} out={out}  耗时={elapsed:.1f}s")

    # 超过警告线时提示
    if inp > TOKEN_WARN_AT:
        print(f"  ⚠️  警告：本轮 input token ({inp}) 超过安全线 {TOKEN_WARN_AT}！")
        print(f"  ⚠️  建议：缩短历史或对大文件内容做截断再传给模型")

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
        # 打印工具结果的 token 估算（粗略：每4字符≈1 token）
        result_tokens = len(result) // 4
        print(f"  → {tc.function.name}()  结果长度≈{result_tokens} tokens")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

print(f"\n累计：input={total_input}  output={total_output}  合计={total_input+total_output}")
