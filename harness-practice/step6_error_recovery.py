"""
练习 B - Step 3：验证错误处理的价值
故意让 Agent 读一个不存在的文件，看它怎么自己恢复
"""
import os, json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

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
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]

def dispatch(name, args):
    try:
        if name == "read_file":
            with open(args["path"], "r") as f:
                return f.read()
        elif name == "write_file":
            with open(args["path"], "w") as f:
                f.write(args["content"])
            return "已写入 " + args["path"]
    except FileNotFoundError:
        return "错误：文件不存在 → " + args.get("path", "?")
    except Exception as e:
        return "错误：" + type(e).__name__ + ": " + str(e)[:120]

task = (
    "读取 /tmp/nonexistent_abc.txt 的内容。"
    "如果文件不存在，就自己创建它，写入 'hello from agent'，再读出来告诉我。"
)
messages = [{"role": "user", "content": task}]

print("=== 错误恢复测试 ===\n")
for turn in range(8):
    resp = client.chat.completions.create(
        model="moonshot-v1-8k", max_tokens=512, tools=tools, messages=messages)
    finish = resp.choices[0].finish_reason
    msg = resp.choices[0].message
    print(f"[轮 {turn+1}] finish={finish}")
    messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

    if finish == "stop":
        print("\n最终回复：" + msg.content)
        break

    for tc in (msg.tool_calls or []):
        args = json.loads(tc.function.arguments)
        result = dispatch(tc.function.name, args)
        print("  → " + tc.function.name + "(" + str(args) + ")")
        print("  ← " + result[:80])
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
