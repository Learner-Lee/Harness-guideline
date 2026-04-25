"""
练习 A - Step 3：完整 Agent Loop
这是 Harness 的心跳：while 循环 + 工具执行 + messages 累积
"""
import os, datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "返回当前的时间字符串",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

# ① 初始 messages
messages = [{"role": "user", "content": "现在几点？用工具查，然后用中文告诉我。"}]

print("=== Agent Loop 开始 ===\n")

for turn in range(10):   # 最多 10 轮，防死循环
    print(f"--- 第 {turn + 1} 轮 ---")

    # ② 调 LLM
    resp = client.chat.completions.create(
        model="moonshot-v1-8k",
        max_tokens=256,
        tools=tools,
        messages=messages,
    )

    msg = resp.choices[0].message
    finish = resp.choices[0].finish_reason
    print(f"finish_reason: {finish}")

    # ③ 把模型回复追加进 messages（无论是文字还是工具调用都要追加）
    messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

    # ④ 判断停止条件
    if finish == "stop":
        print(f"\n✅ 完成！模型最终回复：\n{msg.content}")
        break

    # ⑤ 执行工具，把结果追加回 messages
    if finish == "tool_calls" and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  → 模型要调用工具: {tc.function.name}()")

            # 执行工具
            result = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  → 工具执行结果: {result}")

            # 把结果追加进 messages，role 必须是 "tool"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,   # 必须和 tool_calls 里的 id 对应
                "content": result,
            })

print(f"\n=== 结束，messages 共 {len(messages)} 条 ===")
for i, m in enumerate(messages):
    role = m["role"]
    content = str(m.get("content", ""))[:50]
    print(f"  [{i}] {role}: {content}")
