"""
练习 A - Step 2：注册工具，看模型"想调工具"时的响应结构
目的：亲眼看 finish_reason 和 tool_calls 字段变化
"""
import os, json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

# 注册一个工具：获取当前时间
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "返回当前的时间字符串",
            "parameters": {
                "type": "object",
                "properties": {},   # 这个工具不需要任何参数
            },
        },
    }
]

resp = client.chat.completions.create(
    model="moonshot-v1-8k",
    max_tokens=256,
    tools=tools,
    messages=[{"role": "user", "content": "现在几点了？用工具查一下。"}],
)

print("=== 模型想调工具时的响应结构 ===")
print(f"finish_reason : {resp.choices[0].finish_reason}")
print(f"content       : {resp.choices[0].message.content}")
print()

if resp.choices[0].message.tool_calls:
    for tc in resp.choices[0].message.tool_calls:
        print(f"tool_call id    : {tc.id}")
        print(f"tool_call name  : {tc.function.name}")
        print(f"tool_call args  : {tc.function.arguments}")
else:
    print("tool_calls: None（模型没有调用工具）")
