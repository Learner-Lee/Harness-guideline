"""
练习 A - Step 1：裸调 API，看响应结构
目的：在加工具之前，先搞清楚 resp 对象长什么样
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

resp = client.chat.completions.create(
    model="moonshot-v1-8k",
    max_tokens=256,
    messages=[{"role": "user", "content": "1+1 等于几？"}],
)

print("=== 响应结构 ===")
print(f"finish_reason : {resp.choices[0].finish_reason}")
print(f"role          : {resp.choices[0].message.role}")
print(f"content       : {resp.choices[0].message.content}")
print(f"tool_calls    : {resp.choices[0].message.tool_calls}")
print()
print(f"input tokens  : {resp.usage.prompt_tokens}")
print(f"output tokens : {resp.usage.completion_tokens}")
