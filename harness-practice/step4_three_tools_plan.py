"""
练习 B - Step 1：注册三个工具，看模型的"调用计划"
只看模型想调什么、传什么参数，还不真正执行
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件的文本内容并返回",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件的绝对路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把文本内容写入本地文件（覆盖写）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "文件的绝对路径"},
                    "content": {"type": "string", "description": "要写入的文本内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "执行一条 shell 命令，返回 stdout 输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "要执行的 shell 命令"}
                },
                "required": ["cmd"],
            },
        },
    },
]

messages = [{
    "role": "user",
    "content": "用 shell 命令列出 /tmp 目录下的文件，把结果保存到 /tmp/filelist.txt，再读出来告诉我有哪些文件。"
}]

resp = client.chat.completions.create(
    model="moonshot-v1-8k",
    max_tokens=512,
    tools=tools,
    messages=messages,
)

print("=== 模型的第一步调用计划 ===")
print(f"finish_reason : {resp.choices[0].finish_reason}")
for tc in (resp.choices[0].message.tool_calls or []):
    print(f"\n  工具名 : {tc.function.name}")
    print(f"  参数   : {tc.function.arguments}")
