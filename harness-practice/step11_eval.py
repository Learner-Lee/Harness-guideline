"""
练习 D - Step 3：简单 Eval 套件
测试 Agent 对不同类型任务的完成质量，量化通过率
"""
import os, json, subprocess
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

WORKSPACE = "/tmp/agent_workspace"
os.makedirs(WORKSPACE, exist_ok=True)

# ── 工具定义（复用沙箱版本）────────────────────────────────
tools = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "写入工作区文件",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "在工作区执行 shell 命令（ls/cat/echo/wc）",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]

SHELL_WHITELIST = {"ls", "cat", "echo", "wc", "head", "tail", "find"}

def dispatch(name, args):
    try:
        if name == "write_file":
            path = os.path.join(WORKSPACE, os.path.basename(args["path"]))
            with open(path, "w") as f: f.write(args["content"])
            return "已写入 " + path
        elif name == "run_shell":
            cmd_name = args["cmd"].strip().split()[0]
            if cmd_name not in SHELL_WHITELIST:
                return f"[拒绝] {cmd_name} 不在白名单"
            r = subprocess.run(args["cmd"], shell=True, capture_output=True,
                               text=True, timeout=5, cwd=WORKSPACE)
            return r.stdout or r.stderr or "（无输出）"
    except Exception as e:
        return "错误：" + str(e)[:80]

# ── Agent 运行器 ────────────────────────────────────────────
def run_agent(task, max_turns=8):
    """运行 Agent，返回最终回复文本"""
    messages = [{"role": "user", "content": task}]
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model="moonshot-v1-8k", max_tokens=512, tools=tools, messages=messages)
        finish = resp.choices[0].finish_reason
        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
        if finish == "stop":
            return msg.content or ""
        for tc in (msg.tool_calls or []):
            args = json.loads(tc.function.arguments)
            result = dispatch(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "（未在最大轮数内完成）"

# ── LLM-as-Judge 评分 ───────────────────────────────────────
def judge(task, response, criteria):
    """用轻量模型判断 Agent 回复是否满足标准，返回 True/False"""
    prompt = (
        f"任务：{task}\n"
        f"Agent 回复：{response}\n"
        f"评判标准：{criteria}\n\n"
        f"Agent 的回复是否满足评判标准？只回复 YES 或 NO，不要其他内容。"
    )
    resp = client.chat.completions.create(
        model="moonshot-v1-8k",
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("Y")

# ── 测试用例 ────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC-01",
        "task": "用 echo 命令在工作区创建一个 test.txt，写入 'hello world'，然后告诉我文件已创建。",
        "criteria": "回复中提到文件已创建或写入成功，且包含 test.txt 或 hello world"
    },
    {
        "id": "TC-02",
        "task": "用 shell 命令统计工作区有几个文件，告诉我数量。",
        "criteria": "回复中包含具体的数字，说明工作区文件数量"
    },
    {
        "id": "TC-03",
        "task": "写一个名为 report.txt 的文件，内容是今天的学习总结：我学习了 Agent Harness。",
        "criteria": "回复确认 report.txt 已写入，且提到了 Agent Harness 或学习总结"
    },
    {
        "id": "TC-04",
        "task": "用 ls 列出工作区的所有文件，然后用 wc 统计 test.txt 的行数，告诉我结果。",
        "criteria": "回复中同时包含文件列表信息和 test.txt 的行数"
    },
]

# ── 运行 Eval ────────────────────────────────────────────────
print("=" * 50)
print("  Agent Eval 套件")
print("=" * 50)

passed = 0
results = []

for tc in TEST_CASES:
    print(f"\n[{tc['id']}] {tc['task'][:45]}...")
    response = run_agent(tc["task"])
    ok = judge(tc["task"], response, tc["criteria"])
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  状态   : {status}")
    print(f"  回复   : {response[:80]}...")
    print(f"  标准   : {tc['criteria'][:60]}...")
    if ok: passed += 1
    results.append({"id": tc["id"], "pass": ok})

print("\n" + "=" * 50)
print(f"  通过率：{passed}/{len(TEST_CASES)} = {passed/len(TEST_CASES)*100:.0f}%")
print("=" * 50)

# 保存 eval 结果到文件，方便后续对比
with open(os.path.join(WORKSPACE, "eval_results.json"), "w") as f:
    json.dump({"passed": passed, "total": len(TEST_CASES), "results": results}, f, indent=2)
print("\nEval 结果已保存至 agent_workspace/eval_results.json")
