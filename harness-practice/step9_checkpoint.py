"""
练习 D - Step 1：检查点与恢复
运行两次：第一次模拟"第3轮崩溃"，第二次从检查点恢复
"""
import os, json, time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

CHECKPOINT_FILE = "/tmp/agent_ckpt.json"
SAVE_EVERY      = 2          # 每 2 轮保存一次
CRASH_AT_TURN   = 3          # 模拟第 3 轮崩溃（设为 None 则不崩溃）

tools = [
    {"type": "function", "function": {
        "name": "get_step",
        "description": "返回当前执行步骤编号（模拟多步骤任务）",
        "parameters": {"type": "object",
                       "properties": {"step": {"type": "integer",
                                               "description": "步骤编号 1-5"}},
                       "required": ["step"]}}},
]

# ── 序列化：把含 tool_calls 对象的 message 转成纯字典 ────────
def serialize_messages(messages):
    result = []
    for m in messages:
        entry = {"role": m["role"]}
        # content 可能是 None 或字符串
        entry["content"] = m.get("content") or ""
        # tool_calls 是 SDK 对象，要手动提取
        if m.get("tool_calls"):
            entry["tool_calls"] = [
                {"id": tc.id,
                 "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in m["tool_calls"]
            ]
        # tool_call_id（tool role 用）
        if "tool_call_id" in m:
            entry["tool_call_id"] = m["tool_call_id"]
        result.append(entry)
    return result

def save_checkpoint(messages, turn):
    data = {"turn": turn, "messages": serialize_messages(messages)}
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 检查点已保存（第 {turn} 轮，共 {len(messages)} 条消息）")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"🔄 发现检查点！从第 {data['turn']} 轮恢复，共 {len(data['messages'])} 条消息")
        return data["messages"], data["turn"]
    return None, 0

# ── 启动：先尝试恢复 ────────────────────────────────────────
messages, start_turn = load_checkpoint()
if messages is None:
    messages = [{"role": "user",
                 "content": "请依次调用工具完成步骤 1、2、3、4、5，每步告诉我进度。"}]
    print("🆕 全新开始")
else:
    print(f"   继续从第 {start_turn + 1} 轮执行\n")

print()

# ── Agent Loop ───────────────────────────────────────────────
for turn in range(start_turn, start_turn + 10):

    # 模拟第 CRASH_AT_TURN 轮崩溃
    if CRASH_AT_TURN and turn == CRASH_AT_TURN:
        print(f"\n💥 模拟崩溃！（第 {turn} 轮）")
        print("   → 重新运行此脚本，它会从检查点恢复\n")
        raise SystemExit(1)

    resp = client.chat.completions.create(
        model="moonshot-v1-8k",
        max_tokens=512,
        tools=tools,
        messages=messages,
    )
    finish = resp.choices[0].finish_reason
    msg    = resp.choices[0].message

    print(f"[轮 {turn}] finish={finish}  in={resp.usage.prompt_tokens}")

    messages.append({
        "role":       "assistant",
        "content":    msg.content,
        "tool_calls": msg.tool_calls,
    })

    # 每 SAVE_EVERY 轮保存检查点
    if (turn + 1) % SAVE_EVERY == 0:
        save_checkpoint(messages, turn)

    if finish == "stop":
        print(f"\n✅ 全部完成！\n{msg.content}")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🗑️  检查点已清除")
        break

    for tc in (msg.tool_calls or []):
        args = json.loads(tc.function.arguments)
        result = f"步骤 {args['step']} 完成 ✓"
        print(f"  → get_step(step={args['step']})  →  {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
