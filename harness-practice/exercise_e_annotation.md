# 练习 E：Claude Code 源码中五大组件标注

> 二进制位置：`/root/.local/share/claude/versions/2.1.118`（ELF，239MB，Node.js 打包）
> 提取方式：`strings` + `dd` 偏移截取，偏移 108,793,000 字节处

---

## 完整原始代码段（已提取，约 10KB）

下面是从二进制里还原出的关键 JS 代码。变量名经过 minify（H/$/_/A/f/z...），
但逻辑完整可读。

---

## ① 组件一：Prompt 构造

**对应代码：`DE6`（上下文压缩提示词）+ `messages` 数组管理**

```javascript
// DE6 = 系统级 summarization prompt（硬编码在二进制里）
var ME6 = 1e5;   // 触发压缩的 token 阈值：100,000 tokens
var DE6 = `You have been working on the task described above but have not yet 
completed it. Write a continuation summary that will allow you (or another 
instance of yourself) to resume work efficiently in a future context window 
where the conversation history will be replaced with this summary. ...`;

// 压缩时的 Prompt 构造：把整个 messages 历史 + summarization 指令组装成新请求
let z = await this.client.beta.messages.create({
    model: _,
    messages: [
        ...f,                          // 原始历史 messages
        { role: "user", content: [{ type: "text", text: A }] }  // 追加压缩指令
    ],
    max_tokens: r8(this, oP, "f").params.max_tokens
});
// 压缩完成后，把整个 messages 替换成单条摘要
r8(this, oP, "f").params.messages = [{ role: "user", content: z.content }];
```

**对应你的 harness_v1.py**：`compress_history()` 函数、`system` 参数、`READ_CHAR_LIMIT` 截断逻辑

**关键发现**：Claude Code 的 summarization prompt（`DE6`）是直接硬编码在二进制里的，
长达 500+ 字符，结构化要求模型输出 5 个部分：任务概述、当前状态、重要发现、下一步、需保留上下文。

---

## ② 组件二：LLM 调用

**对应代码：`idH` 类的主循环里的 `client.beta.messages.stream` / `create`**

```javascript
// 流式调用（streaming）
$ = this.client.beta.messages.stream({ ..._ }, r8(this, ldH, "f"));
CK(this, SI, $.finalMessage(), "f");   // SI = 当前 API 调用的 Promise
yield $;

// 非流式调用（non-streaming）
CK(this, SI,
    this.client.beta.messages.create({ ..._, stream: false }, r8(this, ldH, "f")),
    "f");
yield r8(this, SI, "f");
```

**对应你的 harness_v1.py**：
```python
resp = client.chat.completions.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
```

**关键发现**：Claude Code 同时支持流式和非流式，参数从 `r8(this, oP, "f").params` 里读取，
相当于你的 `MODEL / TOOLS / messages` 配置。`ldH` 里还注入了 `x-stainless-helper` 请求头，
用于追踪工具使用情况（telemetry）。

---

## ③ 组件三：工具注册

**对应代码：`H.tools.find(...)` + `tG4` 函数里的工具查找**

```javascript
// 工具注册：在 params 里传入 tools 数组，每个工具必须有 name 和 run 方法
let A = H.tools.find((f) =>
    ("name" in f ? f.name : f.mcp_server_name) === _.name
    //  ↑ 普通工具用 .name      ↑ MCP 工具用 .mcp_server_name
);

if (!A || !("run" in A)) {
    // 工具不存在或没有 run 方法 → 立刻回传错误
    return {
        type: "tool_result",
        tool_use_id: _.id,
        content: `Error: Tool '${_.name}' not found`,
        is_error: true
    };
}
```

**对应你的 harness_v1.py**：`TOOLS` 列表（JSON schema 声明）+ `dispatch()` 的 `if name == "read_file"` 路由

**关键发现**：Claude Code 的工具对象同时包含「描述（schema）」和「执行（run 方法）」，
而你的 harness_v1.py 把这两件事分开了（`TOOLS` 是 schema，`dispatch()` 是执行）。
Claude Code 还原生支持 MCP 工具（用 `mcp_server_name` 字段区分）。

---

## ④ 组件四：工具执行

**对应代码：`tG4` 函数（Tool Dispatcher）**

```javascript
async function tG4(H, $ = H.messages.at(-1)) {
    // 1. 取最后一条 assistant 消息
    if (!$ || $.role !== "assistant" || !$.content || typeof $.content === "string")
        return null;

    // 2. 过滤出所有 tool_use 块（模型可能并行调多个工具）
    let q = $.content.filter((_) => _.type === "tool_use");
    if (q.length === 0) return null;

    // 3. 并行执行所有工具，构造 tool_result 消息
    return {
        role: "user",
        content: await Promise.all(q.map(async (_) => {
            let A = H.tools.find((f) =>
                ("name" in f ? f.name : f.mcp_server_name) === _.name
            );
            // 工具不存在 → 错误回传（不抛出，让模型处理）
            if (!A || !("run" in A))
                return { type: "tool_result", tool_use_id: _.id,
                         content: `Error: Tool '${_.name}' not found`, is_error: true };
            try {
                let f = _.input;
                if ("parse" in A && A.parse) f = A.parse(f);  // 可选的输入预处理
                let z = await A.run(f);                         // 真正执行工具
                return { type: "tool_result", tool_use_id: _.id, content: z };
            } catch (f) {
                // 工具抛异常 → 回传错误字符串，而不是让进程崩溃
                return {
                    type: "tool_result", tool_use_id: _.id,
                    content: f instanceof gZH ? f.content
                             : `Error: ${f instanceof Error ? f.message : String(f)}`,
                    is_error: true
                };
            }
        }))
    };
}
```

**对应你的 harness_v1.py**：`dispatch()` 函数

**关键发现（4条）**：
1. `Promise.all(q.map(...))` = 并行执行多工具，你的版本是串行 `for tc in msg.tool_calls`
2. 工具有可选的 `parse()` 预处理钩子，在输入传给 `run()` 之前做类型转换/验证
3. 错误处理完全一致：catch 异常 → `is_error: true` 回传，不抛出
4. `gZH`（`ToolError` 类）允许工具返回结构化错误内容，不只是字符串

---

## ⑤ 组件五：循环与停止

**对应代码：`idH` 类的 `async*[Symbol.asyncIterator]()` 方法（Generator 实现的 while 循环）**

```javascript
async * [Symbol.asyncIterator]() {
    // 安全阀 ①：最大迭代次数
    while (true) {
        if (r8(this, oP, "f").params.max_iterations &&
            r8(this, ndH, "f") >= r8(this, oP, "f").params.max_iterations)
            break;   // ← 达到 max_iterations，停止

        // 递增轮次计数器
        CK(this, ndH, (H = r8(this, ndH, "f"), H++, H), "f");

        // LLM 调用（流式或非流式）
        $ = this.client.beta.messages.stream({ ..._ }, r8(this, ldH, "f"));
        yield $;   // ← 每轮 yield 给外层消费

        // 安全阀 ②：上下文压缩（token 超 10万 时触发）
        if (!await r8(this, cdH, "m", wE6).call(this)) {
            // 压缩未触发：追加 assistant 消息到 messages
            if (!r8(this, XYH, "f")) {
                let { role: z, content: Y } = await r8(this, SI, "f");
                r8(this, oP, "f").params.messages.push({ role: z, content: Y });
            }
            // 生成 tool_result 消息（调用 tG4）
            let f = await r8(this, cdH, "m", Bz8).call(this,
                r8(this, oP, "f").params.messages.at(-1));

            if (f)
                r8(this, oP, "f").params.messages.push(f);  // ← 追加工具结果，继续循环
            else if (!r8(this, XYH, "f"))
                break;   // ← f 为 null = 没有 tool_use = end_turn，停止循环
        }
    }
}
```

**对应你的 harness_v1.py**：
```python
for turn in range(start_turn, start_turn + MAX_TURNS):   # max_iterations
    resp = client.chat.completions.create(...)
    messages.append({"role": "assistant", ...})           # params.messages.push
    if finish == "stop": break                            # f 为 null → break
    for tc in msg.tool_calls:
        messages.append({"role": "tool", ...})            # params.messages.push(f)
```

**关键发现**：
- Claude Code 用 **ES6 Generator**（`async*`）实现循环，每轮 `yield` 给外部消费流式事件
- 停止条件不是判断 `stop_reason == "end_turn"`，而是判断 `tG4()` 返回 `null`（即没有 `tool_use` 块）
- 压缩阈值 `ME6 = 1e5`（10 万 token）硬编码，超过才触发 summarization

---

## 汇总对照表

| 组件 | 你的 harness_v1.py | Claude Code 源码 | 关键差异 |
|------|-------------------|-----------------|---------|
| ① Prompt 构造 | `system` 参数 + `READ_CHAR_LIMIT` 截断 | `DE6` 硬编码 summarization prompt | CC 用摘要替换整个历史；你用截断 |
| ② LLM 调用 | `client.chat.completions.create()` | `client.beta.messages.stream()` | CC 用流式 + Anthropic beta API |
| ③ 工具注册 | `TOOLS` JSON schema + `dispatch()` 路由 | 工具对象含 `{name, run}` | CC 把 schema 和执行合一；你分离 |
| ④ 工具执行 | 串行 `for tc in tool_calls` | 并行 `Promise.all(q.map(...))` | CC 并行执行多工具 |
| ⑤ 循环与停止 | `for turn in range(MAX_TURNS)` | `while(true)` + Generator yield | CC 用 Generator 支持流式消费 |

---

## 额外发现：2 个你没有但 CC 有的组件

**A. 自动上下文压缩（Compaction）**
- token 超 10 万时，自动调一次 LLM 把整个对话历史压缩成结构化摘要
- 摘要格式：任务概述 / 当前状态 / 重要发现 / 下一步 / 需保留上下文（`DE6`）

**B. MCP 工具原生支持**
- 工具查找时检查 `mcp_server_name` 字段
- 你的 harness_v1.py 目前只支持本地工具，接入 MCP 需要加这一个字段的判断
