# API 参数对照表

本文档整理 OpenAI Chat Completions API 和 Anthropic Messages API 的原生参数，以及 wrapper 的实现状态。

## OpenAI Chat Completions API (`/v1/chat/completions`)

### 请求参数

| 参数 | 类型 | 必填 | 原生描述 | Wrapper 状态 |
|------|------|------|----------|--------------|
| `model` | string | ✅ | 模型 ID | ✅ 支持，含别名映射 |
| `messages` | array | ✅ | 对话消息列表 | ✅ 支持 |
| `temperature` | number | ❌ | 采样温度 (0-2) | ⚠️ 接收但忽略 |
| `top_p` | number | ❌ | 核采样参数 | ⚠️ 接收但忽略 |
| `n` | integer | ❌ | 生成数量 | ⚠️ 固定为 1 |
| `stream` | boolean | ❌ | 是否流式 | ✅ 支持 |
| `stop` | string/array | ❌ | 停止序列 | ❌ SDK 不支持 |
| `max_tokens` | integer | ❌ | 最大生成 tokens | ⚠️ 接收但忽略 |
| `max_completion_tokens` | integer | ❌ | 最大完成 tokens (o1 系列) | ⚠️ 接收但忽略 |
| `presence_penalty` | number | ❌ | 存在惩罚 (-2 to 2) | ⚠️ 接收但忽略 |
| `frequency_penalty` | number | ❌ | 频率惩罚 (-2 to 2) | ⚠️ 接收但忽略 |
| `logit_bias` | object | ❌ | Token 偏置 | ⚠️ 接收但忽略 |
| `logprobs` | boolean | ❌ | 返回 log 概率 | ❌ 不支持 |
| `top_logprobs` | integer | ❌ | 返回 top N log 概率 | ❌ 不支持 |
| `user` | string | ❌ | 用户标识符 | ⚠️ 接收但忽略 |
| `stream_options` | object | ❌ | 流式选项 | ⚠️ 接收但忽略 |
| `tools` | array | ❌ | 工具定义列表 | ❌ 不支持原生格式 |
| `tool_choice` | string/object | ❌ | 工具选择策略 | ❌ 不支持 |
| `response_format` | object | ❌ | 响应格式 (JSON mode) | ❌ 不支持 |
| `seed` | integer | ❌ | 随机种子 | ❌ 不支持 |
| `service_tier` | string | ❌ | 服务层级 | ❌ 不支持 |
| `store` | boolean | ❌ | 存储用于蒸馏 | ❌ 不支持 |
| `reasoning_effort` | string | ❌ | 推理强度 (o1/o3) | ✅ 支持，映射到 thinking |

### Wrapper 扩展参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `session_id` | string | 会话 ID，用于多轮对话 |
| `enable_tools` | boolean | 是否启用内置工具 (custom mode) |
| `allowed_tools` | array | 允许的工具列表 |
| `disallowed_tools` | array | 禁用的工具列表 |
| `max_turns` | integer | 最大对话轮数 |
| `mcp_servers` | object | MCP 服务器配置 |

### 响应格式

| 字段 | 类型 | Wrapper 状态 |
|------|------|--------------|
| `id` | string | ✅ 支持 |
| `object` | string | ✅ 固定 "chat.completion" |
| `created` | integer | ✅ 支持 |
| `model` | string | ✅ 支持 |
| `choices` | array | ✅ 支持 |
| `choices[].index` | integer | ✅ 支持 |
| `choices[].message` | object | ✅ 支持 |
| `choices[].finish_reason` | string | ✅ 支持 |
| `choices[].logprobs` | object | ❌ 始终 null |
| `usage` | object | ✅ 支持 |
| `usage.prompt_tokens` | integer | ✅ 支持 |
| `usage.completion_tokens` | integer | ✅ 支持 |
| `usage.total_tokens` | integer | ✅ 支持 |
| `usage.prompt_tokens_details` | object | ✅ 支持 (cached_tokens) |
| `usage.completion_tokens_details` | object | ⚠️ 占位符 |
| `system_fingerprint` | string | ❌ 始终 null |

---

## Anthropic Messages API (`/v1/messages`)

### 请求参数

| 参数 | 类型 | 必填 | 原生描述 | Wrapper 状态 |
|------|------|------|----------|--------------|
| `model` | string | ✅ | 模型 ID | ✅ 支持 |
| `messages` | array | ✅ | 对话消息列表 | ✅ 支持 |
| `max_tokens` | integer | ✅ | 最大生成 tokens | ⚠️ 接收但忽略 |
| `system` | string/array | ❌ | 系统提示 | ✅ 支持 (仅 string) |
| `temperature` | number | ❌ | 采样温度 (0-1) | ⚠️ 接收但忽略 |
| `top_p` | number | ❌ | 核采样参数 | ⚠️ 接收但忽略 |
| `top_k` | integer | ❌ | Top-K 采样 | ⚠️ 接收但忽略 |
| `stop_sequences` | array | ❌ | 停止序列 | ❌ SDK 不支持 |
| `stream` | boolean | ❌ | 是否流式 | ✅ 支持 |
| `metadata` | object | ❌ | 请求元数据 | ⚠️ 接收但忽略 |
| `thinking` | object | ❌ | 扩展思维配置 | ✅ 支持 |
| `tools` | array | ❌ | 工具定义列表 | ❌ 不支持原生格式 |
| `tool_choice` | object | ❌ | 工具选择策略 | ❌ 不支持 |
| `service_tier` | string | ❌ | 服务层级 | ❌ 不支持 |
| `output_config` | object | ❌ | 输出配置 (结构化输出) | ❌ 不支持 |

### Wrapper 扩展参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `session_id` | string | 会话 ID，用于多轮对话 |
| `allowed_tools` | array | 允许的工具列表 |
| `disallowed_tools` | array | 禁用的工具列表 |
| `mcp_servers` | object | MCP 服务器配置 |

### 响应格式

| 字段 | 类型 | Wrapper 状态 |
|------|------|--------------|
| `id` | string | ✅ 支持 |
| `type` | string | ✅ 固定 "message" |
| `role` | string | ✅ 固定 "assistant" |
| `content` | array | ✅ 支持 (text blocks) |
| `model` | string | ✅ 支持 |
| `stop_reason` | string | ✅ 支持 |
| `stop_sequence` | string | ⚠️ 始终 null |
| `usage` | object | ✅ 支持 |
| `usage.input_tokens` | integer | ✅ 支持 |
| `usage.output_tokens` | integer | ✅ 支持 |

### 消息内容类型

| 类型 | Wrapper 状态 | 说明 |
|------|--------------|------|
| `text` | ✅ 支持 | |
| `image` (base64) | ✅ 兼容 | 缓存到文件，通过 Read 工具读取 |
| `image` (url) | ✅ 兼容 | 下载缓存到文件，通过 Read 工具读取 |
| `document` (text) | ✅ 兼容 | 缓存到文件，通过 Read 工具读取 |
| `document` (base64 PDF) | ✅ 兼容 | 缓存到文件，通过 Read 工具读取 |
| `document` (url) | ✅ 兼容 | 下载缓存到文件，通过 Read 工具读取 |
| `tool_use` | ❌ 不支持原生格式 | |
| `tool_result` | ❌ 不支持原生格式 | |

**注意**: Claude Agent SDK 的 `ContentBlock` 类型仅支持 `TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock`。图片和文档通过文件缓存机制实现兼容：
1. 将二进制/base64 内容保存到 `.claude_media_cache` 目录
2. 在 prompt 中包含文件路径引用，如 `[Image file: /path/to/file.png]`
3. Claude Code 使用内置 Read 工具读取这些文件

---

## 扩展思维 (Extended Thinking) 参数

### OpenAI 兼容方式

```json
{
  "reasoning_effort": "high"
}
```

| 值 | max_thinking_tokens |
|----|---------------------|
| `none` | 禁用 |
| `low` | 8,000 |
| `medium` | 16,000 |
| `high` | 31,999 |

### Anthropic 原生方式

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 16000
  }
}
```

---

## 状态图例

| 符号 | 含义 |
|------|------|
| ✅ | 完全支持 |
| ⚠️ | 接收但不生效 / 部分支持 |
| ❌ | 不支持 |

---

## 未实现功能列表

### 高优先级

1. **原生工具调用** - OpenAI `tools`/`tool_choice` 和 Anthropic `tools`/`tool_choice`
2. **结构化输出** - OpenAI `response_format` 和 Anthropic `output_config`
3. **采样参数** - `temperature`, `top_p`, `top_k` 的实际生效

### 中优先级

4. **最大 tokens 限制** - `max_tokens` / `max_completion_tokens` 的实际生效

### SDK 限制 (无法实现)

5. **停止序列** - `stop` / `stop_sequences` - SDK 不支持

### 低优先级

7. **Log 概率** - `logprobs`, `top_logprobs`
8. **随机种子** - `seed`
9. **服务层级** - `service_tier`
10. **存储** - `store`

---

## 参考链接

- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages)
