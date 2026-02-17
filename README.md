# Claude Code Wrapper

Anthropic Messages API wrapper that exposes Claude Agent SDK as a standard API endpoint.

```
Direct (Anthropic API):
┌────────┐  /v1/messages  ┌──────────────┐       ┌───────────┐
│ Client  │──────────────▶│ Wrapper:8790 │──────▶│ Claude SDK│
└────────┘                └──────────────┘       └───────────┘

With LiteLLM (OpenAI API compatible):
┌────────┐  /v1/chat/completions  ┌─────────────┐  /v1/messages  ┌──────────────┐       ┌───────────┐
│ Client  │──────────────────────▶│ LiteLLM:4000│──────────────▶│ Wrapper:8790 │──────▶│ Claude SDK│
└────────┘                        └─────────────┘               └──────────────┘       └───────────┘
```

## Features

- Anthropic Messages API (`/v1/messages`)
- Model listing (`/v1/models`)
- Adaptive thinking with effort=high (internal, not configurable via API)
- External tool proxy (client `tools` → MCP stdio proxy)
- MCP server configuration
- Hash-based automatic session matching
- SSE streaming
- Multi-arch support (amd64 / arm64)

## Quick Start

### 1. Generate OAuth Token

On a machine where you're logged into Claude:

```bash
claude setup-token
```

Copy the token (valid for ~1 year).

### 2. Create docker-compose.yml

```yaml
services:
  claude-code-wrapper:
    container_name: claude-code-wrapper
    image: ghcr.io/farewellsagittarius/claude-code-wrapper:latest
    ports:
      - "8790:8790"
    environment:
      - INTERNAL_API_KEY=sk-claude-code-wrapper
      - CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}
      - TOOLS=Read,Grep,Glob,WebSearch,WebFetch
      - LOAD_USER_MCP=false
      - LOG_TO_FILE=false
    restart: unless-stopped
```

### 3. Create .env

```bash
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-xxxxx
```

### 4. Start

```bash
docker compose up -d
```

### Local Development

```bash
cp .env.example .env
# Edit .env and set CLAUDE_CODE_OAUTH_TOKEN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `INTERNAL_API_KEY` | Wrapper API auth key | `sk-claude-code-wrapper` |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude OAuth token (via `claude setup-token`) | *(required)* |
| `PORT` | Server port | `8790` |
| `HOST` | Listen address | `0.0.0.0` |
| `TOOLS` | Tool config: unset=all, `""`=none, `"Task,Bash,Read"`=specific | all |
| `LOAD_USER_MCP` | Load user MCP servers from `~/.claude.json` | `true` |
| `EXPOSE_THINKING` | Pass `<thinking>` blocks through to client | `false` |
| `CLAUDE_CWD` | Claude working directory | temp dir |
| `DEBUG_MODE` | Debug mode | `false` |

## Usage

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8790/v1",
    api_key="sk-claude-code-wrapper"
)

response = client.messages.create(
    model="claude-code-opus",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)
```

### Model Aliases

| Name | SDK Model |
|------|-----------|
| `claude-code` | SDK default |
| `claude-code-opus` | opus |
| `claude-code-sonnet` | sonnet |
| `claude-code-haiku` | haiku |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API |
| `/v1/models` | GET | Model listing |
| `/v1/sessions` | GET | Active sessions |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger docs |

## License

MIT
