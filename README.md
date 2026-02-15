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

## Quick Start

### Local (Recommended)

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8790

# Run with custom config
INTERNAL_API_KEY=my-key TOOLS="Read,Grep,Glob" .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8790

# Run with hot-reload (development)
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8790 --reload
```

### Docker

```yaml
# docker-compose.yml
services:
  claude-code-wrapper:
    container_name: claude-code-wrapper
    image: ghcr.io/farewellsagittarius/claude-code-wrapper:latest
    ports:
      - "8790:8790"
    environment:
      - INTERNAL_API_KEY=sk-claude-code-wrapper
      - TOOLS=Read,Grep,Glob,WebSearch,WebFetch
      - LOAD_USER_MCP=false
      - LOG_TO_FILE=false
    volumes:
      - ~/.claude/.credentials.json:/home/claude/.claude/.credentials.json
    restart: unless-stopped
```

```bash
docker compose up -d
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8790` |
| `HOST` | Listen address | `0.0.0.0` |
| `INTERNAL_API_KEY` | Auth key | `sk-claude-code-wrapper` |
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
