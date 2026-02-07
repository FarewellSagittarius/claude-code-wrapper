# Claude Code Wrapper

Anthropic Messages API wrapper that exposes Claude Agent SDK as a standard API endpoint.

```
Direct (Anthropic API):
┌────────┐  /v1/messages  ┌─────────────┐       ┌───────────┐
│ Client  │──────────────▶│ Wrapper:8080 │──────▶│ Claude SDK│
└────────┘                └─────────────┘       └───────────┘

With LiteLLM (OpenAI API compatible):
┌────────┐  /v1/chat/completions  ┌─────────────┐  /v1/messages  ┌─────────────┐       ┌───────────┐
│ Client  │──────────────────────▶│ LiteLLM:4000│──────────────▶│ Wrapper:8080 │──────▶│ Claude SDK│
└────────┘                        └─────────────┘               └─────────────┘       └───────────┘
```

## Features

- Anthropic Messages API (`/v1/messages`)
- Model listing (`/v1/models`)
- External tool proxy (client `tools` → MCP stdio proxy)
- MCP server configuration
- Hash-based automatic session matching
- SSE streaming

## Quick Start

### Docker Deployment

```yaml
# docker-compose.yml
services:
  claude-code-wrapper:
    container_name: claude-code-wrapper
    image: ghcr.io/farewellsagittarius/claude-code-wrapper:latest
    ports:
      - "8080:8790"
    environment:
      - INTERNAL_API_KEY=sk-claude-code-wrapper
      - TOOLS=Read,Grep,Glob,WebSearch,WebFetch
      - LOAD_USER_MCP=false
      - LOG_TO_FILE=false
    volumes:
      - ~/.claude/.credentials.json:/home/claude/.claude/.credentials.json:ro
    restart: unless-stopped
```

```bash
docker-compose up -d
```

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

## Configuration

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8790` |
| `HOST` | Listen address | `0.0.0.0` |
| `INTERNAL_API_KEY` | Auth key | `sk-internal-dev` |
| `TOOLS` | Tool config: unset=all, `""`=none, `"Task,Bash,Read"`=specific | all |
| `LOAD_USER_MCP` | Load user MCP servers | `true` |
| `CLAUDE_CWD` | Claude working directory | temp dir |
| `DEBUG_MODE` | Debug mode | `false` |

## Usage

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8080/v1",
    api_key="sk-internal-dev"
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
