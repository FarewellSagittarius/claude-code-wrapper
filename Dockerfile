FROM python:3.12-slim

# Install Node.js 22 (required for Claude Code CLI on arm64, bundled CLI is x86-64 only)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Install Claude Code CLI (arm64 native via npm, requires root for global install)
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user (SDK rejects --dangerously-skip-permissions under root)
RUN useradd -m -s /bin/bash claude && \
    mkdir -p /home/claude/.claude && \
    chown -R claude:claude /app /home/claude
USER claude

ENV PORT=8790
ENV HOST=0.0.0.0
EXPOSE 8790

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8790"]
