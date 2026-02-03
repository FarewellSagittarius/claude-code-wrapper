"""Main FastAPI application for Claude Code Wrapper."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .routes import anthropic, chat, models, sessions
from .services.claude import ClaudeService
from .services.session import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize services
claude_service = ClaudeService()
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Claude Code Wrapper...")

    # Initialize route services
    chat.init_services(claude_service, session_manager)
    anthropic.init_services(claude_service, session_manager)
    sessions.init_session_manager(session_manager)

    # Start session cleanup
    await session_manager.start_cleanup()

    logger.info(f"Server ready on port {settings.PORT}")
    logger.info(f"Default model: {settings.DEFAULT_MODEL}")
    logger.info(f"Working directory: {claude_service.cwd}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    session_manager.shutdown()


app = FastAPI(
    title="Claude Code Wrapper",
    description="OpenAI-compatible API wrapper for Claude Agent SDK",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/v1")
app.include_router(anthropic.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")
app.include_router(sessions.router, prefix="/v1")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Claude Code Wrapper",
        "version": "1.0.0",
        "description": "OpenAI-compatible API for Claude Agent SDK",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "messages": "/v1/messages",
            "models": "/v1/models",
            "sessions": "/v1/sessions",
            "health": "/health",
        },
        "documentation": "/docs",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
            }
        },
    )


def main():
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE,
    )


if __name__ == "__main__":
    main()
