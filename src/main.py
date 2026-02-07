"""Main FastAPI application for Claude Code Wrapper."""

import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .routes import anthropic, models, sessions
from .services.claude import ClaudeService
from .services.session import SessionManager
from .services.tool_proxy import cleanup_all_proxies


def setup_logging():
    """Configure logging with console and optional file output."""
    log_level = logging.DEBUG if settings.DEBUG_MODE else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # File handler (if enabled)
    if settings.LOG_TO_FILE:
        # Determine log directory (relative to project root or absolute)
        if os.path.isabs(settings.LOG_DIR):
            log_dir = Path(settings.LOG_DIR)
        else:
            # Get project root (parent of src/)
            project_root = Path(__file__).parent.parent
            log_dir = project_root / settings.LOG_DIR

        # Create log directory if needed
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / settings.LOG_FILE

        # Rotating file handler: 10MB max, keep 5 backups
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

        return str(log_file)

    return None


# Configure logging
log_file_path = setup_logging()
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
    anthropic.init_services(claude_service, session_manager)
    sessions.init_session_manager(session_manager)

    # Start session cleanup
    await session_manager.start_cleanup()

    logger.info(f"Server ready on port {settings.PORT}")
    logger.info(f"Working directory: {claude_service.cwd}")
    if log_file_path:
        logger.info(f"Log file: {log_file_path}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    cleanup_all_proxies()
    session_manager.shutdown()


app = FastAPI(
    title="Claude Code Wrapper",
    description="Anthropic Messages API wrapper for Claude Agent SDK",
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
app.include_router(anthropic.router, prefix="/v1")
app.include_router(sessions.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")


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
        "description": "Anthropic Messages API for Claude Agent SDK",
        "endpoints": {
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
