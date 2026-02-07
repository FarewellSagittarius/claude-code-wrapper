"""Base adapter utilities for file handling and caching."""

import base64
import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class FileCache:
    """File cache manager for media and documents."""

    # File extension mapping for MIME types
    MIME_TO_EXT = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
        "application/json": ".json",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "text/html": ".html",
        "text/csv": ".csv",
    }

    _cache_dir: Optional[Path] = None

    @classmethod
    def get_cache_dir(cls, cwd: Optional[str] = None) -> Path:
        """Get or create cache directory for media files."""
        if cls._cache_dir is None:
            if cwd:
                cls._cache_dir = Path(cwd) / ".claude_media_cache"
            else:
                cls._cache_dir = Path(tempfile.gettempdir()) / "claude_media_cache"
            cls._cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._cache_dir

    @staticmethod
    def get_file_hash(data: bytes) -> str:
        """Generate short hash for file content."""
        return hashlib.sha256(data).hexdigest()[:16]

    @classmethod
    async def save(
        cls,
        data: bytes,
        media_type: str,
        filename: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> Path:
        """Save binary data to cache and return file path."""
        cache_dir = cls.get_cache_dir(cwd)
        file_hash = cls.get_file_hash(data)

        # Determine extension
        ext = cls.MIME_TO_EXT.get(media_type, "")
        if not ext and filename:
            ext = Path(filename).suffix

        # Create filename
        if filename:
            safe_name = re.sub(r"[^\w\-.]", "_", Path(filename).stem)[:32]
            cache_filename = f"{safe_name}_{file_hash}{ext}"
        else:
            cache_filename = f"file_{file_hash}{ext}"

        cache_path = cache_dir / cache_filename

        # Write if not exists
        if not cache_path.exists():
            cache_path.write_bytes(data)
            logger.debug(f"Cached file: {cache_path}")

        return cache_path


async def fetch_url(url: str, cwd: Optional[str] = None) -> Optional[Path]:
    """Fetch file from URL and save to cache."""
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").split(";")[0].strip()

            # Try to get filename from URL or Content-Disposition
            filename = None
            cd = response.headers.get("content-disposition", "")
            if "filename=" in cd:
                filename = cd.split("filename=")[-1].strip("\"'")
            if not filename:
                filename = url.split("/")[-1].split("?")[0]

            return await FileCache.save(response.content, content_type, filename, cwd)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def estimate_tokens(text: str) -> int:
    """Roughly estimate token count (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)
