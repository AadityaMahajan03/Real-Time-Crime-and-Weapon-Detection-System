"""FastAPI video streaming router providing full HTTP 206 Partial Content range requests."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/clips", tags=["Clips & Video Streaming"])


def _stream_video_range(
    file_path: Path,
    start: int,
    end: int,
    chunk_size: int = 256 * 1024,
) -> Generator[bytes, None, None]:
    """Yield chunks of binary data within the specified byte range."""
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/{clip_id}/stream")
async def stream_clip(
    clip_id: str,
    request: Request,
    range: Optional[str] = Header(None),
) -> Response:
    """Stream surveillance H.264 video clip with HTTP Range support for seamless browser seeking."""
    # Resolve clip path from database or storage
    from src.fasttrack.core.config import get_config
    from src.fasttrack.storage.db import DatabaseManager
    from src.fasttrack.storage.repositories import ClipRepository

    config = get_config()
    db = DatabaseManager(config.storage.db_path)
    clip_repo = ClipRepository(db)

    clip = clip_repo.get_by_id(clip_id)
    video_path: Optional[Path] = None

    if clip:
        p = Path(clip["raw_clip_path"])
        if p.exists():
            video_path = p
        elif Path(clip["annotated_clip_path"]).exists():
            video_path = Path(clip["annotated_clip_path"])

    if not video_path or not video_path.exists():
        # Fallback check in evidence table
        ev_rows = db.execute_query("SELECT path FROM evidence WHERE id = ?;", (clip_id,))
        if ev_rows and Path(ev_rows[0]["path"]).exists():
            video_path = Path(ev_rows[0]["path"])

    if not video_path or not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video clip '{clip_id}' was not found on disk.",
        )

    file_size = video_path.stat().st_size
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Video file exists but is 0 bytes.",
        )

    # If no Range header, return entire file (HTTP 200)
    if not range:
        return StreamingResponse(
            _stream_video_range(video_path, 0, file_size - 1),
            status_code=status.HTTP_200_OK,
            headers={
                "Content-Length": str(file_size),
                "Content-Type": "video/mp4",
                "Accept-Ranges": "bytes",
            },
        )

    # Parse Range: bytes=start-end
    try:
        range_val = range.replace("bytes=", "").strip()
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            return Response(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                headers={"Content-Range": f"bytes */{file_size}"},
            )

        content_length = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(
            _stream_video_range(video_path, start, end),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers,
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Range header format.",
        )
