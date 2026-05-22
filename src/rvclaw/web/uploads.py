from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any

from rvclaw.utils import ensure_dir


ALLOWED_UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
UPLOAD_ID_RE = re.compile(r"^[a-f0-9]{32}\.(png|jpg|jpeg|webp)$")


def save_upload_bytes(uploads_dir: str | Path, filename: str, content: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError("unsupported upload type; expected png, jpg, jpeg, or webp")
    if not content:
        raise ValueError("uploaded image is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("uploaded image is too large")

    root = ensure_dir(uploads_dir)
    upload_id = f"{uuid.uuid4().hex}{suffix}"
    path = root / upload_id
    path.write_bytes(content)
    return {
        "upload_id": upload_id,
        "image_ref": f"upload:{upload_id}",
        "filename": Path(filename).name,
        "size": len(content),
        "content_type": mimetypes.guess_type(upload_id)[0] or "application/octet-stream",
    }


def resolve_upload_image_ref(uploads_dir: str | Path, image_ref: str) -> Path:
    if not image_ref.startswith("upload:"):
        raise ValueError("image_ref must be a local upload reference")
    upload_id = image_ref.removeprefix("upload:")
    if not UPLOAD_ID_RE.fullmatch(upload_id):
        raise ValueError("invalid upload id")
    root = ensure_dir(uploads_dir).resolve()
    path = (root / upload_id).resolve()
    if root not in [path.parent, *path.parents] or not path.is_file():
        raise ValueError("uploaded image not found")
    return path


def read_upload_bytes(uploads_dir: str | Path, upload_id: str) -> dict[str, Any]:
    if not UPLOAD_ID_RE.fullmatch(upload_id):
        raise ValueError("invalid upload id")
    path = resolve_upload_image_ref(uploads_dir, f"upload:{upload_id}")
    return {
        "upload_id": upload_id,
        "content_type": mimetypes.guess_type(upload_id)[0] or "application/octet-stream",
        "bytes": path.read_bytes(),
    }
