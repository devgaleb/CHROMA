"""CHROMA upload settings shared by image and video inputs."""
import os
import re

DEFAULT_LIMIT = "20gb"

def upload_limit():
    """Return Gradio-compatible file-size limit (default 20 GiB-ish).

    CHROMA_MAX_UPLOAD_SIZE accepts a number of bytes or a value such as
    500mb, 20gb, 1tb. Set to 0 for no application-level size limit.
    """
    raw = os.environ.get("CHROMA_MAX_UPLOAD_SIZE", DEFAULT_LIMIT).strip().lower()
    if raw == "0":
        return None
    if re.fullmatch(r"[1-9][0-9]*", raw):
        return int(raw)
    if not re.fullmatch(r"[1-9][0-9]*(?:kb|mb|gb|tb)", raw):
        raise ValueError("CHROMA_MAX_UPLOAD_SIZE must be bytes, 500mb, 20gb, 1tb, or 0")
    return raw
