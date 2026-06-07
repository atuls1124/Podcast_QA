"""
Small helpers shared across the project: timestamp formatting and
YouTube URL building with the &t=Ns deep-link parameter.
"""

from __future__ import annotations

import re


def extract_video_id(youtube_url: str) -> str:
    """
    Extract the 11-character YouTube video ID from any common YouTube URL
    (watch?v=, youtu.be/, /embed/, /shorts/, /live/).
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[?&#].*)?$",
        r"(?:youtu\.be\/|embed\/|shorts\/|live\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract a YouTube video ID from: {youtube_url}")


def format_timestamp(seconds: float) -> str:
    """
    Convert a number of seconds into a human-readable timestamp.
    Hours are only shown when the value is >= 1 hour.
    """
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def youtube_url_with_timestamp(youtube_url: str, seconds: float) -> str:
    """
    Build a YouTube watch URL that starts playback at `seconds`.
    Uses the official &t=Ns deep-link parameter.
    """
    video_id = extract_video_id(youtube_url)
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"
