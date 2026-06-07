"""
Step 1 & 2 of the pipeline.

Downloads the podcast audio from YouTube with yt-dlp and transcribes it
with OpenAI Whisper, producing a JSON transcript where every entry keeps
its start / end timestamps from the audio.

Output:
    data/transcript/transcript.json
        [
            {"start": 0.0,  "end": 4.2,  "text": "..."},
            {"start": 4.2,  "end": 9.8,  "text": "..."},
            ...
        ]
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yt_dlp
import whisper

from config import AUDIO_DIR, TRANSCRIPT_DIR, WHISPER_MODEL, YOUTUBE_URL


def _find_ffmpeg() -> str | None:
    """
    Locate the ffmpeg binary.

    1. Use `ffmpeg` from PATH (the normal case on macOS / Linux and on
       Windows once the user has restarted their shell after `winget install`).
    2. Fall back to common Windows install locations so a freshly-opened
       PowerShell that hasn't refreshed its environment still works.
    """
    found = shutil.which("ffmpeg")
    if found:
        return os.path.dirname(found)

    candidates = [
        os.environ.get("FFMPEG_LOCATION"),
        r"C:\Users\anoop\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin",
        r"C:\ProgramData\chocolatey\bin",
        r"C:\ffmpeg\bin",
        os.path.expanduser(r"~\ffmpeg\bin"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "ffmpeg.exe")):
            return cand
    return None


# Ensure ffmpeg is on PATH for the current process BEFORE Whisper is used.
# Whisper's audio loader shells out to ffmpeg via subprocess, so if the
# binary isn't visible to this Python interpreter it raises WinError 2.
_ffmpeg_dir = _find_ffmpeg()
if _ffmpeg_dir:
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def download_audio(url: str, output_dir: Path) -> Path:
    """
    Download the best-quality audio stream from `url` and convert it to WAV
    using ffmpeg (handled by yt-dlp's FFmpegExtractAudio postprocessor).

    If a WAV for this video already exists in `output_dir`, the download
    is skipped - this makes the script idempotent and friendly to retries
    on flaky connections.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look up the video id from the URL so we can pre-check for an existing file.
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as probe:
        info = probe.extract_info(url, download=False)
        video_id = info.get("id", "audio")

    existing = output_dir / f"{video_id}.wav"
    if existing.exists() and existing.stat().st_size > 0:
        print(f"[INFO] Reusing existing audio: {existing}")
        return existing

    output_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": {
            # Whisper resamples to 16 kHz mono internally. Producing that
            # format directly makes the WAV ~50x smaller and avoids the
            # "file not found" errors that some Whisper builds throw when
            # reading very large 48 kHz stereo WAVs.
            "ffmpeg_o": ["-ar", "16000", "-ac", "1"],
        },
        "quiet": False,
        "no_warnings": True,
    }

    ffmpeg_dir = _find_ffmpeg()
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir
        print(f"[INFO] Using ffmpeg from: {ffmpeg_dir}")
    else:
        print(
            "[WARN] ffmpeg was not found on PATH. Audio extraction may fail. "
            "Install it via `winget install Gyan.FFmpeg` or `choco install ffmpeg`."
        )

    print(f"[INFO] Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "audio")

    audio_path = output_dir / f"{video_id}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    print(f"[INFO] Audio saved to: {audio_path}")
    return audio_path


def transcribe_audio(audio_path: Path, model_name: str = "base") -> list[dict]:
    """
    Run Whisper on the audio file and return a list of segments, each with
    its start, end, and text fields. Whisper's segment-level timestamps are
    precise enough to deep-link into the YouTube video.
    """
    print(f"[INFO] Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"[INFO] Transcribing: {audio_path}")
    # task="transcribe" preserves the original language; switch to "translate"
    # if you want a forced English transcript for non-English content.
    result = model.transcribe(str(audio_path), verbose=False, task="transcribe")

    segments: list[dict] = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
            }
        )

    print(f"[INFO] Transcription complete: {len(segments)} segments")
    return segments


def main() -> None:
    try:
        audio_path = download_audio(YOUTUBE_URL, AUDIO_DIR)
        segments = transcribe_audio(audio_path, WHISPER_MODEL)

        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TRANSCRIPT_DIR / "transcript.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Transcript saved to: {out_path}")
    except Exception as exc:  # noqa: BLE001 - top-level error reporting
        print(f"[ERROR] transcribe.py failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
