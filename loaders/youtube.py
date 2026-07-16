from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    FetchedTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)


def is_valid_youtube_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    # Fast path checking using a set for O(1) lookups
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        if host == "youtu.be":
            return bool(parsed.path and parsed.path.strip("/"))
        # Query parameters lookup ko optimize kiya gaya hai
        query_params = parse_qs(parsed.query)
        return bool(query_params.get("v") or parsed.path.startswith("/watch"))
    return False


def extract_video_id(url: str) -> str | None:
    if not is_valid_youtube_url(url):
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.lstrip("/") or None
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return video_id
    return None


def _fetch_video_title(video_id: str) -> str | None:
    try:
        request = Request(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
            title = payload.get("title")
            return str(title).strip() if title else None
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def _extract_snippet_text(item: Any) -> str:
    """Return the text of a single transcript snippet, whether it's an object or a dict.

    Hoisted out of `_format_transcript_text` (module level instead of a nested
    function) so the function object isn't rebuilt on every call to
    `_format_transcript_text` — same behavior, one less allocation per call.
    """
    if hasattr(item, "text"):
        return getattr(item, "text", "") or ""
    if isinstance(item, dict):
        return item.get("text", "") or ""
    return ""


def _format_transcript_text(transcript: Any) -> str:
    # Safe check: Agar object ke paas .fetch() method hai (jaise Transcript class), toh pehle data fetch karenge.
    # Isse "Transcript object is not iterable" error poori tarah dur ho jata hai.
    if hasattr(transcript, "fetch"):
        try:
            transcript_data = transcript.fetch()
        except Exception:
            transcript_data = []
    elif isinstance(transcript, FetchedTranscript):
        transcript_data = getattr(transcript, "snippets", []) or []
    else:
        transcript_data = transcript or []

    # Safe and optimized text retrieval using generator expression
    text_parts = (_extract_snippet_text(item) for item in transcript_data)
    return " ".join(part for part in text_parts if part).strip()


def _build_unavailable_payload(video_id: str, url: str, title: str | None, status: str, detail: str) -> dict[str, Any]:
    fallback_title = title or f"YouTube video {video_id}"
    fallback_text = (
        f"Transcript unavailable for this video. The video title is '{fallback_title}'. "
        f"Reason: {detail}"
    )
    return {
        "text": fallback_text,
        "video_id": video_id,
        "url": url,
        "title": fallback_title,
        "source": "youtube",
        "language": "unknown",
        "transcript_status": status,
        "transcript_error": detail,
    }


def _classify_transcript_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, InvalidVideoId):
        return "invalid-video-id", "The provided YouTube URL is invalid or the video ID could not be parsed."
    if isinstance(exc, VideoUnavailable):
        return "unavailable", "The requested YouTube video is unavailable, private, or not accessible."
    if isinstance(exc, VideoUnplayable):
        return "unplayable", "The requested YouTube video cannot be played from this environment."
    if isinstance(exc, TranscriptsDisabled):
        return "disabled", "Captions are disabled for this video."
    if isinstance(exc, IpBlocked):
        return "ip-blocked", "YouTube blocked requests from this environment, so captions could not be retrieved."
    if isinstance(exc, RequestBlocked):
        return "request-blocked", "YouTube blocked the transcript request from this environment."
    if isinstance(exc, CouldNotRetrieveTranscript):
        return "retrieval-failed", "YouTube could not retrieve the transcript for this video."
    if isinstance(exc, NoTranscriptFound):
        return "no-captions", "No captions or transcripts are available for this video."
    if isinstance(exc, YouTubeTranscriptApiException):
        return "transcript-api-error", f"The transcript API reported an error: {exc}"
    return "unknown-error", f"Unable to read the transcript: {exc}"


def load_youtube_transcript(url: str) -> dict[str, Any]:
    if not is_valid_youtube_url(url):
        raise ValueError("Please provide a valid YouTube URL.")

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not identify a video from the provided URL.")

    title = _fetch_video_title(video_id)
    api = YouTubeTranscriptApi()

    transcript = None
    first_error: Exception | None = None

    try:
        transcript_list = api.list(video_id)
    except Exception as exc:  # pragma: no cover - defensive guard
        status, detail = _classify_transcript_error(exc)
        return _build_unavailable_payload(video_id, url, title, status, detail)

    # Target languages ki list
    target_languages = ["en", "hi", "es"]

    # Stratgey execution block safely call kar raha hai list filtering ko
    for strategy_name in ("manually_created", "generated"):
        try:
            if strategy_name == "manually_created":
                transcript = transcript_list.find_manually_created_transcript(target_languages)
            else:
                transcript = transcript_list.find_generated_transcript(target_languages)
            break
        except (NoTranscriptFound, TranscriptsDisabled) as exc:
            if strategy_name == "generated":
                first_error = exc
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            first_error = exc
            break

    if transcript is None:
        if first_error is None:
            status, detail = "no-captions", "No captions or transcripts are available for this video."
        else:
            status, detail = _classify_transcript_error(first_error)
        return _build_unavailable_payload(video_id, url, title, status, detail)

    text = _format_transcript_text(transcript)
    if not text:
        status, detail = "empty-transcript", "The transcript was retrieved but did not contain any readable text."
        return _build_unavailable_payload(video_id, url, title, status, detail)

    return {
        "text": text,
        "video_id": video_id,
        "url": url,
        "title": title or f"YouTube video {video_id}",
        "source": "youtube",
        "language": getattr(transcript, "language", "unknown"),
    }


if __name__ == "__main__":
    sample_url = "https://www.youtube.com/watch?v=vJabNEwZIuc"
    try:
        payload = load_youtube_transcript(sample_url)
        print("Video ID:", payload["video_id"])
        print("Video Title:", payload["title"])
        print("\nPoora Transcript (Pehle 500 characters):")
        print(payload["text"][:500] + "...")
    except Exception as exc:  # pragma: no cover - defensive guard
        print(exc)