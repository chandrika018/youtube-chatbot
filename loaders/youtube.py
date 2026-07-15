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
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        if host == "youtu.be":
            return bool(parsed.path and parsed.path.strip("/") )
        return bool(parse_qs(parsed.query).get("v") or parsed.path.startswith("/watch"))
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


def _format_transcript_text(transcript: FetchedTranscript | list[Any]) -> str:
    if isinstance(transcript, FetchedTranscript):
        snippets = transcript.snippets if hasattr(transcript, "snippets") else []
        text_parts = [getattr(snippet, "text", "") for snippet in snippets if getattr(snippet, "text", "")]
    else:
        text_parts = []
        for item in transcript:
            if hasattr(item, "text"):
                text_parts.append(item.text)
            elif isinstance(item, dict):
                text_parts.append(item.get("text", ""))
    return " ".join(text_parts).strip()


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

    for strategy_name, strategy in (
        ("manually_created", lambda transcript_list_obj: transcript_list_obj.find_manually_created_transcript(languages=["en", "hi", "es"])),
        ("generated", lambda transcript_list_obj: transcript_list_obj.find_generated_transcript(languages=["en", "hi", "es"])),
    ):
        try:
            transcript = strategy(transcript_list)
            break
        except NoTranscriptFound as exc:
            if strategy_name == "generated":
                first_error = exc
            continue
        except TranscriptsDisabled as exc:
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
    sample_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    try:
        payload = load_youtube_transcript(sample_url)
        print(payload["video_id"], len(payload["text"]))
    except Exception as exc:  # pragma: no cover - defensive guard
        print(exc)
