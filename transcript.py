from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def extract_video_id(video_url: str):

    parsed_url = urlparse(video_url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed_url.query).get("v", [None])[0]

    return None

def get_transcript(video_url: str):

    video_id = extract_video_id(video_url)

    if not video_id:
        print("Invalid YouTube URL")
        return None

    try:

        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en", "hi"]
        )

        transcript_text = " ".join(
            chunk.text for chunk in transcript
        )

        print("Transcript Length:", len(transcript_text))

        return transcript_text

    except Exception as e:
        print("No caption available:", e)
        return None


# For testing this file independently
if __name__ == "__main__":

    video_url = input("Enter YouTube URL: ")

    transcript = get_transcript(video_url)

    if transcript:
        print(transcript[:500])
