from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript(video_id: str) -> str:
    """
    Fetch transcript from a YouTube video.

    Args:
        video_id (str): YouTube video ID

    Returns:
        str: Complete transcript as a single string
    """

    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en", "hi"]   # Priority: English, then Hindi
        )
         

        transcript_text = " ".join(chunk.text for chunk in transcript)

        print("Trenascript :", len(transcript_text))

        return transcript_text

    except Exception as e:
        print("No caption available for this video:", e)
        return None


# For testing this file independently
if __name__ == "__main__":
    video_id = input("Enter Video ID: ")

    transcript = get_transcript(video_id)

    if transcript:
        print(transcript)
