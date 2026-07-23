import unittest
import sys
import os

# Ensure backend directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.app.services.youtube import yt_service
from backend.app.services.web_search import search_and_synthesize
from backend.app.services.ai import ai_service
from backend.app.services.database import db

class TestTubeMindBackend(unittest.TestCase):
    def setUp(self):
        # Setup settings overrides if any
        pass

    def test_youtube_id_extraction(self):
        urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ")
        ]
        for url, expected in urls:
            self.assertEqual(yt_service.extract_video_id(url), expected)

    def test_web_search_fallback(self):
        query = "Python fastapi developer documentation tutorial"
        result = search_and_synthesize(query)
        self.assertIn("context", result)
        self.assertIn("sources", result)
        self.assertTrue(len(result["sources"]) >= 0)

    def test_ai_scoring(self):
        mock_video = {
            "title": "SpaceX Spacecraft Development Update",
            "channel_name": "SpaceX",
            "description": "Starship is a fully reusable spacecraft design",
            "views": 1000000,
            "likes": 50000,
            "published_at": "1 day ago"
        }
        scores = ai_service.calculate_recommendation_scores(mock_video, "SpaceX Starship")
        self.assertIn("overall_score", scores)
        self.assertIn("why_recommended", scores)
        self.assertTrue(scores["overall_score"] > 50)

    def test_database_chat_persistence(self):
        session_id = "test-session-123"
        chat = db.create_chat(session_id, "Test Chat Title", "gemini")
        self.assertEqual(chat["id"], session_id)
        
        # Test append
        updated = db.append_chat_message(session_id, "user", "Hello AI!")
        self.assertEqual(len(updated["messages"]), 1)
        self.assertEqual(updated["messages"][0]["content"], "Hello AI!")
        
        # Clean up
        db.delete_chat(session_id)

if __name__ == "__main__":
    unittest.main()
