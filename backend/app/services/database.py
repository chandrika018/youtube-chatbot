import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "db.json")

class FileDatabase:
    def __init__(self):
        self.db_path = DB_FILE
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            self._write_db(self._get_default_schema())

    def _get_default_schema(self) -> Dict[str, Any]:
        return {
            "chats": {},            # session_id -> { id, title, created_at, model, messages: [...] }
            "analyzed_videos": {},  # video_id -> { video_details, analysis_result, timestamp }
            "recommendations": [],  # list of searches and their recommendations
            "bookmarks": [],        # list of bookmarked video details
            "settings": {
                "active_model": "gemini",
                "vector_db": "faiss",
                "api_keys": {}
            },
            "analytics": {
                "videos_analyzed_count": 0,
                "recommendations_generated_count": 0,
                "average_ai_score": 0.0,
                "searched_topics": {}  # topic -> count
            }
        }

    def _read_db(self) -> Dict[str, Any]:
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return self._get_default_schema()

    def _write_db(self, data: Dict[str, Any]):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to database: {e}")

    # --- Chat History Management ---
    def get_chats(self) -> List[Dict[str, Any]]:
        db = self._read_db()
        # Sort chats by created_at desc
        chats = list(db["chats"].values())
        chats.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return chats

    def get_chat(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = self._read_db()
        return db["chats"].get(session_id)

    def create_chat(self, session_id: str, title: str, model: str) -> Dict[str, Any]:
        db = self._read_db()
        chat = {
            "id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "model": model,
            "messages": []
        }
        db["chats"][session_id] = chat
        self._write_db(db)
        return chat

    def append_chat_message(self, session_id: str, role: str, content: str, source_type: str = "transcript") -> Dict[str, Any]:
        db = self._read_db()
        if session_id not in db["chats"]:
            db["chats"][session_id] = {
                "id": session_id,
                "title": "New Chat",
                "created_at": datetime.now().isoformat(),
                "model": "gemini",
                "messages": []
            }
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "source_type": source_type  # "transcript" or "web_search"
        }
        db["chats"][session_id]["messages"].append(message)
        self._write_db(db)
        return db["chats"][session_id]

    def rename_chat(self, session_id: str, new_title: str) -> bool:
        db = self._read_db()
        if session_id in db["chats"]:
            db["chats"][session_id]["title"] = new_title
            self._write_db(db)
            return True
        return False

    def delete_chat(self, session_id: str) -> bool:
        db = self._read_db()
        if session_id in db["chats"]:
            del db["chats"][session_id]
            self._write_db(db)
            return True
        return False

    # --- Analyzed Videos Cache ---
    def get_analyzed_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        db = self._read_db()
        return db["analyzed_videos"].get(video_id)

    def save_analyzed_video(self, video_id: str, video_details: Dict[str, Any], analysis_result: Dict[str, Any]):
        db = self._read_db()
        db["analyzed_videos"][video_id] = {
            "video_details": video_details,
            "analysis_result": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update analytics
        db["analytics"]["videos_analyzed_count"] = len(db["analyzed_videos"])
        
        # Recalculate average AI Score
        scores = []
        for v in db["analyzed_videos"].values():
            if "analysis_result" in v and "scores" in v["analysis_result"]:
                scores.append(v["analysis_result"]["scores"].get("overall_score", 0))
        if scores:
            db["analytics"]["average_ai_score"] = round(sum(scores) / len(scores), 1)
            
        self._write_db(db)

    # --- Bookmarks / Saved Videos ---
    def get_bookmarks(self) -> List[Dict[str, Any]]:
        db = self._read_db()
        return db["bookmarks"]

    def add_bookmark(self, video: Dict[str, Any]) -> bool:
        db = self._read_db()
        video_id = video.get("id") or video.get("video_id")
        if not video_id:
            return False
        # Avoid duplicate bookmarks
        if any(b.get("id") == video_id or b.get("video_id") == video_id for b in db["bookmarks"]):
            return True
        db["bookmarks"].append(video)
        self._write_db(db)
        return True

    def remove_bookmark(self, video_id: str) -> bool:
        db = self._read_db()
        initial_len = len(db["bookmarks"])
        db["bookmarks"] = [b for b in db["bookmarks"] if b.get("id") != video_id and b.get("video_id") != video_id]
        self._write_db(db)
        return len(db["bookmarks"]) < initial_len

    # --- Recommendations and Search ---
    def save_recommendations(self, topic: str, search_mode: str, items: List[Dict[str, Any]]):
        db = self._read_db()
        db["recommendations"].append({
            "topic": topic,
            "search_mode": search_mode,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(items)
        })
        
        # Update analytics
        db["analytics"]["recommendations_generated_count"] += 1
        db["analytics"]["searched_topics"][topic] = db["analytics"]["searched_topics"].get(topic, 0) + 1
        
        self._write_db(db)

    # --- Settings ---
    def get_settings(self) -> Dict[str, Any]:
        db = self._read_db()
        return db.get("settings", {})

    def update_settings(self, new_settings: Dict[str, Any]):
        db = self._read_db()
        db["settings"].update(new_settings)
        self._write_db(db)

    # --- Analytics ---
    def get_analytics(self) -> Dict[str, Any]:
        db = self._read_db()
        return db.get("analytics", {
            "videos_analyzed_count": 0,
            "recommendations_generated_count": 0,
            "average_ai_score": 0.0,
            "searched_topics": {}
        })

db = FileDatabase()
