import uuid
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Query, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.app.services.youtube import yt_service
from backend.app.services.vector_store import vector_store_service
from backend.app.services.web_search import search_and_synthesize
from backend.app.services.ai import ai_service
from backend.app.services.database import db
from backend.app.config.config import settings

app = FastAPI(title="TubeMind AI API", version="1.0.0")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://10.220.239.180:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST/RESPONSE SCHEMAS ---
class SettingsUpdate(BaseModel):
    active_model: str
    vector_db: str
    api_keys: Dict[str, str]

class AnalyzeRequest(BaseModel):
    video_url: str

class ChatCreateRequest(BaseModel):
    title: str
    model: str

class ChatMessageRequest(BaseModel):
    session_id: str
    video_id: str
    message: str

# --- ENDPOINTS ---

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "TubeMind AI Backend"}

# --- Settings ---
@app.get("/api/settings")
def get_settings():
    return db.get_settings()

@app.post("/api/settings")
def update_settings(payload: SettingsUpdate):
    db.update_settings(payload.dict())
    return {"status": "success", "settings": db.get_settings()}

# --- Analytics ---
@app.get("/api/analytics")
def get_analytics():
    return db.get_analytics()

# --- Recommendations Engine ---
@app.get("/api/recommend")
def recommend_videos(
    q: str = Query(..., description="Topic or keywords to search"),
    mode: str = Query("normal", description="Search mode: normal, ai, semantic, trending, latest, channel, playlist")
):
    try:
        # Search YouTube
        raw_videos = yt_service.search_videos(q, search_mode=mode, max_results=12)
        if not raw_videos:
            return []

        # Remove duplicate video IDs
        seen_ids = set()
        deduped_videos = []
        for video in raw_videos:
            video_id = video.get("id")
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                deduped_videos.append(video)
        raw_videos = deduped_videos

        # Retrieve settings
        current_settings = db.get_settings()
        active_model = current_settings.get("active_model", "gemini")
        api_keys = current_settings.get("api_keys", {})

        # Compute AI recommendations scores
        ranked_videos = []
        for video in raw_videos:
            video_id = video.get("id")
            if not video_id:
                continue
                
            scores = ai_service.calculate_recommendation_scores(
                video=video,
                query=q,
                provider=active_model,
                user_keys=api_keys
            )
            
            # Map values to match recommendation card requirement
            ranked_videos.append({
                "id": video_id,
                "title": video.get("title"),
                "description": video.get("description"),
                "thumbnail": video.get("thumbnail"),
                "channel_name": video.get("channel_name"),
                "views": video.get("views", 125000),
                "likes": video.get("likes", 6500),
                "published_date": video.get("published_at"),
                "duration": video.get("duration", "10:00"),
                "category": video.get("category", "Tech"),
                "scores": scores
            })

        # Sort recommendations by overall AI recommendation score descending
        if mode in ["ai", "semantic"]:
            ranked_videos.sort(key=lambda x: x["scores"]["overall_score"], reverse=True)

        # Log recommendation search to database
        db.save_recommendations(q, mode, ranked_videos)
        
        return ranked_videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Video Intelligence Analyzer ---
@app.post("/api/analyze")
def analyze_video(payload: AnalyzeRequest):
    video_url = payload.video_url
    video_id = yt_service.extract_video_id(video_url)
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or Video ID.")

    # Check if analysis is already cached in database
    cached_analysis = db.get_analyzed_video(video_id)
    if cached_analysis:
        print(f"Loading cached intelligence results for video {video_id}...")
        return {
            "video_id": video_id,
            "status": "cached",
            "details": cached_analysis["video_details"],
            "analysis": cached_analysis["analysis_result"]
        }

    # Fetch video details
    video_details = yt_service.get_video_details(video_id)
    if not video_details:
        raise HTTPException(status_code=404, detail="Could not retrieve video details from YouTube.")

    # Retrieve transcript
    transcript = yt_service.get_transcript(video_id)
    source_type = "transcript"
    context_text = ""

    if transcript:
        context_text = transcript
    else:
        # Transcript unavailable -> trigger web search fallback
        source_type = "web_search"
        search_query = f"{video_details.get('title')} {video_details.get('channel_name')}"
        search_result = search_and_synthesize(search_query)
        context_text = search_result["context"]

    # Build and cache FAISS index
    index_success = vector_store_service.create_index(video_id, context_text)
    if not index_success:
        print(f"Warning: Failed to index text into vector store for video {video_id}.")

    # Gather active settings
    current_settings = db.get_settings()
    active_model = current_settings.get("active_model", "gemini")
    api_keys = current_settings.get("api_keys", {})

    # Generate AI summaries & key insights
    try:
        short_summary = ai_service.generate_summary(video_details["title"], context_text, "short", active_model, api_keys)
        medium_summary = ai_service.generate_summary(video_details["title"], context_text, "medium", active_model, api_keys)
        detailed_summary = ai_service.generate_summary(video_details["title"], context_text, "detailed", active_model, api_keys)
        bullet_summary = ai_service.generate_summary(video_details["title"], context_text, "bullet", active_model, api_keys)
        executive_summary = ai_service.generate_summary(video_details["title"], context_text, "executive", active_model, api_keys)
        
        insights = ai_service.generate_insights(video_details["title"], context_text, active_model, api_keys)
    except Exception as e:
        print(f"Error generating AI summaries: {e}")
        # Build mock structures
        short_summary = ai_service._mock_summary(video_details["title"], "short")
        medium_summary = ai_service._mock_summary(video_details["title"], "medium")
        detailed_summary = ai_service._mock_summary(video_details["title"], "detailed")
        bullet_summary = ai_service._mock_summary(video_details["title"], "bullet")
        executive_summary = ai_service._mock_summary(video_details["title"], "executive")
        insights = ai_service._mock_insights(video_details["title"])

    # Calculate recommendation details for video details mapping
    scores = ai_service.calculate_recommendation_scores(video_details, video_details["title"], active_model, api_keys)

    analysis_result = {
        "source_type": source_type,
        "summaries": {
            "short": short_summary,
            "medium": medium_summary,
            "detailed": detailed_summary,
            "bullet": bullet_summary,
            "executive": executive_summary
        },
        "insights": insights,
        "scores": scores
    }

    # Save to JSON database
    db.save_analyzed_video(video_id, video_details, analysis_result)

    return {
        "video_id": video_id,
        "status": "success",
        "details": video_details,
        "analysis": analysis_result
    }

# --- Bookmarks ---
@app.get("/api/bookmarks")
def get_bookmarks():
    return db.get_bookmarks()

@app.post("/api/bookmarks")
def add_bookmark(video: Dict[str, Any] = Body(...)):
    success = db.add_bookmark(video)
    return {"status": "success" if success else "error"}

@app.delete("/api/bookmarks/{video_id}")
def remove_bookmark(video_id: str):
    success = db.remove_bookmark(video_id)
    return {"status": "success" if success else "error"}

# --- Chatbot API ---
@app.get("/api/chat/sessions")
def get_chat_sessions():
    return db.get_chats()

@app.get("/api/chat/session/{session_id}")
def get_chat_session(session_id: str):
    chat = db.get_chat(session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return chat

@app.post("/api/chat/create")
def create_chat_session(payload: ChatCreateRequest):
    session_id = str(uuid.uuid4())
    chat = db.create_chat(session_id, payload.title, payload.model)
    return chat

@app.post("/api/chat/message")
def post_chat_message(payload: ChatMessageRequest):
    session_id = payload.session_id
    video_id = payload.video_id
    query = payload.message

    chat = db.get_chat(session_id)
    if not chat:
        # Create on the fly
        chat = db.create_chat(session_id, "New Chat", "gemini")

    # Fetch analyzed video metadata to use title
    video_cache = db.get_analyzed_video(video_id)
    video_title = "YouTube Video"
    source_type = "transcript"
    
    if video_cache:
        video_title = video_cache["video_details"].get("title", video_title)
        source_type = video_cache["analysis_result"].get("source_type", source_type)

    # Retrieve context from local FAISS store
    context_chunks = vector_store_service.query_index(video_id, query, top_k=4)
    
    if context_chunks:
        context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])
    else:
        # If vector store is empty, retrieve cached video's transcript if any, or trigger fallback search
        transcript = yt_service.get_transcript(video_id)
        if transcript:
            context_text = transcript
        else:
            search_query = f"{video_title} query response helper"
            search_result = search_and_synthesize(search_query)
            context_text = search_result["context"]
            source_type = "web_search"

    # Get settings
    current_settings = db.get_settings()
    active_model = current_settings.get("active_model", "gemini")
    api_keys = current_settings.get("api_keys", {})

    # Append user message
    db.append_chat_message(session_id, "user", query, source_type)

    # Compile chat history for LangChain context
    history_messages = chat["messages"]

    # Generate LLM response with timing measurement
    import time
    start_time = time.time()

    response_text = ai_service.chat(
        session_id=session_id,
        video_title=video_title,
        query=query,
        history=history_messages[:-1],  # exclude current user message as we send it as query
        context=context_text,
        provider=active_model,
        user_keys=api_keys
    )
    
    elapsed_time = time.time() - start_time

    # Output verbose RAG Chat execution logs to stdout for development auditing
    print("\n" + "="*70)
    print(f"[RAG AUDIT] Chat Session: {session_id[:8]}...")
    print(f"[RAG AUDIT] User Question: '{query}'")
    print(f"[RAG AUDIT] Provider Engine: {active_model.upper()}")
    print(f"[RAG AUDIT] Context Size Passed: {len(context_text)} characters")
    print(f"[RAG AUDIT] Response Time: {elapsed_time:.3f}s")
    print(f"[RAG AUDIT] LLM Response Preview:")
    print(f"  {response_text[:200]}...")
    print("="*70 + "\n")

    # Append AI assistant response
    updated_chat = db.append_chat_message(session_id, "assistant", response_text, source_type)
    
    # Rename chat if it is the first exchange
    if len(updated_chat["messages"]) <= 2:
        new_title = query[:25] + ("..." if len(query) > 25 else "")
        db.rename_chat(session_id, new_title)
        updated_chat["title"] = new_title

    return {
        "response": response_text,
        "session_id": session_id,
        "chat": updated_chat,
        "source_type": source_type
    }

@app.post("/api/chat/rename/{session_id}")
def rename_chat_session(session_id: str, title: str = Body(..., embed=True)):
    success = db.rename_chat(session_id, title)
    return {"status": "success" if success else "error"}

@app.delete("/api/chat/{session_id}")
def delete_chat_session(session_id: str):
    success = db.delete_chat(session_id)
    return {"status": "success" if success else "error"}

# --- Downloads ---
@app.get("/api/download-transcript/{video_id}")
def download_transcript(video_id: str):
    transcript = yt_service.get_transcript(video_id)
    if not transcript:
        # Attempt fallback to analyzed text
        cached_analysis = db.get_analyzed_video(video_id)
        if cached_analysis:
            # Render a summary or fallback crawler contents
            transcript = f"Detailed transcript not available.\n\nSummary overview:\n{cached_analysis['analysis_result']['summaries']['medium']}"
        else:
            raise HTTPException(status_code=404, detail="Transcript not available for this video.")

    response = Response(content=transcript, media_type="text/plain")
    response.headers["Content-Disposition"] = f"attachment; filename=transcript_{video_id}.txt"
    return response
