#!/usr/bin/env python3
"""
Evaluation Agent Service
Handles interview evaluation using FER, Whisper, and LLM analysis
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import time
import uuid
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from evaluation_service import InterviewEvaluator
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize evaluator
evaluator = InterviewEvaluator()

class EvaluationRequest(BaseModel):
    """Request model for interview evaluation"""
    session_id: str
    question_id: str
    question_text: str
    expected_duration: int = 60  # seconds

class EvaluationResponse(BaseModel):
    """Response model for interview evaluation"""
    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class EvaluationResults(BaseModel):
    """Detailed evaluation results"""
    # Speech Analysis
    transcript: str
    speech_confidence: float
    speech_clarity: float
    speaking_pace: float  # words per minute
    
    # Emotion Analysis
    dominant_emotion: str
    emotion_confidence: float
    emotion_timeline: List[Dict[str, Any]]
    
    # Content Analysis
    content_score: float
    relevance_score: float
    completeness_score: float
    confidence_level: str
    
    # Overall Assessment
    overall_score: float
    feedback: str
    recommendations: List[str]
    
    # Technical Metrics
    video_duration: float
    face_detection_rate: float
    audio_quality: float

# FastAPI app
app = FastAPI(
    title="Evaluation Agent Service",
    description="AI-powered interview evaluation using FER, Whisper, and LLM analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the service"""
    logger.info("Starting Evaluation Agent Service")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"FER Model path: {settings.FER_MODEL_PATH}")
    logger.info(f"Groq API configured: {bool(settings.GROQ_API_KEY)}")
    
    # Pre-load models
    try:
        await evaluator.initialize()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize models: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "evaluation-agent",
        "device": settings.DEVICE,
        "fer_model_loaded": evaluator.fer_model is not None,
        "whisper_model_loaded": evaluator.whisper_model is not None,
        "groq_configured": bool(settings.GROQ_API_KEY)
    }

@app.post("/evaluate-interview/", response_model=EvaluationResponse)
async def evaluate_interview(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    session_id: str = None,
    question_id: str = None,
    question_text: str = None,
    expected_duration: int = 60
):
    """
    Evaluate interview video for emotions, speech, and content
    """
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file type
        if not video_file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid video file type")
        
        # Save uploaded video
        temp_video_path = os.path.join(settings.TEMP_DIR, f"{task_id}_video.mp4")
        
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Perform evaluation
        logger.info(f"Starting evaluation for task {task_id}")
        
        evaluation_data = {
            "session_id": session_id,
            "question_id": question_id,
            "question_text": question_text,
            "expected_duration": expected_duration
        }
        
        results = await evaluator.evaluate_interview(temp_video_path, evaluation_data)
        
        processing_time = time.time() - start_time
        
        if results["success"]:
            return EvaluationResponse(
                task_id=task_id,
                status="completed",
                results=results["data"],
                processing_time=processing_time
            )
        else:
            return EvaluationResponse(
                task_id=task_id,
                status="failed",
                error=results["error"],
                processing_time=processing_time
            )
            
    except Exception as e:
        logger.error(f"Evaluation endpoint error: {str(e)}")
        processing_time = time.time() - start_time
        
        return EvaluationResponse(
            task_id=task_id,
            status="failed",
            error=str(e),
            processing_time=processing_time
        )

@app.post("/evaluate-audio/")
async def evaluate_audio_only(
    audio_file: UploadFile = File(...),
    question_text: str = None
):
    """
    Evaluate audio-only interview (speech and content analysis)
    """
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file type")
        
        # Save uploaded audio
        temp_audio_path = os.path.join(settings.TEMP_DIR, f"{task_id}_audio.wav")
        
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Perform audio-only evaluation
        results = await evaluator.evaluate_audio_only(temp_audio_path, question_text)
        
        processing_time = time.time() - start_time
        
        # Cleanup
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return {
            "task_id": task_id,
            "status": "completed" if results["success"] else "failed",
            "results": results.get("data"),
            "error": results.get("error"),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Audio evaluation error: {str(e)}")
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "processing_time": time.time() - start_time
        }

@app.delete("/cleanup/{task_id}")
async def cleanup_files(task_id: str):
    """Clean up temporary files for a task"""
    files_to_remove = [
        f"{task_id}_video.mp4",
        f"{task_id}_audio.wav",
        f"{task_id}_frames/",  # Directory
        f"{task_id}_analysis.json"
    ]
    
    removed_files = []
    for filename in files_to_remove:
        file_path = os.path.join(settings.TEMP_DIR, filename)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            removed_files.append(filename)
    
    return {"removed_files": removed_files}

@app.get("/models/status")
async def get_models_status():
    """Get status of all loaded models"""
    return {
        "fer_model": {
            "loaded": evaluator.fer_model is not None,
            "path": settings.FER_MODEL_PATH,
            "type": "TensorFlow/Keras"
        },
        "whisper_model": {
            "loaded": evaluator.whisper_model is not None,
            "size": settings.WHISPER_MODEL_SIZE,
            "type": "OpenAI Whisper"
        },
        "groq_llm": {
            "configured": bool(settings.GROQ_API_KEY),
            "model": settings.GROQ_MODEL,
            "type": "Groq LLM API"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )