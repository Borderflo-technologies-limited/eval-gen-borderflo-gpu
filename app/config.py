#!/usr/bin/env python3
"""
Configuration for Evaluation Agent Service
"""

import os
from pathlib import Path

class Settings:
    """Evaluation Agent Service Settings"""
    
    # Service Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Model Configuration
    FER_MODEL_PATH: str = "models/fer_model_from_images.h5"
    WHISPER_MODEL_SIZE: str = "base"  # tiny, base, small, medium, large
    DEVICE: str = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES', 'all') else 'cpu'
    
    # Groq LLM Configuration
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Processing Configuration
    MAX_FRAMES_FOR_EMOTION: int = 30
    MAX_VIDEO_DURATION: int = 300  # 5 minutes
    EMOTION_CONFIDENCE_THRESHOLD: float = 0.6
    
    # File Storage
    TEMP_DIR: str = "temp"
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/avi", "video/mov", "video/webm"]
    ALLOWED_AUDIO_TYPES: list = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg"]
    
    # Evaluation Scoring
    CONTENT_WEIGHT: float = 0.4
    RELEVANCE_WEIGHT: float = 0.3
    SPEECH_WEIGHT: float = 0.2
    EMOTION_WEIGHT: float = 0.1
    
    # Performance Thresholds
    EXCELLENT_SCORE: float = 8.5
    GOOD_SCORE: float = 7.0
    AVERAGE_SCORE: float = 5.5
    POOR_SCORE: float = 4.0
    
    # Speech Analysis
    OPTIMAL_WPM_MIN: int = 120  # words per minute
    OPTIMAL_WPM_MAX: int = 180
    MIN_RESPONSE_LENGTH: int = 10  # minimum words
    
    # Emotion Analysis
    POSITIVE_EMOTIONS: list = ["happy", "neutral", "surprise"]
    NEGATIVE_EMOTIONS: list = ["angry", "disgust", "fear", "sad"]
    
    # Cleanup Configuration
    AUTO_CLEANUP_HOURS: int = 24
    CLEANUP_INTERVAL_MINUTES: int = 60
    
    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(Path(self.FER_MODEL_PATH).parent, exist_ok=True)
    
    def get_score_rating(self, score: float) -> str:
        """Get qualitative rating for a score"""
        if score >= self.EXCELLENT_SCORE:
            return "Excellent"
        elif score >= self.GOOD_SCORE:
            return "Good"
        elif score >= self.AVERAGE_SCORE:
            return "Average"
        elif score >= self.POOR_SCORE:
            return "Below Average"
        else:
            return "Poor"
    
    def is_optimal_speaking_pace(self, wpm: float) -> bool:
        """Check if speaking pace is within optimal range"""
        return self.OPTIMAL_WPM_MIN <= wpm <= self.OPTIMAL_WPM_MAX

# Global settings instance
settings = Settings()