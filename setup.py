#!/usr/bin/env python3
"""
Setup script for Evaluation Agent Service
Downloads required models and sets up dependencies
"""

import os
import sys
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """Download a file from URL to destination"""
    logger.info(f"Downloading {url} to {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded {destination}")

def setup_fer_model():
    """Set up FER (Facial Emotion Recognition) model"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    fer_model_path = models_dir / "fer_model_from_images.h5"
    
    if not fer_model_path.exists():
        logger.info("FER model not found. Please download it manually:")
        logger.info("1. The model should be available from your project's models/ directory")
        logger.info(f"2. Place fer_model_from_images.h5 at {fer_model_path}")
        logger.warning("Service will run in mock mode without the model")
    else:
        logger.info("FER model found")

def setup_whisper():
    """Set up Whisper model (will be downloaded automatically)"""
    logger.info("Whisper model will be downloaded automatically on first use")
    logger.info("Models are cached in ~/.cache/whisper/")

def setup_directories():
    """Create necessary directories"""
    directories = ["temp", "models", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_groq_config():
    """Check Groq API configuration"""
    groq_key = os.getenv('GROQ_API_KEY')
    
    if groq_key:
        logger.info("Groq API key found in environment")
    else:
        logger.warning("Groq API key not found in environment")
        logger.info("Set GROQ_API_KEY environment variable for LLM evaluation")
        logger.info("Service will run in mock mode for content analysis")

def main():
    """Main setup function"""
    logger.info("Setting up Evaluation Agent Service")
    
    # Create directories
    setup_directories()
    
    # Set up models
    setup_fer_model()
    setup_whisper()
    
    # Check configuration
    check_groq_config()
    
    logger.info("Setup completed!")
    logger.info("To start the service, run: python app/main.py")
    logger.info("Or set GROQ_API_KEY environment variable and run with Docker")

if __name__ == "__main__":
    main()