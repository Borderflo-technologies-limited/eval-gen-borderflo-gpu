#!/usr/bin/env python3
"""
Interview Evaluation Service
Handles FER, Whisper, and LLM-based interview evaluation
"""

import os
import cv2
import numpy as np
import logging
import time
import json
from typing import Dict, Any, List, Tuple
import asyncio
import aiohttp

# ML Libraries
try:
    import tensorflow as tf
    from faster_whisper import WhisperModel
    from moviepy.editor import VideoFileClip
    TENSORFLOW_AVAILABLE = True
    FASTER_WHISPER_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    FASTER_WHISPER_AVAILABLE = False
    logging.warning(
        f"ML libraries not available: {e}. Service will run in mock mode."
    )

from config import settings

logger = logging.getLogger(__name__)


class InterviewEvaluator:
    """Main interview evaluation class"""
    
    def __init__(self):
        self.fer_model = None
        self.whisper_model = None
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
        ]
        
    async def initialize(self):
        """Initialize all models"""
        logger.info("Initializing evaluation models...")
        
        # Load FER model
        await self._load_fer_model()
        
        # Load Whisper model
        await self._load_whisper_model()
        
        logger.info("Model initialization completed")
    
    async def _load_fer_model(self):
        """Load Facial Emotion Recognition model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning(
                "TensorFlow not available, FER will use mock mode"
            )
            return
            
        try:
            if os.path.exists(settings.FER_MODEL_PATH):
                logger.info(
                    f"Loading FER model from {settings.FER_MODEL_PATH}"
                )
                # Load TensorFlow model
                self.fer_model = tf.keras.models.load_model(
                    settings.FER_MODEL_PATH
                )
                logger.info("FER model loaded successfully")
            else:
                logger.warning(
                    f"FER model not found at {settings.FER_MODEL_PATH}"
                )
                self.fer_model = None
                
        except Exception as e:
            logger.error(f"Error loading FER model: {e}")
            self.fer_model = None
    
    async def _load_whisper_model(self):
        """Load Whisper speech-to-text model"""
        if not FASTER_WHISPER_AVAILABLE:
            logger.warning(
                "Faster Whisper not available, speech analysis will use mock mode"
            )
            return
            
        try:
            logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL_SIZE}")
            # Prefer GPU with float16; fall back to int8 on CPU
            device_choice = "cuda" if settings.DEVICE == "cuda" else "cpu"
            compute_choice = "float16" if device_choice == "cuda" else "int8"
            self.whisper_model = WhisperModel(
                settings.WHISPER_MODEL_SIZE,
                device=device_choice,
                compute_type=compute_choice
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
    
    def extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract frames from video for emotion analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame skip to get max_frames evenly distributed
            frame_skip = max(1, total_frames // max_frames)
            
            frame_count = 0
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame using OpenCV"""
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def preprocess_face(self, face_region: np.ndarray) -> np.ndarray:
        """Preprocess face region for FER model"""
        try:
            # Resize to model input size (typically 48x48 for FER)
            face_resized = cv2.resize(face_region, (48, 48))
            
            # Convert to grayscale if needed
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Normalize pixel values
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=(0, -1))
            
            return face_batch
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None
    
    async def analyze_emotions(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze emotions in video frames"""
        if not self.fer_model or not TENSORFLOW_AVAILABLE:
            # Mock emotion analysis
            logger.info("Using mock emotion analysis")
            return {
                "dominant_emotion": "neutral",
                "emotion_confidence": 0.85,
                "emotion_timeline": [
                    {"timestamp": 0.0, "emotion": "neutral", "confidence": 0.85},
                    {"timestamp": 2.5, "emotion": "happy", "confidence": 0.72},
                    {"timestamp": 5.0, "emotion": "neutral", "confidence": 0.78}
                ],
                "face_detection_rate": 0.9,
                "emotion_distribution": {
                    "neutral": 0.6,
                    "happy": 0.25,
                    "nervous": 0.1,
                    "confident": 0.05
                }
            }
        
        try:
            emotion_timeline = []
            emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
            total_faces_detected = 0
            
            for i, frame in enumerate(frames):
                timestamp = i * (len(frames) / 30.0)  # Assume ~30 fps
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                if faces:
                    total_faces_detected += 1
                    # Use the largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    # Extract face region
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Preprocess for model
                    processed_face = self.preprocess_face(face_region)
                    
                    if processed_face is not None:
                        # Predict emotion
                        prediction = self.fer_model.predict(processed_face, verbose=0)
                        emotion_idx = np.argmax(prediction[0])
                        confidence = float(prediction[0][emotion_idx])
                        emotion = self.emotion_labels[emotion_idx]
                        
                        emotion_timeline.append({
                            "timestamp": timestamp,
                            "emotion": emotion,
                            "confidence": confidence
                        })
                        
                        emotion_counts[emotion] += 1
            
            # Calculate statistics
            total_emotions = sum(emotion_counts.values())
            emotion_distribution = {k: v/total_emotions if total_emotions > 0 else 0 
                                  for k, v in emotion_counts.items()}
            
            dominant_emotion = max(emotion_distribution, key=emotion_distribution.get)
            face_detection_rate = total_faces_detected / len(frames) if frames else 0
            
            return {
                "dominant_emotion": dominant_emotion,
                "emotion_confidence": emotion_distribution[dominant_emotion],
                "emotion_timeline": emotion_timeline,
                "face_detection_rate": face_detection_rate,
                "emotion_distribution": emotion_distribution
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return self.analyze_emotions([])  # Return mock data
    
    async def transcribe_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract and transcribe audio from video"""
        if not self.whisper_model or not FASTER_WHISPER_AVAILABLE:
            # Mock transcription
            logger.info("Using mock audio transcription")
            return {
                "transcript": (
                    "Thank you for the question. I believe I am well qualified for "
                    "this position because of my experience in software development "
                    "and my passion for learning new technologies. I have worked on "
                    "several projects that involved similar challenges."
                ),
                "confidence": 0.92,
                "speech_clarity": 0.85,
                "speaking_pace": 150,  # words per minute
                "audio_duration": 15.5,
                "audio_quality": 0.88
            }
        
        try:
            # Extract audio from video
            logger.info("Extracting audio from video")
            video_clip = VideoFileClip(video_path)
            audio_path = video_path.replace('.mp4', '_audio.wav')
            video_clip.audio.write_audiofile(
                audio_path, verbose=False, logger=None
            )
            video_clip.close()
            
            # Transcribe with Faster Whisper
            logger.info("Transcribing audio with Faster Whisper")
            segments, info = self.whisper_model.transcribe(audio_path)
            
            # Extract transcript from segments
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text)
            
            transcript = ' '.join(transcript_parts).strip()
            words = transcript.split()
            word_count = len(words)
            
            # Get audio duration
            audio_clip = VideoFileClip(video_path).audio
            duration = audio_clip.duration
            audio_clip.close()
            
            # Calculate speaking pace (words per minute)
            speaking_pace = (
                (word_count / duration) * 60 if duration > 0 else 0
            )
            
            # Estimate confidence from Whisper segments
            if segments:
                avg_confidence = np.mean([
                    seg.avg_logprob for seg in segments
                ])
                confidence = max(0, min(1, (avg_confidence + 1) / 2))
            else:
                confidence = 0.85  # Default confidence
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return {
                "transcript": transcript,
                "confidence": confidence,
                "speech_clarity": confidence,
                "speaking_pace": speaking_pace,
                "audio_duration": duration,
                "audio_quality": confidence,
                "word_count": word_count
            }
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return self.transcribe_audio("")  # Return mock data
    
    async def analyze_content_with_llm(self, transcript: str, question_text: str) -> Dict[str, Any]:
        """Analyze interview content using Groq LLM"""
        if not settings.GROQ_API_KEY:
            # Mock content analysis
            logger.info("Using mock content analysis")
            return {
                "content_score": 8.2,
                "relevance_score": 8.5,
                "completeness_score": 7.8,
                "confidence_level": "High",
                "feedback": "Good response that addresses the question well. Shows clear understanding of the topic and provides relevant examples.",
                "recommendations": [
                    "Consider providing more specific examples",
                    "Structure your response with clear points",
                    "Expand on your experience with concrete details"
                ],
                "key_strengths": [
                    "Clear communication",
                    "Relevant experience mentioned",
                    "Confident delivery"
                ],
                "areas_for_improvement": [
                    "More specific examples needed",
                    "Could elaborate on technical skills"
                ]
            }
        
        try:
            # Prepare LLM prompt
            prompt = f"""
            Analyze this interview response for a visa application. 
            
            Question: {question_text}
            
            Response: {transcript}
            
            Please evaluate the response on the following criteria:
            1. Content relevance to the question (0-10)
            2. Completeness of the answer (0-10) 
            3. Overall quality and confidence (0-10)
            
            Provide:
            - Scores for each criteria
            - Brief feedback (2-3 sentences)
            - 2-3 specific recommendations for improvement
            - Key strengths (if any)
            - Areas for improvement
            
            Format your response as JSON with keys: content_score, relevance_score, completeness_score, confidence_level, feedback, recommendations, key_strengths, areas_for_improvement
            """
            
            # Make API call to Groq
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": settings.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert interview evaluator for visa applications. Provide constructive, helpful feedback."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(settings.GROQ_API_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # Try to parse JSON response
                        try:
                            analysis = json.loads(content)
                            return analysis
                        except json.JSONDecodeError:
                            # If JSON parsing fails, extract key information
                            return self._parse_llm_response(content)
                    else:
                        logger.error(f"Groq API error: {response.status}")
                        return self.analyze_content_with_llm("", "")  # Return mock data
                        
        except Exception as e:
            logger.error(f"LLM content analysis failed: {e}")
            return self.analyze_content_with_llm("", "")  # Return mock data
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse non-JSON LLM response"""
        # Simple parsing logic - in production, this would be more robust
        return {
            "content_score": 7.5,
            "relevance_score": 8.0,
            "completeness_score": 7.0,
            "confidence_level": "Medium",
            "feedback": response_text[:200] + "..." if len(response_text) > 200 else response_text,
            "recommendations": ["Review the full response for detailed feedback"],
            "key_strengths": ["Response provided"],
            "areas_for_improvement": ["See detailed feedback"]
        }
    
    async def evaluate_interview(self, video_path: str, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main evaluation function - coordinates all analysis"""
        try:
            start_time = time.time()
            
            logger.info(f"Starting comprehensive interview evaluation for {video_path}")
            
            # Extract frames for emotion analysis
            frames = self.extract_frames(video_path)
            
            # Run analyses in parallel
            emotion_task = asyncio.create_task(self.analyze_emotions(frames))
            speech_task = asyncio.create_task(self.transcribe_audio(video_path))
            
            # Wait for both to complete
            emotion_results, speech_results = await asyncio.gather(emotion_task, speech_task)
            
            # Analyze content with LLM
            content_results = await self.analyze_content_with_llm(
                speech_results['transcript'], 
                evaluation_data.get('question_text', '')
            )
            
            # Calculate overall score
            overall_score = (
                content_results['content_score'] * 0.4 +
                content_results['relevance_score'] * 0.3 +
                speech_results['confidence'] * 10 * 0.2 +
                emotion_results['emotion_confidence'] * 10 * 0.1
            )
            
            # Compile final results
            results = {
                # Speech Analysis
                "transcript": speech_results['transcript'],
                "speech_confidence": speech_results['confidence'],
                "speech_clarity": speech_results['speech_clarity'],
                "speaking_pace": speech_results['speaking_pace'],
                
                # Emotion Analysis
                "dominant_emotion": emotion_results['dominant_emotion'],
                "emotion_confidence": emotion_results['emotion_confidence'],
                "emotion_timeline": emotion_results['emotion_timeline'],
                
                # Content Analysis
                "content_score": content_results['content_score'],
                "relevance_score": content_results['relevance_score'],
                "completeness_score": content_results['completeness_score'],
                "confidence_level": content_results['confidence_level'],
                
                # Overall Assessment
                "overall_score": round(overall_score, 2),
                "feedback": content_results['feedback'],
                "recommendations": content_results['recommendations'],
                
                # Technical Metrics
                "video_duration": speech_results['audio_duration'],
                "face_detection_rate": emotion_results['face_detection_rate'],
                "audio_quality": speech_results['audio_quality'],
                
                # Additional Details
                "key_strengths": content_results.get('key_strengths', []),
                "areas_for_improvement": content_results.get('areas_for_improvement', []),
                "emotion_distribution": emotion_results['emotion_distribution'],
                "processing_time": time.time() - start_time
            }
            
            logger.info("Interview evaluation completed successfully")
            
            return {
                "success": True,
                "data": results
            }
            
        except Exception as e:
            logger.error(f"Interview evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def evaluate_audio_only(self, audio_path: str, question_text: str = None) -> Dict[str, Any]:
        """Evaluate audio-only interview (no video/emotion analysis)"""
        try:
            logger.info(f"Starting audio-only evaluation for {audio_path}")
            
            # For audio-only, we'll use Whisper directly on the audio file
            if not self.whisper_model or not FASTER_WHISPER_AVAILABLE:
                # Mock results
                speech_results = {
                    "transcript": "This is a mock transcription for audio-only evaluation.",
                    "confidence": 0.85,
                    "speech_clarity": 0.82,
                    "speaking_pace": 145,
                    "audio_duration": 12.3,
                    "audio_quality": 0.88
                }
            else:
                result = self.whisper_model.transcribe(audio_path)
                # Process results similar to video transcription
                transcript = result['text'].strip()
                words = transcript.split()
                word_count = len(words)
                
                # Estimate duration (Whisper doesn't always provide this for audio files)
                duration = result.get('duration', len(transcript) / 10)  # Rough estimate
                speaking_pace = (word_count / duration) * 60 if duration > 0 else 0
                
                speech_results = {
                    "transcript": transcript,
                    "confidence": 0.85,  # Default for audio-only
                    "speech_clarity": 0.85,
                    "speaking_pace": speaking_pace,
                    "audio_duration": duration,
                    "audio_quality": 0.85
                }
            
            # Analyze content with LLM
            content_results = await self.analyze_content_with_llm(
                speech_results['transcript'], 
                question_text or ""
            )
            
            # Calculate overall score (no emotion component for audio-only)
            overall_score = (
                content_results['content_score'] * 0.5 +
                content_results['relevance_score'] * 0.3 +
                speech_results['confidence'] * 10 * 0.2
            )
            
            results = {
                "transcript": speech_results['transcript'],
                "speech_confidence": speech_results['confidence'],
                "speech_clarity": speech_results['speech_clarity'],
                "speaking_pace": speech_results['speaking_pace'],
                "content_score": content_results['content_score'],
                "relevance_score": content_results['relevance_score'],
                "completeness_score": content_results['completeness_score'],
                "overall_score": round(overall_score, 2),
                "feedback": content_results['feedback'],
                "recommendations": content_results['recommendations'],
                "audio_duration": speech_results['audio_duration'],
                "audio_quality": speech_results['audio_quality'],
                "evaluation_type": "audio_only"
            }
            
            return {
                "success": True,
                "data": results
            }
            
        except Exception as e:
            logger.error(f"Audio-only evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }