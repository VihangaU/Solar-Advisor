"""
Voice input and output handler with language detection using streamlit-webrtc
"""
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from google.cloud import speech_v1
from gtts import gTTS
import tempfile
import os
import logging
from typing import Optional, Tuple
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceHandler:
    """Handle voice input and output with language detection"""

    def __init__(self):
        """Initialize voice handler"""
        self.speech_client = None

        # Supported languages for speech recognition
        self.language_codes = {
            'english': 'en-US',
            'sinhala': 'si-LK'
        }

        # For gTTS output
        self.tts_codes = {
            'english': 'en',
            'sinhala': 'si'
        }

    def _get_speech_client(self):
        """Lazily initialize speech client"""
        if self.speech_client is None:
            try:
                self.speech_client = speech_v1.SpeechClient()
            except Exception as e:
                logger.error(f"Error initializing Speech client: {e}")
                return None
        return self.speech_client

    def recognize_speech(self, audio_bytes: bytes, language: str = 'en-US') -> Optional[str]:
        """
        Recognize speech from audio bytes

        Args:
            audio_bytes: Raw audio data
            language: Language code (e.g., 'en-US', 'si-LK')

        Returns:
            Recognized text or None if failed
        """
        try:
            client = self._get_speech_client()
            if not client:
                return None

            audio = speech_v1.RecognitionAudio(content=audio_bytes)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language,
            )

            response = client.recognize(config=config, audio=audio)

            if response.results:
                return response.results[0].alternatives[0].transcript
            return None

        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return None

    def detect_language_from_speech(self, audio_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to recognize speech in both languages and determine which one worked

        Returns:
            Tuple of (detected_language, recognized_text)
        """
        # Try English first
        try:
            text_en = self.recognize_speech(audio_bytes, 'en-US')
            if text_en and len(text_en.strip()) > 0:
                logger.info(f"Recognized English: {text_en}")
                return ('english', text_en)
        except Exception as e:
            logger.error(f"English recognition error: {e}")

        # Try Sinhala
        try:
            text_si = self.recognize_speech(audio_bytes, 'si-LK')
            if text_si and len(text_si.strip()) > 0:
                logger.info(f"Recognized Sinhala: {text_si}")
                return ('sinhala', text_si)
        except Exception as e:
            logger.error(f"Sinhala recognition error: {e}")

        return (None, None)

    def create_webrtc_recorder(self, key: str = "voice_input") -> Optional[Tuple[str, str]]:
        """
        Create a webrtc recorder widget and process audio

        Args:
            key: Unique key for the webrtc component

        Returns:
            Tuple of (detected_language, recognized_text) or None
        """
        try:
            webrtc_ctx = webrtc_streamer(
                key=key,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]},
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
            )

            if webrtc_ctx.state.playing:
                st.info("🎤 Recording... Speak now")

                # Process audio when available
                if webrtc_ctx.audio_processor and not webrtc_ctx.audio_processor.frames_queue.empty():
                    audio_frame = webrtc_ctx.audio_processor.frames_queue.get()
                    if audio_frame:
                        audio_bytes = audio_frame.to_ndarray().tobytes()
                        result = self.detect_language_from_speech(audio_bytes)
                        return result

            return None

        except Exception as e:
            logger.error(f"Error in webrtc recorder: {e}")
            st.error(f"Error accessing microphone: {str(e)}")
            return None

    def text_to_speech(self, text: str, language: str = 'english') -> Optional[str]:
        """
        Convert text to speech and return audio file path

        Args:
            text: Text to convert to speech
            language: Language of the text ('english' or 'sinhala')

        Returns:
            Path to temporary audio file or None if failed
        """
        try:
            lang_code = self.tts_codes.get(language, 'en')

            tts = gTTS(text=text, lang=lang_code, slow=False)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)

            logger.info(f"Generated speech audio: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None

    def cleanup_audio_file(self, filepath: str):
        """Delete temporary audio file"""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up audio file: {filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up audio file: {str(e)}")
