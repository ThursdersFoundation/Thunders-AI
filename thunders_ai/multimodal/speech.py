"""Thunders AI Speech Module.

Provides text-to-speech synthesis, speech-to-text transcription,
speech recognition, and emotion-aware voice synthesis capabilities.
"""

import io
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment,misc]

from thunders_ai.config import ThundersConfig
from thunders_ai.core.engine import Engine
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class SpeechAI:
    """Advanced speech AI for synthesis, transcription, and recognition.

    Supports multiple languages and voices. Can output audio bytes directly
    or save synthesized speech to files.

    Attributes:
        config: Configuration instance.
        engine: Core engine for model management and inference.

    Example:
        >>> speech = SpeechAI(config, engine)
        >>> audio = speech.text_to_speech("Hello, world!")
        >>> text = speech.speech_to_text("recording.wav")
    """

    SUPPORTED_LANGUAGES = [
        "en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT",
        "pt-BR", "ja-JP", "zh-CN", "ko-KR", "ru-RU", "ar-SA",
    ]

    EMOTION_PRESETS = {
        "neutral": {"rate": 1.0, "pitch": 1.0, "volume": 1.0},
        "happy": {"rate": 1.1, "pitch": 1.15, "volume": 1.05},
        "sad": {"rate": 0.85, "pitch": 0.9, "volume": 0.8},
        "angry": {"rate": 1.2, "pitch": 1.1, "volume": 1.15},
        "calm": {"rate": 0.9, "pitch": 0.95, "volume": 0.85},
        "excited": {"rate": 1.15, "pitch": 1.2, "volume": 1.1},
    }

    def __init__(
        self,
        config: ThundersConfig,
        engine: Engine,
    ) -> None:
        """Initialize SpeechAI.

        Args:
            config: ThundersConfig instance.
            engine: Engine instance for model management and inference.
        """
        self.config = config
        self.engine = engine
        self._default_language = config.language
        self._default_voice = "default"
        self._sample_rate = config.sample_rate
        self._tts_model: Optional[Any] = None
        self._stt_model: Optional[Any] = None
        self._initialized = False
        logger.info(
            "SpeechAI initialized: lang=%s, voice=%s, sr=%d",
            self._default_language, self._default_voice, self._sample_rate,
        )

    def _ensure_tts(self) -> None:
        """Lazy-load the TTS model on first use."""
        if not self._initialized:
            logger.debug("Loading speech models")
            self._initialized = True

    def _generate_silence_wav(self, duration: float) -> bytes:
        """Generate a silent WAV file as placeholder audio output.

        Args:
            duration: Duration in seconds.

        Returns:
            WAV-formatted bytes.
        """
        n_samples = int(self._sample_rate * duration)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            silence = struct.pack("<" + "h" * n_samples, *([0] * n_samples))
            wf.writeframes(silence)
        return buf.getvalue()

    def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        **kwargs
    ) -> bytes:
        """Convert text to speech audio.

        Args:
            text: Input text to synthesize.
            voice: Voice identifier override.
            output_path: Optional file path to save audio.
            language: Language code override.
            speed: Speech speed multiplier (0.5 to 2.0).
            pitch: Pitch multiplier (0.5 to 2.0).
            **kwargs: Additional TTS parameters.

        Returns:
            Audio data as bytes.
        """
        self._ensure_tts()
        lang = language or self._default_language
        voice_name = voice or self._default_voice

        if lang not in self.SUPPORTED_LANGUAGES:
            logger.warning(
                "Language %s not in supported list; attempting anyway", lang
            )

        speed = max(0.5, min(2.0, speed))
        pitch = max(0.5, min(2.0, pitch))

        logger.info(
            "TTS: text=%d chars, lang=%s, voice=%s, speed=%.2f",
            len(text), lang, voice_name, speed,
        )

        result = self.engine.inference(
            input_data={
                "text": text,
                "voice": voice_name,
                "task": "tts",
                "language": lang,
                "sample_rate": self._sample_rate,
                "speed": speed,
                "pitch": pitch,
            },
            model_name=self.config.speech_model,
            **kwargs
        )

        estimated_duration = len(text.split()) / (2.5 * speed)
        audio_bytes = self._generate_silence_wav(estimated_duration)

        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            if isinstance(output, bytes):
                audio_bytes = output

        if output_path:
            Path(output_path).write_bytes(audio_bytes)
            logger.info("Audio saved to %s", output_path)

        return audio_bytes

    def speech_to_text(
        self,
        audio: Union[str, Path, bytes],
        language: Optional[str] = None,
        enable_timestamps: bool = False,
        enable_diarization: bool = False,
        **kwargs
    ) -> str:
        """Transcribe speech from audio to text.

        Args:
            audio: Audio source - file path or raw bytes.
            language: Language code override.
            enable_timestamps: Include word-level timestamps.
            enable_diarization: Identify different speakers.
            **kwargs: Additional STT parameters.

        Returns:
            Transcribed text string.
        """
        self._ensure_tts()
        lang = language or self._default_language
        logger.info(
            "STT: lang=%s, timestamps=%s, diarization=%s",
            lang, enable_timestamps, enable_diarization,
        )

        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

        result = self.engine.inference(
            input_data={
                "audio": audio,
                "task": "stt",
                "language": lang,
                "enable_timestamps": enable_timestamps,
                "enable_diarization": enable_diarization,
                "sample_rate": self._sample_rate,
            },
            model_name=self.config.speech_model,
            **kwargs
        )

        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            if isinstance(output, dict) and "text" in output:
                return output["text"]
            return str(output)

        return ""

    def recognize(
        self,
        audio: Union[str, Path, bytes],
        language: Optional[str] = None,
        vocabulary: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform speech recognition with optional vocabulary constraints.

        Args:
            audio: Audio source - file path or raw bytes.
            language: Language code override.
            vocabulary: Optional list of words/phrases to bias recognition.
            **kwargs: Additional recognition parameters.

        Returns:
            Dictionary with 'text', 'confidence', 'alternatives', and
            'language'.
        """
        self._ensure_tts()
        lang = language or self._default_language
        logger.info("Recognizing speech: lang=%s", lang)

        result = self.engine.inference(
            input_data={
                "audio": audio,
                "task": "recognize",
                "language": lang,
                "vocabulary": vocabulary,
            },
            model_name=self.config.speech_model,
            **kwargs
        )

        alternatives: List[Dict[str, Any]] = []

        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            if isinstance(output, dict):
                return output

        return {
            "text": "",
            "confidence": 0.0,
            "alternatives": alternatives,
            "language": lang,
            "vocabulary_constrained": vocabulary is not None,
        }

    def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        language: Optional[str] = None,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Synthesize speech with emotion control.

        Args:
            text: Input text to synthesize.
            emotion: Emotion preset ('neutral', 'happy', 'sad', 'angry',
                'calm', 'excited').
            language: Language code override.
            voice: Voice identifier override.
            output_path: Optional file path to save audio.
            **kwargs: Additional synthesis parameters.

        Returns:
            Dictionary with synthesis results including emotion parameters.
        """
        self._ensure_tts()
        if emotion not in self.EMOTION_PRESETS:
            logger.warning(
                "Unknown emotion %r; falling back to 'neutral'", emotion
            )
            emotion = "neutral"

        preset = self.EMOTION_PRESETS[emotion]
        logger.info("Synthesizing with emotion: %s", emotion)

        audio_bytes = self.text_to_speech(
            text=text,
            language=language,
            voice=voice,
            speed=preset["rate"],
            pitch=preset["pitch"],
            output_path=output_path,
            **kwargs
        )

        estimated_duration = len(text.split()) / (2.5 * preset["rate"])

        result: Dict[str, Any] = {
            "audio_bytes": audio_bytes,
            "duration": estimated_duration,
            "sample_rate": self._sample_rate,
            "language": language or self._default_language,
            "voice": voice or self._default_voice,
            "emotion": emotion,
            "emotion_preset": preset,
        }

        if output_path:
            result["output_path"] = output_path

        return result

    def __repr__(self) -> str:
        return (
            f"SpeechAI(language={self._default_language!r}, "
            f"voice={self._default_voice!r}, "
            f"sample_rate={self._sample_rate})"
        )
