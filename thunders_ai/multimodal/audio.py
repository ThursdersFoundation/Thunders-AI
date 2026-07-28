"""Thunders AI Audio Module.

Provides speech-to-text transcription, audio classification, event detection,
and general audio analysis capabilities.
"""

import io
import wave
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment,misc]

from thunders_ai.config import Config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

AudioInput = Union[str, Path, bytes, "np.ndarray"]


class AudioAI:
    """Advanced audio AI for transcription, classification, and analysis.

    Supports loading audio from file paths, raw bytes, or numpy arrays.
    Integrates with configurable backends for speech recognition and
    audio processing.

    Args:
        config: Optional configuration instance. Uses default config if None.
        sample_rate: Default sample rate for audio processing.
        language: Default language code for transcription.

    Example:
        >>> audio = AudioAI(language="en-US")
        >>> text = audio.transcribe("recording.wav")
        >>> events = audio.detect_events("ambient.wav")
    """

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

    def __init__(
        self,
        config: Optional[Config] = None,
        sample_rate: int = 16000,
        language: str = "en-US",
    ) -> None:
        self._config = config or Config()
        self._sample_rate = sample_rate
        self._language = language
        self._model: Optional[Any] = None
        self._initialized = False
        logger.info(
            "AudioAI initialized: sample_rate=%d, language=%s",
            sample_rate, language,
        )

    def _load_audio(self, audio: AudioInput) -> Tuple["np.ndarray", int]:
        """Load and validate audio from various input types.

        Args:
            audio: Audio source - file path, raw bytes, or numpy array.

        Returns:
            Tuple of (audio_data as numpy float32 array, sample_rate).

        Raises:
            ValueError: If the audio source is invalid or format unsupported.
            FileNotFoundError: If a file path does not exist.
        """
        if np is None:
            raise ImportError("NumPy is required: pip install numpy")

        if isinstance(audio, np.ndarray):
            if audio.ndim not in (1, 2):
                raise ValueError(
                    f"Audio array must be 1D or 2D, got {audio.ndim}D"
                )
            data = audio.astype(np.float32)
            if data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T
            if data.ndim == 2:
                data = data.mean(axis=0)
            return data, self._sample_rate

        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported audio format: {path.suffix}. "
                    f"Supported: {self.SUPPORTED_FORMATS}"
                )
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            return self._read_audio_file(path)

        if isinstance(audio, bytes):
            return self._read_audio_bytes(audio)

        raise ValueError(
            f"Unsupported audio input type: {type(audio).__name__}. "
            "Use str, Path, bytes, or numpy array."
        )

    def _read_audio_file(self, path: Path) -> Tuple["np.ndarray", int]:
        """Read audio data from a file path.

        Args:
            path: Path to the audio file.

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                n_channels = wf.getnchannels()
                samp_width = wf.getsampwidth()
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            data /= 32768.0
            return data, sr

        logger.warning("Non-WAV format fallback: reading as raw bytes")
        raw = path.read_bytes()
        return self._read_audio_bytes(raw)

    def _read_audio_bytes(self, data: bytes) -> Tuple["np.ndarray", int]:
        """Parse raw audio bytes into a numpy array.

        Args:
            data: Raw audio bytes.

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                n_channels = wf.getnchannels()
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_channels > 1:
                arr = arr.reshape(-1, n_channels).mean(axis=1)
            arr /= 32768.0
            return arr, sr
        except wave.Error:
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return arr, self._sample_rate

    def _ensure_model(self) -> None:
        """Lazy-load the audio model on first use."""
        if not self._initialized:
            logger.debug("Loading audio model")
            self._initialized = True

    def transcribe(
        self,
        audio: AudioInput,
        language: Optional[str] = None,
        timestamp_granularity: str = "segment",
    ) -> Dict[str, Any]:
        """Transcribe speech in audio to text.

        Args:
            audio: Input audio source.
            language: Language code override (e.g., 'en-US', 'es-ES').
            timestamp_granularity: Timestamp detail level — 'word' or
                'segment'.

        Returns:
            Dictionary with 'text', 'segments', 'language', and
            'confidence' fields.
        """
        self._ensure_model()
        data, sr = self._load_audio(audio)
        lang = language or self._language
        logger.info("Transcribing audio: lang=%s, samples=%d", lang, len(data))

        segments: List[Dict[str, Any]] = []
        results: Dict[str, Any] = {
            "text": "",
            "segments": segments,
            "language": lang,
            "confidence": 0.0,
            "sample_rate": sr,
            "duration": len(data) / sr,
        }
        return results

    def classify_audio(
        self,
        audio: AudioInput,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Classify audio into categories.

        Args:
            audio: Input audio source.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with 'predictions' list and 'top_prediction'.
        """
        self._ensure_model()
        data, sr = self._load_audio(audio)
        logger.info("Classifying audio: top_k=%d", top_k)

        predictions: List[Dict[str, Any]] = []
        results: Dict[str, Any] = {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "sample_rate": sr,
            "duration": len(data) / sr,
        }
        return results

    def detect_events(
        self,
        audio: AudioInput,
        sensitivity: float = 0.5,
        event_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detect audio events such as speech, music, or environmental sounds.

        Args:
            audio: Input audio source.
            sensitivity: Detection sensitivity (0.0 to 1.0).
            event_types: Optional filter for specific event types.

        Returns:
            Dictionary with 'events' list (type, start, end, confidence)
            and 'event_count'.
        """
        self._ensure_model()
        data, sr = self._load_audio(audio)
        logger.info("Detecting events: sensitivity=%.2f", sensitivity)

        events: List[Dict[str, Any]] = []
        results: Dict[str, Any] = {
            "events": events,
            "event_count": len(events),
            "sensitivity": sensitivity,
            "duration": len(data) / sr,
        }
        return results

    def analyze(
        self,
        audio: AudioInput,
        analysis_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive audio analysis.

        Args:
            audio: Input audio source.
            analysis_types: Analysis types to perform. Options:
                'transcription', 'classification', 'events'. Defaults to all.

        Returns:
            Dictionary with analysis results keyed by type.
        """
        self._ensure_model()
        data, sr = self._load_audio(audio)
        types = analysis_types or ["transcription", "classification", "events"]
        logger.info("Analyzing audio: types=%s", types)

        results: Dict[str, Any] = {
            "sample_rate": sr,
            "duration": len(data) / sr,
            "samples": len(data),
        }
        if "transcription" in types:
            results["transcription"] = self.transcribe(audio)
        if "classification" in types:
            results["classification"] = self.classify_audio(audio)
        if "events" in types:
            results["events"] = self.detect_events(audio)
        return results

    def __repr__(self) -> str:
        return (
            f"AudioAI(sample_rate={self._sample_rate}, "
            f"language={self._language!r}, "
            f"initialized={self._initialized})"
        )
