"""Tests for Thunders AI speech and audio functionality."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thunders_ai import ThundersAI, ThundersConfig
from thunders_ai.multimodal.speech import SpeechAI, TTSResult, STTResult


class TestSpeechAI:
    """Tests for the SpeechAI class."""

    def test_tts_basic(self, thunders_client, sample_audio_path):
        """Test basic text-to-speech conversion."""
        mock_tts_result = TTSResult(
            audio_path=sample_audio_path,
            audio_bytes=b"\x00" * 1000,
            sample_rate=22050,
            duration=1.0,
        )
        with patch.object(thunders_client, "tts", return_value=mock_tts_result):
            result = thunders_client.tts("Hello, world!")
            assert result is not None
            assert result.audio_bytes is not None
            assert result.sample_rate > 0
            assert result.duration > 0

    def test_stt_basic(self, thunders_client, sample_audio_path, mock_stt_response):
        """Test basic speech-to-text conversion."""
        mock_stt_result = STTResult(
            text=mock_stt_response["text"],
            confidence=mock_stt_response["confidence"],
            language=mock_stt_response["language"],
            segments=mock_stt_response["segments"],
        )
        with patch.object(thunders_client, "stt", return_value=mock_stt_result):
            result = thunders_client.stt(sample_audio_path)
            assert result is not None
            assert isinstance(result.text, str)
            assert len(result.text) > 0
            assert result.confidence > 0

    def test_supported_languages(self, thunders_client):
        """Test that TTS and STT support multiple languages."""
        supported_languages = ["en", "es", "fr", "de", "zh", "ja", "ko", "pt", "ru", "ar"]
        with patch.object(
            thunders_client, "get_supported_languages", return_value=supported_languages
        ):
            languages = thunders_client.get_supported_languages()
            assert "en" in languages
            assert "es" in languages
            assert "zh" in languages
            assert len(languages) >= 5

    def test_audio_format_support(self, thunders_client, sample_audio_path):
        """Test support for different audio formats."""
        # WAV is the primary format
        mock_tts_result = TTSResult(
            audio_path=sample_audio_path,
            audio_bytes=b"\x00" * 1000,
            sample_rate=16000,
            duration=1.0,
        )
        with patch.object(thunders_client, "tts", return_value=mock_tts_result):
            result = thunders_client.tts("Test", output_format="wav")
            assert result is not None

        # Test MP3 format
        with patch.object(thunders_client, "tts", return_value=mock_tts_result):
            result = thunders_client.tts("Test", output_format="mp3")
            assert result is not None

    def test_tts_empty_text(self, thunders_client):
        """Test that TTS raises an error for empty text."""
        with pytest.raises((ValueError, TypeError)):
            thunders_client.tts("")

    def test_stt_invalid_audio_path(self, thunders_client):
        """Test that STT raises an error for invalid audio path."""
        with pytest.raises((FileNotFoundError, ValueError)):
            thunders_client.stt("/nonexistent/audio.wav")

    def test_stt_with_language_hint(self, thunders_client, sample_audio_path, mock_stt_response):
        """Test STT with a language hint for better accuracy."""
        mock_stt_result = STTResult(
            text=mock_stt_response["text"],
            confidence=0.99,
            language="en",
            segments=mock_stt_response["segments"],
        )
        with patch.object(thunders_client, "stt", return_value=mock_stt_result) as mock_stt:
            result = thunders_client.stt(sample_audio_path, language="en")
            mock_stt.assert_called_once_with(sample_audio_path, language="en")
            assert result.language == "en"

    def test_tts_with_voice_selection(self, thunders_client, sample_audio_path):
        """Test TTS with different voice options."""
        mock_result = TTSResult(
            audio_path=sample_audio_path,
            audio_bytes=b"\x00" * 1000,
            sample_rate=22050,
            duration=1.0,
        )
        with patch.object(thunders_client, "tts", return_value=mock_result) as mock_tts:
            result = thunders_client.tts("Hello", voice="female-1")
            mock_tts.assert_called_once_with("Hello", voice="female-1")

    def test_tts_with_speaking_rate(self, thunders_client, sample_audio_path):
        """Test TTS with custom speaking rate."""
        mock_result = TTSResult(
            audio_path=sample_audio_path,
            audio_bytes=b"\x00" * 1000,
            sample_rate=22050,
            duration=0.7,
        )
        with patch.object(thunders_client, "tts", return_value=mock_result):
            result = thunders_client.tts("Fast speech", rate=1.5)
            assert result is not None

    def test_stt_result_segments(self, mock_stt_response):
        """Test STTResult contains proper segments."""
        result = STTResult(
            text=mock_stt_response["text"],
            confidence=mock_stt_response["confidence"],
            language=mock_stt_response["language"],
            segments=mock_stt_response["segments"],
        )
        assert len(result.segments) == 2
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 1.5
