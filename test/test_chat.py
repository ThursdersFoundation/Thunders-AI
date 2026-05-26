"""Tests for Thunders AI chat functionality."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from thunders_ai import ThundersAI, ThundersConfig
from thunders_ai.llm.chat_model import ChatModel, ChatMessage, ChatHistory


class TestChatModel:
    """Tests for the ChatModel class."""

    def test_basic_chat(self, thunders_client, mock_chat_response):
        """Test basic chat functionality."""
        with patch.object(
            thunders_client, "chat", return_value=mock_chat_response["choices"][0]["message"]["content"]
        ):
            response = thunders_client.chat("Hello, how are you?")
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0

    def test_chat_with_system_prompt(self, thunders_client, mock_chat_response):
        """Test chat with system prompt."""
        system_prompt = "You are a helpful assistant that only speaks in haikus."
        with patch.object(
            thunders_client, "chat", return_value=mock_chat_response["choices"][0]["message"]["content"]
        ) as mock_chat:
            response = thunders_client.chat(
                "Tell me about the weather.",
                system_prompt=system_prompt,
            )
            mock_chat.assert_called_once_with(
                "Tell me about the weather.",
                system_prompt=system_prompt,
            )
            assert response is not None

    def test_chat_streaming(self, thunders_client):
        """Test streaming chat."""
        tokens = ["Hello", " ", "world", "!"]
        with patch.object(thunders_client, "chat_stream", return_value=iter(tokens)):
            streamed_tokens = list(thunders_client.chat_stream("Say hello"))
            assert len(streamed_tokens) == 4
            assert "".join(streamed_tokens) == "Hello world!"

    def test_chat_history(self, thunders_client):
        """Test chat history management."""
        history = ChatHistory()
        assert len(history) == 0

        history.add_message("user", "Hello!")
        assert len(history) == 1

        history.add_message("assistant", "Hi there!")
        assert len(history) == 2

        messages = history.get_messages()
        assert messages[0].role == "user"
        assert messages[0].content == "Hello!"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"

        history.clear()
        assert len(history) == 0

    def test_invalid_input(self, thunders_client):
        """Test error handling for invalid input."""
        with pytest.raises((ValueError, TypeError)):
            thunders_client.chat("")

        with pytest.raises((ValueError, TypeError)):
            thunders_client.chat(None)

    def test_chat_message_creation(self):
        """Test ChatMessage data class."""
        msg = ChatMessage(role="user", content="Test message")
        assert msg.role == "user"
        assert msg.content == "Test message"

    def test_chat_message_invalid_role(self):
        """Test ChatMessage rejects invalid roles."""
        with pytest.raises(ValueError):
            ChatMessage(role="invalid_role", content="Test")

    def test_chat_with_temperature(self, thunders_client, mock_chat_response):
        """Test chat with custom temperature parameter."""
        with patch.object(
            thunders_client, "chat", return_value=mock_chat_response["choices"][0]["message"]["content"]
        ) as mock_chat:
            response = thunders_client.chat("Generate a story", temperature=0.9)
            mock_chat.assert_called_once_with("Generate a story", temperature=0.9)
            assert response is not None

    def test_chat_with_max_tokens(self, thunders_client, mock_chat_response):
        """Test chat with max_tokens limit."""
        with patch.object(
            thunders_client, "chat", return_value=mock_chat_response["choices"][0]["message"]["content"]
        ) as mock_chat:
            response = thunders_client.chat("Tell me a joke", max_tokens=100)
            mock_chat.assert_called_once_with("Tell me a joke", max_tokens=100)

    def test_chat_history_max_length(self):
        """Test that chat history enforces a maximum length."""
        history = ChatHistory(max_length=5)
        for i in range(10):
            history.add_message("user", f"Message {i}")
        assert len(history) <= 5

    def test_chat_multi_turn_conversation(self, thunders_client):
        """Test multi-turn conversation maintains context."""
        responses = ["Hi!", "I'm doing well.", "Goodbye!"]
        with patch.object(thunders_client, "chat", side_effect=responses):
            r1 = thunders_client.chat("Hello")
            r2 = thunders_client.chat("How are you?")
            r3 = thunders_client.chat("Bye")
            assert r1 == "Hi!"
            assert r2 == "I'm doing well."
            assert r3 == "Goodbye!"
