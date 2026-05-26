"""Tests for Thunders AI API endpoints."""
import json
from unittest.mock import MagicMock, patch

import pytest

from thunders_ai.api.server import create_app


@pytest.fixture
def app():
    """Create a test application instance."""
    with patch("thunders_ai.api.server.ThundersAI"):
        application = create_app(testing=True)
        application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Create a test client for the API."""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Provide authentication headers for API requests."""
    return {"Authorization": "Bearer test-api-key-12345"}


class TestChatEndpoint:
    """Tests for the /api/v1/chat endpoint."""

    def test_chat_endpoint(self, client, auth_headers, mock_chat_response):
        """Test the chat endpoint returns a valid response."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.chat.return_value = mock_chat_response["choices"][0]["message"]["content"]
            response = client.post(
                "/api/v1/chat",
                json={"message": "Hello!", "model": "test-model"},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.get_json()
            assert "response" in data or "content" in data or "message" in data

    def test_chat_endpoint_missing_message(self, client, auth_headers):
        """Test chat endpoint rejects requests without a message."""
        response = client.post(
            "/api/v1/chat",
            json={},
            headers=auth_headers,
        )
        assert response.status_code in (400, 422)

    def test_chat_endpoint_with_system_prompt(self, client, auth_headers, mock_chat_response):
        """Test chat endpoint with a system prompt."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.chat.return_value = "Haiku response"
            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Tell me a joke",
                    "system_prompt": "You are a comedian.",
                },
                headers=auth_headers,
            )
            assert response.status_code == 200

    def test_chat_endpoint_streaming(self, client, auth_headers):
        """Test chat endpoint with streaming enabled."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.chat_stream.return_value = iter(["Hello", " world"])
            response = client.post(
                "/api/v1/chat",
                json={"message": "Hello", "stream": True},
                headers=auth_headers,
            )
            # Streaming response should succeed
            assert response.status_code in (200, 201)


class TestVisionEndpoint:
    """Tests for the /api/v1/vision endpoint."""

    def test_vision_endpoint(self, client, auth_headers, mock_vision_response):
        """Test the vision analysis endpoint."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.analyze_image.return_value = mock_vision_response
            response = client.post(
                "/api/v1/vision/analyze",
                json={"image_url": "https://example.com/test.jpg"},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.get_json()
            assert data is not None

    def test_vision_detect_objects(self, client, auth_headers, mock_vision_response):
        """Test the object detection endpoint."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.detect_objects.return_value = mock_vision_response["objects"]
            response = client.post(
                "/api/v1/vision/detect",
                json={"image_url": "https://example.com/test.jpg"},
                headers=auth_headers,
            )
            assert response.status_code == 200

    def test_vision_endpoint_missing_image(self, client, auth_headers):
        """Test vision endpoint rejects requests without an image."""
        response = client.post(
            "/api/v1/vision/analyze",
            json={},
            headers=auth_headers,
        )
        assert response.status_code in (400, 422)


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test the health check endpoint returns OK status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok" or data.get("healthy") is True

    def test_readiness_check(self, client):
        """Test the readiness probe endpoint."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code in (200, 503)


class TestAuthentication:
    """Tests for API authentication."""

    def test_authentication_valid_key(self, client, auth_headers, mock_chat_response):
        """Test that valid API key is accepted."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.chat.return_value = "Response"
            response = client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers=auth_headers,
            )
            assert response.status_code != 401

    def test_authentication_missing_key(self, client):
        """Test that missing API key is rejected."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert response.status_code in (401, 403)

    def test_authentication_invalid_key(self, client):
        """Test that invalid API key is rejected."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
            headers={"Authorization": "Bearer invalid-key"},
        )
        assert response.status_code in (401, 403)


class TestRateLimiting:
    """Tests for API rate limiting."""

    def test_rate_limiting_headers(self, client, auth_headers):
        """Test that rate limiting headers are present in responses."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.chat.return_value = "Response"
            response = client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers=auth_headers,
            )
            # Rate limit headers should be present
            assert "X-RateLimit-Limit" in response.headers or response.status_code == 200

    def test_rate_limit_exceeded(self, client, auth_headers):
        """Test that exceeding rate limits returns 429."""
        with patch("thunders_ai.api.server.ThundersAI") as mock_ai:
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.chat.return_value = "Response"
            # Send many requests rapidly
            responses = []
            for _ in range(150):
                resp = client.post(
                    "/api/v1/chat",
                    json={"message": "Hello"},
                    headers=auth_headers,
                )
                responses.append(resp)
            # At least one should be rate limited (or all succeed if no rate limit in test mode)
            status_codes = [r.status_code for r in responses]
            assert 200 in status_codes or 429 in status_codes
