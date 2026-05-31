"""REST API usage example using Thunders AI.

Demonstrates how to interact with the Thunders AI REST API using
the requests library for chat completions, vision analysis, and
speech processing endpoints.
"""

import requests

# Thunders AI API base URL
BASE_URL = "http://localhost:8000/api/v1"

# API key for authentication
API_KEY = "thunders_your_api_key_here"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def check_health() -> None:
    """Check the API server health status."""
    print("=== Health Check ===")
    response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health")
    data = response.json()
    print(f"Status: {data.get('status')}")
    print(f"Version: {data.get('version')}")
    print(f"Service: {data.get('service')}")
    print()


def list_models() -> None:
    """List all available models."""
    print("=== List Models ===")
    response = requests.get(f"{BASE_URL}/models", headers=HEADERS)
    models = response.json()
    for model in models:
        print(f"  {model['id']:<20} type={model['type']:<10} max_tokens={model['max_tokens']}")
    print()


def chat_completion() -> None:
    """Send a chat completion request."""
    print("=== Chat Completion ===")
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the benefits of AI in healthcare?"},
        ],
        "model": "thunders-7b",
        "temperature": 0.7,
        "max_tokens": 512,
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS)
    data = response.json()
    print(f"Model: {data.get('model')}")
    print(f"Response: {data['choices'][0]['message']['content']}")
    print(f"Tokens used: {data['usage']['total_tokens']}")
    print()


def vision_analysis() -> None:
    """Analyze an image using the vision endpoint."""
    print("=== Vision Analysis ===")
    payload = {
        "image_url": "https://example.com/images/medical_scan.jpg",
        "prompt": "Describe any anomalies in this medical image.",
        "model": "thunders-vision",
        "detail": "high",
    }
    response = requests.post(f"{BASE_URL}/vision/analyze", json=payload, headers=HEADERS)
    data = response.json()
    print(f"Model: {data.get('model')}")
    print(f"Description: {data.get('description')}")
    print("Labels:")
    for label in data.get("labels", []):
        print(f"  {label.get('label')}: {label.get('confidence'):.1%}")
    print()


def text_to_speech() -> None:
    """Convert text to speech using the TTS endpoint."""
    print("=== Text-to-Speech ===")
    payload = {
        "text": "Welcome to Thunders AI, your intelligent assistant.",
        "voice": "alloy",
        "model": "thunders-tts",
        "speed": 1.0,
        "format": "mp3",
    }
    response = requests.post(f"{BASE_URL}/speech/tts", json=payload, headers=HEADERS)
    data = response.json()
    print(f"Audio URL: {data.get('audio_url')}")
    print(f"Duration: {data.get('duration_seconds'):.1f}s")
    print(f"Format: {data.get('format')}")
    print()


def speech_to_text() -> None:
    """Convert speech to text using the STT endpoint."""
    print("=== Speech-to-Text ===")
    payload = {
        "audio_url": "https://example.com/audio/recording.wav",
        "model": "thunders-stt",
        "language": "en",
    }
    response = requests.post(f"{BASE_URL}/speech/stt", json=payload, headers=HEADERS)
    data = response.json()
    print(f"Transcribed text: {data.get('text')}")
    print(f"Language: {data.get('language')}")
    print(f"Confidence: {data.get('confidence'):.1%}")
    print(f"Duration: {data.get('duration_seconds'):.1f}s")
    print()


def robotics_navigation() -> None:
    """Plan a navigation path using the robotics endpoint."""
    print("=== Robotics Navigation ===")
    payload = {
        "target_x": 10.0,
        "target_y": 5.0,
        "target_z": 0.0,
        "speed": 1.5,
        "avoid_obstacles": True,
        "planner": "a_star",
    }
    response = requests.post(f"{BASE_URL}/robotics/navigate", json=payload, headers=HEADERS)
    data = response.json()
    print(f"Status: {data.get('status')}")
    print(f"Distance: {data.get('distance_meters'):.2f}m")
    print(f"Estimated time: {data.get('estimated_time_seconds'):.1f}s")
    print("Path waypoints:")
    for wp in data.get("path", []):
        print(f"  ({wp['x']:.1f}, {wp['y']:.1f}, {wp['z']:.1f})")
    print()


def main() -> None:
    """Run all REST API examples."""
    check_health()
    list_models()
    chat_completion()
    vision_analysis()
    text_to_speech()
    speech_to_text()
    robotics_navigation()
    print("All REST API examples completed!")


if __name__ == "__main__":
    main()
