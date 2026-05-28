# Quick Start Guide

Get up and running with Thunders AI in under 5 minutes. This guide walks you through three common use cases: basic chat, vision analysis, and running the API server.

## Prerequisites

Make sure you have Thunders AI installed:

```bash
pip install thunders-ai
```

## Example 1: Basic Chat

The simplest way to use Thunders AI is through the chat interface. This example shows how to create a conversational AI that maintains context across messages.

```python
from thunders_ai import ThundersAI, ThundersConfig

# Initialize with default settings
ai = ThundersAI()

# Single message
response = ai.chat("What is machine learning?")
print(response)

# Multi-turn conversation with system prompt
ai = ThundersAI(config=ThundersConfig(
    system_prompt="You are a friendly science tutor who explains concepts simply."
))

# The AI remembers the conversation context
response1 = ai.chat("What is gravity?")
print(f"Q1: {response1}")

response2 = ai.chat("How does it relate to the solar system?")
print(f"Q2: {response2}")

# Streaming responses for real-time output
print("Streaming: ", end="")
for token in ai.chat_stream("Tell me a short story"):
    print(token, end="", flush=True)
print()

# Custom parameters for creative or precise outputs
creative = ai.chat("Write a poem about AI", temperature=0.9, max_tokens=200)
precise = ai.chat("What is 2+2?", temperature=0.0)
```

### Key Concepts

- **ThundersAI** is the main client class. Create one instance and reuse it.
- **ThundersConfig** lets you customize model selection, device, system prompts, and more.
- **chat()** sends a message and returns the full response as a string.
- **chat_stream()** returns an iterator of tokens for real-time display.
- **temperature** controls randomness (0 = deterministic, 1 = creative).
- **max_tokens** limits the length of the response.

## Example 2: Vision Analysis

Thunders AI can analyze images for object detection, classification, captioning, and more. You can pass a file path, URL, or raw image bytes.

```python
from thunders_ai import ThundersAI

ai = ThundersAI()

# Analyze an image from a local file
result = ai.analyze_image("photos/landscape.jpg")
print("Caption:", result["captions"][0])
print("Labels:", result["labels"])

# Detect specific objects with confidence threshold
objects = ai.detect_objects(
    "photos/street.jpg",
    confidence_threshold=0.7
)
for obj in objects:
    print(f"Found {obj['label']} with {obj['confidence']:.1%} confidence")

# Analyze an image from a URL
result = ai.analyze_image("https://example.com/photo.jpg")

# Analyze from base64-encoded image data
import base64
with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
result = ai.analyze_image(f"data:image/jpeg;base64,{image_b64}")

# Batch processing for multiple images
images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
results = ai.analyze_images(images)
for img, result in zip(images, results):
    print(f"{img}: {result['captions'][0]}")
```

### Vision Features

| Feature | Method | Description |
|---------|--------|-------------|
| Image Analysis | `analyze_image()` | Captions, labels, and scene understanding |
| Object Detection | `detect_objects()` | Bounding boxes with class labels and confidence |
| Face Detection | `detect_faces()` | Face locations with landmark points |
| Image Classification | `classify_image()` | Top-K class predictions |

## Example 3: API Server

Run Thunders AI as a REST API server for integration with web apps, microservices, or any HTTP client.

```python
from thunders_ai.api import serve

# Start the API server on port 8000
serve(host="0.0.0.0", port=8000)
```

Or from the command line:

```bash
thunders-ai serve --host 0.0.0.0 --port 8000
```

### Making API Requests

```bash
# Chat endpoint
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Thunders AI!"}'

# Vision endpoint
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg"}'

# Health check
curl http://localhost:8000/api/v1/health
```

### Python API Client

```python
import requests

API_URL = "http://localhost:8000/api/v1"
HEADERS = {"Authorization": "Bearer your-api-key"}

# Chat
response = requests.post(
    f"{API_URL}/chat",
    json={"message": "Hello!"},
    headers=HEADERS,
)
print(response.json())

# Vision
response = requests.post(
    f"{API_URL}/vision/analyze",
    json={"image_url": "https://example.com/photo.jpg"},
    headers=HEADERS,
)
print(response.json())
```

## Next Steps

- Read the **[API Reference](api_reference.md)** for complete method documentation
- Follow the **[Chatbot Tutorial](tutorials/chatbot.md)** to build a full-featured chatbot
- Explore **[Vision AI](tutorials/vision_ai.md)** for advanced computer vision workflows
- Check **[Deployment](installation.md#docker-setup)** for production deployment options
