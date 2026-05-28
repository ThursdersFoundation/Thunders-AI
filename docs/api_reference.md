# API Reference

Complete reference for the Thunders AI Python SDK. All classes, methods, and configuration options are documented below.

---

## Core Classes

### `ThundersAI`

The main client class for interacting with Thunders AI. Create an instance and use it to access all features.

```python
from thunders_ai import ThundersAI, ThundersConfig

ai = ThundersAI(config=ThundersConfig(model="thunders-7b"))
```

#### Constructor

```python
ThundersAI(config: Optional[ThundersConfig] = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ThundersConfig` | `None` | Configuration object. Uses defaults if not provided. |

#### Methods

##### `chat(message, *, system_prompt=None, temperature=None, max_tokens=None, stream=False)`

Send a chat message and receive a response.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | required | The user message |
| `system_prompt` | `Optional[str]` | `None` | Override the system prompt for this request |
| `temperature` | `Optional[float]` | `None` | Sampling temperature (0.0–2.0) |
| `max_tokens` | `Optional[int]` | `None` | Maximum tokens in the response |
| `stream` | `bool` | `False` | If True, return a token iterator |

**Returns**: `str` if `stream=False`, `Iterator[str]` if `stream=True`

**Raises**: `ValueError` if `message` is empty or None

##### `chat_stream(message, **kwargs)`

Stream chat response tokens one at a time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | required | The user message |
| `**kwargs` | — | — | Additional arguments passed to `chat()` |

**Returns**: `Iterator[str]` — Yields response tokens

##### `chat_batch(messages, **kwargs)`

Process multiple messages in a batch for higher throughput.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[str]` | required | List of user messages |
| `**kwargs` | — | — | Additional arguments passed to `chat()` |

**Returns**: `List[str]` — List of responses

##### `analyze_image(image, *, confidence_threshold=None)`

Analyze an image and return captions, labels, and scene information.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `Union[str, bytes]` | required | File path, URL, base64 string, or raw bytes |
| `confidence_threshold` | `Optional[float]` | `None` | Minimum confidence for detections |

**Returns**: `dict` with keys `captions`, `labels`, `objects`

##### `detect_objects(image, *, confidence_threshold=0.5)`

Detect objects in an image with bounding boxes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `Union[str, bytes]` | required | File path, URL, base64 string, or raw bytes |
| `confidence_threshold` | `float` | `0.5` | Minimum confidence threshold |

**Returns**: `List[dict]` — Each dict has `label`, `confidence`, `bbox`

##### `tts(text, *, voice=None, rate=1.0, output_format="wav")`

Convert text to speech.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Text to convert to speech |
| `voice` | `Optional[str]` | `None` | Voice identifier (e.g., "female-1", "aria") |
| `rate` | `float` | `1.0` | Speaking rate multiplier (0.5–2.0) |
| `output_format` | `str` | `"wav"` | Output audio format (`wav`, `mp3`, `ogg`) |

**Returns**: `TTSResult` with `audio_bytes`, `audio_path`, `sample_rate`, `duration`

##### `stt(audio, *, language=None)`

Convert speech to text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | `Union[str, bytes]` | required | Audio file path or raw audio bytes |
| `language` | `Optional[str]` | `None` | Language hint (e.g., "en", "es") for better accuracy |

**Returns**: `STTResult` with `text`, `confidence`, `language`, `segments`

---

### `ThundersConfig`

Configuration class for customizing Thunders AI behavior.

```python
from thunders_ai import ThundersConfig

config = ThundersConfig(
    model="thunders-7b",
    device="cuda",
    max_tokens=4096,
    temperature=0.7,
)
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"thunders-7b"` | Model identifier or path |
| `device` | `str` | `"auto"` | Compute device (`"cpu"`, `"cuda"`, `"auto"`) |
| `max_tokens` | `int` | `2048` | Default maximum tokens for responses |
| `temperature` | `float` | `0.7` | Default sampling temperature |
| `system_prompt` | `Optional[str]` | `None` | Default system prompt |
| `api_key` | `Optional[str]` | `None` | API key for cloud services |
| `model_path` | `Optional[str]` | `None` | Custom path to local model files |
| `log_level` | `str` | `"INFO"` | Logging verbosity level |

---

## LLM Module

### `ChatModel`

Low-level chat model interface for advanced usage.

```python
from thunders_ai.llm.chat_model import ChatModel

model = ChatModel(model_name="thunders-7b", device="cuda")
response = model.generate("Hello, world!", max_tokens=100)
```

#### Methods

| Method | Description |
|--------|-------------|
| `generate(prompt, **kwargs)` | Generate text from a prompt |
| `generate_stream(prompt, **kwargs)` | Stream generated tokens |
| `count_tokens(text)` | Count the number of tokens in text |
| `get_tokenizer()` | Access the underlying tokenizer |

### `ChatMessage`

Data class representing a single message in a conversation.

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | Message role: `"user"`, `"assistant"`, or `"system"` |
| `content` | `str` | The message content |
| `timestamp` | `Optional[float]` | Unix timestamp |

### `ChatHistory`

Manages conversation history with automatic length management.

| Method | Description |
|--------|-------------|
| `add_message(role, content)` | Add a message to history |
| `get_messages()` | Return all messages as a list |
| `clear()` | Clear all messages |
| `search(query)` | Search messages by content |

---

## Memory Module

### `MemorySystem`

Persistent memory storage for long-term context and user preferences.

| Method | Description |
|--------|-------------|
| `save(memory)` | Save a memory to storage |
| `retrieve(memory_id)` | Retrieve a memory by ID |
| `search(query)` | Search memories by content |
| `delete(memory_id)` | Delete a specific memory |
| `clear()` | Clear all memories |
| `consolidate()` | Merge similar memories |
| `list_all()` | List all stored memories |

### `Memory`

Data class for a single memory entry.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `content` | `str` | Memory content |
| `memory_type` | `MemoryType` | Type of memory |
| `timestamp` | `float` | Creation timestamp |
| `expires_at` | `Optional[float]` | Expiry timestamp |

### `MemoryType`

Enum for memory categories.

| Value | Description |
|-------|-------------|
| `PREFERENCE` | User preferences and settings |
| `FACT` | Factual knowledge |
| `EPISODIC` | Event-based memories |
| `PROCEDURAL` | How-to and process knowledge |

---

## Vision Module

### `VisionAI`

Computer vision processing class.

| Method | Description |
|--------|-------------|
| `analyze(image)` | Full image analysis (captions, labels, objects) |
| `detect_objects(image, threshold)` | Object detection with bounding boxes |
| `detect_faces(image)` | Face detection with landmarks |
| `classify(image, top_k)` | Image classification |

### `VisionResult`

Result of image analysis.

| Field | Type | Description |
|-------|------|-------------|
| `captions` | `List[str]` | Generated image captions |
| `objects` | `List[DetectedObject]` | Detected objects |
| `labels` | `List[str]` | Image-level labels |

### `DetectedObject`

A single detected object.

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Object class label |
| `confidence` | `float` | Detection confidence (0–1) |
| `bbox` | `List[int]` | Bounding box `[x1, y1, x2, y2]` |

---

## Speech Module

### `SpeechAI`

Speech processing class for TTS and STT.

| Method | Description |
|--------|-------------|
| `tts(text, voice, rate, format)` | Text-to-speech conversion |
| `stt(audio, language)` | Speech-to-text conversion |
| `get_supported_languages()` | List supported languages |
| `get_available_voices()` | List available voice options |

### `TTSResult`

Result of text-to-speech conversion.

| Field | Type | Description |
|-------|------|-------------|
| `audio_bytes` | `bytes` | Raw audio data |
| `audio_path` | `str` | Path to saved audio file |
| `sample_rate` | `int` | Audio sample rate in Hz |
| `duration` | `float` | Duration in seconds |

### `STTResult`

Result of speech-to-text conversion.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Transcribed text |
| `confidence` | `float` | Overall confidence (0–1) |
| `language` | `str` | Detected language code |
| `segments` | `List[dict]` | Word-level or sentence-level segments |

---

## Security Module

### `EncryptionManager`

AES-256 encryption for data at rest and in transit.

| Method | Description |
|--------|-------------|
| `encrypt(data)` | Encrypt string or bytes |
| `decrypt(ciphertext)` | Decrypt to original data |

### `TokenManager`

JWT-based authentication and authorization.

| Method | Description |
|--------|-------------|
| `create_token(user_id, scopes, expires_in)` | Create a new JWT |
| `verify_token(token, required_scope)` | Verify and decode a JWT |

### `SandboxExecutor`

Isolated code execution environment with resource limits.

| Method | Description |
|--------|-------------|
| `execute(code, timeout)` | Execute Python code safely |

### `ThreatDetector`

AI-powered threat and injection detection.

| Method | Description |
|--------|-------------|
| `analyze(input_text)` | Analyze text for security threats |

---

## API Server

### `create_app(testing=False)`

Create a Flask/FastAPI application instance.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/health/ready` | GET | Readiness probe |
| `/api/v1/chat` | POST | Chat completion |
| `/api/v1/vision/analyze` | POST | Image analysis |
| `/api/v1/vision/detect` | POST | Object detection |
| `/api/v1/speech/tts` | POST | Text-to-speech |
| `/api/v1/speech/stt` | POST | Speech-to-text |
