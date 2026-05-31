"""Constants and default values for Thunders AI."""

# Package info
PACKAGE_NAME = "thunders-ai"
DEFAULT_MODEL = "thunders-7b"
DEFAULT_VISION_MODEL = "thunders-vision-base"
DEFAULT_SPEECH_MODEL = "thunders-speech-base"

# Model sizes
MODEL_SIZES = {
    "thunders-1b": 1_000_000_000,
    "thunders-3b": 3_000_000_000,
    "thunders-7b": 7_000_000_000,
    "thunders-13b": 13_000_000_000,
    "thunders-70b": 70_000_000_000,
}

# Supported devices
SUPPORTED_DEVICES = ["cpu", "cuda", "mps", "auto"]

# Default generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_CONTEXT_WINDOW = 8192

# API endpoints
API_BASE_URL = "https://api.thunders-ai.dev/v1"
API_CHAT_ENDPOINT = "/chat/completions"
API_VISION_ENDPOINT = "/vision/analyze"
API_SPEECH_ENDPOINT = "/speech"

# Cache and storage
DEFAULT_CACHE_DIR = "~/.thunders_ai/cache"
DEFAULT_MODEL_DIR = "~/.thunders_ai/models"
DEFAULT_LOG_DIR = "~/.thunders_ai/logs"

# Vision constants
DEFAULT_IMAGE_SIZE = 512
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Speech constants
DEFAULT_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
SUPPORTED_LANGUAGES = ["en", "id", "ja", "ko", "zh", "es", "fr", "de", "pt", "ar"]

# Robotics constants
DEFAULT_NAVIGATION_ALGORITHM = "a_star"
SUPPORTED_NAVIGATION_ALGORITHMS = ["a_star", "dijkstra", "rrt", "rrt_star", "dwa"]
DEFAULT_ROBOT_TYPE = "generic"
SUPPORTED_ROBOT_TYPES = ["generic", "drone", "vehicle", "arm", "humanoid"]

# Security
DEFAULT_ENCRYPTION_ALGORITHM = "AES-256-GCM"
TOKEN_EXPIRY_SECONDS = 3600

# HTTP
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_WORKERS = 4

# Memory types
MEMORY_TYPES = ["conversation", "episodic", "semantic", "procedural"]

# Response formats
RESPONSE_FORMATS = ["text", "json", "markdown"]
