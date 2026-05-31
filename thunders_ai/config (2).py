"""Configuration management for Thunders AI.

This module provides the ThundersConfig class that manages all configuration
settings for Thunders AI, including model parameters, device settings,
API credentials, and runtime options.
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class SecurityConfig:
    """Security-related configuration settings.
    
    Attributes:
        jwt_secret_key: Secret key for JWT token signing.
        jwt_expiry_minutes: JWT token expiration time in minutes.
        jwt_algorithm: JWT signing algorithm.
        key_rotation_days: Days before encryption key rotation.
        sandbox_max_memory_mb: Maximum memory for sandboxed execution.
        sandbox_max_time_seconds: Maximum execution time for sandbox.
        sandbox_allow_network: Whether sandbox allows network access.
        encryption_algorithm: Default encryption algorithm.
        hash_algorithm: Default hashing algorithm.
        key_size: Default encryption key size in bits.
        threat_detection_level: Sensitivity level for threat detection.
    """
    jwt_secret_key: Optional[str] = None
    jwt_expiry_minutes: int = 60
    jwt_algorithm: str = "HS256"
    key_rotation_days: int = 90
    sandbox_max_memory_mb: int = 512
    sandbox_max_time_seconds: int = 300
    sandbox_allow_network: bool = False
    encryption_algorithm: str = "AES-256-GCM"
    hash_algorithm: str = "SHA-256"
    key_size: int = 256
    threat_detection_level: str = "medium"


@dataclass
class NeuralConfig:
    """Neural network architecture configuration settings.
    
    Attributes:
        rnn_hidden_size: Default hidden size for RNN models.
        rnn_num_layers: Default number of RNN layers.
        rnn_bidirectional: Whether RNN models default to bidirectional.
        transformer_layers: Default number of transformer layers.
        transformer_heads: Default number of attention heads.
        transformer_hidden: Default hidden dimension for transformers.
        transformer_dropout: Default dropout rate for transformer models.
        diffusion_steps: Default number of diffusion steps.
        diffusion_scheduler: Default noise scheduler type ('ddpm' or 'ddim').
        cnn_architecture: Default CNN backbone architecture.
        rl_algorithm: Default RL algorithm ('dqn', 'ppo', 'sac').
        rl_learning_rate: Default learning rate for RL agents.
        rl_gamma: Default discount factor for RL agents.
        default_dtype: Default tensor data type.
    """
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = False
    transformer_layers: int = 12
    transformer_heads: int = 12
    transformer_hidden: int = 768
    transformer_dropout: float = 0.1
    diffusion_steps: int = 1000
    diffusion_scheduler: str = "ddpm"
    cnn_architecture: str = "resnet50"
    rl_algorithm: str = "ppo"
    rl_learning_rate: float = 3e-4
    rl_gamma: float = 0.99
    default_dtype: str = "float32"


@dataclass
class ThundersConfig:
    """Main configuration class for Thunders AI.
    
    Manages all settings including model selection, device configuration,
    API credentials, logging, and feature-specific options.
    
    Attributes:
        model: Name or path of the model to use.
        device: Compute device ("cpu", "cuda", "mps", "auto").
        api_key: API key for cloud services.
        base_url: Base URL for API endpoints.
        temperature: Default sampling temperature.
        max_tokens: Default maximum tokens for generation.
        top_p: Default top-p sampling parameter.
        top_k: Default top-k sampling parameter.
        log_level: Logging level.
        cache_dir: Directory for caching models and data.
    """
    
    # Model settings
    model: str = "thunders-7b"
    device: str = "auto"
    dtype: str = "float16"
    
    # API settings
    api_key: Optional[str] = None
    base_url: str = "https://api.thunders-ai.dev/v1"
    
    # Generation defaults
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    
    # Memory settings
    context_window: int = 8192
    memory_enabled: bool = True
    memory_type: str = "conversation"
    max_memory_entries: int = 100
    
    # Vision settings
    vision_model: str = "thunders-vision-base"
    image_size: int = 512
    confidence_threshold: float = 0.5
    
    # Speech settings
    speech_model: str = "thunders-speech-base"
    sample_rate: int = 16000
    language: str = "en"
    
    # Robotics settings
    robot_type: str = "generic"
    sensor_fusion: bool = True
    navigation_algorithm: str = "a_star"
    obstacle_avoidance: bool = True
    
    # Security settings
    encryption_enabled: bool = True
    sandbox_enabled: bool = False
    auth_required: bool = False
    
    # Cloud settings
    cloud_enabled: bool = False
    cloud_provider: str = "aws"
    distributed: bool = False
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Cache settings
    cache_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.thunders_ai/cache")
    )
    
    # Custom settings
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Sub-configurations (initialized lazily)
    _security: Optional[SecurityConfig] = field(default=None, init=False, repr=False)
    _neural: Optional[NeuralConfig] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.device == "auto":
            self.device = self._auto_detect_device()
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.api_key is None:
            self.api_key = os.environ.get("THUNDERS_AI_API_KEY")
        self._security = SecurityConfig()
        self._neural = NeuralConfig()
    
    @property
    def security(self) -> SecurityConfig:
        """Get security configuration."""
        if self._security is None:
            self._security = SecurityConfig()
        return self._security
    
    @security.setter
    def security(self, value: SecurityConfig) -> None:
        """Set security configuration."""
        self._security = value
    
    @property
    def neural(self) -> NeuralConfig:
        """Get neural network configuration."""
        if self._neural is None:
            self._neural = NeuralConfig()
        return self._neural
    
    @neural.setter
    def neural(self, value: NeuralConfig) -> None:
        """Set neural network configuration."""
        self._neural = value
    
    @staticmethod
    def _auto_detect_device() -> str:
        """Auto-detect the best available compute device.
        
        Returns:
            Device string: "cuda", "mps", or "cpu".
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    @classmethod
    def from_file(cls, path: str) -> "ThundersConfig":
        """Load configuration from a YAML or JSON file.
        
        Args:
            path: Path to configuration file.
            
        Returns:
            ThundersConfig instance with loaded settings.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file format is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_env(cls) -> "ThundersConfig":
        """Create configuration from environment variables.
        
        Environment variables are prefixed with THUNDERS_AI_.
        For example, THUNDERS_AI_MODEL sets the model parameter.
        
        Returns:
            ThundersConfig instance with environment-based settings.
        """
        env_mapping = {
            "THUNDERS_AI_MODEL": "model",
            "THUNDERS_AI_DEVICE": "device",
            "THUNDERS_AI_API_KEY": "api_key",
            "THUNDERS_AI_BASE_URL": "base_url",
            "THUNDERS_AI_LOG_LEVEL": "log_level",
            "THUNDERS_AI_CACHE_DIR": "cache_dir",
        }
        kwargs = {}
        for env_key, param_name in env_mapping.items():
            value = os.environ.get(env_key)
            if value is not None:
                kwargs[param_name] = value
        return cls(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary of all configuration values.
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    def save(self, path: str) -> None:
        """Save configuration to a file.
        
        Args:
            path: Path to save configuration (YAML or JSON).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        
        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                yaml.dump(data, f, default_flow_style=False)
            elif path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {path.suffix}")


# Alias for backward compatibility
Config = ThundersConfig

# Module-level singleton for global config access
_default_config: Optional[ThundersConfig] = None


def get_config() -> ThundersConfig:
    """Get the global default configuration instance.
    
    Returns a singleton ThundersConfig instance, creating it on first call.
    Subsequent calls return the same instance unless reset.
    
    Returns:
        The global ThundersConfig instance.
        
    Example:
        >>> config = get_config()
        >>> print(config.model)
    """
    global _default_config
    if _default_config is None:
        _default_config = ThundersConfig()
    return _default_config


def set_config(config: ThundersConfig) -> None:
    """Set the global default configuration instance.
    
    Args:
        config: ThundersConfig instance to use as global default.
    """
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Reset the global configuration to a fresh default instance."""
    global _default_config
    _default_config = None
