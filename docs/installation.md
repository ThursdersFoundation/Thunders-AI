# Installation Guide

This guide covers all the ways to install and configure Thunders AI, from a quick pip install to full GPU-accelerated setups and Docker deployments.

## Prerequisites

- **Python 3.9** or later (3.10, 3.11, and 3.12 are fully supported)
- **pip** 21.0 or later
- (Optional) **CUDA 11.8+** for GPU acceleration
- (Optional) **Docker** for containerized deployment

## Quick Install

The fastest way to get started is with pip:

```bash
pip install thunders-ai
```

This installs the core library with CPU-only inference. It's perfect for development, testing, and lightweight workloads.

## GPU Setup

For GPU-accelerated inference (recommended for production workloads and large models), install the GPU extras:

```bash
# With CUDA 12.x support
pip install thunders-ai[gpu]

# With CUDA 11.8 support
pip install thunders-ai[gpu-cu118]
```

### Verifying GPU Access

After installation, verify that Thunders AI can detect your GPU:

```python
from thunders_ai import ThundersAI, ThundersConfig

config = ThundersConfig(device="cuda")
ai = ThundersAI(config=config)
print(ai.device_info())  # Should show GPU details
```

### Troubleshooting GPU Issues

If the GPU is not detected:
1. Ensure NVIDIA drivers are installed (`nvidia-smi` should work)
2. Verify CUDA toolkit version matches the installed package
3. Set `CUDA_VISIBLE_DEVICES` if you have multiple GPUs
4. Check that PyTorch or your backend framework has CUDA support enabled

## Docker Setup

Thunders AI provides official Docker images for both CPU and GPU environments:

```bash
# CPU image
docker pull thundersai/thunders-ai:latest

# GPU image (requires NVIDIA Container Toolkit)
docker pull thundersai/thunders-ai:latest-gpu
```

### Running with Docker

```bash
# CPU container
docker run -p 8000:8000 thundersai/thunders-ai:latest

# GPU container
docker run --gpus all -p 8000:8000 thundersai/thunders-ai:latest-gpu
```

### Custom Docker Compose

For production deployments, use Docker Compose:

```yaml
version: "3.9"
services:
  thunders-ai:
    image: thundersai/thunders-ai:latest-gpu
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    environment:
      - THUNDERS_MODEL_PATH=/models
      - THUNDERS_DEVICE=cuda
    volumes:
      - ./models:/models
```

## Installing from Source

For development or to use unreleased features, install from the GitHub repository:

```bash
git clone https://github.com/thunders-ai/thunders-ai.git
cd thunders-ai
pip install -e ".[dev]"
```

### Development Dependencies

For contributing to the project, install the full development toolchain:

```bash
pip install -e ".[dev,test,docs]"
```

This includes:
- **ruff** — Fast Python linter and formatter
- **mypy** — Static type checker
- **pytest** — Testing framework with coverage plugins
- **sphinx** — Documentation builder

## Optional Dependencies

Thunders AI uses optional dependency groups so you only install what you need:

| Group | Install Command | Description |
|-------|----------------|-------------|
| `gpu` | `pip install thunders-ai[gpu]` | CUDA GPU acceleration |
| `vision` | `pip install thunders-ai[vision]` | Computer vision models and tools |
| `speech` | `pip install thunders-ai[speech]` | TTS and STT engines |
| `robotics` | `pip install thunders-ai[robotics]` | Robotics and navigation |
| `cloud` | `pip install thunders-ai[cloud]` | Cloud deployment tools |
| `all` | `pip install thunders-ai[all]` | Everything |
| `dev` | `pip install thunders-ai[dev]` | Development tools |

## Configuration

Thunders AI is configured through the `ThundersConfig` class or environment variables:

```python
from thunders_ai import ThundersConfig

config = ThundersConfig(
    model="thunders-7b",
    device="cuda",
    max_tokens=4096,
    temperature=0.7,
    api_key="your-api-key",  # or set THUNDERS_API_KEY env var
)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THUNDERS_API_KEY` | — | API key for cloud services |
| `THUNDERS_MODEL_PATH` | `~/.thunders/models` | Path to local model files |
| `THUNDERS_DEVICE` | `auto` | Compute device (`cpu`, `cuda`, `auto`) |
| `THUNDERS_LOG_LEVEL` | `INFO` | Logging verbosity |
| `THUNDERS_CACHE_DIR` | `~/.thunders/cache` | Cache directory for downloaded assets |

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade thunders-ai
```

Check your current version:

```python
import thunders_ai
print(thunders_ai.__version__)
```
