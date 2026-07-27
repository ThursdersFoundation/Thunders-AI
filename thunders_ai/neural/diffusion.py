"""Diffusion Model Module for Thunders AI.

Implements denoising diffusion probabilistic models for image generation
with support for DDPM and DDIM schedulers and text-to-image generation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

SCHEDULER_TYPES = {"ddpm", "ddim"}

if HAS_TORCH:

    class _SimpleUNetBlock(nn.Module):
        """Basic UNet convolutional block with optional time embedding."""

        def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 256) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
            self.act = nn.SiLU()

        def forward(self, x: "torch.Tensor", t_emb: "torch.Tensor") -> "torch.Tensor":
            h = self.act(self.norm1(x))
            h = self.conv1(h)
            t_proj = self.time_mlp(self.act(t_emb))[:, :, None, None]
            h = h + t_proj
            h = self.act(self.norm2(h))
            h = self.conv2(h)
            return h


class _NoiseScheduler:
    """Manages noise scheduling for diffusion processes.

    Supports DDPM (linear) and DDIM scheduling strategies.
    """

    def __init__(self, num_steps: int = 1000, schedule_type: str = "ddpm", beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.num_steps = num_steps
        self.schedule_type = schedule_type.lower()
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_0: "torch.Tensor", noise: "torch.Tensor", timesteps: "torch.Tensor") -> "torch.Tensor":
        """Add noise to clean images at specified timesteps.

        Args:
            x_0: Clean images tensor.
            noise: Standard Gaussian noise tensor of the same shape.
            timesteps: Timestep indices tensor.

        Returns:
            Noisy images tensor.
        """
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x_0.device)[timesteps][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)[timesteps][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def step(self, model_output: "torch.Tensor", timestep: int, sample: "torch.Tensor") -> "torch.Tensor":
        """Compute the previous sample from the current noisy sample.

        Args:
            model_output: Noise prediction from the model.
            timestep: Current timestep index.
            sample: Current noisy sample.

        Returns:
            Denoised sample at the previous timestep.
        """
        t = timestep
        if self.schedule_type == "ddim":
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)
            pred_x0 = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1 - alpha_t_prev) * model_output
            return torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        else:
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_prev_t = self.alphas_cumprod_prev[t]
            pred_x0 = (sample - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            posterior_mean = (
                torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t) * pred_x0
                + torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * sample
            )
            if t > 0:
                noise = torch.randn_like(sample)
                posterior_var = self.posterior_variance[t]
                return posterior_mean + torch.sqrt(posterior_var) * noise
            return posterior_mean


class DiffusionModel:
    """Diffusion model for image generation.

    Implements denoising diffusion with configurable schedulers (DDPM, DDIM)
    and optional text conditioning for text-to-image generation.

    Args:
        image_size: Resolution of generated images (assumes square).
        in_channels: Number of input image channels.
        num_steps: Number of diffusion timesteps.
        scheduler: Noise scheduler type ('ddpm' or 'ddim').
        text_emb_dim: Dimension of text embedding for conditioning (0 = no conditioning).
        device: Device for computation.

    Example::

        model = DiffusionModel(image_size=64, num_steps=1000)
        images = model.generate(batch_size=4)
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        num_steps: Optional[int] = None,
        scheduler: str = "ddpm",
        text_emb_dim: int = 0,
        device: Optional[str] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for DiffusionModel. "
                "Install it with: pip install torch"
            )

        app_cfg = get_config()
        cfg = app_cfg.neural
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_steps = num_steps or cfg.diffusion_steps
        self.scheduler_type = scheduler.lower()
        if self.scheduler_type not in SCHEDULER_TYPES:
            raise ValueError(f"Unsupported scheduler '{scheduler}'. Choose from {SCHEDULER_TYPES}")
        self.text_emb_dim = text_emb_dim
        self.device = device or app_cfg.device

        self._scheduler = _NoiseScheduler(self.num_steps, self.scheduler_type)
        self._build_model()
        logger.info(
            "DiffusionModel initialized: size=%d, steps=%d, scheduler=%s, device=%s",
            self.image_size, self.num_steps, self.scheduler_type, self.device,
        )

    def _build_model(self) -> None:
        """Construct the denoising UNet model."""
        base_channels = 64
        time_emb_dim = 256
        enc_ch = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        in_c = self.in_channels + (self.text_emb_dim if self.text_emb_dim > 0 else 0)

        self._time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self._enc_blocks = nn.ModuleList()
        self._dec_blocks = nn.ModuleList()
        ch_in = in_c
        for ch in enc_ch:
            self._enc_blocks.append(_SimpleUNetBlock(ch_in, ch, time_emb_dim))
            ch_in = ch
        for ch in reversed(enc_ch):
            self._dec_blocks.append(_SimpleUNetBlock(ch * 2, ch, time_emb_dim))

        self._final_conv = nn.Conv2d(enc_ch[0], self.in_channels, 1)
        self._model = nn.ModuleDict({
            "time_embed": self._time_embed,
            "enc_blocks": self._enc_blocks,
            "dec_blocks": self._dec_blocks,
            "final_conv": self._final_conv,
        }).to(self.device)

    def _denoise(self, x: "torch.Tensor", t: "torch.Tensor", text_emb: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Run the denoising network on noisy input.

        Args:
            x: Noisy images of shape (batch, channels, H, W).
            t: Timestep tensor of shape (batch,).
            text_emb: Optional text conditioning embeddings.

        Returns:
            Predicted noise tensor.
        """
        if text_emb is not None:
            text_map = text_emb[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, text_map], dim=1)
        t_emb = self._time_embed(t.float().unsqueeze(-1))
        skips: List[torch.Tensor] = []
        for block in self._enc_blocks:
            x = block(x, t_emb)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        for block in self._dec_blocks:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skips.pop()
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
        return self._final_conv(x)

    def generate(
        self,
        batch_size: int = 1,
        text_embeddings: Optional["torch.Tensor"] = None,
        return_intermediates: bool = False,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        """Generate images by iteratively denoising from pure noise.

        Args:
            batch_size: Number of images to generate.
            text_embeddings: Optional text embeddings for conditioning.
            return_intermediates: Whether to return intermediate denoising steps.

        Returns:
            Generated images tensor, optionally with intermediate steps.
        """
        self._model.eval()
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=self.device)
        intermediates: List[torch.Tensor] = []

        with torch.no_grad():
            for t_idx in reversed(range(self.num_steps)):
                t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
                noise_pred = self._denoise(x, t, text_embeddings)
                x = self._scheduler.step(noise_pred, t_idx, x)
                if return_intermediates and t_idx % max(1, self.num_steps // 10) == 0:
                    intermediates.append(x.clone())

        images = (x.clamp(-1, 1) + 1) / 2
        if return_intermediates:
            return images, intermediates
        return images

    def sample(self, num_samples: int = 1, text_embeddings: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Sample images from the diffusion model's latent space.

        Args:
            num_samples: Number of samples to generate.
            text_embeddings: Optional text conditioning.

        Returns:
            Generated image tensor.
        """
        return self.generate(batch_size=num_samples, text_embeddings=text_embeddings)

    def train_step(
        self, images: "torch.Tensor", text_embeddings: Optional["torch.Tensor"] = None
    ) -> Dict[str, "torch.Tensor"]:
        """Perform a single training step.

        Args:
            images: Clean training images of shape (batch, C, H, W) in range [-1, 1].
            text_embeddings: Optional text conditioning.

        Returns:
            Dictionary with 'loss' tensor.
        """
        self._model.train()
        batch_size = images.size(0)
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(images)
        noisy_images = self._scheduler.add_noise(images, noise, t)
        noise_pred = self._denoise(noisy_images, t, text_embeddings)
        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss}

    def load_pretrained(self, path: str) -> None:
        """Load pretrained model weights from disk.

        Args:
            path: File path to the saved model state dict.
        """
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info("Loaded pretrained diffusion weights from %s", path)
        except FileNotFoundError:
            logger.error("Model file not found: %s", path)
            raise
        except RuntimeError as exc:
            logger.error("Failed to load diffusion weights: %s", exc)
            raise

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: Destination file path.
        """
        torch.save(self._model.state_dict(), path)
        logger.info("Diffusion model saved to %s", path)

    def get_config_info(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "num_steps": self.num_steps,
            "scheduler": self.scheduler_type,
            "text_emb_dim": self.text_emb_dim,
            "device": self.device,
        }
