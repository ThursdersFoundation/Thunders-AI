"""Thunders AI Neural Module.

Provides neural network architectures including Transformer, CNN, RNN,
Diffusion models, and Reinforcement Learning agents.
"""

from thunders_ai.neural.transformer import TransformerModel
from thunders_ai.neural.cnn import CNNModel
from thunders_ai.neural.rnn import RNNModel
from thunders_ai.neural.diffusion import DiffusionModel
from thunders_ai.neural.reinforcement_learning import RLAgent

__all__ = [
    "TransformerModel",
    "CNNModel",
    "RNNModel",
    "DiffusionModel",
    "RLAgent",
]

__version__ = "1.0.0"
