"""Unofficial, source-audited PyTorch reproduction of Shorkie-LM."""

from .checkpoint import load_pretrained
from .data import ShorkieShardDataset, make_stream_loader
from .model import (
    ShorkieConfig,
    ShorkieLM,
    frozen_weighted_mlm_loss,
    make_mlm_batch,
    make_shorkie_adam,
    set_linear_warmup,
    set_shorkie_learning_rate,
    shorkie_l2_penalty,
    weighted_mlm_loss,
)

__version__ = "0.1.0rc1"

__all__ = [
    "ShorkieConfig",
    "ShorkieLM",
    "ShorkieShardDataset",
    "frozen_weighted_mlm_loss",
    "load_pretrained",
    "make_mlm_batch",
    "make_shorkie_adam",
    "make_stream_loader",
    "set_linear_warmup",
    "set_shorkie_learning_rate",
    "shorkie_l2_penalty",
    "weighted_mlm_loss",
]
