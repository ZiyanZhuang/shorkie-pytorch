"""Keras HDF5 -> ShorkieTorch 的确定性逐层权重映射。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from .model import ShorkieLM


def _key(path: str) -> str:
    return "model_weights__" + path.replace("/", "__").replace(":", "_")


def _layer(prefix: str, name: str, tensor: str) -> str:
    return _key(f"{prefix}/{name}/{tensor}:0")


def _copy(dst: torch.Tensor, src: np.ndarray, transpose: tuple[int, ...] | None = None) -> None:
    value = torch.from_numpy(src if transpose is None else src.transpose(transpose)).to(dtype=dst.dtype)
    if tuple(dst.shape) != tuple(value.shape):
        raise ValueError(f"形状不匹配：PyTorch {tuple(dst.shape)}，Keras {tuple(value.shape)}")
    dst.copy_(value)


def _copy_linear(layer: nn.Linear, npz: np.lib.npyio.NpzFile, name: str) -> None:
    _copy(layer.weight.data, npz[_layer(name, name, "kernel")], (1, 0))
    _copy(layer.bias.data, npz[_layer(name, name, "bias")])


def _copy_bn(layer: nn.Module, npz: np.lib.npyio.NpzFile, name: str) -> None:
    bn = layer.bn
    _copy(bn.weight.data, npz[_layer(name, name, "gamma")])
    _copy(bn.bias.data, npz[_layer(name, name, "beta")])
    _copy(bn.running_mean, npz[_layer(name, name, "moving_mean")])
    _copy(bn.running_var, npz[_layer(name, name, "moving_variance")])


@torch.no_grad()
def _load_released_weights(model: ShorkieLM, archive: str | Path) -> None:
    """Map a released Keras archive into an already validated architecture."""
    archive = Path(archive)
    if archive.suffix.lower() != ".npz":
        raise ValueError("权重加载器只接受 tools/export_h5_weights.py 生成的 .npz；请勿直接传入 Keras .h5")
    npz = np.load(archive, allow_pickle=False)
    try:
        # stem: Keras [kernel, in, out] -> torch [out, in, kernel]
        _copy(model.stem.weight.data, npz[_layer("conv1d", "conv1d", "kernel")], (2, 1, 0))
        _copy(model.stem.bias.data, npz[_layer("conv1d", "conv1d", "bias")])
        for i, block in enumerate(model.encoder):
            conv1, conv2 = 1 + 2 * i, 2 + 2 * i
            bn1, bn2 = 2 * i, 2 * i + 1
            _copy_bn(block.first.norm, npz, "batch_normalization" if bn1 == 0 else f"batch_normalization_{bn1}")
            _copy_bn(block.second.norm, npz, f"batch_normalization_{bn2}")
            _copy(block.first.conv.weight.data, npz[_layer(f"conv1d_{conv1}", f"conv1d_{conv1}", "kernel")], (2, 1, 0))
            _copy(block.first.conv.bias.data, npz[_layer(f"conv1d_{conv1}", f"conv1d_{conv1}", "bias")])
            _copy(block.second.conv.weight.data, npz[_layer(f"conv1d_{conv2}", f"conv1d_{conv2}", "kernel")], (2, 1, 0))
            _copy(block.second.conv.bias.data, npz[_layer(f"conv1d_{conv2}", f"conv1d_{conv2}", "bias")])
            scale = "scale" if i == 0 else f"scale_{i}"
            _copy(block.scale.data, npz[_layer(scale, scale, "scale")])
        for i, block in enumerate(model.transformers):
            mha = "multihead_attention" if i == 0 else f"multihead_attention_{i}"
            ln1 = "layer_normalization" if 2 * i == 0 else f"layer_normalization_{2 * i}"
            ln2 = f"layer_normalization_{2 * i + 1}"
            _copy(block.norm1.weight.data, npz[_layer(ln1, ln1, "gamma")]); _copy(block.norm1.bias.data, npz[_layer(ln1, ln1, "beta")])
            _copy(block.norm2.weight.data, npz[_layer(ln2, ln2, "gamma")]); _copy(block.norm2.bias.data, npz[_layer(ln2, ln2, "beta")])
            for target, source in ((block.attention.q, "q_layer"), (block.attention.k, "k_layer"), (block.attention.v, "v_layer"), (block.attention.rk, "r_k_layer"), (block.attention.out, "embedding_layer")):
                # MultiheadAttention 在 HDF5 中有两级同名分组。
                key = _key(f"{mha}/{mha}/{source}/kernel:0")
                _copy(target.weight.data, npz[key], (1, 0))
                if target.bias is not None:
                    _copy(target.bias.data, npz[_key(f"{mha}/{mha}/{source}/bias:0")])
            _copy(block.attention.r_w_bias.data, npz[_layer(mha, mha, "r_w_bias")])
            _copy(block.attention.r_r_bias.data, npz[_layer(mha, mha, "r_r_bias")])
            _copy_linear(block.ff1, npz, "dense" if 2 * i == 0 else f"dense_{2 * i}")
            _copy_linear(block.ff2, npz, f"dense_{2 * i + 1}")
        for i, block in enumerate(model.decoder):
            low_bn, skip_bn = 14 + 2 * i, 15 + 2 * i
            _copy_bn(block.low_bn, npz, f"batch_normalization_{low_bn}")
            _copy_bn(block.skip_bn, npz, f"batch_normalization_{skip_bn}")
            _copy_linear(block.low_dense, npz, f"dense_{16 + 2 * i}")
            _copy_linear(block.skip_dense, npz, f"dense_{17 + 2 * i}")
            sep = "separable_conv1d" if i == 0 else f"separable_conv1d_{i}"
            _copy(block.depthwise.weight.data, npz[_layer(sep, sep, "depthwise_kernel")], (1, 2, 0))
            _copy(block.pointwise.weight.data, npz[_layer(sep, sep, "pointwise_kernel")], (2, 1, 0))
            _copy(block.pointwise.bias.data, npz[_layer(sep, sep, "bias")])
        _copy_linear(model.head, npz, f"dense_{16 + 2 * len(model.decoder)}")
    finally:
        npz.close()


def load_released_supervised_weights(model: ShorkieLM, archive: str | Path) -> None:
    """Load released f0--f7 supervised Shorkie weights."""
    if model.cfg.decoder_repeats != 3 or model.cfg.output_channels != 5215 or model.cfg.target_crop != 64:
        raise ValueError("发布的监督模型必须使用 decoder_repeats=3、output_channels=5215、target_crop=64")
    _load_released_weights(model, archive)


def load_released_lm_weights(model: ShorkieLM, archive: str | Path) -> None:
    """Load the released 165-Saccharomycetales Shorkie LM weights."""
    if model.cfg.decoder_repeats != 7 or model.cfg.output_channels != 4 or model.cfg.target_crop != 0:
        raise ValueError("发布的 LM 必须使用 decoder_repeats=7、output_channels=4、target_crop=0")
    _load_released_weights(model, archive)


