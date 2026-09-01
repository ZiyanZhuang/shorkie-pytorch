"""Unofficial PyTorch implementation of the Shorkie-LM trunk.\n\nThe architecture was ported from the Apache-2.0 Shorkie/Baskerville sources\nat upstream commit c6003ce and independently validated against released weights.

Inputs use channel-last layout [batch, length, channels], matching the original
Keras model.  The output is unnormalised four-base logits; this is intentional
because PyTorch's cross_entropy consumes logits directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ShorkieConfig:
    input_channels: int = 170  # 4 bases + mask token + 165-species one-hot label
    filters_init: int = 96
    filters_end: int = 384
    encoder_repeats: int = 7
    decoder_repeats: int = 7  # Shorkie-LM=7; released supervised f0-f7=3
    transformer_repeats: int = 8
    heads: int = 4
    key_size: int = 64
    position_features: int = 32
    dropout: float = 0.20
    attention_dropout: float = 0.05
    positional_dropout: float = 0.01
    encoder_dropout: float = 0.05
    bn_momentum_tf: float = 0.90
    # Upstream model-global l2_scale=1e-6 is inherited by conv_dna,
    # res_tower, unet_conv and final. transformer_tower overrides both its
    # dense and MHA kernels to 1e-8 in the released params.json.
    trunk_l2_scale: float = 1e-6
    transformer_l2_scale: float = 1e-8
    output_channels: int = 4
    output_activation: str = "linear"  # LM: linear logits; supervised Shorkie: softplus
    target_crop: int = 0  # supervised Shorkie checkpoint: 64 bins at decoder output


def _tf_bn_momentum(momentum: float) -> float:
    # Keras: moving = momentum * moving + (1 - momentum) * batch.
    # PyTorch uses the coefficient of the batch statistic instead.
    return 1.0 - momentum


def _keras_variance_scaling_(tensor: Tensor, *, scale: float, fan: int) -> None:
    """Keras VarianceScaling(truncated_normal) 的分布；不承诺跨框架随机数逐位相同。"""
    # Keras 会校正截断正态的方差（截断于 ±2σ 后仍保持目标 variance）。
    std = math.sqrt(scale / fan) / 0.87962566103423978
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def _init_kernel(module: nn.Module, *, scale: float) -> None:
    if not isinstance(module, (nn.Conv1d, nn.Linear)):
        raise TypeError(type(module))
    _keras_variance_scaling_(module.weight, scale=scale, fan=module.weight.shape[1] * (module.weight.shape[2] if module.weight.ndim == 3 else 1))
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class ChannelLastBatchNorm(nn.Module):
    def __init__(self, channels: int, tf_momentum: float) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, eps=1e-3, momentum=_tf_bn_momentum(tf_momentum))

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class ConvNAC(nn.Module):
    """Keras blocks.conv_nac: norm -> GELU -> Conv1D(same)."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, tf_bn_momentum: float) -> None:
        super().__init__()
        self.norm = ChannelLastBatchNorm(in_ch, tf_bn_momentum)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding="same", bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.gelu(self.norm(x), approximate="tanh").transpose(1, 2)).transpose(1, 2)


class ResidualDownBlock(nn.Module):
    """One upstream res_tower iteration, including its saved skip tensor."""
    def __init__(self, in_channels: int, channels: int, tf_bn_momentum: float, dropout: float) -> None:
        super().__init__()
        self.first = ConvNAC(in_channels, channels, 5, tf_bn_momentum)
        # Upstream omits kernel_size here, so the second convolution is 1 bp.
        self.second = ConvNAC(channels, channels, 1, tf_bn_momentum)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        # baskerville.layers.Scale uses a zero-initialised vector, not a scalar.
        self.scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.first(x)
        y = self.second(skip)
        y = self.dropout(y) * self.scale[None, None, :]
        skip = skip + y
        return self.pool(skip.transpose(1, 2)).transpose(1, 2), skip


def _positional_features(length: int, feature_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Exact central-mask relative-position basis used by baskerville.layers."""
    if feature_size % 2:
        raise ValueError("position feature size must be even")
    basis = feature_size // 2
    positions = torch.arange(-length + 1, length, device=device, dtype=dtype)
    pow_rate = math.exp(math.log(length + 1) / basis)
    widths = torch.pow(torch.tensor(pow_rate, device=device, dtype=dtype), torch.arange(1, basis + 1, device=device, dtype=dtype)) - 1
    central = (widths[None, :] > positions.abs()[:, None]).to(dtype)
    return torch.cat((central, positions.sign()[:, None] * central), dim=-1)[None]


def _relative_shift(x: Tensor) -> Tensor:
    # Transformer-XL relative shift, ported line-for-line from layers.py.
    b, h, t1, t2 = x.shape
    x = torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
    x = x.reshape(b, h, t2 + 1, t1)[:, :, 1:, :]
    x = x.reshape(b, h, t1, t2)
    return x[..., :(t2 + 1) // 2]


class RelativeMultiheadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, key_size: int, position_features: int, attention_dropout: float, positional_dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("model dimension must divide by attention heads")
        self.heads, self.key_size, self.value_size = heads, key_size, dim // heads
        self.position_features = position_features
        self.q = nn.Linear(dim, heads * key_size, bias=False)
        self.k = nn.Linear(dim, heads * key_size, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.rk = nn.Linear(position_features, heads * key_size, bias=False)
        self.r_w_bias = nn.Parameter(torch.empty(1, heads, 1, key_size))
        self.r_r_bias = nn.Parameter(torch.empty(1, heads, 1, key_size))
        # zero_initialized=True in upstream: transformer starts as an identity.
        self.out = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        nn.init.kaiming_normal_(self.r_w_bias); nn.init.kaiming_normal_(self.r_r_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.positional_dropout = nn.Dropout(positional_dropout)

    def _heads(self, x: Tensor, width: int) -> Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, width).permute(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self._heads(self.q(x), self.key_size) * (self.key_size ** -0.5)
        k = self._heads(self.k(x), self.key_size)
        v = self._heads(self.v(x), self.value_size)
        content = torch.matmul(q + self.r_w_bias, k.transpose(-2, -1))
        pos = self.positional_dropout(_positional_features(t, self.position_features, x.device, x.dtype))
        rk = self._heads(self.rk(pos), self.key_size)
        relative = _relative_shift(torch.matmul(q + self.r_r_bias, rk.transpose(-2, -1)))
        weights = self.attention_dropout(torch.softmax(content + relative, dim=-1))
        y = torch.matmul(weights, v).permute(0, 2, 1, 3).reshape(b, t, -1)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, cfg: ShorkieConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-3)
        self.attention = RelativeMultiheadAttention(dim, cfg.heads, cfg.key_size, cfg.position_features, cfg.attention_dropout, cfg.positional_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-3)
        self.ff1 = nn.Linear(dim, 2 * dim)
        self.ff2 = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.ff2(F.relu(self.dropout(self.ff1(self.norm2(x))))))


class UNetConv(nn.Module):
    """Upstream unet_conv with dense path, nearest repeat and SeparableConv1D."""
    def __init__(self, low_channels: int, skip_channels: int, tf_bn_momentum: float) -> None:
        super().__init__()
        self.low_bn = ChannelLastBatchNorm(low_channels, tf_bn_momentum)
        self.skip_bn = ChannelLastBatchNorm(skip_channels, tf_bn_momentum)
        self.low_dense = nn.Linear(low_channels, low_channels)
        self.skip_dense = nn.Linear(skip_channels, low_channels)
        # Keras SeparableConv1D 只有 pointwise 后的一个 bias；depthwise 没有独立 bias。
        self.depthwise = nn.Conv1d(low_channels, low_channels, 3, padding="same", groups=low_channels, bias=False)
        self.pointwise = nn.Conv1d(low_channels, low_channels, 1, bias=True)

    def forward(self, low: Tensor, skip: Tensor) -> Tensor:
        low = self.low_dense(F.gelu(self.low_bn(low), approximate="tanh")).repeat_interleave(2, dim=1)
        skip = self.skip_dense(F.gelu(self.skip_bn(skip), approximate="tanh"))
        if low.shape[1] != skip.shape[1]:
            low = low[:, :skip.shape[1]]
        y = low + skip
        return self.pointwise(self.depthwise(y.transpose(1, 2))).transpose(1, 2)


class ShorkieLM(nn.Module):
    """13.7M-parameter Shorkie LM architecture, PyTorch-native."""
    def __init__(self, cfg: ShorkieConfig = ShorkieConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        # conv_dna has no normalization and linear activation.
        self.stem = nn.Conv1d(cfg.input_channels, cfg.filters_init, 11, padding="same")
        filters = [int(round((cfg.filters_init * math.exp(math.log(cfg.filters_end / cfg.filters_init) / (cfg.encoder_repeats - 1)) ** i) / 32) * 32) for i in range(cfg.encoder_repeats)]
        self.encoder = nn.ModuleList([
            ResidualDownBlock(cfg.filters_init if i == 0 else filters[i - 1], c, cfg.bn_momentum_tf, cfg.encoder_dropout)
            for i, c in enumerate(filters)
        ])
        # 上游 transformer 的 attention_dropout=0.05、position_dropout=0.01，
        # 与外层 residual/FF dropout=0.20 是三个独立概率。
        self.transformers = nn.ModuleList([TransformerBlock(filters[-1], cfg) for _ in range(cfg.transformer_repeats)])
        if not 1 <= cfg.decoder_repeats <= cfg.encoder_repeats:
            raise ValueError("decoder_repeats must be within [1, encoder_repeats]")
        self.decoder = nn.ModuleList([
            UNetConv(filters[-1], skip_channels, cfg.bn_momentum_tf)
            for skip_channels in list(reversed(filters))[:cfg.decoder_repeats]
        ])
        self.head = nn.Linear(filters[-1], cfg.output_channels)
        self._init_from_frozen_defaults()

    def _init_from_frozen_defaults(self) -> None:
        """对齐 params.json：普通 kernel=lecun_normal，transformer/MHA=he_normal，bias=0。"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                _init_kernel(module, scale=1.0)  # Keras lecun_normal: fan_in 模式、scale=1。
        for block in self.transformers:
            for module in (block.ff1, block.ff2, block.attention.q, block.attention.k,
                           block.attention.v, block.attention.rk, block.attention.out):
                _init_kernel(module, scale=2.0)  # transformer 的 kernel_initializer/mha_initializer=he_normal。
            nn.init.zeros_(block.attention.out.weight)
            nn.init.zeros_(block.attention.out.bias)  # zero_initialize=True。
            _keras_variance_scaling_(block.attention.r_w_bias, scale=2.0, fan=block.attention.r_w_bias.numel())
            _keras_variance_scaling_(block.attention.r_r_bias, scale=2.0, fan=block.attention.r_r_bias.numel())

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.cfg.input_channels:
            raise ValueError(f"expected [B,L,{self.cfg.input_channels}], got {tuple(x.shape)}")
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)
        skips: list[Tensor] = []
        for block in self.encoder:
            x, skip = block(x); skips.append(skip)
        for block in self.transformers:
            x = block(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
        if self.cfg.target_crop:
            if 2 * self.cfg.target_crop >= x.shape[1]:
                raise ValueError("target_crop removes the entire sequence")
            x = x[:, self.cfg.target_crop:-self.cfg.target_crop]
        y = self.head(F.gelu(x, approximate="tanh"))
        return F.softplus(y) if self.cfg.output_activation == "softplus" else y


def make_mlm_batch(bases: Tensor, species: Tensor, exon_mask: Tensor, repeat_mask: Tensor, *, mask_rate: float = 0.15, training: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Frozen trainer.prep_mlm contract: RC, 80/10/10 BERT masking, multiplicative weights."""
    b, length, channels = bases.shape
    if channels != 4 or exon_mask.shape != (b, length) or repeat_mask.shape != (b, length):
        raise ValueError("bases 必须为 [B,L,4]，两个注释掩码必须为 [B,L]")
    # TF SeqDataset 的 label 是 [B,1,S]；接受 squeeze 后的 [B,S]，避免接口歧义。
    if species.ndim == 3 and species.shape[1] == 1:
        species = species[:, 0]
    if species.ndim != 2 or species.shape[0] != b:
        raise ValueError("species 必须为 [B,S] 或 TF 原始的 [B,1,S]")
    if training:
        rc = torch.randint(0, 2, (b,), device=bases.device, dtype=torch.bool)
        bases = torch.where(rc[:, None, None], bases.flip((1, 2)), bases)
        exon_mask = torch.where(rc[:, None], exon_mask.flip(1), exon_mask)
        repeat_mask = torch.where(rc[:, None], repeat_mask.flip(1), repeat_mask)
    indices = torch.stack([torch.randperm(length, device=bases.device)[:int(mask_rate * length)] for _ in range(b)])
    chosen = torch.zeros(b, length, device=bases.device, dtype=torch.bool)
    chosen.scatter_(1, indices, True)
    token = torch.zeros(b, length, 1, device=bases.device, dtype=bases.dtype)
    x = torch.cat((bases, token), dim=-1)
    random_bases = F.one_hot(torch.randint(4, (b, length), device=bases.device), 4).to(bases.dtype)
    random_x = torch.cat((random_bases, token), dim=-1)
    mask_x = torch.cat((torch.zeros_like(bases), torch.ones_like(token)), dim=-1)
    kind = torch.multinomial(torch.tensor([.1, .1, .8], device=bases.device), b * length, replacement=True).reshape(b, length)
    # 0=保留原碱基、1=随机碱基、2=[MASK]；与 tf.random.categorical([.1,.1,.8]) 相同。
    masked = x
    masked = torch.where((chosen & (kind == 1))[..., None], random_x, masked)
    masked = torch.where((chosen & (kind == 2))[..., None], mask_x, masked)
    species_full = species[:, None, :].expand(-1, length, -1)
    weights = (exon_mask * .1 + (1 - exon_mask)) * (repeat_mask * .1 + (1 - repeat_mask))
    # 返回 RC 后的 target，确保输入、索引与监督目标在同一坐标系。
    return torch.cat((masked, species_full), dim=-1), bases, indices, weights


def frozen_weighted_mlm_loss(logits: Tensor, bases: Tensor, indices: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Keras SUM_OVER_BATCH_SIZE-equivalent weighted loss plus unweighted NLL."""
    picked_logits = logits.gather(1, indices[..., None].expand(-1, -1, 4))
    targets = bases.argmax(dim=-1).gather(1, indices)
    nll = F.cross_entropy(picked_logits.reshape(-1, 4), targets.reshape(-1), reduction="none").reshape_as(indices)
    picked_weights = weights.gather(1, indices)
    return (nll * picked_weights).mean(), nll.mean()


def shorkie_l2_penalty(model: ShorkieLM) -> Tensor:
    """Exact upstream regularizer routing; kernels only, never bias/norm/Scale."""
    # 模块拓扑在训练期间不变；缓存参数分组，避免在每一步反复递归遍历 Module 树。
    groups = getattr(model, "_l2_kernel_groups", None)
    if groups is None:
        transformer_ids = {
            id(module.weight) for block in model.transformers for module in block.modules()
            if isinstance(module, (nn.Conv1d, nn.Linear))
        }
        trunk, transformer = [], []
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                (transformer if id(module.weight) in transformer_ids else trunk).append(module.weight)
        groups = (tuple(trunk), tuple(transformer))
        # tuple 不是 Parameter/Module，不会影响 state_dict；load_state_dict 后引用仍有效。
        object.__setattr__(model, "_l2_kernel_groups", groups)
    trunk, transformer = groups
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for kernel in trunk:
        penalty = penalty + model.cfg.trunk_l2_scale * kernel.square().sum()
    for kernel in transformer:
        penalty = penalty + model.cfg.transformer_l2_scale * kernel.square().sum()
    return penalty


def set_linear_warmup(optimizer: torch.optim.Optimizer, step: int, *, base_lr: float = 1e-4, warmup_steps: int = 20_000) -> float:
    """与 trainer.WarmUp 一致：step<warmup 时 lr=base_lr*step/warmup，否则为 base_lr。"""
    lr = base_lr * min(step / warmup_steps, 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def set_shorkie_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    base_lr: float = 1e-4,
    warmup_steps: int = 20_000,
    schedule: str = "constant",
    final_lr: float | None = None,
    decay_steps: int = 0,
) -> float:
    """Source warmup plus an optional post-warmup cosine continuation.

    ``step`` is schedule-local. For a branch from a pretrained checkpoint the
    caller starts it at ``warmup_steps`` so the continuation begins at
    ``base_lr`` without replaying warmup.
    """
    if base_lr <= 0 or warmup_steps < 0:
        raise ValueError("base_lr must be positive and warmup_steps non-negative")
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    elif schedule == "constant":
        lr = base_lr
    elif schedule == "cosine":
        if decay_steps <= 0:
            raise ValueError("cosine schedule requires decay_steps > 0")
        floor = base_lr if final_lr is None else final_lr
        if not 0 <= floor <= base_lr:
            raise ValueError("final_lr must be between zero and base_lr")
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        lr = floor + 0.5 * (base_lr - floor) * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def make_shorkie_adam(model: ShorkieLM) -> torch.optim.Adam:
    """冻结 Adam 契约；epsilon=1e-7 是 tf.keras.optimizers.Adam 的默认值。"""
    return torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.7, 0.9), eps=1e-7)



# Public, descriptive alias. The legacy name is retained for numerical-regression tests.
weighted_mlm_loss = frozen_weighted_mlm_loss

