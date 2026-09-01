from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from shorkie_torch import (
    ShorkieConfig,
    ShorkieLM,
    make_mlm_batch,
    shorkie_l2_penalty,
    weighted_mlm_loss,
)


def test_default_parameter_count_and_forward_shape() -> None:
    model = ShorkieLM().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_651_812
    assert sum(tensor.numel() for tensor in model.state_dict().values()) == 13_665_856
    sample = torch.zeros(1, 512, 170)
    sample[:, :, 0] = 1
    sample[:, :, 114] = 1
    with torch.inference_mode():
        output = model(sample)
    assert output.shape == (1, 512, 4)
    assert torch.isfinite(output).all()


def test_mlm_mask_count_channels_and_weight_contract() -> None:
    torch.manual_seed(165)
    batch, length = 16, 512
    codes = torch.arange(length).remainder(4).repeat(batch, 1)
    bases = F.one_hot(codes, 4).float()
    species = F.one_hot(torch.arange(batch).remainder(165), 165).float()
    exon = torch.zeros(batch, length)
    repeat = torch.zeros(batch, length)
    exon[:, :32] = 1
    repeat[:, 16:48] = 1
    inputs, targets, indices, weights = make_mlm_batch(bases, species, exon, repeat, training=False)
    assert inputs.shape == (batch, length, 170)
    assert targets.shape == bases.shape
    assert indices.shape == (batch, int(0.15 * length))
    assert all(torch.unique(row).numel() == row.numel() for row in indices)
    selected_mask_tokens = inputs.gather(1, indices[..., None].expand(-1, -1, 170))[..., 4]
    ratio = float(selected_mask_tokens.mean())
    assert 0.75 < ratio < 0.85
    selected_dna = inputs.gather(1, indices[..., None].expand(-1, -1, 170))[..., :4]
    selected_targets = targets.gather(1, indices[..., None].expand(-1, -1, 4))
    changed_random = (~selected_mask_tokens.bool()) & (selected_dna != selected_targets).any(dim=-1)
    unchanged = (~selected_mask_tokens.bool()) & ~changed_random
    # A random replacement matches the original base 25% of the time, so the
    # observable expectations are 7.5% changed-random and 12.5% unchanged.
    assert 0.05 < float(changed_random.float().mean()) < 0.10
    assert 0.09 < float(unchanged.float().mean()) < 0.16
    assert torch.allclose(weights[:, :16], torch.full((batch, 16), 0.1))
    assert torch.allclose(weights[:, 16:32], torch.full((batch, 16), 0.01))
    assert torch.allclose(weights[:, 32:48], torch.full((batch, 16), 0.1))
    assert torch.allclose(weights[:, 48:], torch.ones(batch, length - 48))


def test_reverse_complement_keeps_targets_and_weights_aligned() -> None:
    torch.manual_seed(9)
    length = 128
    codes = torch.arange(length).remainder(4)[None]
    bases = F.one_hot(codes, 4).float()
    species = F.one_hot(torch.tensor([109]), 165).float()
    exon = torch.zeros(1, length); exon[:, :7] = 1
    repeat = torch.zeros(1, length); repeat[:, 20:29] = 1
    _, targets, _, weights = make_mlm_batch(bases, species, exon, repeat, training=True)
    forward_weights = (exon * 0.1 + (1 - exon)) * (repeat * 0.1 + (1 - repeat))
    is_forward = torch.equal(targets, bases)
    is_reverse = torch.equal(targets, bases.flip((1, 2)))
    assert is_forward or is_reverse
    expected_weights = forward_weights if is_forward else forward_weights.flip(1)
    assert torch.equal(weights, expected_weights)


def test_weighted_loss_matches_manual_formula() -> None:
    logits = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]])
    bases = F.one_hot(torch.tensor([[0, 1]]), 4).float()
    indices = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.1, 1.0]])
    weighted, unweighted = weighted_mlm_loss(logits, bases, indices, weights)
    nll = F.cross_entropy(logits.reshape(-1, 4), torch.tensor([0, 1]), reduction="none")
    assert torch.equal(weighted, (nll * torch.tensor([0.1, 1.0])).mean())
    assert torch.equal(unweighted, nll.mean())
    assert math.isfinite(float(weighted))


def test_l2_routes_trunk_and_transformer_kernels() -> None:
    model = ShorkieLM()
    penalty = shorkie_l2_penalty(model)
    trunk, transformer = model._l2_kernel_groups
    expected = sum((kernel.square().sum() * model.cfg.trunk_l2_scale for kernel in trunk), torch.zeros(()))
    expected += sum((kernel.square().sum() * model.cfg.transformer_l2_scale for kernel in transformer), torch.zeros(()))
    assert trunk and transformer
    assert torch.allclose(penalty.cpu(), expected.cpu(), atol=1e-7, rtol=1e-7)


def test_random_initialization_nll_is_near_log_four() -> None:
    torch.manual_seed(165)
    model = ShorkieLM().eval()
    length = 512
    codes = torch.randint(4, (2, length))
    bases = F.one_hot(codes, 4).float()
    species = F.one_hot(torch.tensor([109, 109]), 165).float()
    annotation = torch.zeros(2, length)
    inputs, targets, indices, weights = make_mlm_batch(
        bases, species, annotation, annotation, training=False
    )
    with torch.inference_mode():
        _, nll = weighted_mlm_loss(model(inputs), targets, indices, weights)
    assert abs(float(nll) - math.log(4)) < 0.05
