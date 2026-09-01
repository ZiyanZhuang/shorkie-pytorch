# Training and checkpoint contract

The production-style entrypoint uses physical batch 8, 150 updates per epoch,
five independently masked validation repeats, source optimizer settings and
best-validation checkpoint selection. Gradient accumulation changes BatchNorm
behavior and is not described as source-equivalent.

Training checkpoints contain model, optimizer, RNG, sampler, telemetry and
resume metadata. They are operational artifacts, not public model files.
Exact epoch-boundary resume requires matching corpus hash, architecture,
optimizer/schedule, seed, worker configuration and sampler cursor. Historical
checkpoints without those fields must be labelled non-exact continuation.

The public conversion command reads a trusted local checkpoint, strictly loads
the model, and writes only tensor state plus a minimal JSON config. It removes
optimizer, RNG, tracking, host, and path metadata.

The validated Blackwell environment used PyTorch 2.11.0+cu128 on compute
capability 12.0. This is a measured environment, not a minimum hardware claim.

