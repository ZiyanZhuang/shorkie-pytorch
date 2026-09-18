# Security policy

Do not open a public issue containing credentials, private paths, unpublished
data, or a malicious checkpoint. Use a private reporting channel available through the repository Security tab; never post sensitive details publicly.

Public inference accepts only `safetensors` plus JSON configuration. The legacy
conversion command uses `torch.load(..., weights_only=False)` because audited
training checkpoints contain optimizer and run state. Run it only on a
checkpoint you created or otherwise trust.

Never load an untrusted `.pt`, `.pth`, or pickle checkpoint. Release bundles
must contain no optimizer, RNG, tracking identifier, hostname, or filesystem
path metadata.
