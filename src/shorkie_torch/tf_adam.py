"""Dense Keras Adam update; epsilon applies to the uncorrected second moment.

Foreach operations reduce Python/kernel-launch overhead without changing the
optimizer equation. No decoupled weight decay: regularization belongs in loss.
"""
import math
import torch


class KerasAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0, betas=(0.7, 0.9), eps=1e-7):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     keras_iterations=0, algorithm='keras_dense_adam_v1'))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get('algorithm') != 'keras_dense_adam_v1':
                raise ValueError('incompatible optimizer state')
            params = [p for p in group['params'] if p.grad is not None]
            if not params:
                continue
            if any(p.grad.is_sparse or p.dtype != torch.float32 for p in params):
                raise ValueError('KerasAdam requires dense FP32 master parameters')
            beta1, beta2 = group['betas']
            group['keras_iterations'] += 1
            step = group['keras_iterations']
            alpha = group['lr'] * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
            buckets = {}
            for p in params:
                state = self.state[p]
                if not state:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                buckets.setdefault(p.device, []).append(p)
            for ps in buckets.values():
                gs = [p.grad for p in ps]
                ms = [self.state[p]['exp_avg'] for p in ps]
                vs = [self.state[p]['exp_avg_sq'] for p in ps]
                # TF: m += (g-m)*(1-beta1); v += (g*g-v)*(1-beta2).
                dm = torch._foreach_sub(gs, ms)
                torch._foreach_mul_(dm, 1 - beta1)
                torch._foreach_add_(ms, dm)
                del dm
                dv = torch._foreach_mul(gs, gs)
                torch._foreach_sub_(dv, vs)
                torch._foreach_mul_(dv, 1 - beta2)
                torch._foreach_add_(vs, dv)
                del dv
                denom = torch._foreach_sqrt(vs)
                torch._foreach_add_(denom, group['eps'])
                updates = torch._foreach_mul(ms, alpha)
                torch._foreach_div_(updates, denom)
                torch._foreach_sub_(ps, updates)
        return loss
