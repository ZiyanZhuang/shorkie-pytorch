import math
from pathlib import Path
import subprocess
import sys

import torch

from shorkie_torch.tf_adam import KerasAdam


def test_keras_adam_epsilon_placement():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = KerasAdam([p], lr=0.01, betas=(0.9, 0.999), eps=1e-7)
    expected, m, v = 1.0, 0.0, 0.0
    for step, gradient in enumerate([1e-5, -2e-5, 3e-5], 1):
        p.grad = torch.tensor([gradient])
        opt.step()
        m = 0.9*m + 0.1*gradient
        v = 0.999*v + 0.001*gradient**2
        expected -= 0.01*math.sqrt(1-0.999**step)/(1-0.9**step)*m/(math.sqrt(v)+1e-7)
        assert abs(p.item()-expected) < 2e-7


def test_supervised_training_cli_imports():
    script = Path(__file__).parents[1] / 'examples/supervised/train_tracks.py'
    run = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'scratch-matched170' in run.stdout
