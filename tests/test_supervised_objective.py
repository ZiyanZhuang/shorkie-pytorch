"""Independent mathematical checks for the explicitly adapted objective."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('coverage_core',Path(__file__).parents[1]/'examples/supervised/supervised_core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

@pytest.mark.parametrize('value',[-1000.,-110.,-20.,0.,30.])
@pytest.mark.parametrize('target',[0.,1.,100.])
def test_extreme_logits(value,target):
 z=torch.full((2,17,3),value,requires_grad=True);y=torch.full_like(z,target)
 loss=core.coverage_loss(y,core.log_softplus(z));g=torch.autograd.grad(loss,z)[0]
 assert torch.isfinite(loss) and torch.isfinite(g).all()
 if target>0 and value < -20:assert (g<0).all()

def test_independent_rate_formula_and_gradient():
 torch.manual_seed(7);z=torch.randn(2,17,3,dtype=torch.float64,requires_grad=True);y=torch.rand_like(z)*10
 p=torch.nn.functional.softplus(z);s=p.sum(1)
 ref=(-(y*torch.log(p/s[:,None,:])).sum(1)+.2*(s-y.sum(1)*torch.log(s))).mean()/17
 got=core.coverage_loss(y,core.log_softplus(z))
 torch.testing.assert_close(got,ref,rtol=1e-12,atol=1e-12)
 a=torch.autograd.grad(got,z,retain_graph=True)[0];b=torch.autograd.grad(ref,z)[0]
 torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
