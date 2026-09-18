"""Public supervised adaptation: normalized-lograte-v4, not epsilon-formula parity.

Reference: calico/shorkie-paper pinned Baskerville metrics.poisson_multinomial,
supervised params and Westminster fold rotation. Apache-2.0 upstream.
"""
import torch
from shorkie_torch.model import ShorkieLM,ShorkieConfig

LOSS_PROTOCOL='normalized-lograte-v4'

def log_softplus(z):
    if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite logits')
    # At z<-20 the asymptotic log(softplus(z))=z differs by <1.1e-9.
    safe=torch.log(torch.nn.functional.softplus(z.clamp_min(-20)))
    return torch.where(z < -20,z,safe)

def coverage_forward(model,x):
    captured=[]
    handle=model.head.register_forward_hook(lambda module,args,out:captured.append(out))
    try:prediction=model(x)
    finally:handle.remove()
    if len(captured)!=1:raise RuntimeError('Expected exactly one output-head invocation')
    return prediction,log_softplus(captured[0])

def coverage_loss(y,log_p,total_weight=.2):
    if y.shape!=log_p.shape or y.ndim!=3:raise ValueError('Expected matching B,L,T')
    if not torch.isfinite(y).all() or not torch.isfinite(log_p).all():raise FloatingPointError('Nonfinite target/log-rate')
    if (y<0).any():raise ValueError('Negative coverage target')
    length=y.shape[1];sy=y.sum(1);log_s=torch.logsumexp(log_p,dim=1)
    total=torch.exp(log_s)
    if not torch.isfinite(sy).all() or not torch.isfinite(total).all():raise FloatingPointError('Nonfinite total')
    # Exact normalized profile; true empty profiles contribute zero.
    profile=-(y*(log_p-log_s[:,None,:])).sum(1)/length
    poisson=(total-sy*log_s)/length
    result=(profile+total_weight*poisson).mean()
    if not torch.isfinite(result):raise FloatingPointError('Nonfinite loss')
    return result

def build_model(num_tracks,checkpoint=None):
    m=ShorkieLM(ShorkieConfig(decoder_repeats=3,output_channels=num_tracks,output_activation='softplus',target_crop=64))
    if checkpoint is not None:
        lm=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'];dest=m.state_dict()
        transfer={k:v for k,v in lm.items() if k in dest and not k.startswith('head.')}
        missing=set(dest)-set(transfer)
        if missing!={'head.weight','head.bias'}:raise ValueError('Unexpected transfer omissions: '+str(missing))
        for k,v in transfer.items():
            if v.shape!=dest[k].shape:raise ValueError('Transfer shape mismatch '+k)
        m.load_state_dict({**dest,**transfer},strict=True)
    return m

def split_ids(folds,test_fold):
    if not 0<=test_fold<8:raise ValueError('Eight genomic folds required')
    import numpy as np
    a=np.asarray(folds);v=(test_fold+1)%8
    if not np.isin(a,np.arange(8)).all():raise ValueError('Invalid genomic fold')
    return {k:np.flatnonzero(mask) for k,mask in {'test':a==test_fold,'valid':a==v,'train':(a!=test_fold)&(a!=v)}.items()}

def reverse_complement_input(x):
    a=x.flip(1).clone();a[:,:,:4]=a[:,:,[3,2,1,0]];return a

def orient_prediction_back(y,strand_pair):
    if sorted(strand_pair)!=list(range(y.shape[-1])) or any(strand_pair[strand_pair[i]]!=i for i in range(len(strand_pair))):raise ValueError('Strand map must be an involution')
    return y.flip(1)[:,:,strand_pair]

def ensemble_logsed(ref_predictions,alt_predictions,gene_bins,track_indices):
    # Fold/orientation and track averaging precede the nonlinear log transform.
    ref=torch.stack(ref_predictions).mean(0)[...,track_indices].mean(-1)
    alt=torch.stack(alt_predictions).mean(0)[...,track_indices].mean(-1)
    return torch.log2(alt[...,gene_bins].sum(-1)+1)-torch.log2(ref[...,gene_bins].sum(-1)+1)
