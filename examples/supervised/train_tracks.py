"""Source-parameter public-track adaptation. All runs require cache READY hash admission.

This is not TF end-to-end parity: corrected blacklist/crop coordinates, public
normalization and NumPy full-epoch permutation differ from author TFRecord shuffle.
"""
import argparse,csv,dataclasses,hashlib,json,os,pathlib,random,signal,time
import numpy as np
import torch
from shorkie_torch.model import ShorkieLM,ShorkieConfig,shorkie_l2_penalty,set_shorkie_learning_rate
from shorkie_torch.tf_adam import KerasAdam
from supervised_core import coverage_loss,coverage_forward,LOSS_PROTOCOL,build_model,split_ids,reverse_complement_input,orient_prediction_back
from build_cache import sha,save_json

STOP=False
def request_stop(*unused):
 global STOP
 STOP=True
def atomic_save(obj,p):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.partial');torch.save(obj,tmp);os.replace(tmp,p)
def rng_state():
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def set_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
 if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])
def cache_open(path):
 p=pathlib.Path(path);r=json.loads((p/'READY.json').read_text())
 for name,h in r['hashes'].items():
  if sha(p/name)!=h:raise ValueError('Cache hash mismatch '+name)
 x=np.load(p/'sequence.npy',mmap_mode='r');y=np.load(p/'targets.npy',mmap_mode='r');folds=np.load(p/'folds.npy')
 if x.shape!=(len(y),16384) or y.shape[1]!=896 or len(folds)!=len(y):raise ValueError('Cache shape mismatch')
 if not np.isfinite(y).all() or np.min(y)<0 or np.max(x)>4:raise ValueError('Cache values rejected')
 return x,y,folds,r
def inputs(code,channels,species_column,device):
 q=torch.as_tensor(np.array(code,copy=True),device=device,dtype=torch.long)
 b=torch.nn.functional.one_hot(q,5)[...,:4].float()
 if channels==4:return b
 z=torch.zeros((*q.shape,channels),device=device);z[...,:4]=b;z[...,species_column]=1
 return z
class Moments:
 def __init__(self,t):self.n=0;self.s=np.zeros((5,t),np.float64)
 def update(self,y,p):
  a=np.asarray(y,np.float64).reshape(-1,self.s.shape[1]);b=np.asarray(p,np.float64).reshape(a.shape)
  if not np.isfinite(a).all() or not np.isfinite(b).all():raise FloatingPointError('Nonfinite evaluation')
  self.n+=len(a);self.s+=np.stack([a.sum(0),b.sum(0),(a*a).sum(0),(b*b).sum(0),(a*b).sum(0)])
 def result(self):
  sy,sp,yy,pp,yp=self.s;vy=yy-sy*sy/self.n;vp=pp-sp*sp/self.n;cov=yp-sy*sp/self.n
  # Undefined constant-track correlations remain NaN, never silently zero.
  r=np.divide(cov,np.sqrt(np.maximum(vy*vp,0)),out=np.full_like(cov,np.nan),where=(vy>0)&(vp>0))
  r2=1-np.divide(yy+pp-2*yp,vy,out=np.full_like(cov,np.nan),where=vy>0)
  return {'pearson':r.tolist(),'r2':r2.tolist(),'mean_pearson':float(np.nanmean(r)),'mean_r2':float(np.nanmean(r2)),'n_bin_observations':self.n,'undefined_pearson_tracks':int(np.isnan(r).sum()),'undefined_r2_tracks':int(np.isnan(r2).sum())}
def evaluate(m,x,y,ids,a,pred_path=None):
 m.eval();mom=Moments(y.shape[-1]);loss_sum=0.;count=0
 pred=np.lib.format.open_memmap(str(pred_path)+'.partial',mode='w+',dtype=np.float32,shape=(len(ids),896,y.shape[-1])) if pred_path else None
 with torch.no_grad():
  for s in range(0,len(ids),a.batch):
   idx=ids[s:s+a.batch];xx=inputs(x[idx],m.cfg.input_channels,a.species_column,a.device);yy=torch.as_tensor(np.array(y[idx],dtype=np.float32),device=a.device)
   pp,logp=coverage_forward(m,xx)
   if a.action=='evaluate' and a.rc:
    pr,logr=coverage_forward(m,reverse_complement_input(xx))
    pp=(pp+orient_prediction_back(pr,a.rc_partner))/2
    logp=torch.logaddexp(logp,orient_prediction_back(logr,a.rc_partner))-np.log(2.)
   loss=coverage_loss(yy,logp);loss_sum+=float(loss)*len(idx);count+=len(idx)
   pred_cpu=pp.cpu().numpy();mom.update(y[idx],pred_cpu)
   if pred is not None:pred[s:s+len(idx)]=pred_cpu
 if pred is not None:pred.flush();del pred;os.replace(str(pred_path)+'.partial',pred_path)
 r=mom.result();r['coverage_loss']=loss_sum/count;r['l2']=float(shorkie_l2_penalty(m));r['selection']=r['mean_pearson']+r['mean_r2']/4
 if y.shape[-1]==78:
  r['modalities']={}
  for label,indices in [('ChIP',list(range(74))),('nascent_RNA_5prime_tag',list(range(74,78)))]:
   sub=Moments(len(indices));sub.n=mom.n;sub.s=mom.s[:,indices];r['modalities'][label]=sub.result()
 if not np.isfinite(r['selection']):raise FloatingPointError('Undefined selection metric')
 return r
def model_for(a,tracks):
 if a.arm=='pretrained':
  if not a.pretrained:raise ValueError('Pretrained arm requires checkpoint path')
  return build_model(tracks,a.pretrained)
 return ShorkieLM(ShorkieConfig(input_channels=4 if a.arm=='scratch' else 170,decoder_repeats=3,output_channels=tracks,output_activation='softplus',target_crop=64))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['train','evaluate']);ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--arm',choices=['pretrained','scratch','scratch-matched170'],required=True);ap.add_argument('--pretrained');ap.add_argument('--resume');ap.add_argument('--checkpoint');ap.add_argument('--device',default='cuda');ap.add_argument('--species-column',type=int,default=114);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--seed',type=int,default=44);ap.add_argument('--epochs',type=int,default=5000);ap.add_argument('--minimum-epochs',type=int,default=50);ap.add_argument('--patience',type=int,default=150);ap.add_argument('--max-hours',type=float,default=3.2);ap.add_argument('--warmup-steps',type=int,default=5000);ap.add_argument('--lr',type=float);ap.add_argument('--max-steps',type=int,default=0);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--tf32',action='store_true');ap.add_argument('--rc',action='store_true');a=ap.parse_args()
 if a.batch!=8:raise ValueError('Main protocol keeps effective batch 8; create separately audited protocol for changes')
 if not 5<=a.species_column<170:raise ValueError('Invalid absolute species column')
 torch.set_num_threads(a.threads);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=a.tf32;torch.backends.cudnn.allow_tf32=a.tf32;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 x,y,folds,ready=cache_open(a.cache);a.rc_partner=ready.get('rc_partner',list(range(y.shape[-1])));ids=split_ids(folds,a.fold);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 # Evaluation reconstructs architecture without needlessly loading pretrain weights.
 if a.action=='evaluate':
  ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=ShorkieLM(ShorkieConfig(**ck['model_config'])).to(a.device)
  if ck['contract']['cache_ready_sha256']!=sha(pathlib.Path(a.cache)/'READY.json') or ck['contract']['fold']!=a.fold:raise ValueError('OOF checkpoint/cache/fold mismatch')
  if ck['contract']['arm']!=a.arm:raise ValueError('Evaluation arm mismatch')
  if ck['contract'].get('loss_protocol')!=LOSS_PROTOCOL:raise ValueError('Wrong evaluation loss protocol')
  m.load_state_dict(ck['model'],strict=True);r=evaluate(m,x,y,ids['test'],a,out/'oof_predictions.npy');np.save(out/'oof_indices.npy',ids['test']);r.update(fold=a.fold,arm=a.arm,checkpoint_sha256=sha(a.checkpoint),cache_ready_sha256=ck['contract']['cache_ready_sha256'],claim='single-fold held-out coverage; not eightfold external ensemble')
  save_json(out/'oof_metrics.json',r);print(json.dumps(r));return
 lr=a.lr if a.lr is not None else (2e-5 if a.arm in ['pretrained','scratch-matched170'] else 5e-4)
 contract={'loss_protocol':LOSS_PROTOCOL,'cache_ready_sha256':sha(pathlib.Path(a.cache)/'READY.json'),'fold':a.fold,'arm':a.arm,'pretrained_sha256':sha(a.pretrained) if a.arm=='pretrained' else None,'batch':a.batch,'lr':lr,'warmup_steps':a.warmup_steps,'minimum_epochs':a.minimum_epochs,'patience':a.patience,'max_epochs':a.epochs,'seed':a.seed,'species_column':a.species_column,'tf32':a.tf32,'code_sha256':sha(__file__),'core_sha256':sha(pathlib.Path(__file__).with_name('supervised_core.py')),'optimizer':'KerasAdam .9/.999 eps1e-7','shuffle':'full permutation each epoch; adapted from TFRecord shuffle256','selection':'mean track Pearson plus mean track R2/4','clipnorm':.1,'augmentation':'none as source training params'}
 m=model_for(a,y.shape[-1]).to(a.device);opt=KerasAdam(m.parameters(),lr=0,betas=(.9,.999),eps=1e-7)
 state={'epoch':0,'cursor':0,'step':0,'best':-float('inf'),'unimproved':0,'order':None,'epoch_loss_sum':0.,'epoch_batches':0}
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False)
  if ck['contract']!=contract:raise ValueError('Resume contract mismatch; no old-loss migration permitted')
  m.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);state=ck['training'];set_rng(ck['rng'])
 elif (out/'latest.pt').exists():raise ValueError('Output already has latest.pt; explicitly resume')
 save_json(out/'contract.json',contract);signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 t0=time.monotonic();last_snapshot=t0;start_step=state['step'];reason='epoch_limit'
 def checkpoint(name):
  atomic_save({'model':m.state_dict(),'model_config':dataclasses.asdict(m.cfg),'optimizer':opt.state_dict(),'training':state,'rng':rng_state(),'contract':contract,'saved_unix':time.time()},out/name)
 while state['epoch']<a.epochs:
  if state['epoch']>=a.minimum_epochs and state['unimproved']>a.patience:reason='source_patience';break
  if state['order'] is None:state['order']=np.random.permutation(ids['train']);state['cursor']=0
  # Source max500 steps and full batches, without inventing extra examples/epoch.
  nb=min(len(state['order'])//a.batch,500);m.train()
  while state['cursor']<nb:
   pos=state['cursor']*a.batch;idx=state['order'][pos:pos+a.batch];xx=inputs(x[idx],m.cfg.input_channels,a.species_column,a.device);yy=torch.as_tensor(np.array(y[idx],dtype=np.float32),device=a.device)
   opt.zero_grad(set_to_none=True);pred,logp=coverage_forward(m,xx);loss=coverage_loss(yy,logp)+shorkie_l2_penalty(m)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; previous latest retained')
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.1,error_if_nonfinite=True);current_lr=set_shorkie_learning_rate(opt,state['step'],base_lr=lr,warmup_steps=a.warmup_steps);opt.step()
   state['step']+=1;state['cursor']+=1;state['epoch_loss_sum']+=float(loss.detach());state['epoch_batches']+=1
   elapsed=time.monotonic()-t0
   if state['step']%20==0:print(json.dumps({'stage':'train','epoch':state['epoch'],'step':state['step'],'lr':current_lr,'loss':float(loss.detach()),'bp_s':(state['step']-start_step)*a.batch*16384/elapsed,'elapsed_s':elapsed}),flush=True)
   if time.monotonic()-last_snapshot>=3600:
    checkpoint(f'hourly-{int(time.time())}-step{state["step"]}.pt');checkpoint('latest.pt');last_snapshot=time.monotonic()
   if STOP or elapsed>=a.max_hours*3600 or (a.max_steps and state['step']-start_step>=a.max_steps):
    reason='signal' if STOP else 'budget_limit';checkpoint('latest.pt');save_json(out/'STOPPED.json',{'reason':reason,'epoch':state['epoch'],'cursor':state['cursor'],'step':state['step'],'resumable':True,'complete':False});return
  val=evaluate(m,x,y,ids['valid'],a);improved=val['selection']>state['best']
  if improved:state['best']=val['selection'];state['unimproved']=0
  else:state['unimproved']+=1
  record={'epoch':state['epoch'],'step':state['step'],'train_total_loss':state['epoch_loss_sum']/state['epoch_batches'],'valid':val,'improved':improved,'elapsed_s':time.monotonic()-t0,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None}
  state['epoch']+=1;state['order']=None;state['cursor']=0;state['epoch_loss_sum']=0.;state['epoch_batches']=0
  checkpoint('latest.pt')
  if improved:checkpoint('best.pt')
  with open(out/'metrics.jsonl','a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
 checkpoint('latest.pt');save_json(out/'STOPPED.json',{'reason':reason,'epoch':state['epoch'],'step':state['step'],'resumable':True,'converged_under_rule':reason=='source_patience'})
if __name__=='__main__':main()
