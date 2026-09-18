"""Strict all-fold admission and track-level OOF metrics; no test-driven selection."""
import argparse,csv,json,pathlib
import numpy as np
from scipy.stats import spearmanr
from train_tracks import Moments,cache_open
from build_cache import sha,save_json
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--run-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 x,y,f,ready=cache_open(a.cache);root=pathlib.Path(a.run_root);out=pathlib.Path(a.out);out.mkdir(exist_ok=True);records=[];files=[]
 for arm in ['pretrained','scratch']:
  preds=np.empty(y.shape,np.float32);seen=np.zeros(len(y),np.uint8)
  for fold in range(8):
   d=root/f'fold{fold}-{arm}';stop=json.loads((d/'STOPPED.json').read_text())
   if stop.get('reason')!='source_patience':raise ValueError(f'{arm} fold{fold} has not met source patience; do not present as completed main benchmark')
   od=d/'oof';m=json.loads((od/'oof_metrics.json').read_text());ids=np.load(od/'oof_indices.npy');pred=np.load(od/'oof_predictions.npy',mmap_mode='r')
   if m['fold']!=fold or m['arm']!=arm or m['cache_ready_sha256']!=sha(pathlib.Path(a.cache)/'READY.json'):raise ValueError('OOF metadata mismatch')
   if m['checkpoint_sha256']!=sha(d/'best.pt'):raise ValueError('OOF predictions came from different best checkpoint')
   if set(ids.tolist())!=set(np.flatnonzero(f==fold).tolist()) or len(set(ids))!=len(ids):raise ValueError('OOF identities do not equal heldout partition')
   if pred.shape!=(len(ids),896,y.shape[-1]) or not np.isfinite(pred).all():raise ValueError('OOF invalid prediction shape/value')
   preds[ids]=pred;seen[ids]+=1;files.append({'arm':arm,'fold':fold,'prediction_sha256':sha(od/'oof_predictions.npy'),'checkpoint_sha256':m['checkpoint_sha256']})
  if not np.all(seen==1):raise ValueError('Each window must be evaluated exactly once by its heldout model')
  mom=Moments(y.shape[-1]);mom.update(y,preds);metrics=mom.result()
  for t in range(y.shape[-1]):
   rho=float(spearmanr(np.asarray(y[:,:,t]).ravel(),preds[:,:,t].ravel()).statistic)
   records.append({'arm':arm,'track_index':t,'modality':'ChIP' if t<74 else 'nascent_RNA_5prime_tag','Pearson':metrics['pearson'][t],'Spearman':rho,'R2':metrics['r2'][t]})
 with open(out/'per-track-oof.csv','w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 summary={}
 for arm in ['pretrained','scratch']:
  summary[arm]={}
  for modality in ['ChIP','nascent_RNA_5prime_tag']:
   rr=[r for r in records if r['arm']==arm and r['modality']==modality]
   summary[arm][modality]={m:{'mean':float(np.nanmean([r[m] for r in rr])),'median':float(np.nanmedian([r[m] for r in rr])),'n_tracks':len(rr)} for m in ['Pearson','Spearman','R2']}
 save_json(out/'summary.json',{'status':'ALL_16_MODELS_SOURCE_PATIENCE_AND_OOF_COMPLETE','metrics':summary,'provenance':files,'limitations':['public signal and altered genomic split; not original test set','overlapping same-fold windows repeat bases; intervals are not independent bootstrap units','74 ChIP and 2 RNA conditions x 2 strands are not independent replicate counts','source scratch learning rate and input channels differ; pretrained-vs-scratch not a pure single-factor causal estimate']})
if __name__=='__main__':main()
