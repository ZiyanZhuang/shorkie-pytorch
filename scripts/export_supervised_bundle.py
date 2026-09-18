"""Export a trusted local checkpoint using an allowlist; never copy private metadata."""
import argparse,dataclasses,hashlib,json,pathlib
import torch
from safetensors.torch import save_file,load_file
from shorkie_torch.model import ShorkieLM,ShorkieConfig

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=pathlib.Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 assert sha(a.checkpoint)==a.expected_sha256,'Checkpoint does not match evaluated model'
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 c=ck['contract'];assert c['loss_protocol']=='normalized-lograte-v4'
 cfg=ShorkieConfig(**ck['model_config']);assert cfg.output_channels==78 and cfg.decoder_repeats==3 and cfg.target_crop==64 and cfg.output_activation=='softplus'
 model=ShorkieLM(cfg);model.load_state_dict(ck['model'],strict=True)
 state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
 assert all(torch.isfinite(v).all() for v in state.values())
 a.out.mkdir(parents=True,exist_ok=False);file=a.out/'model.safetensors'
 save_file(state,str(file),metadata={'format':'pt','model_type':'shorkie_coverage'})
 loaded=load_file(str(file));assert all(torch.equal(v,loaded[k]) for k,v in state.items())
 config={'format_version':1,'model_type':'shorkie_coverage','architecture':dataclasses.asdict(cfg),'state_file':file.name,'state_sha256':sha(file)}
 (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 public={k:c[k] for k in ['loss_protocol','fold','arm','batch','lr','warmup_steps','minimum_epochs','patience','max_epochs','seed','species_column','tf32','optimizer','shuffle','selection','clipnorm','augmentation']}
 public.update(source_checkpoint_sha256=a.expected_sha256,tensor_count=len(state),parameter_count=sum(v.numel() for v in model.parameters()),training_epoch=ck['training']['epoch'],training_step=ck['training']['step'],export_validation='strict load, finite tensors, exact safetensors roundtrip')
 (a.out/'training_summary.json').write_text(json.dumps(public,indent=2)+'\n')
 print(json.dumps({'fold':c['fold'],'arm':c['arm'],'state_sha256':config['state_sha256'],'status':'PASS'}),flush=True)
if __name__=='__main__':main()
