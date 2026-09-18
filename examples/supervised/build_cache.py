"""Public ChIP adaptation of pinned hound_data/read/write; atomic per-track cache."""
import argparse,csv,gzip,hashlib,heapq,json,math,os,pathlib,random,time
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save_json(p,obj):
 p=pathlib.Path(p);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(obj,indent=2),encoding='utf8');q.replace(p)
def fasta(p):
 out={};name=None
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:
  for line in f:
   if line.startswith('>'):name=line[1:].split()[0];out[name]=[]
   else:out[name].append(line.strip().upper())
 return {k:''.join(v) for k,v in out.items() if k!='chrM'}
def bed(p):
 out={}
 for l in open(p):
  if not l.strip() or l.startswith('#'):continue
  c,a,b=l.split()[:3];out.setdefault(c,[]).append((int(a),int(b)))
 return {c:sorted(x) for c,x in out.items()}
def split_blocks(seqs,gaps):
 blocks=[]
 for c,s in seqs.items():
  start=0
  for a,b in gaps.get(c,[]):
   if a>start:blocks.append((c,start,a))
   start=max(start,b)
  if start<len(s):blocks.append((c,start,len(s)))
 heap=[(-(b-a),(c,a,b)) for c,a,b in blocks if b-a>=14336];heapq.heapify(heap)
 while -heap[0][0]>786432:
  neg,(c,a,b)=heapq.heappop(heap);m=a+(b-a)//2
  for x,y in [(a,m),(m,b)]:heapq.heappush(heap,(-(y-x),(c,x,y)))
 return [b for n,b in heap]
def fold_blocks(blocks,seed=44):
 rng=np.random.RandomState(seed);aim=math.ceil(sum(b-a for c,a,b in blocks)/8);sizes=np.zeros(8);folds=[[] for _ in range(8)]
 for length,(c,a,b) in sorted([(b-a,(c,a,b)) for c,a,b in blocks],reverse=True):
  gap=np.maximum(aim-sizes,0);f=int(rng.choice(8,p=gap/gap.sum()));folds[f].append((c,a,b));sizes[f]+=length
 merged=[]
 for f,bs in enumerate(folds):
  dst=[]
  for c,a,b in sorted(bs):
   if dst and dst[-1][0]==c and dst[-1][2]==a:dst[-1]=(c,dst[-1][1],b)
   else:dst.append((c,a,b))
  merged.extend((c,a,b,f) for c,a,b in dst)
 return merged
def bin_mask(c,start,black):
 # Deliberate coordinate correction: output bin zero begins at INPUT+1024.
 # Pinned annotate_unmap uses INPUT start with an output-length array.
 origin=start+1024;out=np.zeros(896,bool)
 for a,b in black.get(c,[]):
  lo=max(a,origin);hi=min(b,origin+14336)
  if lo>=hi:continue
  s=(lo-origin)//16;e=math.ceil((hi-origin)/16)
  if origin+(s+1)*16-lo<1.6:s+=1
  if hi-(origin+(e-1)*16)<1.6:e-=1
  out[s:e]=True
 return out
def transform(values,start,intervals,umap):
 a=values.astype(np.float32,copy=True)
 if np.isinf(a).any() or np.any(a[np.isfinite(a)]<0):raise ValueError('Invalid coverage')
 baseline=float(np.nan_to_num(np.percentile(a,50)))
 for lo,hi in intervals:
  left=max(0,lo-start);right=min(len(a),hi-start)
  if left<right:a[left:right]=np.clip(a[left:right],0,baseline)
 a[np.isnan(a)]=baseline
 y=a[1024:-1024].reshape(896,16).sum(1,dtype=np.float32)
 mask=y>100000;y[mask]=99999+np.sqrt(y[mask]-100000+1)
 return np.clip(y,0,65504).astype(np.float16)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--targets',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--out',required=True);ap.add_argument('--blacklist',required=True);ap.add_argument('--gaps',required=True);args=ap.parse_args()
 out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
 tracks=list(csv.DictReader(open(args.targets,encoding='utf-8-sig'),delimiter='\t'))
 contract={'targets_sha256':sha(args.targets),'reference_sha256':sha(args.reference),'blacklist_sha256':sha(args.blacklist),'gaps_sha256':sha(args.gaps),'builder_sha256':sha(__file__),'nuclear_only':True,'seed':44,'input':16384,'bin':16,'crop_bp':1024,'stride':6165,'folds':8,'signal_scale':1,'soft_clip':100000,'global_clip_pct':.9999999,'coordinate_policy':'corrected_output_origin_plus1024','blacklist_slice':'clamped_to_window','claim':'public ChIP method adaptation; not original fold identity/normalization'}
 cp=out/'contract.json'
 if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Existing cache contract differs; use a new output directory')
 save_json(cp,contract);seqs=fasta(args.reference);black=bed(args.blacklist);blocks=fold_blocks(split_blocks(seqs,bed(args.gaps)))
 rows=[];random.seed(44)
 for f in range(8):
  rr=[]
  for c,a,b,ff in blocks:
   if ff!=f:continue
   for s in range(a,b-16384,6165):
    u=bin_mask(c,s,black)
    if u.mean()<.5:rr.append((c,s,s+16384,f))
  random.shuffle(rr);rows.extend(rr)
 if len(set(r[3] for r in rows))!=8:raise ValueError('Empty fold')
 with open(out/'windows.tsv','w',newline='') as h:
  w=csv.writer(h,delimiter='\t');w.writerow(['chrom','start','end','fold']);w.writerows(rows)
 with open(out/'blocks.tsv','w',newline='') as h:
  w=csv.writer(h,delimiter='\t');w.writerow(['chrom','start','end','fold']);w.writerows(blocks)
 lut=np.full(256,4,np.uint8)
 for i,c in enumerate('ACGT'):lut[ord(c)]=i
 x=np.stack([lut[np.frombuffer(seqs[c][s:e].encode(),np.uint8)] for c,s,e,f in rows]);np.save(out/'sequence.npy',x)
 umap=np.stack([bin_mask(c,s,black) for c,s,e,f in rows]);np.save(out/'unmap.npy',umap);np.save(out/'folds.npy',np.array([r[3] for r in rows],np.uint8))
 import pyBigWig
 start=time.monotonic();done=0
 for ti,t in enumerate(tracks):
  p=out/f'track-{ti:03d}.npy';rec=out/f'track-{ti:03d}.json'
  if rec.exists() and p.exists() and json.loads(rec.read_text())['cache_sha256']==sha(p):continue
  if sha(t['file'])!=t['sha256']:raise ValueError('Source hash mismatch '+t['identifier'])
  bw=pyBigWig.open(t['file']);bychrom={};y=np.empty((len(rows),896),np.float16)
  for c in seqs:
   if bw.chroms().get(c)!=len(seqs[c]):raise ValueError('Reference mismatch '+c)
   indices=[i for i,r in enumerate(rows) if r[0]==c]
   vals=bw.values(c,0,len(seqs[c]),numpy=True)
   for i in indices:
    _,s,e,_=rows[i];y[i]=transform(vals[s:e],s,black.get(c,[]),umap[i])
  bw.close();threshold=float(np.percentile(y.astype(np.float64),99.99999));np.minimum(y,threshold,out=y)
  # Source write-time unmappable clipping happens after global target clipping.
  for i in range(len(y)):y[i,umap[i]]=np.minimum(y[i,umap[i]],np.percentile(y[i].astype(np.float64),50))
  if not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid cache')
  with open(str(p)+'.partial','wb') as h:np.save(h,y)
  os.replace(str(p)+'.partial',p)
  save_json(rec,{'identifier':t['identifier'],'source_sha256':t['sha256'],'cache_sha256':sha(p),'shape':list(y.shape),'sum':float(y.sum(dtype=np.float64)),'maximum':float(y.max()),'global_clip_threshold':threshold})
  done+=1;elapsed=time.monotonic()-start
  print(json.dumps({'completed_track':ti+1,'total_tracks':len(tracks),'elapsed_s':elapsed,'estimated_remaining_s':elapsed/done*(len(tracks)-ti-1)}),flush=True)
 yp=out/'targets.npy'
 mm=np.lib.format.open_memmap(str(yp)+'.partial',mode='w+',dtype=np.float16,shape=(len(rows),896,len(tracks)))
 for i in range(len(tracks)):mm[:,:,i]=np.load(out/f'track-{i:03d}.npy')
 mm.flush();del mm;os.replace(str(yp)+'.partial',yp)
 # Cross-fold intervals cannot overlap by construction; check independently.
 for c in seqs:
  rr=sorted([r for r in rows if r[0]==c],key=lambda r:r[1])
  for i,r in enumerate(rr):
   for q in rr[i+1:]:
    if q[1]>=r[2]:break
    if q[3]!=r[3]:raise AssertionError('Cross-fold leakage')
 receipt={'status':'CACHE_READY_PUBLIC_CHIP_ADAPTATION','n_windows':len(rows),'n_tracks':len(tracks),'fold_counts':np.bincount([r[3] for r in rows],minlength=8).tolist(),'shapes':{'sequence':list(x.shape),'targets':[len(rows),896,len(tracks)]},'hashes':{k:sha(out/k) for k in ['contract.json','sequence.npy','targets.npy','folds.npy','windows.tsv']},'runtime_seconds':time.monotonic()-start,'cross_fold_input_overlap':0,'original_normalization_recovered':False,'coordinate_corrections':['unmappable output origin +1024','blacklist intersect slice clamp'],'training_started':False}
 save_json(out/'READY.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
