import pathlib,json,csv,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=pathlib.Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
Q=pathlib.Path(parser.parse_args().output);Q.mkdir(parents=True,exist_ok=True)
s=json.loads((P/'summary.json').read_text())['metrics']
def ours(mod,arm):return s[arm][mod]['Pearson']['median']
rows=[('ChIP','Paper\nChIP-exo',.315,.356,'Figure 3C archived panel values'),('ChIP','Paper\nChIP-MNase',.424,.446,'Figure 3C archived panel values'),('ChIP','Ours\nPublic ChIP (74)',ours('ChIP','scratch'),ours('ChIP','pretrained'),'pooled OOF median'),('RNA','Paper figure\nInduction RNA',.703,.776,'Figure 3C archived panel values'),('RNA','Paper text\nInduction RNA',.67,.78,'manuscript text; conflicts with panel baseline'),('RNA','Paper figure\n1000-strain RNA',.579,.629,'Figure 3C archived panel values'),('RNA','Ours\nNascent RNA (4)',ours('nascent_RNA_5prime_tag','scratch'),ours('nascent_RNA_5prime_tag','pretrained'),'pooled OOF median')]
with open(Q/'comparison-source.csv','w',newline='',encoding='utf8') as f:
 w=csv.writer(f);w.writerow(['modality','group','scratch','pretrained','source']);w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
colors=['#8999A8','#3E739B']
fig,axs=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[3,4]},sharey=True)
for ax,mod in zip(axs,['ChIP','RNA']):
 rr=[r for r in rows if r[0]==mod];x=np.arange(len(rr))
 for j in range(2):
  b=ax.bar(x+(j-.5)*.34,[r[2+j] for r in rr],.34,color=colors[j],label=['Random init / scratch','Pretrained'][j])
  for bar,r in zip(b,rr):
   if r[1].startswith('Ours'):bar.set_hatch('//')
   ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.018,f'{bar.get_height():.3f}',ha='center',fontsize=9)
 ax.set_xticks(x,[r[1] for r in rr]);ax.set_ylim(0,1);ax.set_title(mod+' | median bin-level Pearson');ax.axvline(len(rr)-1.5,color='#aaaaaa',lw=.8,ls='--')
axs[0].set_ylabel('Median Pearson across tracks');axs[0].legend(loc='upper left',fontsize=9)
fig.text(.5,.02,'Different assays, track sets and splits: descriptive reference, not a matched benchmark.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.08,1,1])
for ext in ['png','pdf','svg']:fig.savefig(Q/f'paper-reference.{ext}',dpi=250,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.5));g=np.array([r[3]-r[2] for r in rows]);x=np.arange(len(rows));b=ax.bar(x,g,color=['#3E739B' if r[1].startswith('Ours') else '#8999A8' for r in rows])
for i,v in enumerate(g):ax.text(i,v+.003,f'+{v:.3f}',ha='center',fontsize=10)
ax.set_xticks(x,[r[1] for r in rows]);ax.set_ylabel('Pretrained median minus scratch median');ax.set_ylim(0,.14);ax.set_title('Within-study pretraining gain (different protocols)');fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(Q/f'pretraining-gain-reference.{ext}',dpi=250,bbox_inches='tight')
print(json.dumps({'chip_delta_vs_exo':ours('ChIP','pretrained')-.356,'rna_delta_vs_induction_figure':ours('nascent_RNA_5prime_tag','pretrained')-.776,'rna_delta_vs_strains':ours('nascent_RNA_5prime_tag','pretrained')-.629,'chip_gain':g[2],'rna_gain':g[-1]}))
