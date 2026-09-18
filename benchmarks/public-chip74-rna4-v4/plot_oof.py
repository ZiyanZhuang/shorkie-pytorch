import pathlib,json,argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=pathlib.Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
Q=pathlib.Path(parser.parse_args().output);Q.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(P/'per-track-oof.csv');metrics=['Pearson','Spearman','R2']
s=d[d.arm=='scratch'].set_index('track_index');p=d[d.arm=='pretrained'].set_index('track_index')
gain=p[metrics]-s[metrics];gain['modality']=s.modality;gain.to_csv(Q/'paired-gains.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
colors=['#8899AA','#3E739B'];fig,axs=plt.subplots(2,3,figsize=(10,6.8))
lines=['# 合并 OOF 逐轨道分析','', '8折预测按held-out窗口索引合并，再逐轨道计算指标；不是对8折指标求平均。每条轨道包含1855个窗口×896个bin；同折重叠窗口会重复基因组位置。以下均为描述性统计，无独立生物学重复显著性声明。','', '| 模态 | 指标 | scratch均值 | pretrained均值 | 配对平均差 | 改善轨道数 |','|---|---|---:|---:|---:|---:|']
for i,(mod,label) in enumerate([('ChIP','ChIP (74 tracks)'),('nascent_RNA_5prime_tag','Nascent RNA (4 tracks)')]):
 ids=s.index[s.modality==mod]
 for j,m in enumerate(metrics):
  a=axs[i,j];vals=[s.loc[ids,m].mean(),p.loc[ids,m].mean()];a.bar([0,1],vals,color=colors,width=.6)
  a.set_xticks([0,1],['Scratch','Pretrained']);a.set_title(label+' | '+('R²' if m=='R2' else m));a.axhline(0,color='#444444',lw=.6)
  for k,v in enumerate(vals):a.annotate(f'{v:.3f}',(k,v),xytext=(0,5 if v>=0 else -14),textcoords='offset points',ha='center',fontsize=10)
  a.margins(y=.25);g=gain.loc[ids,m];lines.append(f'| {label} | {m} | {vals[0]:.6f} | {vals[1]:.6f} | {g.mean():+.6f} | {(g>0).sum()}/{len(ids)} |')
fig.tight_layout(pad=1.6)
for ext in ['png','pdf','svg']:fig.savefig(Q/f'modality-comparison.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(12,7),sharex=True)
for ax,m in zip(axs,metrics):
 g=gain[m];ax.bar(g.index,g,color=np.where(g>=0,'#3E739B','#BB8176'),width=.8);ax.axhline(0,color='black',lw=.6);ax.axvline(73.5,color='#777777',ls='--',lw=.8);ax.set_ylabel('Δ '+('R²' if m=='R2' else m))
axs[0].set_title('Per-track gain: pretrained minus scratch | ChIP 0–73, RNA 74–77');axs[-1].set_xlabel('Track index (cache order)');axs[-1].set_xticks(list(range(0,74,5))+[74,77]);fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(Q/f'per-track-gains.{ext}',dpi=220,bbox_inches='tight')
lines+=['','## RNA逐轨道','', '| 轨道索引 | scratch Pearson | pretrained Pearson | ΔPearson | scratch R² | pretrained R² |','|---|---:|---:|---:|---:|---:|']
for t in range(74,78):lines.append(f'| {t} | {s.loc[t,"Pearson"]:.5f} | {p.loc[t,"Pearson"]:.5f} | {gain.loc[t,"Pearson"]:+.5f} | {s.loc[t,"R2"]:.5f} | {p.loc[t,"R2"]:.5f} |')
lines+=['','## 解释边界','','RNA是两个WT条件×正负链，不是4个独立条件或生物学重复。ChIP轨道也可能来自相关实验，轨道改善计数不等于独立重复显著性。柱状图为轨道均值，不添加虚构置信区间。','', 'Pearson/Spearman衡量线性/秩相关，R²同时受预测幅度和偏差影响；负R²表示该轨道平方误差大于用整体目标均值预测的基线。不能用相关系数提升代替幅度校准。','', '两分支的学习率和输入通道存在差异，收益应称为当前预训练微调流程相对scratch基线的收益，不能归因为预训练唯一因素。公开数据、划分和normalized-lograte-v4损失具有明确适配差异，不与论文数字做无条件高低结论。','', '后续优先审计下降轨道、低R²轨道的稀疏性及尺度校准，再决定额外训练或算法改动；暂不根据OOF测试分数反复调参。']
(Q/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:13]))
