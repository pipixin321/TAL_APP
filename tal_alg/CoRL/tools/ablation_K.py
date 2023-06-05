import os
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
from pylab import mpl
mpl.rcParams['font.size'] = 14
plt.switch_backend('agg')

mAP_results=[53.6,53.7,53.6,54.6,54.8,54.0,53.6,53.9,53.8]
vid_acc=[88.6,89.1,89.5,91.4,92.4,91.0,90.0,90.3,90.2]
N=len(mAP_results)

fig,ax1=plt.subplots(figsize=(7,3))
plt.grid(axis='y')

plt.plot(range(1,N+1),mAP_results,'.-',markersize=10,color='blue',label='mAP')
ax1.set_xlim(0,len(mAP_results)+1)
ax1.set_ylim(53,56)
plt.xticks(range(0,N+1),[str(i) for i in range(N+1)])
ax1.set_xlabel('K')
ax1.set_ylabel('mAP AVG(0.1:0.7) (%)')
for i in range(N):
    plt.text(i+1-0.3,mAP_results[i]+0.1,str(mAP_results[i]))
plt.legend(loc='upper left')

ax2=ax1.twinx()
plt.plot(range(1,N+1),vid_acc,'.-',markersize=10,color='orange',label='vid acc')
ax2.set_ylabel('video acc (%)')
ax2.set_ylim(85,94)
for i in range(N):
    plt.text(i+1-0.3,vid_acc[i]+0.2,str(vid_acc[i]))

plt.legend(loc='upper right')
plt.savefig(os.path.join("./tools/figs","ablation_K_curve.png"),bbox_inches="tight",dpi=400)
plt.savefig(os.path.join("./tools/figs","ablation_K_curve.pdf"),bbox_inches="tight",dpi=400)