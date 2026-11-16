import json,glob,sys,statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Noto Sans CJK JP"
cells=defaultdict(list)
for p in glob.glob(sys.argv[1]+"/runs/*.json"):
    d=json.load(open(p)); cells[(d["arm"]["D"],round(d["spec"]["rate"],2))]+=[r for s in d["segments"] for r in s["requests"] if r.get("token_itl_ms") and r.get("ttft_ms") is not None]
INK,MUTE,GRID="#1f1f1e","#6b6a64","#e4e3dd"; C1,C2="#2a78d6","#eb6834"
rates=[0.1,0.34,0.51,0.76]; Ds=[16,24,34,44,64]
fig,axes=plt.subplots(1,4,figsize=(13,4.3),sharey=True,facecolor="white")
for ax,rt in zip(axes,rates):
    for i,D in enumerate(Ds):
        rq=cells[(D,rt)]; t=st.fmean(r["ttft_ms"] for r in rq)/1000; g=st.fmean(sum(r["token_itl_ms"]) for r in rq)/1000
        ax.barh(i,t,color=C1,height=0.62,label="첫 토큰까지 (대기 + prefill)" if i==0 else None)
        ax.barh(i,g,left=t+0.12,color=C2,height=0.62,label="첫 토큰 뒤 생성" if i==0 else None)
        ax.text(t+g+0.5,i,f"{t+g:.1f}",va="center",fontsize=9.5,color=INK)
        ax.text(max(t/2,0.1),i,f"{t:.1f}",va="center",ha="center",fontsize=8.5,color="white") if t>1.6 else None
    ax.set_title(f"도착률 {rt:.2f} req/s",fontsize=11,color=INK,loc="left")
    ax.set_yticks(range(len(Ds))); ax.set_yticklabels([f"D{D} (prefill {108-D} SM)" for D in Ds],fontsize=9.5,color=INK)
    ax.invert_yaxis(); ax.set_xlim(0,41); ax.grid(axis="x",color=GRID,lw=0.8); ax.set_axisbelow(True)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTE,length=0,labelsize=9)
    ax.set_xlabel("요청당 평균 시간 (초)",fontsize=9.5,color=MUTE)
axes[0].legend(loc="upper center",bbox_to_anchor=(2.2,-0.2),ncol=2,frameon=False,fontsize=10)
fig.suptitle("decode 몫(D)을 바꾸면 시간이 어디로 옮겨 가는가",x=0.012,ha="left",fontsize=14,fontweight="bold",color=INK,y=0.99)
fig.text(0.012,0.905,"긴 입력(중앙값 6,469토큰) · 출력 1,024토큰 고정 · 2026-10-29 용량 사다리, seed 2개 합산, 버스트 셀 · 막대 끝 숫자 = 완료 시각",fontsize=9.5,color=MUTE)
fig.subplots_adjust(top=0.80,left=0.115,right=0.99,bottom=0.25,wspace=0.06)
fig.savefig(sys.argv[2],dpi=150)
