import json,glob,sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"]=False
S="/home/wonho/Experiments/KISTI/_vessl_fetch/pdmux-motivation-pre-ladder-938c3be-20261113032517/results/motivation_pre_ladder_runs/pdmux-motivation-pre-ladder-938c3be-20261113032517/raw/runs/*arxiv_summ*r0.50625_s20261122.json"
RID="arxiv_summ-s20261122-00010"
runs={}
for p in glob.glob(S):
    d=json.load(open(p)); arm="vLLM" if d["engine"]=="vllm" else ("PD-mux" if "bundle" in d["arm"] else "SG mixed")
    runs[arm]=[r for s in d["segments"] for r in s["requests"]]
INK,MUTE,GRID,BAND="#1f1f1e","#6b6a64","#e4e3dd","#d9d8d0"
COL={"vLLM":"#2a78d6","SG mixed":"#eb6834","PD-mux":"#1baf7a"}
fig,axes=plt.subplots(3,1,figsize=(10,8.4),sharex=True,sharey=True,facecolor="white")
XMAX=9.0
for ax,arm in zip(axes,("vLLM","SG mixed","PD-mux")):
    rq=runs[arm]; me=next(r for r in rq if r["request_id"]==RID); t0=me["t_sent_mono"]
    for r in rq:
        if r is me or r.get("ttft_ms") is None: continue
        a=r["t_sent_mono"]-t0; b=a+r["ttft_ms"]/1000
        if b>0 and a<XMAX: ax.axvspan(max(a,0),min(b,XMAX),color=BAND,lw=0,zorder=0)
    t=[x-t0 for x in me["token_t_mono"]]; n=list(range(1,len(t)+1))
    ax.step([0]+t,[0]+n,where="post",color=COL[arm],lw=2,zorder=3)
    ax.plot([t[0]],[1],"o",ms=7,color=COL[arm],mec="white",mew=1.5,zorder=4)
    itl=me["token_itl_ms"]; gaps=[(t[i],itl[i]) for i in range(len(itl)) if itl[i]>100]
    tot=sum(g for _,g in gaps)
    ax.annotate(f"첫 토큰 {me['ttft_ms']/1000:.2f} s",(t[0],1),xytext=(t[0]+0.08,-4),textcoords="data",fontsize=9.5,color=INK,va="top")
    ax.text(0.012,0.93,arm,transform=ax.transAxes,fontsize=13,fontweight="bold",color=INK,va="top")
    msg=(f"100 ms 넘는 멈춤 {len(gaps)}번, 한 번에 약 {round(sum(g for _,g in gaps)/len(gaps))} ms, 합계 {tot/1000:.1f} s" if gaps else f"100 ms 넘는 멈춤 없음 (가장 긴 간격 {max(itl):.0f} ms)")
    ax.text(0.012,0.80,msg+f"\n마지막 토큰 {t[-1]:.1f} s",transform=ax.transAxes,fontsize=10,color=MUTE,va="top",linespacing=1.5)
    if gaps:
        gt,gl=max(gaps,key=lambda g:g[1]); k=next(i for i in range(len(itl)) if t[i]==gt)
        ax.annotate("멈춤: 토큰 수가 그대로인 평평한 구간",(gt+gl/2000,k+1),xytext=(gt+0.7,k-38),fontsize=9.5,color=INK,
                    arrowprops=dict(arrowstyle="-",color=MUTE,lw=0.8))
    ax.set_xlim(0,XMAX); ax.set_ylim(-14,len(t)+8)
    ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTE,length=0,labelsize=9.5)
    ax.set_ylabel("받은 토큰 수",color=MUTE,fontsize=10)
axes[-1].set_xlabel("이 요청을 보낸 뒤 지난 시간 (초)",color=MUTE,fontsize=10)
axes[0].text(0.988,0.07,"회색 띠 = 다른 요청이 첫 토큰을 기다리는 구간(대기 + prefill)",transform=axes[0].transAxes,fontsize=9.5,color=MUTE,ha="right",
             bbox=dict(fc="white",ec="none",pad=2))
fig.suptitle("같은 요청 하나가 세 구성에서 토큰을 받는 모습",x=0.065,ha="left",fontsize=15,fontweight="bold",color=INK,y=0.985)
fig.text(0.065,0.94,f"arXiv 요약, 입력 {me['input_tokens']:,}토큰 · 출력 {len(t)}토큰, 도착률 0.51 req/s · 사전 사다리 자료(격자 선택용, 인용 불가)",fontsize=10,color=MUTE)
fig.subplots_adjust(top=0.90,left=0.065,right=0.985,bottom=0.07,hspace=0.12)
fig.savefig(sys.argv[1],dpi=150)
