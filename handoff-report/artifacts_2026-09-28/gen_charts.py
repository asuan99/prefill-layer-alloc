import sys,re,math
p=sys.argv[1]; s=open(p).read()
a=s.index('<h3 style="margin-top:22px">지금까지 관측된 것'); b=s.index('<h2>5. 어떤 요청이')
W,H=360,268; L=56; T=44; BASE_B=H-78
def svg_open(vb,label): return f'<svg viewBox="0 0 {vb}" role="img" aria-label="{label}" style="max-width:100%;height:auto;font-family:var(--sans);font-size:11.5px;color:var(--ink)">'
AH='<defs><marker id="ahH" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>'
def frame(title,yl):
    R=W-14;B=BASE_B
    return [f'<text x="{W/2:.0f}" y="15" text-anchor="middle" font-weight="700" font-size="12">{title}</text>',f'<text x="{L}" y="{T-10}" font-size="10" fill="var(--ink-2)">{yl}</text>',f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="currentColor"/><line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="currentColor"/>'],R,B
def yticks(o,R,B,ymin,ymax,step,fmt):
    v=ymin
    while v<=ymax+1e-9:
        y=B-(B-T)*(v-ymin)/(ymax-ymin); o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="currentColor" opacity=".12"/><text x="{L-4}" y="{y+3.5:.1f}" text-anchor="end" font-size="9.5" fill="var(--ink-3)">{fmt.format(v)}</text>'); v+=step
def cats(o,R,B,labels):
    n=len(labels)
    for i,lab in enumerate(labels):
        x=L+(R-L)*(i+0.5)/n; ls=lab.split('\n')
        o.append(f'<text x="{x:.1f}" y="{B+13}" text-anchor="middle" font-size="9.5" fill="var(--ink-2)">{ls[0]}</text>')
        if len(ls)>1: o.append(f'<text x="{x:.1f}" y="{B+24}" text-anchor="middle" font-size="9" fill="var(--ink-3)">{ls[1]}</text>')
def xtitle(o,R,B,t): o.append(f'<text x="{(L+R)/2:.0f}" y="{B+38}" text-anchor="middle" font-size="10" fill="var(--ink-2)">{t}</text>')
def scope(o,lines):
    for i,l in enumerate(lines): o.append(f'<text x="{W/2:.0f}" y="{H-20+i*11}" text-anchor="middle" fill="var(--ink-3)" font-size="9.3">{l}</text>')
def X(R,i,n): return L+(R-L)*(i+0.5)/n
def Y(B,v,ymin,ymax): return B-(B-T)*(v-ymin)/(ymax-ymin)
def bars(o,R,B,data,ymin,ymax,fmt,bw=34):
    n=len(data)
    for i,(v,sd,c) in enumerate(data):
        x=X(R,i,n); y=Y(B,v,ymin,ymax); o.append(f'<rect x="{x-bw/2:.1f}" y="{y:.1f}" width="{bw}" height="{B-y:.1f}" fill="{c}" opacity=".8"/>'); top=y
        if sd is not None:
            y1=Y(B,v+sd,ymin,ymax); y2=Y(B,v-sd,ymin,ymax); o.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="currentColor"/><line x1="{x-4:.1f}" y1="{y1:.1f}" x2="{x+4:.1f}" y2="{y1:.1f}" stroke="currentColor"/><line x1="{x-4:.1f}" y1="{y2:.1f}" x2="{x+4:.1f}" y2="{y2:.1f}" stroke="currentColor"/>'); top=min(y,y1)
        o.append(f'<text x="{x:.1f}" y="{top-5:.1f}" text-anchor="middle" font-size="9.5" font-weight="600">{fmt.format(v)}</text>')
G='var(--ok)';P='var(--gate)'
o,R,B=frame('prefill: 층 종류별 SM 민감도 비 (Diff B)','attention ÷ SSM'); ymin,ymax=0.8,1.5; yticks(o,R,B,ymin,ymax,0.1,'{:.1f}'); n=6
o.append(f'<line x1="{L}" y1="{Y(B,1.0,ymin,ymax):.1f}" x2="{R}" y2="{Y(B,1.0,ymin,ymax):.1f}" stroke="var(--gate)" stroke-dasharray="4 3"/><text x="{L+6}" y="{Y(B,1.0,ymin,ymax)+13:.1f}" fill="var(--gate)" font-size="9.5">1.0 = 재배분 레버 없음</text>')
o.append(f'<rect x="{X(R,0,n)-8:.1f}" y="{Y(B,1.34,ymin,ymax):.1f}" width="16" height="{Y(B,1.24,ymin,ymax)-Y(B,1.34,ymin,ymax):.1f}" fill="var(--accent)" opacity=".35"/><text x="{X(R,0,n)+12:.1f}" y="{Y(B,1.34,ymin,ymax)-4:.1f}" font-size="9.5" fill="var(--accent)">1.24–1.34</text>')
o.append(f'<circle cx="{X(R,1,n):.1f}" cy="{Y(B,1.198,ymin,ymax):.1f}" r="4" fill="var(--accent)"/><text x="{X(R,1,n)+7:.1f}" y="{Y(B,1.198,ymin,ymax)-5:.1f}" font-size="9.5" fill="var(--accent)">1.20</text>')
for i in range(2,6): o.append(f'<circle cx="{X(R,i,n):.1f}" cy="{Y(B,1.0,ymin,ymax):.1f}" r="4" fill="var(--accent)"/>')
o.append(f'<rect x="{X(R,5,n)-8:.1f}" y="{Y(B,1.04,ymin,ymax):.1f}" width="16" height="{Y(B,0.96,ymin,ymax)-Y(B,1.04,ymin,ymax):.1f}" fill="var(--accent)" opacity=".25"/><text x="{X(R,4,n):.1f}" y="{Y(B,1.0,ymin,ymax)-9:.1f}" text-anchor="middle" font-size="9.5" fill="var(--accent)">≈1.0 (≥8000: 0.96–1.04)</text>')
cats(o,R,B,['256','512','1024','2000','4000','8000']); xtitle(o,R,B,'입력 길이 L (토큰)'); scope(o,['KISTI A100 · Zamba2-2.7B · triton · CUDA graph OFF 마이크로 · 독립 측정 1회','NemotronH 계열로 옮겨 쓰지 않음'])
ca=svg_open(f'{W} {H}','prefill에서 attention 층과 SSM 층의 SM 민감도 비. 입력 256 토큰에서 1.24–1.34, 512에서 1.20, 1024 이상에서 약 1.0으로 층 종류별 재배분 레버가 사라진다.')+AH+"".join(o)+'</svg>'
o,R,B=frame('decode: SM 16 → 92일 때 토큰 간격 감소 비 (8B급)','SM16 ÷ SM92'); ymin,ymax=1.0,3.5; yticks(o,R,B,ymin,ymax,0.5,'{:.1f}')
arms=[('Qwen2.5-7B\ntransformer · 점추정',2.388,None,None),('Nemotron-H 8B\nhybrid · 점추정',2.687,None,None),('Codestral-7B\npure SSM · n=6',3.058,3.056,3.060),('Zamba2-7B\nhybrid · n=6',3.114,3.077,3.155)]
for i,(lab,v,lo,hi) in enumerate(arms):
    x=X(R,i,4)
    if lo: o.append(f'<line x1="{x:.1f}" y1="{Y(B,lo,ymin,ymax):.1f}" x2="{x:.1f}" y2="{Y(B,hi,ymin,ymax):.1f}" stroke="var(--ok)" stroke-width="2"/>')
    o.append(f'<circle cx="{x:.1f}" cy="{Y(B,v,ymin,ymax):.1f}" r="5" fill="var(--ok)"/><text x="{x:.1f}" y="{Y(B,v,ymin,ymax)-11:.1f}" text-anchor="middle" font-size="10" fill="var(--ok)" font-weight="600">{v:.2f}×</text>')
cats(o,R,B,[a_[0] for a_ in arms]); xtitle(o,R,B,'모델 (prefill 16 SM 고정 · decode 배치 16 · ITL p50)'); scope(o,['KISTI A100 · 레버 존재만 확립','모델 간 순서·격차는 주장하지 않음(인용 정지)'])
cb=svg_open(f'{W} {H}','8B급 모델 4종에서 decode SM을 16에서 92로 늘렸을 때 토큰 간격이 줄어드는 비율. transformer 2.39, Nemotron-H 2.69, pure SSM 3.06, Zamba2 3.11. 모델 간 순서는 주장하지 않는다.')+AH+"".join(o)+'</svg>'
o,R,B=frame('층 종류별 재배분의 실행 비용 (decode 지연)','TPOT (ms)'); ymin,ymax=0,140; yticks(o,R,B,ymin,ymax,20,'{:.0f}'); bars(o,R,B,[(42,None,G),(124,None,P),(85,None,'var(--warn)')],ymin,ymax,'{:g} ms',bw=48); cats(o,R,B,['phase 전체 단일 배분\nagnostic','층 종류별 재배분\ncoordinated','재배분 + 드레인 최적화\nOPT']); scope(o,['KISTI A100 · Zamba2 · 양쪽 모두 CUDA graph OFF(eager)','크기보다 방향으로 인용'])
cc=svg_open(f'{W} {H}','층 종류별로 SM을 재배분하는 방식의 decode 지연. phase 전체 단일 배분 42 ms, 층 종류별 재배분 124 ms, 드레인 최적화 후 85 ms.')+AH+"".join(o)+'</svg>'
def he0(title,yl,data,labels,ymin,ymax,step,fmt,scopes,aria):
    o,R,B=frame(title,yl); yticks(o,R,B,ymin,ymax,step,fmt.split('|')[0]); bars(o,R,B,data,ymin,ymax,fmt.split('|')[1],bw=30); cats(o,R,B,labels); scope(o,scopes); return svg_open(f'{W} {H}',aria)+AH+"".join(o)+'</svg>'
cd1=he0('실시간 조절 vs 고정 배분 — 느슨한 SLO','goodput (req/s)',[(2.846,0.055,G),(3.039,0.130,G),(3.171,0.025,G),(3.220,0.013,G),(3.132,0.019,P),(2.934,0.306,P)],['D16\n고정','D24\n고정','D34\n고정','D44\n고정','조절\n+게이트','조절'],2.4,3.6,0.2,'{:.1f}|{:.2f}',['KISTI A100 · Zamba2-2.7B · 부하 변화 trace · n≥4(조절+게이트 n=9) · 평균±SD','TTFT≤3 s 기준 · D44 − 조절+게이트 = 0.088 (5.4σ)'],'느슨한 SLO에서 고정 배분 D16 2.85, D24 3.04, D34 3.17, D44 3.22 req/s 대 실시간 조절 3.13(게이트 있음)과 2.93(없음). 고정 D44가 가장 높다.')
cd2=he0('실시간 조절 vs 고정 배분 — 빡빡한 SLO','목표 달성률 (%)',[(33,None,G),(41,None,G),(49.6,3.9,G),(73.2,4.8,G),(44.3,2.7,P),(40.6,0.4,P)],['D16\n고정','D24\n고정','D34\n고정','D44\n고정','조절\n+게이트','조절'],0,100,20,'{:.0f}|{:g}%',['KISTI A100 · Zamba2-2.7B · 채팅 SLO(TTFT 300 ms / ITL 50 ms)로 재조율 · n=4','도착률 8/s(용량 경계) · D44 − 조절+게이트 = 28.9%p (≈10σ)'],'빡빡한 SLO에서 목표 달성률: 고정 D16 33%, D24 41%, D34 49.6%, D44 73.2%, 실시간 조절 44.3%(게이트)와 40.6%. 고정 D44가 가장 높다.')
o,R,B=frame('분할 서빙(PD-mux) vs 기본 서빙 — goodput 이득','이득 (%)'); ymin,ymax=0,200; yticks(o,R,B,ymin,ymax,50,'{:.0f}'); bars(o,R,B,[(40.5,None,'var(--accent)'),(185.8,None,'var(--accent)'),(27.0,None,'var(--accent)')],ymin,ymax,'+{:g}%',bw=48); cats(o,R,B,['Zamba2\n부하 2/s','Zamba2\n부하 3/s','Granite-4\n부하 4/s']); scope(o,['KISTI A100 · CUDA graph ON · n=5 짝지음 · 인용 가능한 값 3개뿐','Granite 3/s(+11.7%)는 CI가 0을 포함해 미검증 · 원인 귀속 없음'])
ce=svg_open(f'{W} {H}','분할 서빙을 켰을 때 기본 서빙 대비 goodput 이득: Zamba2 부하 2/s에서 40.5%, 3/s에서 185.8%, Granite-4 부하 4/s에서 27.0%. 운영 설정에서 측정된 값 3개뿐이다.')+AH+"".join(o)+'</svg>'
o,R,B=frame('처리 한계와 동거 비율 — 두 워크로드','처리 한계 (req/s)'); ymin,ymax=0,4; yticks(o,R,B,ymin,ymax,1,'{:.0f}'); bars(o,R,B,[(3.05,None,G),(0.696,None,G)],ymin,ymax,'{:g} req/s',bw=60)
for i,res in enumerate(['동거 3–10%','동거 43–98%']): x=X(R,i,2); v=[3.05,0.696][i]; o.append(f'<text x="{x:.1f}" y="{Y(B,v,ymin,ymax)-20:.1f}" text-anchor="middle" font-size="10" fill="var(--gate)">{res}</text>')
cats(o,R,B,['채팅형 A\n입력 256 · 출력 512','문서 요약형 B\n입력 8192 · 출력 64']); scope(o,['KISTI A100 · Nano-9B-v2 · 고정 D44 설정 · n=2 seed · 소수점 인용 금지','동거 비율은 가중 규약에 따라 달라짐 · VESSL에서 재측정(3-b)'])
cf=svg_open(f'{W} {H}','처리 한계: 채팅형 shape A 약 3.05 req/s, 문서 요약형 shape B 약 0.70 req/s. 동거 비율은 A에서 3–10%, B에서 43–98%.')+AH+"".join(o)+'</svg>'
o,R,B=frame('prefill 조각 수 — 입력 길이와 예산에 따라','조각 수'); ymin,ymax=0,60; yticks(o,R,B,ymin,ymax,10,'{:.0f}')
def xl_(v): return L+(R-L)*(math.log2(v)-7)/(14-7)
for v in [128,256,512,1024,2048,4096,8192,16384]: o.append(f'<text x="{xl_(v):.1f}" y="{B+13}" text-anchor="middle" font-size="9.5" fill="var(--ink-2)">{v if v<1000 else str(v//1024)+"K"}</text>')
for bud,c in ((65536,'var(--accent)'),(8192,'var(--warn)'),(2048,'var(--gate)')):
    pts=[f'{xl_(v):.1f},{B-(B-T)*min(max(1,math.ceil(56/max(1,bud//v))),60)/60:.1f}' for v in range(128,16385,64)]; o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2"/>')
o.append(f'<text x="{L+6}" y="{T+12}" font-size="9.5" fill="var(--gate)">예산 2048</text><text x="{L+6}" y="{T+24}" font-size="9.5" fill="var(--warn)">예산 8192</text><text x="{L+6}" y="{T+36}" font-size="9.5" fill="var(--accent)">예산 65536 (기본)</text>')
o.append(f'<line x1="{xl_(1170):.1f}" y1="{T}" x2="{xl_(1170):.1f}" y2="{B}" stroke="currentColor" stroke-dasharray="3 3" opacity=".5"/><text x="{xl_(1170)+4:.1f}" y="{B-13}" font-size="9.5" fill="var(--ink-2)">1170 미만: 조각 1개</text>')
xtitle(o,R,B,'입력 길이 (토큰, 로그축)'); scope(o,['엔진 규칙 ceil(56 ÷ max(1, 예산 ÷ 입력)) — 계산값(예보), 56층 Nano-9B-v2','옛 자료 재집계: 채팅형 전부 1개, 문서 요약형 7 또는 14개'])
cg=svg_open(f'{W} {H}','prefill 조각 수. 기본 예산 65536에서는 입력 약 1170 토큰 아래가 조각 1개이고, 8192 토큰에서 7개, 16K에서 14개. 예산을 2048로 줄이면 짧은 입력도 여러 조각이 된다.')+AH+"".join(o)+'</svg>'
o=[f'<text x="{W/2:.0f}" y="15" text-anchor="middle" font-weight="700" font-size="12">decode 속도 vs 문맥 길이 × SM 수</text>',f'<rect x="{L}" y="{T}" width="{W-14-L}" height="{BASE_B-T}" fill="none" stroke="var(--gate)" stroke-dasharray="5 4" rx="6"/>',f'<text x="{W/2:.0f}" y="{T+50}" text-anchor="middle" font-size="12" fill="var(--gate)" font-weight="700">쓸 수 있는 측정이 없음</text>',f'<text x="{W/2:.0f}" y="{T+72}" text-anchor="middle" font-size="10.5" fill="var(--ink-2)">옛 값은 계측 결함으로 인용 금지,</text>',f'<text x="{W/2:.0f}" y="{T+86}" text-anchor="middle" font-size="10.5" fill="var(--ink-2)">운영 설정(CUDA graph ON) 측정 자체가 없음</text>',f'<text x="{W/2:.0f}" y="{T+112}" text-anchor="middle" font-size="11" fill="var(--accent)" font-weight="600">→ 3-a가 VESSL에서 처음 잰다</text>',f'<text x="{W/2:.0f}" y="{T+130}" text-anchor="middle" font-size="10" fill="var(--ink-3)">hybrid 특성이 시스템에 나타나는 핵심 곡선</text>']
ch=svg_open(f'{W} {H}','decode 속도가 문맥 길이와 SM 수에 따라 어떻게 변하는지는 쓸 수 있는 측정이 없다. 옛 값은 계측 결함으로 인용 금지이며 운영 설정 측정은 3-a가 처음 잰다.')+AH+"".join(o)+'</svg>'
old=s[a:b]; new=old
blocks=re.findall(r'<svg viewBox="0 0 360 \d+"[\s\S]*?</svg>',old); assert len(blocks)==9, len(blocks)
for blk,svg in zip(blocks,(ca,cb,cc,cd1,cd2,ce,cf,cg,ch)): new=new.replace(blk,svg,1)
s=s[:a]+new+s[b:]; open(p,'w').write(s)
import xml.etree.ElementTree as ET
for x in (ca,cb,cc,cd1,cd2,ce,cf,cg,ch): ET.fromstring(x)
css=re.search(r'<style>(.*?)</style>',s,re.S).group(1)
open(p.rsplit('/',1)[0]+'/figH_check.html','w').write(f'<!doctype html><html><head><meta charset="utf-8"><style>{css} body{{padding:16px;width:1000px}}</style></head><body><div class="wrap">{new}</div></body></html>')
print('ok')
