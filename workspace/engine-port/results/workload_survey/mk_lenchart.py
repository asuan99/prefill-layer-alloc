import math
# group, paper, workload, lo, typ, hi, approx(bool)
R=[
("pd","MuxWise (ASPLOS'26)","ShareGPT",4,226,1024,0),
("pd","MuxWise (ASPLOS'26)","LooGLE",3380,30000,81000,0),
("pd","MuxWise (ASPLOS'26)","OpenThoughts (추론)",311,709,4633,0),
("pd","MuxWise (ASPLOS'26)","Mooncake Conversation",891,7538,123000,0),
("pd","MuxWise (ASPLOS'26)","Mooncake Tool&Agent",891,8596,123000,0),
("pd","Bullet (ASPLOS'26)","ShareGPT · Azure-Code · arXiv-Sum",200,2000,32000,1),
("pd","DuetServe (ICML'26)","Azure-Code / Azure-Conv",None,2047,None,0),
("pd","DuetServe (ICML'26)","Mooncake Conversation (1k 샘플)",None,12035,None,0),
("pd","Nexus (arXiv 2507.06608)","ShareGPT / ArXiv-Sum",432,3832,6894,0),
("pd","Nexus (arXiv 2507.06608)","Long Data Collections",5461,5905,9817,0),
("gen","DistServe (OSDI'24)","ShareGPT · HumanEval · LongBench(≤2048)",50,800,2048,1),
("gen","Sarathi-Serve (OSDI'24)","sharegpt4 (≤8k)",None,1730,8192,0),
("gen","Sarathi-Serve (OSDI'24)","arxiv_summarization (≤16k)",None,7059,16384,0),
("gen","Mooncake (FAST'25)","Kimi Conversation",None,12035,126195,0),
("lc","LoongServe (SOSP'24)","L-Eval",2700,None,210500,0),
("lc","LoongServe (SOSP'24)","LV-Eval",15100,None,497300,0),
("lc","Medha (arXiv 2409.17264)","Medha-SWE (자체 구축)",None,393000,839000,0),
("lc","Medha (arXiv 2409.17264)","혼합 long 5% (최대 10M)",128000,None,10000000,1),
("lc","Vertumnus (arXiv 2609.04774)","L-Eval / Tool-Agent / Production",None,8596,81724,0),
("lc","P-PAS (arXiv 2608.15171)","고정 20k–30k / 출력 32",20000,25000,30000,0),
("lc","Context Parallelism (MLSys'25)","고정 128k–1M",128000,None,1000000,0),
("hy","Marconi (MLSys'25)","LMSys · ShareGPT · SWE-bench",300,5000,30000,1),
("hy","Jenga (arXiv 2503.18292)","arXiv-QA (Ministral 평균 92k)",55000,92408,110000,0),
("hy","SQD (arXiv 2609.13134)","Chat 32k / RAG 256k / Agentic 1M",32000,None,1000000,0),
("us","이 프로젝트 (제안)","Mooncake Conversation × Nano-9B-v2",100,6909,126195,0),
]
COL={"pd":"var(--g-pd)","gen":"var(--g-gen)","lc":"var(--g-lc)","hy":"var(--g-hy)","us":"var(--g-us)"}
W=1000; L=300; Rr=975; top=64; rh=22
XMIN=100
def x(v): v=max(v,XMIN); return L+(math.log10(v)-2)/(7-2)*(Rr-L)
D=[]; prev=None
for r in R:
    if r[1]!=prev: D.append(("H",r)); prev=r[1]
    D.append(("W",r))
n=len(D); bot=top+rh*n
H=bot+52
o=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="선행 연구별 평가 워크로드의 입력 길이 범위(로그 축). PD-mux 계열은 1k–120k, long-context 계열은 100k–10M에 분포한다.">']
# plot background for label column separation
o.append(f'<rect x="{L}" y="{top}" width="{Rr-L}" height="{bot-top}" class="plotbg"/>')
for t,lab in [(100,"100"),(1000,"1k"),(10000,"10k"),(100000,"100k"),(1000000,"1M"),(10000000,"10M")]:
    xx=x(t); o.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bot}" class="grid"/>')
    o.append(f'<text x="{xx:.1f}" y="{top-6}" text-anchor="middle" class="axl">{lab}</text>')
    o.append(f'<text x="{xx:.1f}" y="{bot+16}" text-anchor="middle" class="axl">{lab}</text>')
# reference lines + labels in header band (two lanes, no overlap with rows)
xc=x(8600); xl=x(131072)
o.append(f'<line x1="{xc:.1f}" y1="{top-30}" x2="{xc:.1f}" y2="{bot}" class="xover"/>')
o.append(f'<text x="{xc-6:.1f}" y="{top-34}" text-anchor="end" class="ann us">KV = state 경계 ≈8.6k</text>')
o.append(f'<line x1="{xl:.1f}" y1="{top-48}" x2="{xl:.1f}" y2="{bot}" class="lim"/>')
o.append(f'<text x="{xl+6:.1f}" y="{top-38}" class="ann lm">Nano-9B-v2 최대 131k</text>')
o.append(f'<text x="{L}" y="{top-48}" class="axl">입력 토큰 수 (로그 축)</text>')
for i,(kind,(g,p,w,lo,typ,hi,ap)) in enumerate(D):
    y=top+i*rh+rh/2
    c=COL[g]
    if kind=="H":
        if i>0: o.append(f'<line x1="8" y1="{y-rh/2:.1f}" x2="{Rr}" y2="{y-rh/2:.1f}" class="sep"/>')
        o.append(f'<rect x="8" y="{y-6:.1f}" width="11" height="11" rx="2" fill="{c}"/>')
        o.append(f'<text x="26" y="{y+5:.1f}" class="pp">{p}</text>')
        continue
    o.append(f'<text x="{L-12}" y="{y+5:.1f}" text-anchor="end" class="wl">{w}</text>')
    if lo is not None or hi is not None:
        a=lo if lo else typ; b=hi if hi else typ
        dash='stroke-dasharray="5 4"' if ap else ''
        o.append(f'<line x1="{x(a):.1f}" y1="{y:.1f}" x2="{x(b):.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="6" stroke-linecap="round" {dash} opacity=".8"/>')
        if a<XMIN:
            o.append(f'<path d="M{L+3} {y:.1f} l7 -5 v10 z" fill="{c}"/>')
    if typ: o.append(f'<circle cx="{x(typ):.1f}" cy="{y:.1f}" r="5" fill="{c}" class="dot"/>')
o.append(f'<text x="{L}" y="{bot+38}" class="axl">막대 = 보고된 최소(또는 중앙값)~최대 · 점 = 평균 또는 중앙값 · 점선 = 그림에서 읽은 근사 · ◀ = 100 미만에서 시작</text>')
o.append('</svg>')
open("/tmp/claude-1000/-home-wonho-Experiments-KISTI/4d722672-c9b9-4d1e-b32c-8d68a59299bc/scratchpad/survey/lenchart.svg","w").write("\n".join(o))
print(H)
