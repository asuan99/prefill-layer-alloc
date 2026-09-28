import sys,re,math
S=sys.argv[1]
rp=f'{S}/pdmux_roadmap.html'; vp=f'{S}/pdmux_review.html'
r=open(rp).read(); v=open(vp).read()
AH='<defs><marker id="ahH1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>'
def svg_open(vb,label): return f'<svg viewBox="0 0 {vb}" role="img" aria-label="{label}" style="max-width:100%;height:auto;font-family:var(--sans);font-size:11.5px;color:var(--ink)">'
W,H=960,300
def frame(x0,pw,title,xl,yl):
    L=x0+52;R=x0+pw-12;T=44;B=H-70
    o=[f'<text x="{x0+pw/2:.0f}" y="16" text-anchor="middle" font-weight="700" font-size="12">{title}</text>',f'<text x="{L}" y="{T-10}" font-size="10" fill="var(--ink-2)">{yl}</text>',f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="currentColor"/><line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="currentColor"/>',f'<text x="{(L+R)/2:.0f}" y="{B+30}" text-anchor="middle" font-size="10" fill="var(--ink-2)">{xl}</text>']
    return o,L,R,T,B
Ls=[256,512,1024,2048,4096,8192,16384,32768]; attn=[0.093,0.164,0.429,1.363,4.789,17.686,69.270,273.449]; mamba=[0.930,0.887,0.991,1.779,3.413,6.744,13.199,24.949]
o,L,R,T,B=frame(0,320,'prefill: 층당 시간의 증가 (실측)','입력 길이 L (토큰, 로그축)','L=256 대비 배율 (로그축)')
xmin,xmax=math.log2(256),math.log2(32768); ymin,ymax=0,math.log10(4000)
X=lambda q: L+(R-L)*(math.log2(q)-xmin)/(xmax-xmin); Y=lambda q: B-(B-T)*(math.log10(q)-ymin)/(ymax-ymin)
for t in (1,10,100,1000): o.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{R}" y2="{Y(t):.1f}" stroke="currentColor" opacity=".12"/><text x="{L-4}" y="{Y(t)+3.5:.1f}" text-anchor="end" font-size="9.5" fill="var(--ink-3)">{t}×</text>')
for q in Ls: o.append(f'<text x="{X(q):.1f}" y="{B+13}" text-anchor="middle" font-size="9.5" fill="var(--ink-2)">{q if q<1000 else str(q//1024)+"K"}</text>')
o.append(f'<line x1="{X(256):.1f}" y1="{Y(1):.1f}" x2="{X(32768):.1f}" y2="{Y(128):.1f}" stroke="currentColor" stroke-dasharray="3 3" opacity=".35"/><text x="{X(32768)-2:.1f}" y="{Y(128)+12:.1f}" text-anchor="end" font-size="9.5" fill="var(--ink-3)">∝ L</text>')
o.append(f'<line x1="{X(256):.1f}" y1="{Y(1):.1f}" x2="{X(16384):.1f}" y2="{Y(4096):.1f}" stroke="currentColor" stroke-dasharray="3 3" opacity=".35"/><text x="{X(16384)-30:.1f}" y="{Y(4096)+4:.1f}" text-anchor="end" font-size="9.5" fill="var(--ink-3)">∝ L²</text>')
pa=" ".join(f'{X(l):.1f},{Y(a/attn[0]):.1f}' for l,a in zip(Ls,attn)); pm=" ".join(f'{X(l):.1f},{Y(m/mamba[0]):.1f}' for l,m in zip(Ls,mamba))
o.append(f'<polyline points="{pa}" fill="none" stroke="var(--accent)" stroke-width="2"/><polyline points="{pm}" fill="none" stroke="var(--ok)" stroke-width="2"/>')
for l,a in zip(Ls,attn): o.append(f'<circle cx="{X(l):.1f}" cy="{Y(a/attn[0]):.1f}" r="3" fill="var(--accent)"/>')
for l,m in zip(Ls,mamba): o.append(f'<circle cx="{X(l):.1f}" cy="{Y(m/mamba[0]):.1f}" r="3" fill="var(--ok)"/>')
o.append(f'<text x="{X(8192)-6:.1f}" y="{Y(attn[5]/attn[0])-8:.1f}" text-anchor="end" font-size="10.5" fill="var(--accent)" font-weight="600">attention: 2,940× (초선형)</text>')
o.append(f'<text x="{X(32768)-2:.1f}" y="{Y(mamba[7]/mamba[0])+16:.1f}" text-anchor="end" font-size="10.5" fill="var(--ok)" font-weight="600">Mamba: 27× (선형 이하)</text>')
for i,l_ in enumerate(['KISTI A100 · Zamba2-2.7B · triton · CUDA graph OFF 마이크로 · 배치 1 · 108 SM','버킷: attention=코어만, Mamba=mixer 전체 → 두 선의 절대값 비교 금지, 증가 형태만']): o.append(f'<text x="160" y="{H-16+i*11}" text-anchor="middle" fill="var(--ink-3)" font-size="9.3">{l_}</text>')
p1="".join(o)
o,L,R,T,B=frame(320,320,'decode: 토큰 1개당 비용 (개념)','문맥 길이 →','시간')
o.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="currentColor" marker-end="url(#ahH1)"/>')
n=40; pa=" ".join(f'{L+(R-L-8)*i/n:.1f},{B-(B-T-10)*(0.15+0.7*i/n):.1f}' for i in range(n+1)); pm=" ".join(f'{L+(R-L-8)*i/n:.1f},{B-(B-T-10)*0.18:.1f}' for i in range(n+1))
o.append(f'<polyline points="{pa}" fill="none" stroke="var(--accent)" stroke-width="2"/><polyline points="{pm}" fill="none" stroke="var(--ok)" stroke-width="2"/>')
o.append(f'<text x="{R-4}" y="{T+22}" text-anchor="end" fill="var(--accent)" font-weight="600">attention ∝ 문맥 (KV 읽기)</text><text x="{R-4}" y="{B-(B-T-10)*0.18-8:.1f}" text-anchor="end" fill="var(--ok)" font-weight="600">SSM 일정 (고정 상태)</text>')
o.append(f'<rect x="{L+6}" y="{T+34}" width="200" height="40" rx="4" fill="var(--gate-soft)" stroke="var(--gate)" stroke-dasharray="4 3"/><text x="{L+106}" y="{T+50}" text-anchor="middle" font-size="10" fill="var(--gate)" font-weight="600">쓸 수 있는 실측 없음</text><text x="{L+106}" y="{T+64}" text-anchor="middle" font-size="9.5" fill="var(--gate)">검토 페이지 3-a가 VESSL에서 잰다</text>')
o.append(f'<text x="480" y="{H-16}" text-anchor="middle" fill="var(--ink-3)" font-size="9.3">개념 — 옛 측정은 계측 결함으로 인용 금지</text>')
p2="".join(o)
o,L,R,T,B=frame(640,320,'요청 1개가 쥐는 상태 메모리 (config 계산)','문맥 길이 (토큰, 로그축)','MB (로그축)')
kv_per_tok=4*2*8*128*2/1e6; ssm=27*128*80*128*2/1e6; dense_per_tok=56*2*8*128*2/1e6
ctxs=[1024,2048,4096,8192,16384,32768,65536,131072]; xmin,xmax=10,17; ymin,ymax=1,4
X3=lambda q: L+(R-L)*(math.log2(q)-xmin)/(xmax-xmin); Y3=lambda mb: B-(B-T)*(math.log10(max(mb,10))-ymin)/(ymax-ymin)
for t,lab in ((10,'10 MB'),(100,'100 MB'),(1000,'1 GB'),(10000,'10 GB')): o.append(f'<line x1="{L}" y1="{Y3(t):.1f}" x2="{R}" y2="{Y3(t):.1f}" stroke="currentColor" opacity=".12"/><text x="{L-4}" y="{Y3(t)+3.5:.1f}" text-anchor="end" font-size="9" fill="var(--ink-3)">{lab}</text>')
for q in ctxs[::2]: o.append(f'<text x="{X3(q):.1f}" y="{B+13}" text-anchor="middle" font-size="9.5" fill="var(--ink-2)">{q//1024}K</text>')
pk=" ".join(f'{X3(c):.1f},{Y3(kv_per_tok*c):.1f}' for c in ctxs); pd=" ".join(f'{X3(c):.1f},{Y3(dense_per_tok*c):.1f}' for c in ctxs); ps=" ".join(f'{X3(c):.1f},{Y3(ssm):.1f}' for c in ctxs)
o.append(f'<polyline points="{pd}" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3" opacity=".5"/><polyline points="{pk}" fill="none" stroke="var(--accent)" stroke-width="2"/><polyline points="{ps}" fill="none" stroke="var(--ok)" stroke-width="2"/>')
o.append(f'<text x="{X3(131072)-2:.1f}" y="{Y3(dense_per_tok*131072)+12:.1f}" text-anchor="end" font-size="9.5" fill="var(--ink-3)">56층 전부 attention이면</text>')
o.append(f'<text x="{X3(131072)-2:.1f}" y="{Y3(kv_per_tok*131072)-6:.1f}" text-anchor="end" font-size="10.5" fill="var(--accent)" font-weight="600">attention KV (4층): 16 KB/토큰</text>')
o.append(f'<text x="{X3(1024)+4:.1f}" y="{Y3(ssm)-6:.1f}" font-size="10.5" fill="var(--ok)" font-weight="600">SSM 상태 (27층): 약 71 MB 고정</text>')
for i,l_ in enumerate(['Nano-9B-v2 HF config(kv heads 8 · head 128 · mamba heads 128×80 · state 128 · bf16)로 계산','실측 아님 · conv 상태·활성값 제외 · 16K 문맥에서 KV 268 MB vs SSM 71 MB']): o.append(f'<text x="800" y="{H-16+i*11}" text-anchor="middle" fill="var(--ink-3)" font-size="9.3">{l_}</text>')
p3="".join(o)
h1=svg_open(f'{W} {H}','Hybrid 모델의 두 token mixer: 실측된 prefill 층당 시간은 입력 256에서 32768 토큰으로 갈 때 attention이 약 2,940배, Mamba가 약 27배 늘어 attention은 초선형, Mamba는 선형 이하다. decode 토큰당 비용은 개념도로 attention이 문맥에 비례하고 SSM은 일정하며 실측은 없다. 요청당 상태 메모리는 config로 계산해 attention KV가 토큰당 16 KB로 문맥에 비례하고 SSM 상태는 약 71 MB로 고정이다.')+AH+p1+p2+p3+'</svg>'
a=r.index('<figure class="fig">'); b=r.index('</figure>',a)+9
r=r[:a]+f'<figure class="fig">{h1}<figcaption>그림 H1. 왼쪽(실측): 입력이 128배 길어질 때 attention 층의 prefill 시간은 약 2,940배, Mamba 층은 약 27배 늘었다. attention은 L²에 가깝고 Mamba는 짧은 입력에서 고정 비용이 커 선형 이하다(두 선의 절대값은 계측 버킷이 달라 비교하지 않는다). 가운데(개념): decode에서 attention은 문맥만큼 KV를 읽고 SSM은 고정 상태만 갱신한다 — 실측은 검토 페이지 3-a. 오른쪽(계산): Nano-9B-v2는 attention이 4층뿐이라 KV가 토큰당 16 KB이고, 27개 Mamba 층의 상태는 약 71 MB로 고정이다.</figcaption></figure>'+r[b:]
old_f='H3-f. 운영 설정에서 분할 서빙을 켜면 꼬리 goodput이 오른다. 단 이득은 꼬리에 있고 중앙 decode는 5–45% 느려지며, 원인이 "SM 분할 자체"인지는 미확립.'
new_f='H3-f. SGLang 엔진 서빙 측정(<code>bench_serving</code>, 합성 요청 입력 2000·출력 96 토큰, 도착률 2–6/s, <code>--enable-pdmux</code>, CUDA graph ON, KISTI A100). 분할 서빙을 켜면 꼬리 goodput이 오르지만 이득은 꼬리에 있고 중앙 decode는 5–45% 느려지며, 원인이 "SM 분할 자체"인지는 미확립.'
assert old_f in r; r=r.replace(old_f,new_f)
r=r.replace('>KISTI A100 · CUDA graph ON · n=5 짝지음 · 인용 가능한 값 3개뿐</text>','>KISTI A100 · SGLang bench_serving · CUDA graph ON · n=5 짝지음 · 인용 가능한 값 3개</text>')
i0=r.index('<figure class="chart"><svg viewBox="0 0 360 268" role="img" aria-label="decode 속도가 문맥 길이와 SM 수에 따라'); i1=r.index('</figure>',i0)+9; r=r[:i0]+r[i1:]
r=r.replace('정본이 인용을 금지한 값은 싣지 않았고, 그 자리는 "측정 없음"으로 비워 두었다.','정본이 인용을 금지한 값은 싣지 않았다. 아직 측정이 없는 축(decode 속도 vs 문맥 × SM)은 검토 페이지의 실행 단계 3-a에 적었다.')
sec6=open(f'{S}/sec6.html').read()
r=r.replace('<footer>짝 페이지', sec6+'\n<footer>짝 페이지',1)
gap='<div class="note" style="margin-top:10px;border-left-color:var(--gate)"><b>채워야 할 측정 공백 — decode 속도 vs 문맥 길이 × SM 수.</b> hybrid 특성이 시스템에 드러나는 핵심 곡선(attention 4층의 몫이 문맥과 함께 커져 decode에 필요한 SM이 움직이는가)인데, 지금 쓸 수 있는 측정이 없다. 옛 값은 계측 결함 3건으로 인용 금지이고, 운영 설정(CUDA graph ON) 측정은 존재하지 않는다. 3-a가 이 곡선을 처음 잰다. 모델 구조 때문이라고 말하려면 구조가 다른 모델 2–3종이 필요하다.</div>'
anchor='<div class="grid2">\n    <div><h4>3-a. decode 속도 곡선'
assert anchor in v; v=v.replace(anchor, gap+'\n'+anchor,1)
open(rp,'w').write(r); open(vp,'w').write(v)
import xml.etree.ElementTree as ET; ET.fromstring(h1)
css=re.search(r'<style>(.*?)</style>',r,re.S).group(1)
open(f'{S}/figH1_check.html','w').write(f'<!doctype html><html><head><meta charset="utf-8"><style>{css} body{{padding:16px;width:1000px}}</style></head><body><div class="wrap"><figure class="fig">{h1}</figure></div></body></html>')
print('report',len(r),'review',len(v))
