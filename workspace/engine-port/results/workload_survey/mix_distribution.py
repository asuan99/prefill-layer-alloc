import json,sys,math,random
from tokenizers import Tokenizer
D,OC,W1=sys.argv[1:4]
tok=Tokenizer.from_file(D+'/nano_tokenizer.json')
n=lambda s:len(tok.encode(s,add_special_tokens=False).ids)
ld=[];c={}
for l in open(D+'/longdep_qa.jsonl',encoding='utf-8'):
    r=json.loads(l);k=r['doc_id']
    if k not in c:c[k]=n(r['context'])
    a=r['answer'];a=' '.join(map(str,a)) if isinstance(a,list) else str(a)
    ld.append((c[k]+n(r['question']),max(1,n(a))))
w1=[tuple(x) for x in json.load(open(W1))]
oc=[]
for conv in json.load(open(OC)):
    h=0
    for it in conv['items']:
        t=n(it['value'])
        if it['from'] in('gpt','assistant') and h>0 and h+t<=8192: oc.append((h,t))
        h+=t
json.dump(oc,open(sys.argv[4],'w')); json.dump(ld,open(sys.argv[5],'w'))
def q(v,p):
    v=sorted(v);return v[min(len(v)-1,max(0,math.ceil(p*len(v))-1))]
def st(name,I,O):
    m=sum(I)/len(I);sd=(sum((x-m)**2 for x in I)/len(I))**.5
    print(f"{name:24s} n={len(I):6d} in mean={m:8.0f} p50={q(I,.5):6d} p90={q(I,.9):6d} p99={q(I,.99):6d} max={max(I):6d} CV={sd/m:5.2f} p99/p50={q(I,.99)/max(1,q(I,.5)):7.1f} out mean={sum(O)/len(O):6.1f} p50={q(O,.5)}")
st('ShareGPT first',[a for a,b in w1],[b for a,b in w1]); st('openchat rounds',[a for a,b in oc],[b for a,b in oc]); st('LooGLE longdep',[a for a,b in ld],[b for a,b in ld])
T=lambda L:9+0.111*L
R=random.Random(7)
print('mix: f_long by request; floor model 9+0.111 L ms extrapolated')
for f in (.5,.2,.1,.05,.02,.01):
    N=200000;x=[(R.choice(ld) if R.random()<f else R.choice(w1)) for _ in range(N)]
    I=[a for a,b in x];O=[b for a,b in x];st(f'mix long={f:.0%}',I,O)
    work=sum(T(a) for a in I)
    long_work=sum(T(a) for a,b in x if a>=12000)
    tok_share=sum(a for a in I if a>=12000)/sum(I)
    ET=work/N; lam=.7/(ET/1000)
    print(f"   prefill-ms share long={long_work/work:.3f} token share={tok_share:.3f} E[T]={ET:.0f}ms rate@rho0.7={lam:.3f} long/300s={lam*f*300:.1f} short/300s={lam*(1-f)*300:.0f} decode tokens/s={lam*sum(O)/N:.0f}")
gap=sum(1 for a,b in w1 if 3897<a<12361)
print('gap count',gap,'W1 max',max(a for a,b in w1),'longdep min',min(a for a,b in ld))
