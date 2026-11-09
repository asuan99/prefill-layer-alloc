import json,sys,math
from tokenizers import Tokenizer
D=sys.argv[1]
tok=Tokenizer.from_file(D+'/nano_tokenizer.json')
n=lambda s:len(tok.encode(s,add_special_tokens=False).ids)
def q(v,p):
    v=sorted(v);return v[min(len(v)-1,max(0,math.ceil(p*len(v))-1))]
def st(lab,v):
    print(f"{lab:26s} n={len(v):5d} mean={sum(v)/len(v):9.1f} min={min(v):6d} p50={q(v,.5):6d} p90={q(v,.9):6d} p99={q(v,.99):6d} max={max(v):7d}")
cache={};IN=[];OUT=[];res={}
for t in ['longdep_qa','shortdep_qa','shortdep_cloze','summarization']:
    I=[];O=[];qs=[]
    for l in open(f'{D}/{t}.jsonl',encoding='utf-8'):
        r=json.loads(l)
        k=r['doc_id']
        if k not in cache: cache[k]=n(r['context'])
        a=r['answer']; a=' '.join(map(str,a)) if isinstance(a,list) else str(a)
        qt=n(r['question']) if r['question'] else 0
        I.append(cache[k]+qt); O.append(n(a)); qs.append(qt)
    st(t+' in',I); st(t+' out(answer)',O); res[t]=(I,O); IN+=I; OUT+=O
st('ALL in',IN); st('ALL out',OUT); st('docs (unique)',list(cache.values()))
for th in (3380,8800,20480,81000,94800,131072):
    print('share in >',th, round(sum(x>th for x in IN)/len(IN),4), ' docs', round(sum(x>th for x in cache.values())/len(cache),4))
QA=res['longdep_qa'][0]+res['shortdep_qa'][0]; 
sel=[x for x in IN if 3380<=x<=81000]; print('in [3380,81000]',len(sel),round(sum(sel)/len(sel),1))
out3=[o for t in ('longdep_qa','shortdep_qa') for o in res[t][1]]; st('QA out',out3)
