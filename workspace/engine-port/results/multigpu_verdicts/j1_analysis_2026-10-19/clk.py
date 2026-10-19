import json, csv, sys, datetime as dt, collections, statistics as st, numpy as np
CLK=sys.argv[1]; rows=json.load(open(sys.argv[2]))
T=[];M=[];W=[];TH=[];TEMP=[]
with open(CLK) as f:
    for i,row in enumerate(csv.reader(f)):
        if i==0 or len(row)<6 or row[5].strip()!='0': continue
        try:
            t=dt.datetime.strptime(row[0].strip().replace('/','-'),'%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()
            T.append(t);M.append(float(row[1].split()[0]));W.append(float(row[2].split()[0]));TEMP.append(float(row[3]));TH.append(int(row[4].strip(),16))
        except Exception: pass
T=np.array(T);M=np.array(M);W=np.array(W);TH=np.array(TH);TEMP=np.array(TEMP)
pc=(TH&0x4)!=0; idle=(TH&0x1)!=0
print('GPU0 samples',len(T),'pwrcap frac %.3f'%pc.mean(),'other bits', sorted(set(TH.tolist())))
print('overall clk median',np.median(M),'temp median',np.median(TEMP),'temp max',TEMP.max(),'power p50/p95/max',np.percentile(W,[50,95,100]))
print('when pwrcap: clk p5/p50/mean',np.percentile(M[pc],[5,50]),M[pc].mean(),' power p50',np.median(W[pc]))
print('when not cap & busy (W>200): clk mean',M[(~pc)&(W>200)].mean())
# per cell
agg=collections.defaultdict(list)
for r in rows:
    s=(T>=r['t0'])&(T<=r['t0']+r['wall'])
    r['clk_mean']=float(M[s].mean()); r['cap']=float(pc[s].mean()); r['temp']=float(TEMP[s].mean()); r['pw']=float(W[s].mean())
    tag='HIx' if r['ext'] else r['rung']
    agg[(r['shape'],tag,r['arm'])].append(r)
print('shape rung arm  n  cap_frac  clk_mean  temp  W_mean  t_order(h from start)')
t00=min(r['t0'] for r in rows)
for k in sorted(agg):
    v=agg[k]
    print(k, len(v), '%.2f'%np.mean([x['cap'] for x in v]), '%.0f'%np.mean([x['clk_mean'] for x in v]), '%.1f'%np.mean([x['temp'] for x in v]),'%.0f'%np.mean([x['pw'] for x in v]), '%.2f'%((np.mean([x['t0'] for x in v])-t00)/3600))
# ratios of clock relevant to estimands (lambda* cells = final HI rung)
def fin(shape,arm):
    v=agg[(shape,'HIx',arm)] or agg[(shape,'HI',arm)]
    return np.mean([x['clk_mean'] for x in v])
for sh in 'ABX':
    print(sh,'clk ratio agn/fsym %.4f  agn/fused %.4f d44/fsym %.4f fsym/fused %.4f'%(fin(sh,'agnostic')/fin(sh,'fused_sym'),fin(sh,'agnostic')/fin(sh,'fused'),fin(sh,'d44')/fin(sh,'fused_sym'),fin(sh,'fused_sym')/fin(sh,'fused')))
# idle trend: temperature at idle over time
for h in range(0,9):
    s=(T>=t00+h*3600)&(T<t00+(h+1)*3600)
    if s.any(): print('hour',h,'cap %.2f clk %.0f temp %.1f W %.0f'%(pc[s].mean(),M[s].mean(),TEMP[s].mean(),W[s].mean()))
json.dump(rows,open(sys.argv[2].replace('.json','_clk.json'),'w'))
