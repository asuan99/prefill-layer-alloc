import json,sys,glob,os
d=sys.argv[1]
for f in sorted(glob.glob(os.path.join(d,'*.jsonl'))):
    q=[json.loads(l) for l in open(f)]
    n=len(q); ok=[x for x in q if x.get('completion_s') is not None and not x.get('error')]
    arr=[x['scheduled_arrival_s'] for x in q]
    # semantics check
    ft=[x['sent_s']+x['ttft_ms']/1000 for x in ok]
    viol_abs=sum(1 for x,t in zip(ok,ft) if x['completion_s']<t-1e-3)   # if completion_s absolute, must be >= first token time
    sumitl=[x['sent_s']+(x['ttft_ms']+sum(x['token_itl_ms']))/1000 for x in ok]
    resid=max(abs(x['completion_s']-s) for x,s in zip(ok,sumitl))
    endA=[x['completion_s'] for x in ok]            # absolute
    endB=[x['sent_s']+x['completion_s'] for x in ok]  # audit_r1_sus.py
    print(os.path.basename(f), 'n',n,'ok',len(ok),'err',n-len(ok),'arr_last %.1f'%max(arr),
          'viol_abs',viol_abs,'max|completion-(sent+ttft+sumITL)| %.3fs'%resid)
    LA,LB=max(endA),max(endB)
    print('  defA last_end %.1f  n/last %.2f  in600 %d  n/600 %.2f'%(LA,len(ok)/LA,sum(e<=600 for e in endA),len(ok)/600))
    print('  defB last_end %.1f  n/last %.2f'%(LB,len(ok)/LB))
    for W in (120,):
        rowsA=[];rowsB=[]
        k=0
        while k*W< max(LB,LA):
            lo,hi=k*W,(k+1)*W
            a=sum(lo<=t<hi for t in arr); s=sum(lo<=t<hi for t in ft)
            cA=sum(lo<=t<hi for t in endA); cB=sum(lo<=t<hi for t in endB)
            rowsA.append((lo,a,s,cA,cB)); k+=1
        print('  win  arr ftok compA compB  (counts per 120s)')
        for r in rowsA: print('  %5d %4d %4d %4d %4d'%r)
        # defB windowed rate over full windows inside [0,LB)
        fullB=[r[4]/W for r in rowsA if r[0]+W<=LB]; fullA=[r[3]/W for r in rowsA if r[0]+W<=LA]
        print('  defB full-window rate min %.2f max %.2f'%(min(fullB),max(fullB)),' defA full-window rate min %.2f max %.2f'%(min(fullA),max(fullA)))
