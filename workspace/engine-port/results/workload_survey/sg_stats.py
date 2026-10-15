import json,ast,sys
d=json.load(open(sys.argv[1]))
for ts,v in list(d.items())[:4]:
    print("t=",ts, "keys",list(v.keys()))
    for k,raw in v.items():
        try: pm=ast.literal_eval(raw)
        except Exception: continue
        if not isinstance(pm,dict): continue
        it=sorted(pm.items()); tot=sum(p for _,p in it); c=0; qs={}
        mean=sum(x*p for x,p in it)/tot
        for x,p in it:
            c+=p/tot
            for q in (.5,.9,.99):
                if q not in qs and c>=q: qs[q]=x
        print("   %-16s mean %7.0f p50 %6s p90 %6s p99 %6s max %6s"%(k,mean,qs.get(.5),qs.get(.9),qs.get(.99),it[-1][0]))
