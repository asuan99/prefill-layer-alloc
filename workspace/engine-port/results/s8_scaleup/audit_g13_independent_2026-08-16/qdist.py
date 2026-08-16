import math, random
def gammainc(a,x):
    if x<=0: return 0.0
    if x < a+1.0:
        term=1.0/a; s=term; n=1
        while n<10000:
            term*=x/(a+n); s+=term
            if abs(term)<abs(s)*1e-16: break
            n+=1
        return s*math.exp(-x+a*math.log(x)-math.lgamma(a))
    tiny=1e-300; b=x+1.0-a; c=1/tiny; d=1/b; h=d
    for i in range(1,10000):
        an=-i*(i-a); b+=2.0; d=an*d+b
        if abs(d)<tiny: d=tiny
        c=b+an/c
        if abs(c)<tiny: c=tiny
        d=1/d; delt=d*c; h*=delt
        if abs(delt-1)<1e-16: break
    return 1.0-math.exp(-x+a*math.log(x)-math.lgamma(a))*h
def chi2q(p,nu):
    lo,hi=1e-14,max(50.0,10*nu+80)
    for _ in range(300):
        mid=0.5*(lo+hi)
        if gammainc(nu/2.0,mid/2.0)<p: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)
def betacf(a,b,x):
    tiny=1e-300; qab=a+b; qap=a+1; qam=a-1
    c=1.0; d=1-qab*x/qap
    if abs(d)<tiny: d=tiny
    d=1/d; h=d
    for m in range(1,500):
        m2=2*m
        aa=m*(b-m)*x/((qam+m2)*(a+m2))
        d=1+aa*d; c=1+aa/c
        if abs(d)<tiny: d=tiny
        if abs(c)<tiny: c=tiny
        d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d=1+aa*d; c=1+aa/c
        if abs(d)<tiny: d=tiny
        if abs(c)<tiny: c=tiny
        d=1/d; de=d*c; h*=de
        if abs(de-1)<1e-15: break
    return h
def betai(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
    if x < (a+1)/(a+b+2): return bt*betacf(a,b,x)/a
    return 1-bt*betacf(b,a,1-x)/b
def fcdf(x,d1,d2):
    if x<=0: return 0.0
    return betai(d1/2.0,d2/2.0,d1*x/(d1*x+d2))
def fq(p,d1,d2):
    lo,hi=1e-12,1e6
    for _ in range(300):
        mid=math.sqrt(lo*hi)
        if fcdf(mid,d1,d2)<p: lo=mid
        else: hi=mid
    return math.sqrt(lo*hi)
