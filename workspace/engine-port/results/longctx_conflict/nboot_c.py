"""GPU 0. 만장일치 요건이 n에 대해 역행하는가 — 규칙 자신의 결함 점검."""
import numpy as np, null_rate_c as M

def run(n_boot, eff, vote="unanimous", n_req=160, n_rep=250, seed=23):
    rng = np.random.default_rng(seed)
    tc = M.t_crit(M.ALPHA/M.M_FAMILY, n_boot-1)
    fires = 0
    for _ in range(n_rep):
        gp = np.empty((M.N_CTX, n_boot, M.N_SPLIT, len(M.S_GRID), len(M.ITL_GRID)))
        for c in range(M.N_CTX):
            for b in range(n_boot):
                for k in range(M.N_SPLIT):
                    shift = -eff if ((c==0 and k==0) or (c==M.N_CTX-1 and k==2)) else 0.0
                    d = rng.normal(0, M.SIG_BOOT)
                    ttft = np.exp(np.log(M.MED_TTFT)+shift+d+rng.normal(0,M.SIG_TTFT,n_req))*M.FLOOR
                    itl  = np.exp(np.log(M.MED_ITL)+d+rng.normal(0,M.SIG_ITL,n_req))
                    ok_t = ttft[:,None] <= M.S_GRID[None,:]; ok_i = itl[:,None] <= M.ITL_GRID[None,:]
                    gp[c,b,k] = (ok_t[:,:,None] & ok_i[:,None,:]).mean(axis=0)
        cm = gp.mean(axis=1)
        usable = ~((cm<0.10).all(axis=1) | (cm>0.95).all(axis=1))
        pa = gp.argmax(axis=2)                       # (ctx, boot, s, itl)
        maj = cm.argmax(axis=1)                      # (ctx, s, itl) 평균 기준 승자
        agree = (pa == maj[:, None]).sum(axis=1)     # 몇 부팅이 그 승자에 동의하나
        if vote == "unanimous":
            ok_vote = agree == n_boot
        elif vote == "supermajority":                # >= ceil(0.75 n)
            ok_vote = agree >= int(np.ceil(0.75*n_boot))
        else:                                        # majority
            ok_vote = agree > n_boot/2
        v = usable & ok_vote
        win = maj
        H,W = len(M.S_GRID), len(M.ITL_GRID)
        def comps(mask):
            seen=np.zeros_like(mask,bool); out=[]
            for i in range(H):
                for j in range(W):
                    if mask[i,j] and not seen[i,j]:
                        st=[(i,j)]; cp=[]; seen[i,j]=True
                        while st:
                            x,y=st.pop(); cp.append((x,y))
                            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                                u,w_=x+dx,y+dy
                                if 0<=u<H and 0<=w_<W and mask[u,w_] and not seen[u,w_]:
                                    seen[u,w_]=True; st.append((u,w_))
                        out.append(cp)
            return out
        hit=False
        for c1 in range(M.N_CTX):
            for c2 in range(c1+1,M.N_CTX):
                diff = v[c1]&v[c2]&(win[c1]!=win[c2])
                for cp in comps(diff):
                    if len(cp)<4: continue
                    ok=True
                    for c in (c1,c2):
                        idx=tuple(np.array(cp).T)
                        band=gp[c][:,:,idx[0],idx[1]].mean(axis=2)
                        w0=win[c][cp[0]]
                        order=np.argsort(band.mean(axis=0))
                        runner=order[-2] if order[-1]==w0 else order[-1]
                        d_=band[:,w0]-band[:,runner]; sd=d_.std(ddof=1)
                        if sd==0: ok=False; break
                        hw=tc*sd/np.sqrt(n_boot)
                        rel=abs(d_.mean())/max(band.mean(axis=0).max(),1e-12)
                        if not ((d_.mean()-hw>0 or d_.mean()+hw<0) and rel>=M.FLOOR_PRACT):
                            ok=False; break
                    if ok: hit=True
                if hit: break
            if hit: break
        fires += hit
    return fires/n_rep

if __name__ == "__main__":
    print(" 투표규칙        n=4     n=6     n=8    | 귀무(n=6)")
    for vote in ("unanimous","supermajority","majority"):
        row=[run(n,0.30,vote) for n in (4,6,8)]
        null6=run(6,0.0,vote)
        print(f" {vote:14s} {row[0]:.3f}   {row[1]:.3f}   {row[2]:.3f}   |  {null6:.3f}")
