"""GPU 0. 규칙 변이 비교 — 검정 입력을 '띠'(데이터 선택)에서 'USABLE 전체'(비선택)로 옮기면
비단조성이 사라지는가, 그리고 검정력은?"""
import numpy as np, null_rate_c as M

def run(n_boot, eff, vote, test_on, n_req=160, n_rep=300, seed=101, sig_boot=None):
    rng = np.random.default_rng(seed); tc = M.t_crit(M.ALPHA/M.M_FAMILY, n_boot-1)
    sb = M.SIG_BOOT if sig_boot is None else sig_boot
    H,W = len(M.S_GRID), len(M.ITL_GRID); fires=0
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
    for _ in range(n_rep):
        gp = np.empty((M.N_CTX, n_boot, M.N_SPLIT, H, W))
        for c in range(M.N_CTX):
            for b in range(n_boot):
                for k in range(M.N_SPLIT):
                    shift = -eff if ((c==0 and k==0) or (c==M.N_CTX-1 and k==2)) else 0.0
                    d = rng.normal(0, sb)
                    ttft = np.exp(np.log(M.MED_TTFT)+shift+d+rng.normal(0,M.SIG_TTFT,n_req))*M.FLOOR
                    itl  = np.exp(np.log(M.MED_ITL)+d+rng.normal(0,M.SIG_ITL,n_req))
                    ok_t = ttft[:,None] <= M.S_GRID[None,:]; ok_i = itl[:,None] <= M.ITL_GRID[None,:]
                    gp[c,b,k] = (ok_t[:,:,None] & ok_i[:,None,:]).mean(axis=0)
        cm = gp.mean(axis=1)
        usable = ~((cm<0.10).all(axis=1) | (cm>0.95).all(axis=1))
        pa = gp.argmax(axis=2); maj = cm.argmax(axis=1)
        agree = (pa == maj[:,None]).sum(axis=1)
        need = {"unanimous": n_boot, "supermajority": int(np.ceil(0.75*n_boot)),
                "majority": n_boot//2+1}[vote]
        v = usable & (agree >= need); win = maj
        hit=False
        for c1 in range(M.N_CTX):
            for c2 in range(c1+1,M.N_CTX):
                diff = v[c1]&v[c2]&(win[c1]!=win[c2])
                for cp in comps(diff):
                    if len(cp)<4: continue
                    ok=True
                    for c in (c1,c2):
                        cells = cp if test_on=="band" else list(map(tuple, np.argwhere(usable[c])))
                        idx=tuple(np.array(cells).T)
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
        fires+=hit
    return fires/n_rep

if __name__=="__main__":
    print("검정입력  투표          n=4    n=6    n=8   | 귀무 n=6  귀무 n=8")
    for test_on in ("band","usable"):
        for vote in ("unanimous","supermajority"):
            pw=[run(n,0.30,vote,test_on) for n in (4,6,8)]
            n0=[run(n,0.0,vote,test_on) for n in (6,8)]
            print(f" {test_on:8s} {vote:14s} {pw[0]:.3f}  {pw[1]:.3f}  {pw[2]:.3f}  |  {n0[0]:.3f}     {n0[1]:.3f}")

def sd_map(n_boot=6, n_req=160, n_rep=200, seed=77):
    """S0-D가 잴 '실현 페어드 SD'와 규칙 검정력의 대응표."""
    out=[]
    for sig in (0.02, 0.04, 0.06, 0.10):
        rng=np.random.default_rng(seed); sds=[]
        for _ in range(60):
            g=np.empty((n_boot,M.N_SPLIT,len(M.S_GRID),len(M.ITL_GRID)))
            for b in range(n_boot):
                for k in range(M.N_SPLIT):
                    d=rng.normal(0,sig)
                    t=np.exp(np.log(M.MED_TTFT)+d+rng.normal(0,M.SIG_TTFT,n_req))*M.FLOOR
                    i_=np.exp(np.log(M.MED_ITL)+d+rng.normal(0,M.SIG_ITL,n_req))
                    g[b,k]=((t[:,None]<=M.S_GRID[None,:])[:,:,None]&(i_[:,None]<=M.ITL_GRID[None,:])[:,None,:]).mean(axis=0)
            cm=g.mean(axis=0); us=~((cm<0.10).all(axis=0)|(cm>0.95).all(axis=0))
            if us.any():
                band=g[:,:,us].mean(axis=2); base=band.mean(axis=0).max()
                if base>0: sds.append((band[:,0]-band[:,1]).std(ddof=1)/base)
        import importlib
        pw=run(n_boot,0.30,"supermajority","band",n_req=n_req,n_rep=n_rep,seed=seed,sig_boot=sig)
        out.append((sig,float(np.mean(sds)),pw))
    return out
