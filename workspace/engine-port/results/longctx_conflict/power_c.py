"""GPU 0. (c)의 반쪽 — 귀무 발화율이 0이면 다음 질문은 '검정력이 있는가'다.
대안: ctx 최저에서 split0이, ctx 최고에서 split2가 각각 e만큼 빠른 TTFT를 갖는다
(= argmax가 실제로 이동하는 세계). 등록 규칙이 그것을 잡아내는 비율을 잰다."""
import numpy as np, null_rate_c as M

def run(n_req, eff, n_rep=300, sig_boot=M.SIG_BOOT, seed=11):
    rng = np.random.default_rng(seed)
    fires = 0; gaps = []
    for _ in range(n_rep):
        gp = np.empty((M.N_CTX, M.N_BOOT, M.N_SPLIT, len(M.S_GRID), len(M.ITL_GRID)))
        for c in range(M.N_CTX):
            for b in range(M.N_BOOT):
                for k in range(M.N_SPLIT):
                    shift = 0.0
                    if c == 0 and k == 0: shift = -eff
                    if c == M.N_CTX-1 and k == 2: shift = -eff
                    d = rng.normal(0, sig_boot)
                    ttft = np.exp(np.log(M.MED_TTFT)+shift+d+rng.normal(0,M.SIG_TTFT,n_req))*M.FLOOR
                    itl  = np.exp(np.log(M.MED_ITL)+d+rng.normal(0,M.SIG_ITL,n_req))
                    ok_t = ttft[:,None] <= M.S_GRID[None,:]
                    ok_i = itl[:,None]  <= M.ITL_GRID[None,:]
                    gp[c,b,k] = (ok_t[:,:,None] & ok_i[:,None,:]).mean(axis=0)
        cm = gp.mean(axis=1)
        usable = ~((cm<0.10).all(axis=1) | (cm>0.95).all(axis=1))
        pa = gp.argmax(axis=2); unan = (pa==pa[:,:1]).all(axis=1); win = pa[:,0]
        v = usable & unan
        if usable[0].any():
            g = cm[0][:, usable[0]]
            gaps.append((np.sort(g.mean(axis=1))[-1]-np.sort(g.mean(axis=1))[-2]))
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
                        hw=M.TC[M.M_FAMILY]*sd/np.sqrt(M.N_BOOT)
                        rel=abs(d_.mean())/max(band.mean(axis=0).max(),1e-12)
                        if not ((d_.mean()-hw>0 or d_.mean()+hw<0) and rel>=M.FLOOR_PRACT):
                            ok=False; break
                    if ok: hit=True
                if hit: break
            if hit: break
        fires += hit
    return fires/n_rep, (np.mean(gaps) if gaps else float('nan'))

if __name__ == "__main__":
    print(" N_req  logshift  실현 goodput 간극   P(ARGMAX_MOVES)")
    for n_req in (60, 160, 320):
        for eff in (0.0, 0.15, 0.30, 0.50):
            p, gap = run(n_req, eff, n_rep=200)
            print(f" {n_req:5d}  {eff:8.2f}  {gap:16.3f}   {p:.3f}")
