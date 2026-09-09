"""GPU 0. §4.5 (d) 선결조건 — 4셀·3대조(m=3) 격자에서 규칙 검정력 재산출.
셀 = A,B,C,D. 등록 대조: P=(A,B) iso-C · S1=(A,C) iso-R · S2=(D,B) iso-R.
rev3 표는 N_CTX=4·m=4 격자 산출이라 그대로 못 쓴다(감사 차단 B4)."""
import numpy as np, null_rate_c as M

CELLS = ["A", "B", "C", "D"]
CONTRASTS = [("A", "B"), ("A", "C"), ("D", "B")]      # m = 3
H, W = len(M.S_GRID), len(M.ITL_GRID)

def comps(mask):
    seen = np.zeros_like(mask, bool); out = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                st = [(i, j)]; cp = []; seen[i, j] = True
                while st:
                    x, y = st.pop(); cp.append((x, y))
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        u, v = x+dx, y+dy
                        if 0 <= u < H and 0 <= v < W and mask[u, v] and not seen[u, v]:
                            seen[u, v] = True; st.append((u, v))
                out.append(cp)
    return out

def run(n_boot, eff_map, n_req=160, n_rep=300, seed=5, sig_boot=None, m=3,
        only_contrast=None):
    """eff_map[cell][split] = log-shift on median TTFT (음수면 그 split이 유리).
    only_contrast: None이면 3대조 중 하나라도 발화하면 hit, 아니면 그 대조만."""
    rng = np.random.default_rng(seed)
    sb = M.SIG_BOOT if sig_boot is None else sig_boot
    tc = M.t_crit(M.ALPHA/m, n_boot-1)
    fires = 0
    tested = CONTRASTS if only_contrast is None else [only_contrast]
    for _ in range(n_rep):
        gp = {}
        for c in CELLS:
            g = np.empty((n_boot, M.N_SPLIT, H, W))
            for b in range(n_boot):
                for k in range(M.N_SPLIT):
                    shift = eff_map.get(c, {}).get(k, 0.0)
                    d = rng.normal(0, sb)
                    t = np.exp(np.log(M.MED_TTFT)+shift+d+rng.normal(0, M.SIG_TTFT, n_req))*M.FLOOR
                    i_ = np.exp(np.log(M.MED_ITL)+d+rng.normal(0, M.SIG_ITL, n_req))
                    g[b, k] = ((t[:, None] <= M.S_GRID[None, :])[:, :, None]
                               & (i_[:, None] <= M.ITL_GRID[None, :])[:, None, :]).mean(axis=0)
            gp[c] = g
        cm = {c: gp[c].mean(axis=0) for c in CELLS}
        usable = {c: ~((cm[c] < 0.10).all(axis=0) | (cm[c] > 0.95).all(axis=0)) for c in CELLS}
        win = {c: cm[c].argmax(axis=0) for c in CELLS}
        need = int(np.ceil(0.75*n_boot))
        voted = {c: (gp[c].argmax(axis=1) == win[c][None]).sum(axis=0) >= need for c in CELLS}
        hit = False
        for (c1, c2) in tested:
            v1 = usable[c1] & voted[c1]; v2 = usable[c2] & voted[c2]
            diff = v1 & v2 & (win[c1] != win[c2])
            for cp in comps(diff):
                if len(cp) < 4: continue
                ok = True
                for c in (c1, c2):
                    idx = tuple(np.array(cp).T)
                    band = gp[c][:, :, idx[0], idx[1]].mean(axis=2)
                    w0 = win[c][cp[0]]
                    order = np.argsort(band.mean(axis=0))
                    runner = order[-2] if order[-1] == w0 else order[-1]
                    d_ = band[:, w0] - band[:, runner]; sd = d_.std(ddof=1)
                    if sd == 0: ok = False; break
                    hw = tc*sd/np.sqrt(n_boot)
                    rel = abs(d_.mean())/max(band.mean(axis=0).max(), 1e-12)
                    if not ((d_.mean()-hw > 0 or d_.mean()+hw < 0) and rel >= M.FLOOR_PRACT):
                        ok = False; break
                if ok: hit = True
            if hit: break
        fires += hit
    return fires/n_rep

if __name__ == "__main__":
    NULL = {}
    ALT_P = {"A": {0: -0.30}, "B": {2: -0.30}}          # iso-C 대조에서 승자가 갈림
    print("== 4셀·3대조(m=3) 격자, 초다수결+띠, N_req=160 ==")
    print("  n    귀무(전체)   P 검정력   P 귀무(P만)")
    for n in (4, 6, 8):
        print(f"  {n}   {run(n, NULL, n_rep=250):9.3f}   "
              f"{run(n, ALT_P, n_rep=250, only_contrast=('A','B')):8.3f}   "
              f"{run(n, NULL, n_rep=250, only_contrast=('A','B')):10.3f}")
    print("\n== SD 민감도 (n=6, P 대조) ==")
    print("  SIG_BOOT   P 검정력")
    for sig in (0.02, 0.04, 0.06, 0.10):
        print(f"    {sig:.2f}     {run(6, ALT_P, n_rep=250, sig_boot=sig, only_contrast=('A','B')):.3f}")
