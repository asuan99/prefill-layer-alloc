"""
Decode-step roofline cost model for DENSE hybrid LLMs (Mamba-2 / Gated DeltaNet
mixers + softmax attention + dense FFN), built directly from HF config.json.

Layer types and per-layer cost (one decode step, B sequences, L cached tokens each):

  Attention (GQA / MHA, optional output gate)
    W  = h*Hq*d*g + 2*h*Hkv*d + Hq*d*h            (g = 2 if gated, else 1)
    F  = 2*W*B + 4*Hq*d*L*B
    M  = W*bw + 2*Hkv*d*L*B*bkv

  Mamba-2
    din = H*P ; C = din + 2*G*N ; S = H*P*N
    W  = h*(din + C + H) + din*h + C*K
    F  = 2*W*B + 6*S*B + 2*C*K*B
    M  = W*bw + 2*S*B*bs + 2*C*(K-1)*B*bkv

  Gated DeltaNet (GDN)
    C = 2*Hk*dk + Hv*dv ; S = Hv*dk*dv
    W  = 2*h*Hk*dk + 2*h*Hv*dv + 2*h*Hv + Hv*dv*h + C*K   (q,k | v,gate | a,b | out | conv)
    F  = 2*W*B + 7*S*B + 2*C*K*B                (decay, S·k, rank-1 delta write, S·q)
    M  = W*bw + 2*S*B*bs + 2*C*(K-1)*B*bkv

  Dense FFN:  ReLU² -> W = 2*h*f ; SwiGLU -> W = 3*h*f ;  F = 2*W*B ; M = W*bw
  LM head:    W = h*V ; F = 2*W*B ; M = W*bw

  T_layer = max(F/P_peak, M/BW), summed over layers (no cross-layer overlap).

Memory capacity:
  W_total = sum of layer weights + embedding (+ LM head unless tied)
  cache/seq = n_attn*L*2*Hkv*d*bkv + n_rec*(S*bs + C*(K-1)*bkv)
  B_max = floor((0.9*HBM - W_total) / cache/seq)

Assumptions: A100-SXM4-80GB roofline, single GPU, bf16 weights & KV, recurrent state
dtype configurable (default fp32 for every model so mixers compare on equal footing).
Relative trends only; no kernel launch, no prefill, no TP.
"""
import json, os, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "cfgs")
PEAK_FLOPS, PEAK_BW, HBM = 312e12, 2.039e12, 80e9
BW_B, KV_B = 2, 2


def jload(f):
    c = json.load(open(os.path.join(CFG, f)))
    return c.get("text_config", c), c


# ---------------------------------------------------------------- layer specs
def attn_spec(h, Hq, Hkv, d, gated):
    return dict(kind="attn", h=h, Hq=Hq, Hkv=Hkv, d=d,
                W=h * Hq * d * (2 if gated else 1) + 2 * h * Hkv * d + Hq * d * h)


def mamba_spec(h, H, P, N, G, K):
    din = H * P
    C = din + 2 * G * N
    return dict(kind="mamba", S=H * P * N, C=C, K=K,
                W=h * (din + C + H) + din * h + C * K, ops=6)


def gdn_spec(h, Hk, Hv, dk, dv, K):
    C = 2 * Hk * dk + Hv * dv
    return dict(kind="gdn", S=Hv * dk * dv, C=C, K=K,
                W=2 * h * Hk * dk + 2 * h * Hv * dv + 2 * h * Hv + Hv * dv * h + C * K, ops=7)


def ffn_spec(h, f, act):
    return dict(kind="ffn", W=(2 if act == "relu2" else 3) * h * f)


# ---------------------------------------------------------------- model builders
def nemotron(fname, name, role):
    c, _ = jload(fname)
    h, p = c["hidden_size"], c["hybrid_override_pattern"]
    d = c.get("head_dim") or c.get("attention_head_dim") or h // c["num_attention_heads"]
    spec = {
        "M": mamba_spec(h, c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"],
                        c["n_groups"], c["conv_kernel"]),
        "*": attn_spec(h, c["num_attention_heads"], c["num_key_value_heads"], d, False),
        "-": ffn_spec(h, c["intermediate_size"], "relu2"),
    }
    layers = [spec[x] for x in p]
    return dict(name=name, role=role, mixer="Mamba-2", h=h, V=c["vocab_size"], tied=False,
                layers=layers, ctx=c.get("max_position_embeddings"))


def qwen35(fname, name, role):
    t, top = jload(fname)
    h, n = t["hidden_size"], t["num_hidden_layers"]
    lt = t.get("layer_types") or ["full_attention" if (i + 1) % t["full_attention_interval"] == 0
                                  else "linear_attention" for i in range(n)]
    A = attn_spec(h, t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"],
                  t.get("attn_output_gate", True))
    G = gdn_spec(h, t["linear_num_key_heads"], t["linear_num_value_heads"],
                 t["linear_key_head_dim"], t["linear_value_head_dim"], t["linear_conv_kernel_dim"])
    F = ffn_spec(h, t["intermediate_size"], "swiglu")
    layers = []
    for x in lt:
        layers += [A if x == "full_attention" else G, F]
    tied = top.get("tie_word_embeddings", t.get("tie_word_embeddings", False))
    return dict(name=name, role=role, mixer="GDN", h=h, V=t["vocab_size"], tied=tied,
                layers=layers, ctx=t.get("max_position_embeddings"))


def olmo_hybrid(fname, name, role):
    c, _ = jload(fname)
    h = c["hidden_size"]
    A = attn_spec(h, c["num_attention_heads"], c["num_key_value_heads"],
                  c.get("head_dim") or h // c["num_attention_heads"], False)
    G = gdn_spec(h, c["linear_num_key_heads"], c["linear_num_value_heads"],
                 c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"])
    F = ffn_spec(h, c["intermediate_size"], "swiglu")
    layers = []
    for x in c["layer_types"]:
        layers += [A if x == "full_attention" else G, F]
    return dict(name=name, role=role, mixer="GDN", h=h, V=c["vocab_size"],
                tied=c.get("tie_word_embeddings", False), layers=layers,
                ctx=c.get("max_position_embeddings"))


def llama(fname, name, role):
    c, _ = jload(fname)
    h = c["hidden_size"]
    A = attn_spec(h, c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], False)
    F = ffn_spec(h, c["intermediate_size"], "swiglu")
    return dict(name=name, role=role, mixer="none", h=h, V=c["vocab_size"],
                tied=c.get("tie_word_embeddings", False),
                layers=[A, F] * c["num_hidden_layers"], ctx=c.get("max_position_embeddings"))


MODELS = [
    # A: mixer comparison at ~7B non-embedding
    nemotron("nvidia_NVIDIA-Nemotron-Nano-9B-v2.json", "Nano-9B-v2", "A"),
    qwen35("Qwen_Qwen3.5-9B.json", "Qwen3.5-9B", "A"),
    olmo_hybrid("allenai_Olmo-Hybrid-7B.json", "Olmo-Hybrid-7B", "A"),
    llama("unsloth_Meta-Llama-3.1-8B.json", "Llama-3.1-8B", "A"),
    # D: scale axis (Qwen3.5 dense)
    qwen35("Qwen_Qwen3.5-0.8B.json", "Qwen3.5-0.8B", "D"),
    qwen35("Qwen_Qwen3.5-2B.json", "Qwen3.5-2B", "D"),
    qwen35("Qwen_Qwen3.5-4B.json", "Qwen3.5-4B", "D"),
    qwen35("Qwen_Qwen3.5-27B.json", "Qwen3.5-27B", "D"),
    # E: pruning lineage (Nemotron Mamba-2 dense)
    nemotron("nvidia_NVIDIA-Nemotron-Nano-12B-v2.json", "Nano-12B-v2", "E"),
    nemotron("nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16.json", "Nemotron3-Nano-4B", "E"),
    # aux
    nemotron("nvidia_Nemotron-H-8B-Base-8K.json", "Nemotron-H-8B", "aux"),
]


# ---------------------------------------------------------------- cost
def layer_cost(s, B, L, bs):
    if s["kind"] == "attn":
        F = 2 * s["W"] * B + 4 * s["Hq"] * s["d"] * L * B
        M = s["W"] * BW_B + 2 * s["Hkv"] * s["d"] * L * B * KV_B
    elif s["kind"] in ("mamba", "gdn"):
        F = 2 * s["W"] * B + s["ops"] * s["S"] * B + 2 * s["C"] * s["K"] * B
        M = s["W"] * BW_B + 2 * s["S"] * B * bs + 2 * s["C"] * (s["K"] - 1) * B * KV_B
    else:
        F, M = 2 * s["W"] * B, s["W"] * BW_B
    return F, M


LABEL = {"attn": "Attention", "mamba": "Mamba-2", "gdn": "GDN", "ffn": "Dense FFN", "head": "LM head"}


def step(m, B, L, bs):
    out = {}
    for s in m["layers"]:
        F, M = layer_cost(s, B, L, bs)
        k = LABEL[s["kind"]]
        o = out.setdefault(k, dict(t=0.0, F=0.0, M=0.0))
        o["t"] += max(F / PEAK_FLOPS, M / PEAK_BW); o["F"] += F; o["M"] += M
    W = m["h"] * m["V"]
    F, M = 2 * W * B, W * BW_B
    out["LM head"] = dict(t=max(F / PEAK_FLOPS, M / PEAK_BW), F=F, M=M)
    return out


def weights(m):
    return (sum(s["W"] for s in m["layers"]) + m["h"] * m["V"] * (1 if m["tied"] else 2)) * BW_B


def cache_seq(m, L, bs):
    kv = sum(L * 2 * s["Hkv"] * s["d"] * KV_B for s in m["layers"] if s["kind"] == "attn")
    st = sum(s["S"] * bs + s["C"] * (s["K"] - 1) * KV_B for s in m["layers"]
             if s["kind"] in ("mamba", "gdn"))
    return kv, st


def bmax(m, L, bs):
    kv, st = cache_seq(m, L, bs)
    return max(0, int((0.9 * HBM - weights(m)) // (kv + st)))


def summarize(bs):
    rows = []
    for m in MODELS:
        kinds = [s["kind"] for s in m["layers"]]
        n_attn, n_rec = kinds.count("attn"), kinds.count("mamba") + kinds.count("gdn")
        kv8, st = cache_seq(m, 8192, bs)
        r = dict(model=m["name"], set=m["role"], mixer=m["mixer"], ctx=m["ctx"],
                 params_B=round(weights(m) / BW_B / 1e9, 2), weights_GB=round(weights(m) / 1e9, 1),
                 n_rec=n_rec, n_attn=n_attn,
                 kv_per_tok_KiB=round(kv8 / 8192 / 1024, 1), state_per_seq_MiB=round(st / 2**20, 1))
        for L in (8192, 131072):
            if m["ctx"] and L > m["ctx"]:
                r[f"L{L//1024}K"] = None
                continue
            b = bmax(m, L, bs)
            sb = step(m, b, L, bs) if b else None
            t = sum(v["t"] for v in sb.values()) if sb else None
            share = {k: round(v["t"] / t, 2) for k, v in sb.items()} if sb else None
            s1 = step(m, 1, L, bs)
            r[f"L{L//1024}K"] = dict(
                Bmax=b, tok_s=round(b / t) if b else 0,
                step_ms_B1=round(sum(v["t"] for v in s1.values()) * 1e3, 2),
                share_at_Bmax=share)
        rows.append(r)
    return rows


def plot(bs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core = [m for m in MODELS if m["role"] == "A"]
    batches = [1, 4, 16, 64, 256]
    order = ["Mamba-2", "GDN", "Attention", "Dense FFN", "LM head"]
    col = {"Mamba-2": "#1D9E75", "GDN": "#0F6E56", "Attention": "#D85A30",
           "Dense FFN": "#888780", "LM head": "#B4B2A9"}
    hat = {"Mamba-2": "", "GDN": "\\\\", "Attention": "//", "Dense FFN": "..", "LM head": ""}
    ctxs = [8192, 65536]                    # 64K keeps Olmo-Hybrid in range
    fig, axes = plt.subplots(len(ctxs), len(core), figsize=(15, 7.5), sharey=True)
    for i, L in enumerate(ctxs):
        for j, m in enumerate(core):
            ax = axes[i][j]
            b = bmax(m, L, bs)
            rows = [step(m, B, L, bs) for B in batches]
            bot = [0.0] * len(batches)
            for k in order:
                if k not in rows[0]:
                    continue
                tot = [sum(v["t"] for v in r.values()) for r in rows]
                sh = [r[k]["t"] / t for r, t in zip(rows, tot)]
                ax.bar(range(len(batches)), sh, bottom=bot, color=col[k], hatch=hat[k],
                       edgecolor="white", linewidth=0.5, label=k)
                bot = [x + y for x, y in zip(bot, sh)]
            for x, B in enumerate(batches):
                if B > b:
                    ax.axvspan(x - .5, x + .5, color="black", alpha=.08, lw=0)
            ax.set_xticks(range(len(batches))); ax.set_xticklabels(batches)
            ax.set_title(f"{m['name']} | L={L//1024}K | Bmax≈{b}", fontsize=10)
            ax.set_xlabel("batch size")
            if j == 0:
                ax.set_ylabel("share of roofline decode time")
    hs, ls = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in ls:
                hs.append(hh); ls.append(ll)
    fig.legend(hs, ls, loc="upper center", ncol=5, frameon=False)
    fig.text(.5, .005, f"A100-SXM4-80GB roofline, single GPU, bf16 weights/KV, recurrent state {bs*8}-bit. "
             "Shaded: exceeds 80 GB.", ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, .02, 1, .95))
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-bytes", type=int, default=4, help="recurrent state dtype bytes (4=fp32, 2=bf16)")
    a = ap.parse_args()
    res = summarize(a.state_bytes)
    json.dump(res, open(os.path.join(HERE, f"dense_summary_s{a.state_bytes*8}.json"), "w"), indent=1)
    plot(a.state_bytes, os.path.join(HERE, f"dense_layer_share_s{a.state_bytes*8}.png"))
    for r in res:
        def f(x):
            if x is None: return "  (ctx limit)"
            return f"B={x['Bmax']:<4d} {x['tok_s']:>6d} tok/s"
        print(f"{r['set']:3s} {r['model']:18s} {r['mixer']:7s} P={r['params_B']:5.2f}B W={r['weights_GB']:5.1f}GB "
              f"rec:attn={r['n_rec']}:{r['n_attn']} KV={r['kv_per_tok_KiB']:5.1f}KiB S={r['state_per_seq_MiB']:5.1f}MiB "
              f"| 8K {f(r['L8K'])} | 128K {f(r['L128K'])}")
