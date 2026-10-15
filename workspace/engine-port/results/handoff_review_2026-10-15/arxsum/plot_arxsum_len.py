"""arXiv-Sum(ccdv/arxiv-summarization, document/test) input-length distribution, Nano-9B-v2 tokenizer.
usage: python3 -I plot_arxsum_len.py <dir with test.parquet and tok/> <out.png>"""
import sys, os, pyarrow.parquet as pq, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tokenizers import Tokenizer
D, out = sys.argv[1], sys.argv[2]
arts = pq.read_table(os.path.join(D, "test.parquet"), columns=["article"]).column("article").to_pylist()
tk = Tokenizer.from_file(os.path.join(D, "tok", "nvidia_NVIDIA-Nemotron-Nano-9B-v2.json"))
I = [len(e.ids) for e in tk.encode_batch(arts, add_special_tokens=False)]
ink, mut, c1 = "#0b0b0b", "#52514e", "#2a78d6"
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist([min(i, 40000) for i in I], bins=range(0, 40001, 1000), color=c1, edgecolor="white", linewidth=1)
for x, lab in ((8760, "KV = state (≈8.8k)\nNano-9B-v2 fp32 state"), (16384, "16k filter\n(keep 91.7%)")):
    ax.axvline(x, color=mut, lw=1.2, ls="--"); ax.text(x + 400, ax.get_ylim()[1] * (.86 if x < 10000 else .55), lab, color=mut, fontsize=8)
ax.set_xlabel("input tokens (article, Nano-9B-v2 tokenizer; >40k folded into last bin)", color=ink)
ax.set_ylabel("requests", color=ink); ax.set_title("arXiv-Sum test split (n=6,440): input length", color=ink, fontsize=11, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e4e3df", lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(out, dpi=150)
