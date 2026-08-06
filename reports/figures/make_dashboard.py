#!/usr/bin/env python3
"""Build the single-file HTML dashboard from canon.py + the figure set.

    python reports/figures/make_dashboard.py

Renders web-resolution copies of every figure (both themes), quantises them to a
64-colour palette -- these are flat-colour charts, so that is lossless to the eye
and cuts the payload ~4x -- embeds them as data URIs, and writes one
self-contained page. No external requests: the Artifact CSP blocks them.

Korean narrative, English figures: no CJK font ships with the venv's matplotlib,
and the figures are meant to be submission-ready anyway.
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import canon as K          # noqa: E402
import style as S          # noqa: E402
import make_figures as M   # noqa: E402

OUT = HERE / "dashboard.html"
RAW = json.loads((HERE / "raw_extracted.json").read_text())

WEB_DPI = 96
QUANT_COLORS = 64


def render_data_uri(n: int, mode: str) -> str:
    slug, fn = M.FIGS[n]
    c = S.apply(mode)
    fig = fn(c)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=WEB_DPI)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB").quantize(
        colors=QUANT_COLORS, method=Image.MEDIANCUT, dither=Image.NONE)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# ── page pieces ───────────────────────────────────────────────────────────────
GRADE_CLASS = {
    "confirmed": "ok", "scoped": "ok", "serving": "ok", "strong": "ok",
    "micro": "warn", "partial": "warn", "strong/partial": "warn",
    "unverified": "mid", "unconfirmed": "mid", "unaudited": "mid",
    "barred": "bad", "retracted": "bad", "refuted": "bad",
}


def chip(text: str, grade: str) -> str:
    return f'<span class="chip {GRADE_CLASS.get(grade, "warn")}">{esc(text)}</span>'


def figure_block(n: int, uris: dict, caption_html: str, grade: str,
                 grade_label: str, title: str) -> str:
    cls = GRADE_CLASS.get(grade, "warn")
    return f"""
<figure class="fig stripe-{cls}">
  <figcaption>
    <div class="fig-head">
      <h3>{esc(title)}</h3>
      {chip(grade_label, grade)}
    </div>
    <div class="fig-note">{caption_html}</div>
  </figcaption>
  <div class="fig-img">
    <img class="fig-light" src="{uris[(n, 'light')]}" alt="{esc(title)}">
    <img class="fig-dark" src="{uris[(n, 'dark')]}" alt="{esc(title)}">
  </div>
</figure>"""


def table(headers: list[str], rows: list[list], caption: str = "",
          numeric_from: int = 1) -> str:
    NUM = ' class="num"'
    th = "".join(f"<th{NUM if i >= numeric_from else ''}>{esc(h)}</th>"
                 for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f"<td{NUM if i >= numeric_from else ''}>{esc(cell)}</td>"
                      for i, cell in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return (f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def build() -> str:
    uris = {(n, m): render_data_uri(n, m) for n in sorted(M.FIGS)
            for m in ("light", "dark")}

    # ── stat tiles ────────────────────────────────────────────────────────────
    tiles = [
        ("PD 분리 이득", "2–22×", "fused 대비, rate 4–6", "scoped",
         "비운영점(no-cudagraph) 한정"),
        ("layer-aware 死", "42 → 124 ms", "decode TPOT, 조율 구현 후", "serving",
         "창립 가설 최종 반증"),
        ("동적 제어 死", "5.4 σ", "d44 static > bind+GATE", "serving",
         "관대·tight SLO 양쪽"),
        ("decode-SM 레버", "2.36–2.91×", "ITL p50, decode SM 16→92", "confirmed",
         "레버 존재만 — 정책 이득 아님"),
        ("철회 원장", "15 건", "게시 후 되돌린 결과", "bad",
         "9건이 자체 적대적 감사로 적발"),
        ("성능 판정 (space 축)", "0 건", "C2 이후 프론티어", "mid",
         "구현 완료 ≠ 성능 주장"),
    ]
    tiles_html = "".join(f"""
    <div class="tile stripe-{GRADE_CLASS.get(g, 'warn')}">
      <div class="tile-k">{esc(k)}</div>
      <div class="tile-v">{esc(v)}</div>
      <div class="tile-s">{esc(s)}</div>
      <div class="tile-r">{esc(r)}</div>
    </div>""" for k, v, s, g, r in tiles)

    # ── tables (the dataviz relief rule: a table view always exists) ──────────
    he0 = K.HE0_LENIENT
    t_he0 = table(
        ["정책", "goodput (req/s)", "sd", "n"],
        [[lbl, f"{m:.3f}", (f"±{sd:.3f}" if sd else "—"), n]
         for lbl, m, sd, n, _ in he0["rows"]],
        "변화 trace (rate 3↔12, 3라운드, duration 합산) · cudagraph ON · TTFT ≤ 3 s")

    c2 = K.C2
    t_c2 = table(
        ["arm"] + [f"SM {s}" for s in c2["sm"]] + ["16→92"],
        [[arm.split("  ")[0] + "  " + arm.split("  ")[1]]
         + [("—" if v is None else f"{v:.2f}") for v in vals]
         + [f'{c2["headline"][arm.split()[0]][2]:.2f}×']
         for arm, vals in c2["arms"].items()],
        "decode ITL p50 (ms), batch=1, 파티션·batch 동시 매칭 · prefill 16 SM 고정")

    db = K.DIFF_B
    t_db = table(
        ["계열"] + [str(l) for l in db["L"]],
        [[name] + [("—" if v is None else f"{v:.3f}") for v in vals]
         for name, vals in db["series"].items()],
        "Diff B = attn / mamba SM-민감도비 · micro, no-cudagraph, 셀당 런 1개")

    pf = K.PREFILL_AXIS
    t_pf = table(
        ["arm"] + [f"P{s}" for s in pf["sm"]] + ["비", "ε 16→92", "게이트"],
        [[arm.split("  ")[0] + "  " + arm.split("  ")[1]]
         + [f"{v:.2f}" for v in vals]
         + [f'{pf["ratio"][arm.split()[0]]:.3f}×',
            f'{pf["elasticity_16_92"][arm.split()[0]]:.3f}',
            pf["gate"][arm.split()[0]]]
         for arm, vals in pf["arms"].items()],
        "TTFT~L 회귀 기울기 (µs/token), n=4 rep · decode 16 SM 고정 · 미감사")

    st = K.STICKY
    t_st = table(
        ["셀", "pooled p50 (ms)", "t95 CI", "사전등록 구간", "판정"],
        [[cell, f"{p50:.2f}", f"[{lo:.3f}, {hi:.3f}]", f"[{plo}, {phi}]",
          "HIT" if hit else "MISS (아래로)"]
         for cell, p50, lo, hi, plo, phi, hit in st["itl"]],
        "job 873015 · SPLIT 모집단 per-token p50 · 블록 n=8 · 사전등록 PREREG_S2")

    ser = RAW["p1_4model_goodput"]["series"]
    p1_rows = []
    for model in ("Nemotron-H-8B", "Granite-4.0-h-micro", "Falcon-H1-3B"):
        if model not in ser or "fused" not in ser[model]:
            continue
        f = {r["rate"]: r["goodput"] for r in ser[model]["fused"]["rows"]}
        a = {r["rate"]: r["goodput"] for r in ser[model]["agnostic"]["rows"]}
        for pol, d in (("fused", f), ("agnostic", a)):
            p1_rows.append([f"{model} · {pol}"] + [f"{d.get(r, float('nan')):.2f}"
                                                   for r in (1, 2, 3, 4, 6)])
    t_p1 = table(["모델 · 정책", "rate 1", "rate 2", "rate 3", "rate 4", "rate 6"],
                 p1_rows,
                 "goodput @ (TTFT ≤ 3 s ∧ TPOT ≤ 60 ms), req/s · "
                 "synthetic in2000/out96 · no-cudagraph, n=1/셀")

    gates = "".join(f"<li>{esc(g[3:] if g[1] == '.' else g[4:])}</li>" for g in K.GATES)

    # ── controller anatomy: what slo-aware / bind / bind+GATE actually are ────
    ctrl_cards = "".join(f"""
    <article class="ctrl">
      <div class="ctrl-head">
        <h4>{esc(ct["id"])}</h4>
        <span class="ctrl-gp">{ct["goodput"]:.3f} req/s</span>
      </div>
      <p class="ctrl-full">{esc(ct["full"])}</p>
      <p class="ctrl-code"><code>{esc(ct["fn"])}()</code> · line {ct["line"]}<br>
         <code>{esc(ct["env"])}</code></p>
      <dl>
        <dt>신호</dt><dd>{esc(ct["signal"])}</dd>
        <dt>규칙</dt><dd>{esc(ct["rule"])}</dd>
        <dt class="bad">무엇이 깨졌나</dt><dd>{esc(ct["broke"])}</dd>
      </dl>
    </article>""" for ct in K.CONTROLLERS)

    t_causes = table(
        ["후보 死因", "판정", "근거"],
        [[d["cause"], d["verdict"], d["evidence"]] for d in K.DEATH_CAUSES],
        "동적이 지는 이유 — 추론이 아니라 직접 계측으로 후보를 제거했다", numeric_from=99)

    gr = K.GATE_RATIONALE
    t_little = table(
        ["static split", "running-batch 동시성 N", "TTFT (s)"],
        [[s, n, f"{t:.2f}"] for s, n, t in gr["littles_law"]],
        "게이트 congestion 가드의 설계 근거 (가드 docstring에 기록된 값) — "
        "⚠ TTFT magnitude는 폐기된 stationary 벤치(n=1)라 인용 금지")

    t_env = table(
        ["env", "기본값", "역할"],
        [["PDMUX_SLO_MODE", "(unset)", "binding → Step E 컨트롤러로 전환"],
         ["PDMUX_SLO_FEAS_GATE", "(unset)", "설정 시 Step G 게이트 활성 (off면 byte-identical)"],
         ["PDMUX_TTFT_SLO_MS", "3000", "prefill-slack 분모"],
         ["PDMUX_TPOT_SLO_MS", "60", "decode-slack 분모 · ITL 가드 임계"],
         ["PDMUX_TPOT_HI / _LO", "0.85 / 0.65", "v7b의 decode-ward / prefill-ward 발화 임계"],
         ["PDMUX_QDEPTH_TARGET", "4", "v7b prefill-ward 게이팅 조건 (qdepth > 4)"],
         ["PDMUX_SLO_PF_URGENCY", "0.5", "prefill이 '급하다'고 볼 slack 비율"],
         ["PDMUX_SLO_ANCHOR_IDX", "(lo+hi)//2", "둘 다 여유일 때 drift할 tuned-static anchor"],
         ["PDMUX_SLO_DWELL", "3", "스위치 후 최소 체류 스텝 (진동 억제)"],
         ["PDMUX_SLO_SAT_LATCH", "20", "포화 감지 후 anchor에 latch할 스텝 수"],
         ["PDMUX_SLO_FEAS_OCC", "0.85", "congestion 가드: batch ≥ 0.85·cap이면 거부"],
         ["PDMUX_SLO_FEAS_MARGIN", "0.9", "ITL 가드: 예측 ITL < 0.9·TPOT_SLO"]],
        "런타임 손잡이 — 전부 " + K.CTRL_SRC.split("/")[-1] + " 에서 읽는다",
        numeric_from=99)

    # ── Diff B: the three series are not three measurements of one thing ─────
    dbs = K.DIFF_B_STEPS
    db_steps = "".join(f"""
    <article class="step">
      <div class="step-head">
        <span class="step-n">{st["step"]}</span>
        <div>
          <h4>{esc(st["frm"])} <span class="arr">→</span> {esc(st["to"])}</h4>
          <p class="step-axis">{esc(st["axis"])}</p>
        </div>
      </div>
      <dl>
        <dt>무엇을 바꿨나</dt><dd>{esc(st["what"])}</dd>
        <dt>왜</dt><dd>{esc(st["why"])}</dd>
        <dt>비대칭 / 함의</dt><dd>{esc(st["asymmetry"])}</dd>
        <dt>검증</dt><dd>{esc(st["proof"])}</dd>
        <dt>효과</dt><dd class="mono">{esc(st["effect"])}</dd>
      </dl>
    </article>""" for st in dbs)

    fb = K.FIRST_BLOCK_INFLATION
    t_fb = table(
        ["층 타입"] + [f"L={x}" for x in fb["L"]],
        [["attn (분자)"] + [f"{v:.3f}" for v in fb["attn"]],
         ["mamba (분모)"] + [f"{v:.3f}" for v in fb["mamba"]]],
        "첫 블록 ÷ steady 값 (clean WIDE, B=1, SM=108) — 1.0이면 warm-up 편향 없음")

    esc_hatch = table(
        ["단계", "결과", "결정 수치"],
        [[e["stage"], e["result"], e["num"]] for e in K.ESCAPE_HATCHES],
        "동적 제어의 탈출구 4단계 — 전부 봉쇄", numeric_from=99)

    def _rk(kind: str) -> str:
        if "refut" in kind or kind in ("retracted", "retired", "citation-barred"):
            return "bad"
        if kind in ("benchmark retired", "abandoned", "downgraded"):
            return "mid"
        return "warn"

    retr = "".join(f"""
      <li class="retr stripe-{_rk(kind)}">
        <span class="rw">{esc(what)}</span>
        <span class="rk">{esc(kind)}</span>
        <span class="ry">{esc(why)}</span>
      </li>""" for date, what, kind, why in K.RETRACTIONS)

    return f"""<title>PD-mux 실험 수치 시각화 — 연구 arc 전체</title>
<style>
:root {{
  color-scheme: light;
  --ground:#f6f7f9; --panel:#ffffff; --panel-2:#fbfcfd;
  --ink:#111419; --ink-2:#4b535f; --ink-3:#79828f;
  --rule:#e3e7ed; --rule-2:#eef1f5;
  --accent:#2a78d6; --accent-soft:#eaf2fd;
  --ok:#0ca30c; --warn:#b8820c; --mid:#ec835a; --bad:#d03b3b;
  --mono: ui-monospace,"SFMono-Regular","Cascadia Mono",Menlo,Consolas,monospace;
  --sans: ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
          "Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif;
  --maxw: 1180px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground:#101317; --panel:#181c21; --panel-2:#1d2228;
    --ink:#f0f3f7; --ink-2:#a8b2be; --ink-3:#7c8592;
    --rule:#272c33; --rule-2:#20252b;
    --accent:#3987e5; --accent-soft:#16243a;
    --ok:#3fc23f; --warn:#e0a52a; --mid:#ef9068; --bad:#e05d5d;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground:#101317; --panel:#181c21; --panel-2:#1d2228;
  --ink:#f0f3f7; --ink-2:#a8b2be; --ink-3:#7c8592;
  --rule:#272c33; --rule-2:#20252b;
  --accent:#3987e5; --accent-soft:#16243a;
  --ok:#3fc23f; --warn:#e0a52a; --mid:#ef9068; --bad:#e05d5d;
}}

* {{ box-sizing: border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:var(--maxw); margin:0 auto; padding:0 20px 88px; }}

/* ── masthead ─────────────────────────────────────────────── */
header.mast {{
  border-bottom:1px solid var(--rule); background:var(--panel);
  padding:38px 0 26px; margin-bottom:34px;
}}
.eyebrow {{
  font-family:var(--mono); font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-3); margin-bottom:14px;
}}
h1 {{
  font-size:clamp(27px,3.9vw,42px); line-height:1.14; letter-spacing:-.021em;
  font-weight:700; margin:0 0 16px; text-wrap:balance; max-width:19ch;
}}
.verdict {{
  font-size:clamp(16px,1.9vw,19px); line-height:1.55; color:var(--ink-2);
  max-width:64ch; margin:0 0 22px;
}}
.verdict b {{ color:var(--ink); font-weight:650; }}
.mast-meta {{
  display:flex; flex-wrap:wrap; gap:8px 22px; font-family:var(--mono);
  font-size:11.5px; color:var(--ink-3); border-top:1px solid var(--rule-2);
  padding-top:16px;
}}

/* ── tiles ────────────────────────────────────────────────── */
.tiles {{
  display:grid; gap:12px; margin-bottom:44px;
  grid-template-columns:repeat(auto-fit,minmax(224px,1fr));
}}
.tile {{
  background:var(--panel); border:1px solid var(--rule); border-radius:3px;
  padding:15px 16px 14px 18px; position:relative;
}}
.tile-k {{ font-size:12.5px; color:var(--ink-2); letter-spacing:.01em; }}
.tile-v {{
  font-family:var(--mono); font-size:26px; font-weight:600; line-height:1.2;
  margin:5px 0 3px; font-variant-numeric:tabular-nums; letter-spacing:-.02em;
}}
.tile-s {{ font-size:12px; color:var(--ink-3); line-height:1.45; }}
.tile-r {{
  font-size:11.5px; color:var(--ink-3); margin-top:9px;
  padding-top:8px; border-top:1px dashed var(--rule);
}}

/* severity stripe — encodes evidence grade, not decoration */
.stripe-ok, .stripe-warn, .stripe-mid, .stripe-bad {{ border-left-width:3px; border-left-style:solid; }}
.stripe-ok {{ border-left-color:var(--ok); }}
.stripe-warn {{ border-left-color:var(--warn); }}
.stripe-mid {{ border-left-color:var(--mid); }}
.stripe-bad {{ border-left-color:var(--bad); }}

/* ── sections ─────────────────────────────────────────────── */
section {{ margin:0 0 52px; scroll-margin-top:20px; }}
.sec-head {{
  display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
  border-bottom:1px solid var(--rule); padding-bottom:11px; margin-bottom:22px;
}}
.sec-head h2 {{
  font-size:22px; letter-spacing:-.014em; font-weight:680; margin:0;
}}
.sec-head .sec-sub {{ font-size:13.5px; color:var(--ink-3); }}
.lede {{ max-width:70ch; color:var(--ink-2); margin:0 0 22px; font-size:15px; }}
.lede b {{ color:var(--ink); font-weight:640; }}

/* ── figures ──────────────────────────────────────────────── */
.fig {{
  margin:0 0 26px; background:var(--panel); border:1px solid var(--rule);
  border-radius:3px; overflow:hidden;
}}
.fig figcaption {{ padding:16px 20px 14px; border-bottom:1px solid var(--rule-2); }}
.fig-head {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:7px; }}
.fig-head h3 {{ font-size:16.5px; font-weight:650; margin:0; letter-spacing:-.01em; }}
.fig-note {{ font-size:13.5px; color:var(--ink-2); max-width:78ch; }}
.fig-note b {{ color:var(--ink); font-weight:640; }}
.fig-note code {{
  font-family:var(--mono); font-size:12px; background:var(--panel-2);
  border:1px solid var(--rule-2); border-radius:2px; padding:.5px 4px;
}}
.fig-img {{ overflow-x:auto; background:var(--panel-2); }}
.fig-img img {{ display:block; width:100%; min-width:760px; height:auto; }}
.fig-dark {{ display:none; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) .fig-light {{ display:none; }}
  :root:not([data-theme="light"]) .fig-dark {{ display:block; }}
}}
:root[data-theme="dark"] .fig-light {{ display:none; }}
:root[data-theme="dark"] .fig-dark {{ display:block; }}
:root[data-theme="light"] .fig-light {{ display:block; }}
:root[data-theme="light"] .fig-dark {{ display:none; }}

/* ── chips ────────────────────────────────────────────────── */
.chip {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.07em;
  text-transform:uppercase; padding:3px 8px; border-radius:2px;
  border:1px solid currentColor; white-space:nowrap; font-weight:600;
}}
.chip.ok {{ color:var(--ok); }}
.chip.warn {{ color:var(--warn); }}
.chip.mid {{ color:var(--mid); }}
.chip.bad {{ color:var(--bad); }}

/* ── tables ───────────────────────────────────────────────── */
details.data {{
  border:1px solid var(--rule); border-radius:3px; background:var(--panel);
  margin:0 0 26px;
}}
details.data > summary {{
  cursor:pointer; padding:12px 18px; font-size:13.5px; color:var(--ink-2);
  font-weight:600; list-style:none; display:flex; align-items:center; gap:9px;
}}
details.data > summary::-webkit-details-marker {{ display:none; }}
details.data > summary::before {{
  content:"+"; font-family:var(--mono); color:var(--accent); font-size:15px;
  line-height:1;
}}
details.data[open] > summary::before {{ content:"−"; }}
details.data > summary:hover {{ color:var(--ink); }}
details.data > summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:-2px; }}
.tw {{ overflow-x:auto; border-top:1px solid var(--rule-2); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
caption {{
  text-align:left; padding:12px 18px 4px; font-size:12px; color:var(--ink-3);
  font-family:var(--mono); line-height:1.5;
}}
th, td {{
  padding:7px 14px; text-align:left; border-bottom:1px solid var(--rule-2);
  white-space:nowrap;
}}
th {{
  font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink-3); font-weight:600; background:var(--panel-2);
}}
td.num, th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
tbody tr:last-child td {{ border-bottom:none; }}

/* ── retraction ledger ────────────────────────────────────── */
ol.gates {{ margin:0; padding-left:1.35em; max-width:78ch; color:var(--ink-2); font-size:14.5px; }}
ol.gates li {{ margin-bottom:9px; }}
ol.gates li::marker {{ font-family:var(--mono); color:var(--accent); font-size:12.5px; }}
ul.ledger {{ list-style:none; margin:0; padding:0; }}
ul.ledger li {{
  display:grid; gap:2px 14px; padding:11px 0 11px 14px;
  border-bottom:1px solid var(--rule-2);
  grid-template-columns:1fr auto;
}}
.rw {{ font-size:14px; font-weight:600; grid-row:1; }}
.rk {{
  font-family:var(--mono); font-size:10.5px; text-transform:uppercase;
  letter-spacing:.06em; grid-row:1; text-align:right;
}}
li.stripe-bad .rk {{ color:var(--bad); }}
li.stripe-mid .rk {{ color:var(--mid); }}
li.stripe-warn .rk {{ color:var(--warn); }}
.ry {{ grid-column:1/3; font-size:12.5px; color:var(--ink-3); }}

/* ── sub-head + controller cards ──────────────────────────── */
.sub-head {{
  margin:40px 0 10px; padding-bottom:9px; border-bottom:1px dashed var(--rule);
}}
.sub-head h3 {{ font-size:17px; font-weight:660; margin:0; letter-spacing:-.01em; }}
.sub-head code {{
  font-family:var(--mono); font-size:14px; color:var(--accent); font-weight:600;
}}
.ctrl-grid {{
  display:grid; gap:12px; margin:0 0 26px;
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr));
}}
.ctrl {{
  background:var(--panel); border:1px solid var(--rule); border-radius:3px;
  padding:16px 18px 14px; display:flex; flex-direction:column;
}}
.ctrl-head {{
  display:flex; align-items:baseline; justify-content:space-between; gap:10px;
}}
.ctrl-head h4 {{
  margin:0; font-size:16px; font-weight:680; letter-spacing:-.012em;
  font-family:var(--mono);
}}
.ctrl-gp {{
  font-family:var(--mono); font-size:12px; color:var(--ink-3);
  font-variant-numeric:tabular-nums; white-space:nowrap;
}}
.ctrl-full {{ margin:3px 0 0; font-size:12.5px; color:var(--ink-2); }}
.ctrl-code {{
  margin:9px 0 0; padding:8px 10px; background:var(--panel-2);
  border:1px solid var(--rule-2); border-radius:2px; font-size:11.5px;
  color:var(--ink-3); line-height:1.7; overflow-x:auto;
}}
.ctrl-code code {{ font-family:var(--mono); color:var(--accent); }}
.ctrl dl {{ margin:13px 0 0; display:grid; gap:3px 0; }}
.ctrl dt {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-3); margin-top:9px; font-weight:600;
}}
.ctrl dt.bad {{ color:var(--bad); }}
.ctrl dd {{ margin:0; font-size:13px; color:var(--ink-2); line-height:1.6; }}

/* ── numbered transformation steps ────────────────────────── */
.step-grid {{
  display:grid; gap:12px; margin:0 0 24px;
  grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
}}
.step {{
  background:var(--panel); border:1px solid var(--rule); border-radius:3px;
  padding:16px 18px 15px;
}}
.step-head {{ display:flex; gap:12px; align-items:flex-start; }}
.step-n {{
  flex:0 0 auto; width:24px; height:24px; border-radius:50%;
  background:var(--accent-soft); color:var(--accent); font-family:var(--mono);
  font-size:12.5px; font-weight:700; display:flex; align-items:center;
  justify-content:center; margin-top:1px;
}}
.step-head h4 {{ margin:0; font-size:14.5px; font-weight:660; letter-spacing:-.008em; }}
.step-head .arr {{ color:var(--ink-3); font-weight:400; padding:0 2px; }}
.step-axis {{
  margin:3px 0 0; font-size:11.5px; color:var(--accent); font-family:var(--mono);
  letter-spacing:.01em;
}}
.step dl {{ margin:13px 0 0; }}
.step dt {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.07em;
  text-transform:uppercase; color:var(--ink-3); margin-top:10px; font-weight:600;
}}
.step dd {{ margin:0; font-size:13px; color:var(--ink-2); line-height:1.6; }}
.step dd.mono {{
  font-family:var(--mono); font-size:11.5px; color:var(--ink); margin-top:2px;
}}

/* ── callout ──────────────────────────────────────────────── */
.callout {{
  border:1px solid var(--rule); border-left:3px solid var(--accent);
  background:var(--panel); padding:16px 20px; border-radius:3px;
  font-size:14px; color:var(--ink-2); margin:0 0 26px; max-width:82ch;
}}
.callout b {{ color:var(--ink); }}
.callout .ct {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--accent); display:block; margin-bottom:7px;
}}

footer.foot {{
  border-top:1px solid var(--rule); padding-top:22px; margin-top:12px;
  font-family:var(--mono); font-size:11.5px; color:var(--ink-3); line-height:1.75;
}}
footer.foot a {{ color:var(--accent); }}
@media (max-width:640px) {{
  ul.ledger li {{ grid-template-columns:1fr; }}
  .rk {{ text-align:left; }}
  .ry {{ grid-column:1; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>

<header class="mast"><div class="wrap">
  <div class="eyebrow">prefill-layer-alloc · hybrid-LLM PD-mux serving</div>
  <h1>PD 분리는 이득, 그 위의 정책은 전부 死</h1>
  <p class="verdict">
    창립 가설은 <b>“hybrid 모델의 층 타입(attention vs mamba)마다 SM을 다르게 주면
    type-agnostic 분할을 이긴다”</b>였다. 시뮬레이터에서는 살아 있었고,
    <b>서빙 실측에서는 모든 형태가 죽었다</b> — decode-side → coordinated → prefill-side로
    세 번 후퇴한 뒤, 마지막 피난처였던 시간축 동적 제어마저 best-static을 넘지 못했다.
    실무 권고는 <b>peak decode 부하 기준 decode-heavy static 고정</b>이다.
  </p>
  <div class="mast-meta">
    <span>A100 80GB PCIe ×1 · 108 SM green context</span>
    <span>sglang v0.5.10 lineage · torch 2.9.1+cu130 · bf16</span>
    <span>정본: PROJECT_STATUS.md · CONSENSUS.md · research_arc.md</span>
  </div>
</div></header>

<div class="wrap">

<div class="tiles">{tiles_html}
</div>

<section id="arc">
  <div class="sec-head"><h2>가설은 어떻게 죽었나</h2>
    <span class="sec-sub">세 축 · 12 단계</span></div>
  <p class="lede">
    이 아크를 관통하는 두 축이 있다. 하나는 <b>Diff A(층 타입 간 <i>비용비</i>)</b>와
    <b>Diff B(층 타입 간 <i>SM 민감도비</i> = 재배분의 진짜 lever)</b>의 혼동 —
    가설이 요구한 건 Diff B인데 관찰되던 건 Diff A였다. 다른 하나는
    <b>micro-measurement가 서빙을 예측하지 못한다</b>는 것으로, 이 실패가 네 번 결론을 뒤집었다.
  </p>
  {figure_block(1, uris,
    "layer-type 축과 time 축의 모든 판정은 <b>실엔진 서빙 측정</b>이다. space 축이 현재 "
    "열린 프론티어이며, <b>아직 성능 판정은 0건</b>이다.",
    "serving", "arc 전체", "연구 arc — 12 단계 kill-chain")}
</section>

<section id="alive">
  <div class="sec-head"><h2>살아남은 것</h2>
    <span class="sec-sub">positive 축 — 둘 다 조건부다</span></div>
  <p class="lede">
    살아남은 positive는 셋뿐이고(<b>PD 분리 이득 · 운영점 cudagraph · decode-SM 레버</b>),
    셋 다 <b>동적 제어의 전제조건이지 동적 제어가 이긴다는 증거가 아니다</b>.
    그리고 최근 감사에서 둘이 흔들렸다.
  </p>
  {figure_block(2, uris,
    "이 4-모델 캠페인은 <b>전부 <code>--disable-cuda-graph</code></b>, 즉 비운영점이고 "
    "셀당 n=1이다. 더 중요한 건 반대 증거다 — fused의 死因은 TPOT가 60 ms SLO를 넘는 것인데 "
    "<b>cudagraph가 바로 그 벽을 없앤다</b>(rate 4: 82.41 → 54.04 ms = SLO 통과). "
    "운영점 대조는 어느 모델에서도 측정된 적이 없다.",
    "scoped", "스코프 축소", "P1 — PD 분리가 fused를 이긴다")}
  <details class="data"><summary>수치 보기 — 4모델 goodput</summary>{t_p1}</details>

  {figure_block(6, uris,
    "prefill을 16 SM에 고정한 채 decode만 16→92로 올리면 ITL p50이 2.36–2.91× 개선되고, "
    "<b>4개 arm(pure SSM / pure Transformer / additive hybrid / substitutive hybrid)이 "
    "사실상 동일</b>하다. 모델 계열은 SM 민감도가 아니라 <b>절대 비용</b>에서 갈린다. "
    "단 이건 <b>등량곡선</b>이라 레버의 존재만 확립하며, 2.36–2.91×는 <b>끝점 비</b>다 — "
    "국소 탄력도가 16→24 구간(0.77–0.88)과 44→92 구간(0.09–0.35)에서 4× 다르므로 "
    "<b>D=44 이상 구간에 적용 금지</b>.",
    "confirmed", "CONFIRMED (scoped)", "C2 — 운영점의 decode-SM 레버")}
  <details class="data"><summary>수치 보기 — C2 arm × SM</summary>{t_c2}</details>
</section>

<section id="dead">
  <div class="sec-head"><h2>죽은 것</h2>
    <span class="sec-sub">layer-type 축 全형태 · 시간축 동적 제어</span></div>
  <p class="lede">
    두 negative는 <b>서로 독립인 증거</b> 위에 서 있어서, 어느 하나가 흔들려도 다른 하나는 유지된다.
    layer-type 쪽 뼈대는 <b>작동하는 구현으로 end-to-end goodput을 직접 비교한 결과</b>라
    Diff A/Diff B 기전 서사가 전부 틀려도 판정은 바뀌지 않는다.
  </p>
  {figure_block(3, uris,
    "S1에서 “첫 layer-aware는 이진 strawman이고 내 구현이 미조율이었다”는 반론이 나왔고, "
    "실제로 같은 54 SM이 미조율 121 ms · 조율 42 ms였다. <b>반증된 건 가설이 아니라 내 측정</b>이었고 "
    "가설은 되살아났다. 그래서 S2에서 coordinated per-type을 <b>진짜로 구현</b>했고 — "
    "<b>124 ms</b>가 나왔다. <code>COORD_OPT</code>로 47%를 회수해도(124→85 ms) 부호는 견고하다.",
    "serving", "서빙 실증", "layer-aware — 조율이 문제가 아니었다")}

  <div class="sub-head" id="controllers">
    <h3>동적 제어 기법 해부 — <code>slo-aware</code> / <code>bind</code> / <code>bind + GATE</code>가 무엇인가</h3>
  </div>
  <p class="lede">
    아래 그래프의 “dynamic” 세 arm은 <b>전부 같은 메커니즘</b>이다 — prefill-layer-span마다
    green-context split 인덱스를 ±1 이동시키고, 스위치 후 최소 3스텝 체류한다.
    <b>다른 것은 결정 규칙 하나뿐</b>이며, 정본이 아니라 엔진 소스가 출처다
    (<code>{esc(K.CTRL_SRC)}</code>).
  </p>
  <div class="ctrl-grid">{ctrl_cards}</div>

  <div class="callout">
    <span class="ct">게이트가 비대칭인 이유</span>
    {esc(gr["asymmetry"])}
    <br><br><b>congestion 가드가 주(主)인 이유</b> — {esc(gr["why_congestion_primary"])}
  </div>
  <details class="data"><summary>수치 보기 — 게이트 설계 근거 (Little's law)</summary>{t_little}
    <p style="padding:10px 18px 14px;margin:0;font-size:12.5px;color:var(--ink-3)">
      {esc(gr["littles_note"])}
    </p>
  </details>

  {figure_block(12, uris,
    "패널 A는 세 컨트롤러의 신호·규칙·무엇이 깨졌는지를 나란히 놓은 것이고, "
    "패널 B는 <b>각 컨트롤러가 실제로 앉은 자리</b>를 static 곡선 위에 겹친 것이다. "
    "<code>bind</code>는 dec_sm 16–24에서 진동하며 최적(44)에 <b>한 번도 도달하지 못하고</b>, "
    "<code>bind + GATE</code>는 d34에 얼어붙는데 <b>자기가 얼어붙은 static보다도 0.039 낮다</b> "
    "— 그게 정착 비용이다. 즉 손실은 <b>스위칭이 아니라 앉은 위치</b>다.",
    "serving", "서빙 · n≥4", "컨트롤러 해부 — 그리고 어디에 앉았나")}
  <details class="data"><summary>수치 보기 — 死因 후보 제거</summary>{t_causes}</details>
  <details class="data"><summary>수치 보기 — 런타임 손잡이(env)</summary>{t_env}</details>

  <div class="callout">
    <span class="ct">그래서 게이트는 무엇이었나</span>
    <b>지능적 제어가 아니라 one-way ratchet auto-tuner였다.</b>
    <code>d24→d34</code>로 <b>한 번 이동한 뒤 prefill-ward 복귀를 113회 전부 거부</b>했다 —
    <code>bs=47 ≥ 0.85×48</code>이 상시 참이었기 때문이다. 그래서 fig 04에서
    <code>bind + GATE</code>가 <code>bind</code>보다 높은 건 제어가 똑똑해서가 아니라
    <b>static 하나로 굳었기 때문</b>이고, 게다가 <b>틀린 static</b>이다(정지 규칙 tpot&lt;51 ms가
    최적점에 못 미쳐 발동). 게이트의 측정된 가치는 성능이 아니라 <b>견고성</b>이다 —
    붕괴 1/4 → 0/9, 분산 16× 타이트.
  </div>

  {figure_block(4, uris,
    "§1-16은 한때 tight SLO에서 동적이 이기는 것처럼 보인다고 기록했으나 §1-17이 반증했다 — "
    "<b>재스코어는 3s로 튜닝된 컨트롤러가 정착한 위치를 사후 채점</b>한 것이었다. "
    "그 SLO로 <b>재튜닝해서 직접 측정</b>하면 동적은 오히려 더 나빠진다. "
    "⇒ “decode-heavy static 지배”는 SLO 엄격도와 무관한 결론이다.",
    "serving", "HE0 · n≥4", "동적 제어는 best-static을 못 넘는다")}
  <details class="data"><summary>수치 보기 — HE0 정책 순위</summary>{t_he0}
    <p style="padding:10px 18px 14px;margin:0;font-size:12.5px;color:var(--ink-3)">
      정본 불일치 1건: <code>research_arc.md</code>는 d16을 <b>2.817</b>로,
      종합 문서 2건은 <b>2.846</b>으로 적는다. 여기서는 최신·중복 기재된 2.846을 썼다.
    </p>
  </details>

  {figure_block(5, uris,
    "차별의 ~95%가 과부하 phase에서 나오고 저부하 phase는 split에 무관심하다. "
    "decode 과다공급이 저부하에서 거의 무해하므로 <b>HI의 최적이 LO에서도 공짜</b>다 ⇒ "
    "“항상 HI 최적을 쓴다”=decode-heavy static이 정의상 최선이고 동적은 과도만 지불한다. "
    "<b>동적이 이기려면 regime 간 최적이 충돌해야 하는데 이 워크로드엔 그 구간이 없다.</b> "
    "부기: LO 지표는 도착률에 <b>천장 절단</b>돼 있어 무신호이지 레버 부재의 증명은 아니다.",
    "serving", "구조적 이유", "왜 동적 제어는 쫓아갈 대상이 없나")}
  <details class="data"><summary>수치 보기 — 탈출구 4단계</summary>{esc_hatch}</details>
</section>

<section id="measure">
  <div class="sec-head"><h2>측정 자체를 의심하다</h2>
    <span class="sec-sub">벤치 폐기 · 계측 결함 · 기전 서사 재작성</span></div>
  <p class="lede">
    이 프로젝트에서 가장 값비싼 교훈 둘은 정책이 아니라 <b>계측</b>에서 나왔다.
    하나는 goodput이 CDF의 가장 가파른 지점에서 평가되고 있었다는 것,
    다른 하나는 비대칭 버킷과 리셋 안 되는 누산기가 <b>조용히 상향 편향된 비율</b>을 만들었다는 것이다.
  </p>
  {figure_block(9, uris,
    "동일 config·동일 프롬프트 fingerprint인데 goodput이 2× 흔들렸다. 설명 대상은 3.5×가 아니라 "
    "<b>3%</b>였다 — 증폭기는 rate 8이 하필 <b>TTFT ≈ SLO 경계</b>에 앉은 것이다. "
    "과부하 큐의 TTFT 평탄역이 3% 결손에 1.5 s → 3.7 s로 이동해 임계선을 넘으면 400/400이 206/400이 된다. "
    "⇒ <b>stationary ShareGPT r8 = 정책 비교 벤치로 폐기</b>.",
    "serving", "벤치 폐기", "메트릭 절벽 — 노이즈가 아니었다")}

  {figure_block(8, uris,
    "“lever는 짧은 L에서 열린다”는 <b>철회됐다</b>. 두 독립 job이 보고값은 4.5% 어긋나는데 "
    "steady 값은 <b>0.08%로 일치</b>한다 — 1.32 vs 1.38 차이는 물리가 아니라 “몇 번 돌았나”였다. "
    "<b>판정은 불변이고 바뀐 건 기전 서사뿐</b> — 서빙 실증이 판정을 지탱한다.",
    "barred", "계측 결함 · 강등", "Diff B — 철회가 서사를 바꾼 방식")}

  <div class="sub-head" id="diffb-series">
    <h3>세 계열의 차이 — <code>as reported</code> / <code>steady</code> / <code>policy unit</code></h3>
  </div>
  <p class="lede">
    이 셋은 <b>같은 것을 세 번 잰 값이 아니다.</b> <b>서로 직교하는 두 축</b>을 순서대로 적용한 결과다 —
    ① <b>표본을 정리</b>하고(같은 추정량·같은 단위), ② <b>단위를 바꾼다</b>(같은 표본·다른 집계).
    이 둘을 뭉뚱그린 것이 바로 “lever는 짧은 L에서 열린다”가 확증처럼 보이게 만든 원인이다.
  </p>
  <div class="step-grid">{db_steps}</div>

  <div class="callout">
    <span class="ct">어느 것이 가설에 답하는가</span>
    <b>정책 단위다.</b> 창립 가설이 요구한 건 <b>런타임 정책이 당길 수 있는 lever</b>였고,
    정책은 커널 하나가 아니라 <b>window 단위</b>로 SM을 재배분한다. 커널 단위 값은
    기전(mechanism) 층위의 양이다.
    <br><br>
    <b>단, 정책 단위는 독립 증거가 아니다</b> — window의 약 90%가 분모에 있는 그 mamba mixer라
    커널 비가 무엇이든 <code>R_policy</code>는 구조적으로 1에 끌린다.
    “정책 단위에서 소멸”은 <i>정책이 착취할 게 없다</i>는 진술이지, <i>lever가 없다</i>는 두 번째 확증이 아니다.
    <br><br>
    ⚠ <code>n_indep = 1</code>(셀당 런 1개) — 위 CI는 <b>런 내부 블록 정밀도</b>이지 재현성이 아니다.
    그리고 <b>L=256은 점값이 아니라 밴드 [1.24, 1.34]로만</b> 인용할 수 있다.
  </div>
  <details class="data"><summary>수치 보기 — Diff B 보고 vs steady vs 정책 단위</summary>{t_db}</details>
  <details class="data"><summary>수치 보기 — ①의 기전: 첫 블록 부풀림 (attn vs mamba)</summary>{t_fb}
    <p style="padding:10px 18px 14px;margin:0;font-size:12.5px;color:var(--ink-3)">
      {esc(fb["closure"])}
    </p>
  </details>
</section>

<section id="frontier">
  <div class="sec-head"><h2>현재 프론티어</h2>
    <span class="sec-sub">space 축 — 성능 판정 0건</span></div>
  <p class="lede">
    긴장 A(<b>HE2</b>: 최적이 안 움직인다 vs <b>C2</b>: 레버는 실재한다)는 아직 닫히지 않았다.
    프론티어에서 필요한 대상은 <b>ITL(D) vs TTFT(108−D)</b>인데 그건 미측정이고,
    그 전에 측정 층 자체의 모순을 먼저 풀어야 했다.
  </p>
  {figure_block(7, uris,
    "C2의 거울상 캠페인. 여기서 <b>반증된 추론을 기록해 둔다</b>: “prefill 기울기(5.0×)가 "
    "decode 기울기(2.4–2.9×)보다 크니 프론티어에서 prefill이 더 급하다”는 <b>지지되지 않는다</b>. "
    "국소 탄력도의 <i>구조</i>가 다르기 때문이다 — prefill ε는 거의 상수(0.71–1.05)인데 "
    "decode ε는 9× 변하며 D≳44에서 포화한다. <b>SM 1개를 옮길 때의 부호가 동작점마다 뒤집힌다.</b>",
    "unaudited", "미감사 · 정본 인용 금지", "prefill 축 — 그리고 여기서 반증된 추론")}
  <details class="data"><summary>수치 보기 — prefill 축 기울기</summary>{t_pf}</details>

  {figure_block(10, uris,
    "같은 arm·같은 서버 플래그·매칭 batch에서 두 job의 “decode 16 SM” ITL이 <b>2.6× 달랐다</b>"
    "(28.79 vs 11.09 ms). 이것이 하위 결과 전부가 딛고 선 바닥이었다. "
    "<code>PDMUX_STICKY_PARTITION</code>으로 파티션을 실제로 붙잡자 답이 나왔다 — "
    "<b>872077의 ‘d16’ 셀은 애초에 16 SM에 있지 않았다</b>(decode-busy 시간의 96.2%를 D108에서 보냄). "
    "라벨을 붙잡지 않으면 셀 라벨은 per-token 효과가 없고(비 1.001), 붙잡으면 크다(2.401).",
    "serving", "사전등록 · n=8 블록", "sticky partition — 2.6× 모순의 해소")}
  <details class="data"><summary>수치 보기 — 사전등록 판독</summary>{t_st}
    <p style="padding:10px 18px 14px;margin:0;font-size:12.5px;color:var(--ink-3)">
      companion d54는 <b>아래로 빗나갔고</b>(12.03 &lt; 13) 사전등록에 그 분기가 없다.
      이 결과는 throughput·goodput·정책·<code>G_LEVER</code>/<code>G_FLAT</code> 주장이 아니다 —
      둘은 여전히 <b>UNDETERMINED</b>다.
    </p>
  </details>
</section>

<section id="evidence">
  <div class="sec-head"><h2>증거 사다리와 철회 원장</h2>
    <span class="sec-sub">감사 반영 기준</span></div>
  <div class="callout">
    <span class="ct">원장을 공개하는 이유</span>
    15건 중 9건은 리뷰어가 아니라 <b>프로젝트 자신의 적대적 감사</b>가 잡아냈고,
    각각이 상설 방법론 게이트가 됐다. 반복되는 실패 유형은 넷이다 —
    <b>항등식을 증거로 쓰기</b>(pin 게이트, <code>kv_mamba_occupancy</code>, <code>g</code>),
    <b>micro를 서빙 예측으로 쓰기</b>(4회 반전),
    <b>사후에 끝점 고르기</b>(3회 재발),
    <b>공유 전제를 검정하지 않고 틀린 두 가설 사이에서만 논쟁하기</b>.
  </div>
  {figure_block(11, uris,
    "왼쪽은 무엇이 어느 등급으로 확립됐는지, 오른쪽은 무엇이 게시된 뒤 되돌려졌는지다. "
    "각 항목의 <b>부기(rider)는 장식이 아니라 하중을 받는다</b> — 스코프 문구를 떼고 인용하면 "
    "그 수치는 정본을 벗어난다.",
    "warn", "등급 · 원장", "증거 사다리 + 철회 15건")}
  <ul class="ledger">{retr}</ul>
</section>

<section id="gates">
  <div class="sec-head"><h2>방법론 게이트</h2>
    <span class="sec-sub">아크가 실제로 가르친 것</span></div>
  <p class="lede">
    아래 10개는 이 프로젝트에서 <b>결론을 뒤집은 실패로부터 신설</b>된 규칙이다.
    9·10번이 가장 최근에 추가됐다.
  </p>
  <ol class="gates">{gates}</ol>
</section>

<footer class="foot"><div>
  정본 위계 — PROJECT_STATUS.md &gt; reports/paper/ &gt; reports/CONSENSUS.md &gt; 기타.<br>
  figure 소스 — <span style="color:var(--ink-2)">reports/figures/</span>
  (<span style="color:var(--ink-2)">canon.py</span> 정본 수치 ·
  <span style="color:var(--ink-2)">extract_raw.py</span> 원자료 파싱 ·
  <span style="color:var(--ink-2)">make_figures.py</span> 렌더 ·
  <span style="color:var(--ink-2)">make_dashboard.py</span> 이 페이지). 논문용 PDF 11종 동봉.<br>
  figure 텍스트는 영문 — venv matplotlib에 CJK 폰트가 없고, 그림 자체는 투고용이다.<br>
  이 페이지는 <b style="color:var(--ink-2)">상태 요약</b>이며 새 성능 판정을 만들지 않는다.
  충돌 시 정본이 우선한다.
</div></footer>

</div>
"""


def main():
    OUT.write_text(build(), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
