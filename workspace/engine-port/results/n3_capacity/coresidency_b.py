#!/usr/bin/env python3
"""작업 B (2026-10-12) -- N3 shape B 원자료의 동거율 *기술값* (관문 아님, 판단 규칙 없음).

사양 = `results/n2_static_grid/VERDICT_coloc_gate_rule_2026-10-12.md` R-3·R-5·R-6·R-7·R-8·CG-1…CG-5.
STOP/GO 판정을 하지 않는다(사용자 결정 2026-10-12 "안 0: 관문 폐기").

  f      = n3_realized_mix.phase_estimators()["coresidency_a_decode"]   (그대로 호출, 재구현 안 함)
           = |PF ∩ DEC| / |DEC|,  DEC = ∪[t_launch, t_sync],  PF = 짝지은 prefill span 합집합,
             bench 창 [첫 work event, 마지막 work event](marks 안) 절단
  병기   coresidency_a_wall (= |PF∩DEC|/|창|, CG-1: 이미 공개된 값), decode_active_frac (= |DEC|/|창|)
  g      = |PF ∩ DEC| / |PF|  (신규, prefill 분모)                     -- 이 파일
  f_nokdw, g_nokdw = prefill_kernel_done_wait=True 인 decode iteration을 교집합(분자)에서 뺀 판;
           분모(|DEC|, |PF|)는 그대로                                   -- 이 파일
  교차 검증: 이 파일이 따로 만든 PF·DEC로 계산한 |PF∩DEC|가 n3_realized_mix의 co(= wall×창)와
           셀마다 1e-9 s 이내로 같아야 한다(다르면 그 셀 값 산출 중단).

제외(셀 단위, 개수 보고): dropped_events > 0 · R7 표식(phase_events_disabled_<name> 또는 서버 로그
문구) · 짝 없는 span(span_seq 기준 start/end 불일치·중복, 또는 start/end 교대 깨짐) · 채점 무효 셀
(n3_analyze.collect의 invalid; 기술값 보수 처리).

E2 r2 원자료는 읽지 않는다(PHASE_EVENTS OFF). N2 Job-PROFILE(n2p_*) 원자료는 읽지 않는다(블라인드).

--selftest : R-5 self-test (기지값 합성 시간선 3개 · 변이 KILL 전수 · 무변이 대조 · 하향 편향 방향 시험).
             실패하면 exit 1, 값 산출 안 함.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import statistics
import sys
import types
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import n3_realized_mix as M  # noqa: E402  (정본 추정량; 이 파일은 수정하지 않는다)

JOBS = {
    "J3_38ed3f4": "n3b_pdmux-n3-capacity-38ed3f4-20261006092813",
    "J4_93c3cca": "n3b_pdmux-n3-capacity-93c3cca-20261007064124",
}
PRIMARY_RATES = (0.15, 0.22)
PE_DISABLED = "PDMUX_PHASE_EVENTS disabled after failure"
CO_TOL_S = 1e-9


# ------------------------------------------------------------------------------- core (mutable in selftest)
def paired_spans(events):
    """n3_realized_mix.phase_estimators와 같은 순차 짝짓기(start가 열리고 다음 end가 닫는다)."""
    spans, open_s = [], None
    for e in events:
        k = e.get("event")
        if k == "prefill_span_start":
            open_s = e
        elif k == "prefill_span_end" and open_s is not None:
            spans.append((open_s["timestamp_monotonic_s"], e["timestamp_monotonic_s"]))
            open_s = None
    return spans


def span_pairing_problems(events):
    """짝 없는 span 검출(span_seq 기준 + 교대 순서)."""
    st, en, order = {}, {}, []
    for e in events:
        k = e.get("event")
        if k == "prefill_span_start":
            st.setdefault(e.get("span_seq"), []).append(e)
            order.append("s")
        elif k == "prefill_span_end":
            en.setdefault(e.get("span_seq"), []).append(e)
            order.append("e")
    probs = []
    no_end = [s for s in st if s not in en]
    no_start = [s for s in en if s not in st]
    dup = [s for s in set(st) | set(en) if len(st.get(s, [])) > 1 or len(en.get(s, [])) > 1]
    if no_end:
        probs.append(f"start without end: {len(no_end)}")
    if no_start:
        probs.append(f"end without start: {len(no_start)}")
    if dup:
        probs.append(f"duplicate span_seq: {len(dup)}")
    if any(order[i] != ("s" if i % 2 == 0 else "e") for i in range(len(order))):
        probs.append("start/end alternation broken")
    for s in st:
        if s in en and len(st[s]) == 1 and len(en[s]) == 1 and \
                en[s][0]["timestamp_monotonic_s"] < st[s][0]["timestamp_monotonic_s"]:
            probs.append(f"end before start (span_seq {s})")
            break
    return probs


def exclusion_reasons(events, r7_marker):
    bad = []
    dmax = max([int(e.get("dropped_events", 0) or 0) for e in events] or [0])
    if dmax > 0:
        bad.append(f"dropped_events_max={dmax}")
    if r7_marker:
        bad.append("R7 marker (PHASE_EVENTS disabled mid-boot)")
    bad += [f"unpaired span: {p}" for p in span_pairing_problems(events)]
    return bad


def coresidency(events, marks, mod=M):
    """f(정본 그대로) + 병기값 + g + kdw 제외판.  mod = n3_realized_mix (selftest에서 변이본 주입)."""
    ph = mod.phase_estimators(events, marks, M.TARGET_PAIR)
    if ph.get("empty"):
        return {"empty": True, "window_source": ph.get("window_source")}
    lo, hi = mod.bench_window(events, marks)[0]
    wlen = hi - lo
    dec = [e for e in events if e.get("event") == "decode_iteration" and lo <= float(e["t_launch"]) <= hi]
    pf = M.clip(paired_spans(events), lo, hi)
    dv = M.clip([(float(e["t_launch"]), float(e["t_sync"])) for e in dec], lo, hi)
    dv_nk = M.clip([(float(e["t_launch"]), float(e["t_sync"])) for e in dec
                    if not e.get("prefill_kernel_done_wait")], lo, hi)
    co = M.measure(M.intersect(pf, dv))
    co_nk = M.measure(M.intersect(pf, dv_nk))
    pfm = M.measure(pf)
    dm = M.measure(dv)
    co_m = ph["coresidency_a_wall"] * ph["window_s"]
    # 창 절단으로 잘려 나간 DEC·PF 길이(하향 편향 크기의 상한 기술값)
    dv_raw = [(float(e["t_launch"]), float(e["t_sync"])) for e in dec]
    trunc_dec = M.measure(dv_raw) - dm
    out = {
        "window_s": ph["window_s"], "window_source": ph["window_source"],
        "f_coresidency_a_decode": ph["coresidency_a_decode"],
        "coresidency_a_wall": ph["coresidency_a_wall"],
        "decode_active_frac": ph["decode_active_frac"],
        "g_prefill_denominator": co / pfm if pfm else float("nan"),
        "f_nokdw": co_nk / dm if dm else float("nan"),
        "g_nokdw": co_nk / pfm if pfm else float("nan"),
        "prefill_active_frac": pfm / wlen if wlen else float("nan"),
        "co_s": co, "co_s_realized_mix": co_m, "pf_s": pfm, "dec_s": dm,
        "kdw_iter_frac": (sum(1 for e in dec if e.get("prefill_kernel_done_wait")) / len(dec)) if dec else float("nan"),
        "kdw_co_share": (1.0 - co_nk / co) if co else float("nan"),
        "w_dec_target_pct": ph["w_dec"]["target_pct"],
        "n_decode_iterations": ph["n_decode_iterations"], "n_prefill_spans_in_window": ph["n_prefill_spans"],
        "decode_bs_max": max([int(e.get("bs", 0)) for e in dec] or [0]),
        "decode_bs_mean": (sum(int(e.get("bs", 0)) for e in dec) / len(dec)) if dec else float("nan"),
        "dec_truncated_by_window_s": trunc_dec,
        "dropped_events_max": ph["dropped_events_max"],
    }
    out["co_crosscheck_ok"] = abs(co - co_m) <= CO_TOL_S and abs(dm - ph["decode_active_frac"] * wlen) <= CO_TOL_S
    return out


def cell(events, marks, r7_marker, mod=M, xcheck=True):
    """제외 규칙 적용 후 값.  제외 셀은 값을 내지 않는다."""
    bad = exclusion_reasons(events, r7_marker)
    if bad:
        return {"excluded": True, "reasons": bad}
    r = coresidency(events, marks, mod)
    if r.get("empty"):
        return {"excluded": True, "reasons": [f"no bench window ({r.get('window_source')})"]}
    if xcheck and not r["co_crosscheck_ok"]:
        return {"excluded": True, "reasons": ["co cross-check vs n3_realized_mix failed"], "debug": r}
    r["excluded"] = False
    return r


# ------------------------------------------------------------------------------- self-test (R-5)
DT = 0.001          # 오라클 래스터 해상도; 모든 끝점은 0.01 격자 -> 중점 표본이 정확


def _ev_span(seq, s, e, dropped=0):
    return [{"event": "prefill_span_start", "timestamp_monotonic_s": s, "span_seq": seq, "prefill_sms": 64,
             "decode_sms": 44, "running_bs": 0, "n_new": 1, "queue_len": 0, "cap": 48, "dropped_events": dropped},
            {"event": "prefill_span_end", "timestamp_monotonic_s": e, "span_seq": seq, "dropped_events": dropped}]


def _ev_dec(tl, ts, kdw=False, inflight=False, dropped=0):
    return {"event": "decode_iteration", "t_launch": tl, "t_sync": ts, "timestamp_monotonic_s": ts, "bs": 1,
            "prefill_sms": 64 if inflight else 0, "decode_sms": 44 if inflight else 108,
            "prefill_in_flight": inflight, "prefill_kernel_done_wait": kdw, "dropped_events": dropped}


def _sorted(ev):
    return sorted(ev, key=lambda e: (e["timestamp_monotonic_s"], 0 if e["event"] == "prefill_span_end" else 1))


def _oracle(spans, decs, marks):
    """독립 오라클: 창을 직접 계산하고(work event 정의), 0.001 s 중점 래스터로 집합 측도를 센다.
    구간 대수(union/intersect)를 쓰지 않는다."""
    work = [s for s, _ in spans] + [e for _, e in spans] + [tl for tl, _, _ in decs]
    work = [t for t in work if marks["bench_start"] <= t <= marks["bench_end"]]
    lo, hi = min(work), max(work)
    dec_in = [(tl, ts, k) for tl, ts, k in decs if lo <= tl <= hi]
    n = int(round((hi - lo) / DT))
    cnt = {"pf": 0, "dec": 0, "co": 0, "co_nk": 0}
    for i in range(n):
        t = lo + (i + 0.5) * DT
        p = any(a <= t <= b for a, b in spans)
        d = any(a <= t <= b for a, b, _ in dec_in)
        dn = any(a <= t <= b for a, b, k in dec_in if not k)
        cnt["pf"] += p
        cnt["dec"] += d
        cnt["co"] += p and d
        cnt["co_nk"] += p and dn
    w = n
    return {"f": cnt["co"] / cnt["dec"], "g": cnt["co"] / cnt["pf"], "wall": cnt["co"] / w,
            "daf": cnt["dec"] / w, "f_nokdw": cnt["co_nk"] / cnt["dec"], "g_nokdw": cnt["co_nk"] / cnt["pf"]}


def _timelines():
    """기존 selftest_synthetic(0.5 s 등간격 decode + span 1개)과 다른 생성 규칙 3종."""
    T = []
    # T1 수기 설계: 불규칙 간격 decode, 창 밖 warm-up span, 창 끝을 넘는 span, 창 밖 decode
    marks = {"bench_start": 0.0, "bench_end": 42.0}
    spans = [(-10.0, -8.0), (10.0, 14.0), (30.0, 31.0), (38.0, 45.0)]
    decs = [(-9.0, -8.5, False), (5.0, 6.0, False), (8.0, 11.0, False), (12.0, 13.0, True), (13.5, 16.0, False),
            (20.0, 22.0, False), (30.5, 30.8, True), (40.0, 41.0, False)]
    T.append(("T1_handbuilt", spans, decs, marks,
              {"f": 2.8 / 9.8, "g": 2.8 / 7.0, "wall": 2.8 / 35.0, "daf": 9.8 / 35.0,
               "f_nokdw": 1.5 / 9.8, "g_nokdw": 1.5 / 7.0}))
    # T2 맞붙은 span(끝=다음 시작) + span 경계를 가로지르는 decode, 0.37/0.13 교대 주기(비등간격)
    marks = {"bench_start": 1.0, "bench_end": 60.0}
    spans = [(3.0, 5.2), (5.2, 9.0), (21.11, 21.9), (40.0, 47.5)]
    decs, t, i = [], 1.5, 0
    while t < 55.0:
        d = 0.37 if i % 2 == 0 else 0.13
        decs.append((round(t, 2), round(t + d - 0.04, 2), i % 5 == 0))
        t = round(t + d + (0.20 if i % 7 == 3 else 0.0), 2)
        i += 1
    T.append(("T2_adjacent_spans", spans, decs, marks, None))
    # T3 prefill-heavy, 희소 decode(span 안에 큰 간격), LCG 결정적 의사난수, 0.01 격자
    marks = {"bench_start": 0.0, "bench_end": 200.0}
    x = 12345

    def rnd():
        nonlocal x
        x = (1103515245 * x + 12345) % (2 ** 31)
        return x / 2 ** 31
    spans, t = [], 2.0
    while t < 180.0:
        ln = round(2.0 + 8.0 * rnd(), 2)
        spans.append((round(t, 2), round(t + ln, 2)))
        t = round(t + ln + 0.5 + 6.0 * rnd(), 2)
    decs, t = [], 1.0
    while t < 190.0:
        ln = round(0.02 + 0.3 * rnd(), 2)
        decs.append((round(t, 2), round(t + ln, 2), rnd() < 0.3))
        t = round(t + ln + 0.05 + 2.5 * rnd(), 2)
    T.append(("T3_prefill_heavy_lcg", spans, decs, marks, None))
    return T


def _events(spans, decs, dropped=0):
    ev = []
    for i, (s, e) in enumerate(spans):
        ev += _ev_span(i + 1, s, e)
    ev += [_ev_dec(a, b, k) for a, b, k in decs]
    ev = _sorted(ev)
    if dropped:
        ev[len(ev) // 2] = dict(ev[len(ev) // 2], dropped_events=dropped)
    return ev


def _mutant_module(name, src, repl):
    """소스 치환 변이본 모듈.  치환 문자열이 정확히 1회 나타나지 않으면 변이가 공허 -> 예외."""
    for a, b in repl:
        c = src.count(a)
        if c != 1:
            raise RuntimeError(f"mutation {name}: pattern occurs {c}x (must be 1): {a[:60]!r}")
        src = src.replace(a, b)
    mod = types.ModuleType(f"mut_{name}")
    mod.__dict__["__file__"] = str(HERE / f"mut_{name}.py")
    exec(compile(src, f"<mut_{name}>", "exec"), mod.__dict__)
    return mod


M_SRC = inspect.getsource(M)
SELF_SRC = None   # main()에서 채움


def _self_mutant(name, repl):
    """이 파일의 cell/coresidency/exclusion을 변이시킨 모듈(정본 M은 원본 그대로 주입)."""
    m = _mutant_module(name, SELF_SRC, repl)
    return m


MUTATIONS = [
    # (이름, 대상, 치환목록, 죽어야 할 추정량)
    ("M1_f_denominator_decode_to_wall", "M",
     [('out["coresidency_a_decode"] = co / dm if dm else', 'out["coresidency_a_decode"] = co / wlen if wlen else')], ("f",)),
    ("M2_f_intersection_to_union", "M",
     [("co = measure(intersect(pf, dv))", "co = measure(union(pf + dv))")], ("f",)),
    ("M3_f_dec_includes_gaps", "M",
     [('dv = clip([(float(e["t_launch"]), float(e["t_sync"])) for e in dec], lo, hi)',
       'dv = clip([(float(a["t_launch"]), float(b["t_launch"])) for a, b in zip(dec, dec[1:])] + '
       '[(float(e["t_launch"]), float(e["t_sync"])) for e in dec[-1:]], lo, hi)')], ("f",)),
    ("M4_f_span_pairing_broken", "M",
     [('        if k == "prefill_span_start":\n            open_s = e\n',
       '        if k == "prefill_span_start":\n            if open_s is not None:\n'
       '                spans.append((open_s["timestamp_monotonic_s"], e["timestamp_monotonic_s"]))\n'
       '            open_s = e\n'),
      ('elif k == "prefill_span_end" and open_s is not None:', 'elif False:')], ("f",)),
    ("M5_f_window_clip_removed", "M",
     [("pf = clip(spans, lo, hi)", "pf = list(spans)"),
      ('dv = clip([(float(e["t_launch"]), float(e["t_sync"])) for e in dec], lo, hi)',
       'dv = [(float(e["t_launch"]), float(e["t_sync"])) for e in dec]')], ("f",)),
    ("G1_g_denominator_prefill_to_decode", "SELF",
     [('"g_prefill_denominator": co / pfm if pfm else float("nan"),\n', '"g_prefill_denominator": co / dm if dm else float("nan"),\n')], ("g",)),
    ("G1b_g_denominator_prefill_to_wall", "SELF",
     [('"g_prefill_denominator": co / pfm if pfm else float("nan"),\n', '"g_prefill_denominator": co / wlen if wlen else float("nan"),\n')], ("g",)),
    ("G2_g_intersection_to_union", "SELF",
     [("    co = M.measure(M.intersect(pf, dv))\n", "    co = M.measure(M.union(pf + dv))\n")], ("g",)),
    ("G3_g_dec_includes_gaps", "SELF",
     [('    dv = M.clip([(float(e["t_launch"]), float(e["t_sync"])) for e in dec], lo, hi)\n',
       '    dv = M.clip([(float(a["t_launch"]), float(b["t_launch"])) for a, b in zip(dec, dec[1:])] + '
       '[(float(e["t_launch"]), float(e["t_sync"])) for e in dec[-1:]], lo, hi)\n')], ("g",)),
    ("G4_g_span_pairing_broken", "SELF",
     [('        if k == "prefill_span_start":\n            open_s = e\n        elif k == "prefill_span_end" and open_s is not None:',
       '        if k == "prefill_span_start":\n            if open_s is not None:\n'
       '                spans.append((open_s["timestamp_monotonic_s"], e["timestamp_monotonic_s"]))\n'
       '            open_s = e\n        elif False:')], ("g",)),
    ("G5_g_window_clip_removed", "SELF",
     [("    pf = M.clip(paired_spans(events), lo, hi)\n", "    pf = list(paired_spans(events))\n")], ("g",)),
    ("K1_nokdw_filter_removed", "SELF",
     [('if not e.get("prefill_kernel_done_wait")], lo, hi)\n', '], lo, hi)\n')],
     ("f_nokdw", "g_nokdw")),
    ("X1_dropped_invalidation_removed", "SELF",
     [("    if dmax > 0:\n", "    if False:\n")], ("excl_dropped",)),
    ("X2_R7_invalidation_removed", "SELF",
     [("    if r7_marker:\n", "    if False:\n")], ("excl_r7",)),
    ("X3_unpaired_invalidation_removed", "SELF",
     [("    bad += [f\"unpaired span: {p}\" for p in span_pairing_problems(events)]\n", "    pass\n")],
     ("excl_unpaired",)),
]


def _run_battery(mod_self, mod_m, T, xcheck=True):
    """mod_self의 cell()을 mod_m 주입으로 실행 -> 추정량별 (통과 여부, 상세)."""
    res = {}
    for name, spans, decs, marks, hand in T:
        ora = _oracle(spans, decs, marks)
        r = mod_self.cell(_events(spans, decs), marks, False, mod_m, xcheck)
        if r.get("excluded"):
            for k in ("f", "g", "f_nokdw", "g_nokdw", "wall", "daf"):
                res.setdefault(k, []).append((name, False, f"clean timeline excluded: {r.get('reasons')}"))
            continue
        got = {"f": r["f_coresidency_a_decode"], "g": r["g_prefill_denominator"], "wall": r["coresidency_a_wall"],
               "daf": r["decode_active_frac"], "f_nokdw": r["f_nokdw"], "g_nokdw": r["g_nokdw"]}
        for k, v in got.items():
            ok = abs(v - ora[k]) < 1e-6 and (hand is None or abs(v - hand[k]) < 1e-9)
            res.setdefault(k, []).append((name, ok, f"got {v:.9f} oracle {ora[k]:.9f}" +
                                          (f" hand {hand[k]:.9f}" if hand else "")))
    # 제외 규칙 시험: 오염된 셀은 반드시 제외되어야 한다
    name, spans, decs, marks, _ = T[0]
    ev_drop = _events(spans, decs, dropped=1)
    ev_unp = [e for e in _events(spans, decs) if not (e["event"] == "prefill_span_end" and e["span_seq"] == 2)]
    res["excl_dropped"] = [("T1+dropped=1", mod_self.cell(ev_drop, marks, False, mod_m).get("excluded") is True, "")]
    res["excl_r7"] = [("T1+R7", mod_self.cell(_events(spans, decs), marks, True, mod_m).get("excluded") is True, "")]
    res["excl_unpaired"] = [("T1-end(seq2)", mod_self.cell(ev_unp, marks, False, mod_m).get("excluded") is True, "")]
    return {k: (all(ok for _, ok, _ in v), v) for k, v in res.items()}


def selftest():
    T = _timelines()
    lines, fails = [], []
    # 0. 오라클 자체 점검: T1 수기값과 래스터 오라클 일치
    o1 = _oracle(*T[0][1:4])
    for k, v in T[0][4].items():
        if abs(o1[k] - v) > 1e-6:
            fails.append(f"oracle vs hand T1 {k}: {o1[k]} vs {v}")
    lines.append(f"oracle(T1) = {json.dumps({k: round(v, 6) for k, v in o1.items()})}")
    for name, spans, decs, marks, _ in T:
        o = _oracle(spans, decs, marks)
        lines.append(f"{name}: spans={len(spans)} decs={len(decs)} f={o['f']:.6f} g={o['g']:.6f} "
                     f"wall={o['wall']:.6f} f_nokdw={o['f_nokdw']:.6f}")
    # 1. 무변이 대조: (a) 이 모듈 그대로, (b) 소스 재-exec 사본(변이 경로 자체가 결과를 바꾸지 않음)
    this = sys.modules[__name__]
    base = _run_battery(this, M, T)
    copy_self = _mutant_module("nomut_self", SELF_SRC, [])
    copy_m = _mutant_module("nomut_m", M_SRC, [])
    base2 = _run_battery(copy_self, copy_m, T)
    for tag, b in (("original", base), ("re-exec copy", base2)):
        bad = [k for k, (ok, _) in b.items() if not ok]
        lines.append(f"no-mutation control [{tag}]: {'PASS' if not bad else 'FAIL ' + str(bad)}")
        if bad:
            fails.append(f"no-mutation control [{tag}] failed on {bad}: " +
                         "; ".join(d for k in bad for _, ok, d in b[k][1] if not ok))
    # 2. 변이 KILL 전수
    for mname, target, repl, must in MUTATIONS:
        try:
            if target == "M":
                mm, ms = _mutant_module(mname, M_SRC, repl), copy_self
            else:
                mm, ms = copy_m, _self_mutant(mname, repl)
            b = _run_battery(ms, mm, T)
            bv = _run_battery(ms, mm, T, xcheck=False)   # 값 단독 KILL(교차검증 배제)
        except RuntimeError as ex:
            fails.append(str(ex))
            lines.append(f"{mname}: VACUOUS ({ex})")
            continue
        except Exception as ex:   # 변이가 런타임 오류를 내면 그것도 검출(KILL)이지만 기록
            lines.append(f"{mname}: KILLED (exception {type(ex).__name__}: {ex})")
            continue
        killed = [k for k in must if not b[k][0]]
        vkilled = [k for k in must if not bv[k][0]]
        status = "KILLED" if len(killed) == len(must) and len(vkilled) == len(must) else "SURVIVED"
        detail = {k: [(n, d[:90]) for n, ok, d in b[k][1] if not ok][:1] for k in must}
        vdetail = {k: f"{sum(1 for _, ok, _ in bv[k][1] if not ok)}/{len(bv[k][1])} timelines; first: " +
                   str([(n, d[:90]) for n, ok, d in bv[k][1] if not ok][:1]) for k in must}
        lines.append(f"{mname}: {status} (must kill {list(must)}; pipeline {detail}; value-only(no x-check) {vdetail})")
        if status != "KILLED":
            fails.append(f"mutation {mname} SURVIVED on {[k for k in must if k not in killed]}")
    # 3. 하향 편향 방향 시험 -- 오염이 값을 진실보다 *낮추는* 경로를 제외 규칙이 막는지
    name, spans, decs, marks, hand = T[0]
    truth = hand["f"]
    # (a) span_end 1개 유실 -> 순차 짝짓기가 span을 잃음 -> 정본 f가 진실보다 낮아짐; 제외 규칙이 막아야 함
    ev_unp = [e for e in _events(spans, decs) if not (e["event"] == "prefill_span_end" and e["span_seq"] == 2)]
    f_unp = M.phase_estimators(ev_unp, marks, M.TARGET_PAIR)["coresidency_a_decode"]
    excl = cell(ev_unp, marks, False).get("excluded")
    lines.append(f"downward-bias (a) lost span_end: raw f {f_unp:.6f} vs truth {truth:.6f} "
                 f"({'LOWER' if f_unp < truth - 1e-12 else 'not lower'}); wrapper excluded={excl}")
    if not (f_unp < truth - 1e-12 and excl):
        fails.append("downward-bias (a): lost span_end did not lower f or was not excluded")
    # (b) 동거 decode iteration 유실(+dropped_events 1) -> f 하향; 제외되어야 함
    decs_b = [d for d in decs if d[0] != 8.0]
    ev_b = _events(spans, decs_b, dropped=1)
    f_b = M.phase_estimators(ev_b, marks, M.TARGET_PAIR)["coresidency_a_decode"]
    excl_b = cell(ev_b, marks, False).get("excluded")
    lines.append(f"downward-bias (b) lost co-resident decode + dropped=1: raw f {f_b:.6f} vs truth {truth:.6f} "
                 f"({'LOWER' if f_b < truth - 1e-12 else 'not lower'}); wrapper excluded={excl_b}")
    if not (f_b < truth - 1e-12 and excl_b):
        fails.append("downward-bias (b) not lower or not excluded")
    # (c) PHASE_EVENTS 중도 비활성(R7): t>12 이후 사건 소실 -> 창이 짧아지고 f 변화; R7 표식이면 제외
    ev_c = [e for e in _events(spans, decs) if e["timestamp_monotonic_s"] <= 12.0]
    excl_c = cell(ev_c, marks, True).get("excluded")
    lines.append(f"R7 truncation: wrapper excluded={excl_c}")
    if not excl_c:
        fails.append("R7 truncated cell not excluded")
    # (d) 창 절단 방향: 마지막 decode가 hi(=마지막 t_launch)에서 잘린다 -> 정본 f는 이 iteration을 0으로 셈.
    #     절단 크기는 실제 셀마다 dec_truncated_by_window_s로 공시한다(값 정정 안 함).
    r = cell(_events(spans, decs), marks, False)
    lines.append(f"window truncation (d): T1 dec_truncated_by_window_s = {r['dec_truncated_by_window_s']:.3f} (expected 1.000)")
    if abs(r["dec_truncated_by_window_s"] - 1.0) > 1e-9:
        fails.append("window truncation size wrong")
    # (e) 분모 순서 불변량: wall <= f (|DEC| <= |창|), kdw 제외판 <= 원판
    for name_, sp, dc, mk, _ in T:
        rr = cell(_events(sp, dc), mk, False)
        if not (rr["coresidency_a_wall"] <= rr["f_coresidency_a_decode"] + 1e-12 and
                rr["f_nokdw"] <= rr["f_coresidency_a_decode"] + 1e-12 and rr["g_nokdw"] <= rr["g_prefill_denominator"] + 1e-12):
            fails.append(f"ordering invariant broken on {name_}")
    lines.append("ordering invariants (wall<=f, f_nokdw<=f, g_nokdw<=g): checked on 3 timelines")
    print("\n".join(lines))
    if fails:
        print("CORESIDENCY_SELFTEST_FAILED")
        for f in fails:
            print("  " + f)
        return 1, lines
    print(f"CORESIDENCY_SELFTEST_OK ({len(MUTATIONS)}/{len(MUTATIONS)} mutations KILLED, no-mutation controls PASS x2, "
          f"downward-bias tests PASS)")
    return 0, lines


# ------------------------------------------------------------------------------- data
def r7_marker(out_dir: Path, name: str) -> bool:
    if (out_dir / f"phase_events_disabled_{name}").exists():
        return True
    srv = out_dir / f"srv_{name}.log"
    return srv.exists() and PE_DISABLED in srv.read_text(errors="replace")


def run_data():
    import n3_analyze as N  # 채점 유효성(collect) 재사용
    import n3_plan as P
    res = {"spec": "VERDICT_coloc_gate_rule_2026-10-12.md R-3/R-5/R-6/R-7/R-8; descriptive only, no gate",
           "jobs": {}}
    for jtag, jdir in JOBS.items():
        out_dir = HERE / jdir
        units, _cells, invalid, by_name = N.collect(out_dir, "B")
        on_units = [u for u in units if u["kind"] in ("grid", "obs_on")]
        jr = {"dir": jdir, "cells": {}, "excluded": {}, "not_run_or_scoring_invalid": {}}
        for u in on_units:
            name = u["name"]
            if name in invalid:
                jr["not_run_or_scoring_invalid"][name] = invalid[name]
                continue
            tel = out_dir / f"tel_{name}.jsonl"
            ev = M.read_events(tel)
            marks = M.read_marks(out_dir / f"marks_{name}.txt")
            c = cell(ev, marks, r7_marker(out_dir, name))
            c.update(rate=u["rate"], seed=u["seed"], kind=u["kind"], name=name)
            # 러너가 쓴 mix_<name>.json의 CO-a(wall)과 일치 확인(같은 파일을 같은 코드로 읽었는지)
            mx = N.mix_of(out_dir, name)
            if mx and not c.get("excluded"):
                c["mix_json_co_a_wall"] = mx.get("phase", {}).get("coresidency_a_wall")
                c["mix_json_match"] = (c["mix_json_co_a_wall"] is not None and
                                       abs(c["mix_json_co_a_wall"] - c["coresidency_a_wall"]) < 1e-12)
            (jr["excluded"] if c.get("excluded") else jr["cells"])[name] = c
        # rate별 요약(seed별 값 + Job 안 mean±SD·min·max; Job 간 풀링 없음)
        by_rate = {}
        for c in jr["cells"].values():
            by_rate.setdefault(c["rate"], []).append(c)
        summ = {}
        for rate in sorted(by_rate):
            cs = sorted(by_rate[rate], key=lambda c: c["seed"])
            s = {"n_seeds": len(cs), "underpowered_n_below_4": len(cs) < 4, "seeds": [c["seed"] for c in cs]}
            for k in ("f_coresidency_a_decode", "g_prefill_denominator", "coresidency_a_wall", "decode_active_frac",
                      "f_nokdw", "g_nokdw", "kdw_iter_frac", "w_dec_target_pct", "dec_truncated_by_window_s",
                      "decode_bs_max"):
                v = [c[k] for c in cs]
                s[k] = {"per_seed": v, "mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else float("nan"),
                        "min": min(v), "max": max(v)}
            summ[f"{rate:.2f}"] = s
        jr["by_rate"] = summ
        jr["n_cells"] = len(jr["cells"])
        jr["n_excluded"] = len(jr["excluded"])
        jr["n_not_run_or_scoring_invalid"] = len(jr["not_run_or_scoring_invalid"])
        jr["all_co_crosscheck_ok"] = all(c["co_crosscheck_ok"] for c in jr["cells"].values())
        jr["all_mix_json_match"] = all(c.get("mix_json_match", False) for c in jr["cells"].values())
        res["jobs"][jtag] = jr
    return res


def main():
    global SELF_SRC
    SELF_SRC = Path(__file__).read_text()
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(HERE / "CORESIDENCY_B_2026-10-12.json"))
    a = ap.parse_args()
    rc, lines = selftest()
    if a.selftest or rc:
        if rc:
            print("self-test failed -> no values computed (R-5)")
        return rc
    res = run_data()
    res["selftest"] = {"status": "OK", "log": lines}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    for jtag, jr in res["jobs"].items():
        print(f"== {jtag}: cells {jr['n_cells']} excluded {jr['n_excluded']} not_run/invalid {jr['n_not_run_or_scoring_invalid']} "
              f"co_xcheck {jr['all_co_crosscheck_ok']} mix_match {jr['all_mix_json_match']}")
        for r, s in jr["by_rate"].items():
            print(f"  r={r} n={s['n_seeds']} f={[round(x, 4) for x in s['f_coresidency_a_decode']['per_seed']]} "
                  f"g={[round(x, 4) for x in s['g_prefill_denominator']['per_seed']]} "
                  f"wall={[round(x, 4) for x in s['coresidency_a_wall']['per_seed']]}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
