#!/usr/bin/env python3
"""Step 0 집계 도구 (수리판, 2026-08-22) — HOLB 아티팩트의 decode 간극을 층화 집계한다.

★★ 이 스크립트의 산출로 인과 주장을 하지 마라 ★★
    2026-08-22 감사(`audit_step0_2026-08-22/VERDICT.md`)가 이 재분석의 **인과 귀속을
    `REFUTED`** 로 판정했다: `stream_key` 변화(=전환) 경계의 간극 증가는 파티션 변경에
    귀속될 수 없다 — 인덱스가 **바뀌지 않은** adjust 경계가 같은 데이터에서 더 큰 값을
    낸다.  ⇒ 여기서 나오는 `delta_med`는 **전환 기계 비용이 아니다.** 서술 통계일 뿐이고,
    정본(`CONSENSUS.md` / `PROJECT_STATUS.md` / 논문)에 인용할 수 없다.
    이 파일의 수리 범위는 **집계 도구를 정직하게 만드는 것**뿐이다(감사 결함 1·2·3).
    수리는 결론을 되살리지 않는다.

GPU 0 · 새 측정 0 · 새 결정량 0 · 새 판정어 0.  입력은 이미 존재하는
`results/p1_gates/gate2/*.holb.jsonl`(계측 채널 `src/multiplex/holb_probe.py`, **읽기 전용**).

--------------------------------------------------------------------------
수리 1 — 층 정의를 코드 사실대로 고치고, 섞여 있던 두 축을 분리했다
--------------------------------------------------------------------------
수리 전 docstring은 `strict`를 *"그 간극 동안 다른 스트림 forward가 없었다"* 로,
`ambiguous`를 *"prefill forward가 실제로 돌았다(n_other_fw>0)"* 로 적었다.  **거짓이다.**
실제 구현(`../../src/multiplex/holb_probe.py:69-73`, 분기 `:200-212`):

    GAP_STRICT    = "strict"      # 간극 안 **모든** 관측에서 decode pending > 0
    GAP_AMBIGUOUS = "ambiguous"   # 간극 안 **어느** 관측에서 pending == 0 (요청 도착 유휴)

즉 `gap_class`의 축은 **decode 대기 여부**이지 다른 스트림 실행 여부가 아니다.  두 축은
독립이며 실제로 어긋난다(이 격자 실측: `strict`의 7.1%가 `n_other_fw>0`).  그래서 이제
**2×2**로 집계한다 — 셀 이름 자체가 두 축을 분리해 읽는다:

    축 W (decode 수요)  : wait = gap_class=="strict"     (간극 내내 decode가 대기)
                          idle = gap_class=="ambiguous"  (간극 어딘가에서 decode 수요 0)
    축 O (동시 실행)    : solo = n_other_fw == 0         (간극 안에 해소된 비-decode forward 없음)
                          conc = n_other_fw >  0         (있었음; pdmux arm에선 prefill 스트림)

    셀 = wait_solo · wait_conc · idle_solo · idle_conc

`gap_class`가 `first`/`invalid`인 이벤트는 **어느 셀에도 넣지 않고** `n_unclassified`로
따로 센다(둘 다 축 W의 상태가 아니다).  전환 체인(`stream_key`)은 그래도 전진시킨다.

--------------------------------------------------------------------------
수리 2 — `phase == "measure"` 필터를 제거했다 (거짓 통제였다)
--------------------------------------------------------------------------
`holb_probe.py:469`가 **모든** gap 이벤트를 리터럴 `"measure"`로 방출한다(이 격자
155,571건 전부).  따라서 수리 전의 `phase in (None,"measure")` 필터는 **0건을 제거**했고,
그것을 *"워밍업 제외"* 라 주석 단 것은 **거짓 통제**였다(교훈 #9/#53).
지금은 **필터가 없다.**  대신 관측된 `phase` 값 집합과 제거 건수(정의상 0)를 보고한다.

★ 한계(통제인 척하지 않기 위해 명시): **이 채널은 워밍업을 구분하지 않는다.**  이 분석에
   워밍업 제외는 **적용되어 있지 않다.**  워밍업 통제가 필요하면 다른 축을 가져와야 한다.

--------------------------------------------------------------------------
수리 3 — 건강도 게이트가 이 분석의 가장 취약한 실패 모드를 본다
--------------------------------------------------------------------------
수리 전엔 `dropped_events`(텔레메트리 **큐** 카운터)만 봤다.  이 분석이 가장 취약한 건
`dropped_spans`(프로브 **FIFO**, `holb_probe.py:368`)다 — 그게 >0이면 decode 체인이 끊겨
**간극 부풀림과 `stream_key` 오라벨이 동시에** 난다.  지금은 `holb_summary` 이벤트에서
`dropped_spans`·`n_errors`·`n_invalid_gaps`·`n_inconsistent_gaps`·`timeline_over_host`
(그리고 기존 `dropped_events`)를 읽어 **게이트**로 건다.  `holb_summary`가 아예 없으면
건강도를 확인할 수 없으므로 fail-closed로 제외한다.

★ 게이트 #21 준수: 제외된 아티팩트는 **폐기가 아니라 보고**된다 — 파일명과 사유를 화면과
   JSON 양쪽에 남긴다.  "측정 실패"를 조용히 지우지 않는다.

--------------------------------------------------------------------------
변이 테스트 (교훈 #53)
--------------------------------------------------------------------------
수리 3건은 각각, **그 수리를 되돌린 변이본에서 반드시 실패하는** 검사와 짝지어져 있다.
검사 본체는 언제나 무변이 파일에서 오므로 변이본이 자기 심사관을 약화시킬 수 없다.

    python3 step0_switch_gap.py --selftest           # 무변이 배터리 + 변이 3종
    python3 step0_switch_gap.py --selftest-mutants   # 변이만
    python3 step0_switch_gap.py                      # 실제 아티팩트 집계 (GPU 0)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics as st
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GATE2 = os.path.join(HERE, "..", "p1_gates", "gate2")
# 감사된 수리-전 산출물 `STEP0_SWITCH_GAP_2026-08-22.json`은 증거이므로 덮어쓰지 않는다.
OUT_JSON = os.path.join(HERE, "STEP0_SWITCH_GAP_REPAIRED_2026-08-22.json")
SEED = 1
B = 10000

CITATION_STOP = (
    "인용 금지: 이 산출의 어떤 수치도 전환 비용·정책 근거로 인용할 수 없다 "
    "(audit_step0_2026-08-22/VERDICT.md = REFUTED). 집계 도구 검증용."
)

# --- BEGIN repair-1 ---
# 축 정의는 계측 채널의 사실을 그대로 옮긴 것이다(holb_probe.py:69-73, 200-212).
GAP_STRICT = "strict"        # 간극 내 모든 관측에서 decode pending > 0
GAP_AMBIGUOUS = "ambiguous"  # 간극 내 어느 관측에서 decode pending == 0

CELLS = ("wait_solo", "wait_conc", "idle_solo", "idle_conc")
CELL_DESC = {
    "wait_solo": "decode 대기 O · 다른 스트림 forward X",
    "wait_conc": "decode 대기 O · 다른 스트림 forward O",
    "idle_solo": "decode 수요 끊김 · 다른 스트림 forward X",
    "idle_conc": "decode 수요 끊김 · 다른 스트림 forward O",
}


def cell_of(e):
    """2×2 셀 이름, 또는 축 W의 상태가 아니면 None.

    축 W는 `gap_class`(=decode 대기 여부)에서, 축 O는 `n_other_fw`(=다른 스트림 forward
    실행 여부)에서 **각각 따로** 읽는다.  두 축을 하나로 접지 않는 것이 수리 1의 요점이다.
    """
    gc = e["gap_class"]
    if gc not in (GAP_STRICT, GAP_AMBIGUOUS):
        return None                       # "first" / "invalid": 축 W의 상태가 아니다
    wait = gc == GAP_STRICT
    conc = int(e["n_other_fw"] or 0) > 0
    return ("wait_" if wait else "idle_") + ("conc" if conc else "solo")
# --- END repair-1 ---

PHASE_NOTE = (
    "이 채널은 워밍업을 구분하지 않는다: holb_probe.py:469가 모든 gap 이벤트를 리터럴 "
    "\"measure\"로 방출한다. 따라서 워밍업 제외는 적용되어 있지 않다(통제 아님)."
)

# --- BEGIN repair-2 ---
def phase_channel(ev):
    """(kept, info) — **필터하지 않는다.**  `phase` 채널의 변별력을 관측해 보고만 한다.

    수리 전의 `phase in (None,"measure")` 필터는 정의상 0건을 제거하는 항등식이었고,
    그것을 "워밍업 제외"라 부른 것이 거짓 통제였다(교훈 #9/#53).  통제인 척하는 대신
    관측된 값 집합·제거 건수·한계를 그대로 적는다.
    """
    values = sorted({str(e["phase"]) for e in ev})
    kept = list(ev)                                   # 필터 없음
    return kept, {
        "values": values,
        "discriminates_warmup": len(values) > 1,
        "n_removed": len(ev) - len(kept),
        "note": PHASE_NOTE,
    }
# --- END repair-2 ---

TLH_TOL = 0.01   # timeline_over_host 허용대 (관측치는 ~1e-4 수준)

# --- BEGIN repair-3 ---
def health_verdict(h):
    """이 아티팩트를 **집계에서 제외**해야 할 사유 목록(빈 리스트 = 포함).

    가장 취약한 실패 모드는 `dropped_spans`(프로브 FIFO 포화)다: 그게 >0이면 decode
    체인이 끊겨 간극 부풀림과 `stream_key` 오라벨이 동시에 난다.  건강도를 확인할 수
    없는 아티팩트(요약 이벤트 없음)도 fail-closed로 제외한다.  제외는 폐기가 아니라
    보고다(게이트 #21) — 호출자가 파일명과 사유를 그대로 출력한다.
    """
    reasons = []
    if not h.get("has_summary"):
        return ["no holb_summary event -- health unverifiable (fail-closed)"]
    for f in ("dropped_spans", "n_errors", "n_invalid_gaps",
              "n_inconsistent_gaps", "dropped_events"):
        v = h.get(f)
        if v is None:
            reasons.append(f"{f} missing from holb_summary")
        elif v:
            reasons.append(f"{f}={v}")
    t = h.get("timeline_over_host")
    if t is None:
        reasons.append("timeline_over_host missing from holb_summary")
    elif not (1.0 - TLH_TOL <= float(t) <= 1.0 + TLH_TOL):
        reasons.append(f"timeline_over_host={float(t):.5f} outside 1+-{TLH_TOL}")
    return reasons
# --- END repair-3 ---

HEALTH_COUNTERS = ("dropped_spans", "n_errors", "n_invalid_gaps",
                   "n_inconsistent_gaps", "dropped_events")


def med(xs):
    return st.median(xs) if xs else float("nan")


def read_artifact(path):
    """(gap 이벤트 목록[프로그램 순서], health 요약) — 계측 채널은 읽기만 한다."""
    gaps = []
    health = {"has_summary": False, "timeline_over_host": None}
    counters = {k: None for k in HEALTH_COUNTERS}
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ev = d.get("event")
        if ev == "holb_gap":
            gaps.append(d)
            de = d.get("dropped_events")
            if de is not None:
                counters["dropped_events"] = max(counters["dropped_events"] or 0, int(de))
        elif ev == "holb_summary":
            health["has_summary"] = True
            for k in HEALTH_COUNTERS:
                v = d.get(k)
                if v is not None:
                    counters[k] = max(counters[k] or 0, int(v))
            t = d.get("timeline_over_host")
            if t is not None:
                health["timeline_over_host"] = float(t)
    health.update(counters)
    return gaps, health


def boundaries(gaps):
    """연속한 decode 간극을 (전환 여부) × (2×2 셀)로 라벨한 경계 레코드 목록.

    `stream_key`가 바뀌면 전환.  셀이 None인 레코드(first/invalid)도 목록에 남겨
    `n_unclassified`로 세되 어느 셀에도 넣지 않는다 — 전환 체인은 그래도 전진한다.
    """
    prev_key = None
    out = []
    for d in gaps:
        if d.get("gap_ms") is None:
            continue
        key = d.get("stream_key")
        if key is None:
            continue
        if prev_key is not None:
            rec = {
                "gap_ms": float(d["gap_ms"]),
                "switch": key != prev_key,
                "gap_class": d.get("gap_class"),
                "n_other_fw": int(d.get("n_other_fw") or 0),
                "other_ms": float(d.get("other_stream_fw_ms") or 0.0),
                "bs": d.get("decode_bs"),
                "phase": d.get("phase"),
            }
            rec["cell"] = cell_of(rec)
            out.append(rec)
        prev_key = key
    return out


def _cell_stats(sw, no):
    return {"n_sw": len(sw), "n_no": len(no),
            "med_sw": med(sw), "med_no": med(no),
            "delta_med": (med(sw) - med(no)) if sw and no else float("nan"),
            "mean_sw": st.fmean(sw) if sw else float("nan"),
            "mean_no": st.fmean(no) if no else float("nan")}


def summarize(path):
    """아티팩트 1개 → 행 1개(건강도 포함).  게이트는 호출자(analyze)가 건다."""
    gaps, health = read_artifact(path)
    ev = boundaries(gaps)
    kept, ph = phase_channel(ev)
    row = {
        "file": os.path.basename(path),
        "health": health,
        "phase_channel": ph,
        "n_boundary": len(kept),
        "n_switch": sum(1 for e in kept if e["switch"]),
        "n_unclassified": sum(1 for e in kept if e["cell"] is None),
        "cells": {},
    }
    for name in CELLS:
        sw = [e["gap_ms"] for e in kept if e["switch"] and e["cell"] == name]
        no = [e["gap_ms"] for e in kept if not e["switch"] and e["cell"] == name]
        row["cells"][name] = _cell_stats(sw, no)
    # 진단 (판정 아님): 꼬리 지배도 · 축 O의 구성 비대칭 · idle_conc 잔차
    allg = sorted((e["gap_ms"] for e in kept), reverse=True)
    tot = sum(allg)
    row["top5pct_share"] = (sum(allg[:max(1, len(allg) // 20)]) / tot) if tot else float("nan")
    cls = [e for e in kept if e["cell"] is not None]
    for tag, want in (("switch", True), ("noswitch", False)):
        sub = [e for e in cls if e["switch"] is want]
        row[f"conc_frac_{tag}"] = (sum(1 for e in sub if e["n_other_fw"] > 0) / len(sub)
                                   if sub else float("nan"))
    amb = [e for e in kept if e["cell"] == "idle_conc" and e["switch"]]
    row["idle_conc_switch_residual_med"] = (
        med([max(0.0, e["gap_ms"] - e["other_ms"]) for e in amb]) if amb else float("nan"))
    return row


def boot_ci(vals, stat=st.median, b=B, seed=SEED):
    """아티팩트(rep)를 1차 단위로 하는 부트스트랩.  경계는 독립 반복이 아니다."""
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 3:
        return (float("nan"), float("nan"), len(vals))
    rng = random.Random(seed)
    draws = sorted(stat([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(b))
    return (draws[int(0.025 * b)], draws[int(0.975 * b)], len(vals))


def analyze(files):
    """건강도 게이트 → 통과분만 집계.  제외분은 사유와 함께 보고한다(게이트 #21)."""
    admitted, excluded = [], []
    for f in files:
        row = summarize(f)
        reasons = health_verdict(row["health"])
        if reasons:
            excluded.append({"file": row["file"], "reasons": reasons,
                             "health": row["health"], "n_boundary": row["n_boundary"]})
        else:
            admitted.append(row)

    rep = {
        "frame": ("Step 0 집계 도구 (수리판). GPU 0 · 새 측정 0 · 새 결정량 0 · 성능 판정 0건."),
        "citation_stop": CITATION_STOP,
        "n_artifacts_seen": len(files),
        "n_artifacts_admitted": len(admitted),
        "admitted": [r["file"] for r in admitted],
        "excluded": excluded,
        "n_boundary": sum(r["n_boundary"] for r in admitted),
        "n_switch": sum(r["n_switch"] for r in admitted),
        "n_unclassified": sum(r["n_unclassified"] for r in admitted),
        "per_artifact": admitted,
        "cells": {},
    }
    vals, disc, nrem = set(), False, 0
    for r in admitted:
        info = r["phase_channel"] or {}
        vals |= set(info.get("values") or [])
        disc = disc or bool(info.get("discriminates_warmup"))
        nrem += int(info.get("n_removed") or 0)
    rep["phase_channel"] = {"values": sorted(vals), "discriminates_warmup": disc,
                            "n_removed": nrem, "note": PHASE_NOTE}

    for name in CELLS:
        d = [r["cells"][name]["delta_med"] for r in admitted]
        lo, hi, n = boot_ci(d)
        msw = [r["cells"][name]["med_sw"] for r in admitted
               if not math.isnan(r["cells"][name]["med_sw"])]
        mno = [r["cells"][name]["med_no"] for r in admitted
               if not math.isnan(r["cells"][name]["med_no"])]
        rep["cells"][name] = {
            "desc": CELL_DESC.get(name, ""),
            "n_sw": sum(r["cells"][name]["n_sw"] for r in admitted),
            "n_no": sum(r["cells"][name]["n_no"] for r in admitted),
            "n_artifacts_with_delta": n,
            "med_of_med_sw": med(msw), "med_of_med_no": med(mno),
            "delta_med_median": med([x for x in d if not math.isnan(x)]),
            "delta_med_ci95": [lo, hi],
        }
    for k, agg in (("top5pct_share", "top5pct_share"),
                   ("conc_frac_switch", "conc_frac_switch"),
                   ("conc_frac_noswitch", "conc_frac_noswitch"),
                   ("idle_conc_switch_residual_med", "idle_conc_switch_residual_med")):
        xs = [r[agg] for r in admitted if not math.isnan(r[agg])]
        rep[k + "_median"] = med(xs)
        rep[k + "_range"] = [min(xs), max(xs)] if xs else [float("nan"), float("nan")]
    return rep


def _f(x, w=9, p=3):
    return ("{:>%d}" % w).format("--" if x is None or (isinstance(x, float) and math.isnan(x))
                                 else ("{:.%df}" % p).format(x))


def render(rep):
    """화면 출력 문자열.  제외 아티팩트는 반드시 이름과 사유가 보인다(게이트 #21)."""
    L = []
    L.append("!! 인과 귀속 REFUTED (audit_step0_2026-08-22/VERDICT.md) — 아래 수치는 "
             "서술 통계이고 전환 비용이 아니다. 정본 인용 불가.")
    L.append(f"아티팩트 {rep['n_artifacts_seen']}개 중 건강도 게이트 통과 "
             f"{rep['n_artifacts_admitted']}개 · 경계 {rep['n_boundary']}개 "
             f"· 전환 {rep['n_switch']}개 · 미분류(first/invalid) {rep['n_unclassified']}개")
    if rep["excluded"]:
        L.append(f"EXCLUDED {len(rep['excluded'])}개 (폐기 아님 — 사유 병기):")
        for x in rep["excluded"]:
            L.append(f"  - {x['file']}: {'; '.join(x['reasons'])}")
    else:
        L.append("EXCLUDED 0개 (건강도 게이트: dropped_spans·n_errors·n_invalid_gaps·"
                 "n_inconsistent_gaps·dropped_events·timeline_over_host)")
    ph = rep["phase_channel"]
    L.append(f"phase 채널: 값 {ph['values']} · 워밍업 변별 {ph['discriminates_warmup']} "
             f"· 필터 제거 {ph['n_removed']}건 — {ph['note']}")
    L.append("")
    # `n_art` = delta 를 낸 아티팩트 수. 임계가 아니라 **사실**이다 -- 셀이 몇 개의
    # 1차 단위 위에 서 있는지를 행 자체가 드러내야 오독을 막는다(희소 셀 경고).
    L.append(f"{'cell':10s} {'n_sw':>6s} {'n_no':>7s} {'n_art':>5s} {'med(sw)':>9s} "
             f"{'med(no)':>9s} {'delta_med':>10s} {'delta_med 95%CI (rep-boot)':>28s}  설명")
    for name, c in rep["cells"].items():
        lo, hi = c["delta_med_ci95"]
        ci = "[--, --]" if math.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"
        L.append(f"{name:10s} {c['n_sw']:6d} {c['n_no']:7d} "
                 f"{c['n_artifacts_with_delta']:5d} {_f(c['med_of_med_sw'])} "
                 f"{_f(c['med_of_med_no'])} {_f(c['delta_med_median'], 10)} "
                 f"{ci:>28s}  {c['desc']}")
    L.append("")
    L.append(f"진단 · 꼬리 지배도(상위 5%): 중앙 {_f(rep['top5pct_share_median'], 5)} "
             f"· 범위 [{rep['top5pct_share_range'][0]:.3f}, {rep['top5pct_share_range'][1]:.3f}]")
    L.append(f"진단 · 축 O 구성 비대칭  P(conc|전환)={_f(rep['conc_frac_switch_median'], 5, 5)} "
             f"· P(conc|비전환)={_f(rep['conc_frac_noswitch_median'], 5, 5)} "
             "(두 팔의 동시-실행 구성이 다르다 = 축 O는 전환 여부와 교락)")
    L.append(f"진단 · idle_conc 전환 간극에서 다른 스트림 실행분을 뺀 잔차(중앙): "
             f"{_f(rep['idle_conc_switch_residual_med_median'], 5)} ms")
    L.append(CITATION_STOP)
    return "\n".join(L)


# ==========================================================================
# 자기검사 + 변이 테스트 (교훈 #53)
# ==========================================================================
L1 = "L1 2x2 cell counts read axis W from gap_class and axis O from n_other_fw"
L2 = "L2 exactly the four named cells, and they partition the boundaries"
L3 = "L3 the structurally sparse cell is still reported (n=0 not omitted)"
L5 = "L5 first/invalid gaps enter no cell and are counted as unclassified"
P1 = "P1 no phase filtering happens: every boundary survives, n_removed == 0"
P2 = "P2 the phase channel's discriminating power is reported honestly"
H1 = "H1 dropped_spans>0 artefact is excluded (probe FIFO, the exposed mode)"
H2 = "H2 n_errors / n_invalid_gaps / n_inconsistent_gaps / timeline_over_host gate too"
H3 = "H3 an artefact with no holb_summary is excluded (health unverifiable)"
H4 = "H4 exclusions are reported with file and reason, not silently discarded"
H5 = "H5 the aggregate is computed from admitted artefacts only"

_FIX = []


def _fixtures():
    """합성 아티팩트 4종.  GPU 0, 실제 캠페인 파일을 건드리지 않는다."""
    if _FIX:
        return _FIX[0]
    td = tempfile.mkdtemp(prefix="step0_fix_")
    m = GAP_AMBIGUOUS
    # (gap_class, gap_ms, n_other_fw, stream_key, phase)
    mix = [
        ("first", None, 1, "A", "measure"),          # 건너뜀 (gap_ms None)
        (GAP_STRICT, 0.4, 0, "A", "measure"),        # prev_key 확립, 경계 아님
        (GAP_STRICT, 0.5, 0, "A", "measure"),        # no-switch wait_solo
        (GAP_STRICT, 1.5, 0, "B", "measure"),        # switch    wait_solo
        (GAP_STRICT, 2.0, 3, "B", "measure"),        # no-switch wait_conc  (축 어긋남)
        (GAP_STRICT, 9.0, 2, "A", "measure"),        # switch    wait_conc  (축 어긋남)
        (m,          50.0, 0, "A", "measure"),       # no-switch idle_solo  (축 어긋남)
        ("invalid",  -0.3, 0, "A", "measure"),       # 미분류, 체인은 전진
        (m,          80.0, 4, "B", "measure"),       # switch    idle_conc
        (m,          60.0, 4, "B", "measure"),       # no-switch idle_conc
        (m,          90.0, 4, "A", "measure"),       # switch    idle_conc
        (GAP_STRICT, 0.6, 0, "A", "measure"),        # no-switch wait_solo
        (GAP_STRICT, 1.6, 0, "B", "measure"),        # switch    wait_solo
    ]
    ph = [
        ("first", None, 0, "A", "measure"),
        (GAP_STRICT, 0.4, 0, "A", "warmup"),
        (GAP_STRICT, 0.5, 0, "A", "warmup"),         # 경계 1
        (GAP_STRICT, 0.6, 0, "A", "warmup"),         # 경계 2
        (GAP_STRICT, 0.7, 0, "A", "measure"),        # 경계 3
        (GAP_STRICT, 0.8, 0, "B", "measure"),        # 경계 4
        (GAP_STRICT, 0.9, 0, "B", "measure"),        # 경계 5
        (GAP_STRICT, 1.0, 0, "A", "measure"),        # 경계 6
    ]
    fx = {"dir": td}
    fx["mix"] = _write_fixture(os.path.join(td, "mix.holb.jsonl"), mix, {})
    fx["phase"] = _write_fixture(os.path.join(td, "phase.holb.jsonl"), ph, {})
    fx["ds"] = _write_fixture(os.path.join(td, "dropspans.holb.jsonl"), mix,
                              {"dropped_spans": 4})
    fx["err"] = _write_fixture(os.path.join(td, "unhealthy.holb.jsonl"), mix,
                               {"n_errors": 2, "n_invalid_gaps": 1,
                                "n_inconsistent_gaps": 3, "timeline_over_host": 1.4})
    fx["nosum"] = _write_fixture(os.path.join(td, "nosummary.holb.jsonl"), mix, None)
    _FIX.append(fx)
    return fx


def _write_fixture(path, specs, summary):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "holb_open", "phase": "startup"}) + "\n")
        for i, (gc, gap, nof, key, phase) in enumerate(specs):
            fh.write(json.dumps({
                "event": "holb_gap", "phase": phase, "gap_ms": gap, "gap_class": gc,
                "n_other_fw": nof, "other_stream_fw_ms": 0.0,
                "stream_key": key, "decode_bs": 1, "dropped_events": 0,
                "pend_min_in_gap": 1 if gc == GAP_STRICT else 0, "seq": i}) + "\n")
        if summary is not None:
            s = {"event": "holb_summary", "phase": "running", "dropped_spans": 0,
                 "n_errors": 0, "n_invalid_gaps": 0, "n_inconsistent_gaps": 0,
                 "timeline_over_host": 1.0002, "dropped_events": 0}
            s.update(summary)
            fh.write(json.dumps(s) + "\n")
    return path


def _checks(ns, fx):
    """L*/P*/H* 배터리를 `ns`(무변이=globals(), 또는 exec된 변이본)에 대해 실행.

    검사 본체는 언제나 무변이 파일에서 온다 ⇒ 변이본은 자기 심사관을 약화시킬 수 없다.
    반환 [(name, status, detail)] · status in ok/fail/raised · **raised는 어디서도 통과가 아니다.**
    """
    analyze, render_ = ns["analyze"], ns["render"]
    res = []

    def ck(name, fn):
        try:
            good = bool(fn())
        except Exception as e:  # noqa: BLE001
            res.append((name, "raised", repr(e)))
            return
        res.append((name, "ok" if good else "fail", ""))

    def counts(rep):
        return {k: (v.get("n_sw"), v.get("n_no")) for k, v in rep["cells"].items()}

    def l1():
        rep = analyze([fx["mix"]])
        return counts(rep) == {"wait_solo": (2, 2), "wait_conc": (1, 1),
                               "idle_solo": (0, 1), "idle_conc": (2, 1)}

    def l2():
        rep = analyze([fx["mix"]])
        if set(rep["cells"]) != {"wait_solo", "wait_conc", "idle_solo", "idle_conc"}:
            return False
        tot = sum(c["n_sw"] + c["n_no"] for c in rep["cells"].values())
        return tot + rep["n_unclassified"] == rep["n_boundary"] == 11

    def l3():
        c = analyze([fx["mix"]])["cells"].get("idle_solo") or {}
        return c.get("n_sw") == 0 and c.get("n_no") == 1

    def l5():
        rep = analyze([fx["mix"]])
        return rep["n_unclassified"] == 1

    def p1():
        rep = analyze([fx["phase"]])
        return (rep["n_boundary"] == 6
                and (rep.get("phase_channel") or {}).get("n_removed") == 0)

    def p2():
        a = (analyze([fx["phase"]]).get("phase_channel") or {})
        b = (analyze([fx["mix"]]).get("phase_channel") or {})
        return (a.get("values") == ["measure", "warmup"]
                and a.get("discriminates_warmup") is True
                and b.get("values") == ["measure"]
                and b.get("discriminates_warmup") is False)

    def h1():
        rep = analyze([fx["mix"], fx["ds"]])
        ex = {x["file"]: "; ".join(x["reasons"]) for x in rep["excluded"]}
        return (rep["admitted"] == ["mix.holb.jsonl"]
                and set(ex) == {"dropspans.holb.jsonl"}
                and "dropped_spans=4" in ex["dropspans.holb.jsonl"])

    def h2():
        rep = analyze([fx["mix"], fx["err"]])
        ex = {x["file"]: "; ".join(x["reasons"]) for x in rep["excluded"]}
        r = ex.get("unhealthy.holb.jsonl", "")
        return (rep["admitted"] == ["mix.holb.jsonl"]
                and all(k in r for k in ("n_errors=2", "n_invalid_gaps=1",
                                         "n_inconsistent_gaps=3", "timeline_over_host=")))

    def h3():
        rep = analyze([fx["mix"], fx["nosum"]])
        ex = {x["file"]: "; ".join(x["reasons"]) for x in rep["excluded"]}
        return (rep["admitted"] == ["mix.holb.jsonl"]
                and "holb_summary" in ex.get("nosummary.holb.jsonl", ""))

    def h4():
        txt = render_(analyze([fx["mix"], fx["ds"], fx["nosum"]]))
        return ("dropspans.holb.jsonl" in txt and "nosummary.holb.jsonl" in txt
                and "EXCLUDED" in txt and "dropped_spans=4" in txt)

    def h5():
        rep = analyze([fx["mix"], fx["ds"]])
        return rep["n_artifacts_admitted"] == 1 and rep["n_boundary"] == 11

    for name, fn in ((L1, l1), (L2, l2), (L3, l3), (L5, l5), (P1, p1), (P2, p2),
                     (H1, h1), (H2, h2), (H3, h3), (H4, h4), (H5, h5)):
        ck(name, fn)
    return res


# -- 변이본: 각 수리를 **되돌린** 소스 ------------------------------------
_MUT_A_BODY = '''
# 수리 전: 층을 docstring이 주장하던 대로 `n_other_fw` 한 축으로 접는다.
GAP_STRICT = "strict"
GAP_AMBIGUOUS = "ambiguous"
CELLS = ("strict", "ambig")
CELL_DESC = {"strict": "다른 스트림 forward 없음", "ambig": "다른 스트림 forward 있었음"}


def cell_of(e):
    return "strict" if int(e["n_other_fw"] or 0) == 0 else "ambig"
'''

_MUT_B_BODY = '''
def phase_channel(ev):
    # 워밍업 제외
    return [e for e in ev if e.get("phase") in (None, "measure")], {"note": "워밍업 제외"}
'''

_MUT_C_BODY = '''
def health_verdict(h):
    # 수리 전: 텔레메트리 큐 카운터만 본다(프로브 FIFO는 안 본다).
    return ["dropped_events"] if h.get("dropped_events") else []
'''


def _block_span(src, tag):
    b = "# --- " + "BEGIN " + tag + " ---"
    e = "# --- " + "END " + tag + " ---"
    if src.count(b) != 1 or src.count(e) != 1:
        raise AssertionError(f"marker {tag}: begin x{src.count(b)}, end x{src.count(e)}")
    i, j = src.index(b) + len(b), src.index(e)
    if j <= i:
        raise AssertionError(f"marker {tag} out of order")
    return i, j


def _mut(tag, body):
    def go(src):
        i, j = _block_span(src, tag)
        return src[:i] + "\n" + body + "\n" + src[j:]
    return go


# 변이 이름 -> (소스 변이, **정확히** 실패해야 하는 검사 집합)
MUTANTS = {
    "A_revert_layer_axis_to_n_other_fw": (_mut("repair-1", _MUT_A_BODY), {L1, L2, L3, L5}),
    "B_revert_phase_filter_as_warmup":   (_mut("repair-2", _MUT_B_BODY), {P1, P2}),
    "C_revert_health_gate_to_dropped_events": (_mut("repair-3", _MUT_C_BODY), {H1, H2, H3, H4, H5}),
}


def selftest():
    fx = _fixtures()
    res = _checks(globals(), fx)
    ok = True
    for name, status, detail in res:
        print(f"  [{'PASS' if status == 'ok' else 'FAIL'}] {name}"
              + (f"  <{status}: {detail}>" if status != "ok" else ""))
        ok = ok and status == "ok"
    print("BASELINE ALL PASS" if ok else "BASELINE FAILURES PRESENT")
    return 0 if ok else 1


def selftest_mutants():
    """변이본은 **명명된 검사에서만** 실패해야 한다.  다른 예외로 죽으면 불합격."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    fx = _fixtures()
    ok = True
    for name, (mutate, expected) in MUTANTS.items():
        try:
            mutated = mutate(src)
        except AssertionError as e:
            print(f"  [FAIL] mutant {name}: anchor not found ({e}) -- 하네스가 낡았다")
            ok = False
            continue
        if mutated == src:
            print(f"  [FAIL] mutant {name}: source unchanged -- 변이 없음")
            ok = False
            continue
        ns = {"__name__": "_mutant", "__file__": os.path.abspath(__file__)}
        try:
            exec(compile(mutated, f"<mutant:{name}>", "exec"), ns)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] mutant {name}: did not compile/exec ({e!r})")
            ok = False
            continue
        status = {n: s for n, s, _ in _checks(ns, fx)}
        raised = sorted(n.split()[0] for n, s in status.items() if s == "raised")
        failed = {n for n, s in status.items() if s == "fail"}
        missing = sorted(n.split()[0] for n in expected - failed)
        extra = sorted(n.split()[0] for n in failed - expected)
        good = not raised and not missing and not extra
        ids = " ".join(sorted(n.split()[0] for n in failed)) or "(none)"
        print(f"  [{'PASS' if good else 'FAIL'}] mutant {name}: "
              f"{len(failed)}/{len(status)} checks FAIL -> {ids}"
              + (f"; MISSING (통과해버림) {missing}" if missing else "")
              + (f"; RAISED {raised}" if raised else "")
              + (f"; UNEXPECTED {extra}" if extra else ""))
        ok = ok and good
    print("MUTANTS ALL PASS" if ok else "MUTANT FAILURES PRESENT")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("patterns", nargs="*", default=None,
                    help="GATE2 아래 glob (기본: g2_*_agnostic_rep*.holb.jsonl)")
    ap.add_argument("--selftest", action="store_true", help="무변이 배터리 + 변이 3종")
    ap.add_argument("--selftest-mutants", action="store_true", help="변이만")
    ap.add_argument("--gate2", default=GATE2)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    if a.selftest_mutants:
        return selftest_mutants()
    if a.selftest:
        # 교훈 #53: 배터리와 그것을 반증 가능하게 만드는 변이는 한 덩어리다.
        return selftest() or selftest_mutants()

    pats = a.patterns or ["g2_*_agnostic_rep*.holb.jsonl"]
    files = sorted(f for p in pats for f in glob.glob(os.path.join(a.gate2, p)))
    if not files:
        print("no artifacts matched")
        return 2
    rep = analyze(files)
    print(render(rep))
    json.dump(rep, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
