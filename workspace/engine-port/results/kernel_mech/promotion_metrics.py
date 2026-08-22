#!/usr/bin/env python3
"""승격안 6문장 집계 도구 (2026-08-22) — 차단 B1 해소용 **단일 구현**.

★★ 이것은 **집계 도구**다.  결정 규칙도, 판정어도, 인과 서술도 만들지 않는다 ★★

왜 이 파일이 존재하는가
-----------------------
`PROMOTION_DRAFT_SWITCH_2026-08-22.md`의 승격 감사가 **차단 B1**을 냈다:

    "문장 2-5를 계산하는 커밋된 코드가 0건이고, 같은 추정량의 손 구현 3개가 `s` 상한을
     0.038 / 0.0396 / 0.0429로 냈다(+-13% 산포).  정본에 3자리로 적을 근거가 없다."

⇒ 요구사항은 **단일 구현 + 그 구현의 SHA 등재**다(선례: gate #13 `g13_analyze.py`).
이 파일이 승격안 문장 2·3·4·5·6과 스코프 사실의 **모든 수치를 한 번에** 낸다.
JSON에 자기 SHA256을 적으므로 정본은 그 SHA로 이 구현을 고정 인용할 수 있다.

무엇을 하지 않는가 (범위 밖 — 어기면 이 도구가 다시 감사 대상이 된다)
---------------------------------------------------------------------
* GPU 0 · 새 측정 0 · 새 결정량 0 · 새 판정어 0.  입력은 이미 존재하는 아티팩트뿐이다.
* **인과 서술 금지.**  `s <= ...`는 *"가법성 가정 하의 상한"* 이라고만 적는다.
  문장 2의 순서관계는 *"비식별 논증"*(귀속 불가를 보이는 것)이라고만 적는다.
* 선행 `step0_switch_gap.py`는 **읽지도 고치지도 않는다**(별건 수리 완료본).  경계·층·건강도
  규약은 그 파일에서 확립된 것을 **독립 재구현**했고, 다른 점은 아래 "규약" 절에 전부 적었다.

규약 (감사가 구현 의존이라 지적한 항목은 전부 여기서 명시된다)
--------------------------------------------------------------
1. **경계** = 같은 아티팩트 안에서 **연속한 두 `holb_gap` 이벤트**.  각 `holb_gap`은 decode
   forward 1개를 서술한다(`holb_probe.py:445-455`): `gap_ms` = 직전 decode end -> 이번 decode
   start, `decode_dur_ms`/`stream_key`/`decode_bs` = **이번(=착지) decode**의 값.
   ⇒ 경계의 **착지** 속성은 언제나 뒤쪽 이벤트에서 읽는다.
2. **첫 이벤트 처리 규칙 (감사 지적: 155,531 vs 155,551)**.  파일의 첫 gap 이벤트는
   `gap_class=="first"`이고 `gap_ms is None`이다 — 앞선 decode가 없어서 간극을 못 잰 것이지
   `stream_key`/`decode_bs`가 없는 게 아니다.  두 규약을 **둘 다** 계산해 보고한다:
       seeded   : first 이벤트를 **체인의 씨앗(prev)** 으로만 쓰고 경계는 만들지 않는다.
       unseeded : first 이벤트를 아예 버린다(다음 이벤트가 새 체인의 씨앗이 된다).
   차이는 **아티팩트당 정확히 경계 1개**(전부 정상상태 셀).  기본값 = `seeded`.
3. **층**은 두 축을 **접지 않는다**(step0 수리에서 확립된 사실).
       축 W: `gap_class` -- strict <=> 간극 내 모든 관측에서 decode pending>0 (`holb_probe.py:69-73`)
       축 O: `n_other_fw` -- 간극 안에 해소된 다른 스트림 forward가 있었는가
   문장 2·3·4는 **축 W의 strict 층**에서 계산하고, 축 O는 **별개 축으로 보고**만 한다.
4. **착지 스트림 식별**: 아티팩트별로 `decode_dur_ms` 중앙값이 **큰** stream_key = `PART`
   (분할 파티션, config target 74/34), **작은** 쪽 = `FULL`(무분할 108).  ★target이 아니라
   관측량으로 식별한다(Stage 0 교훈: pin은 realized로 검증).
5. **건강도 게이트**: `dropped_spans`·`n_errors`·`n_invalid_gaps`·`n_inconsistent_gaps`·
   `dropped_events` 중 하나라도 >0이면 제외, `timeline_over_host`가 1+-1% 밖이면 제외,
   `holb_summary`가 없으면 fail-closed 제외.  ★아티팩트당 요약 이벤트가 **여러 개**이므로
   카운터는 **최댓값**, `timeline_over_host`는 **1에서 가장 먼 값**으로 게이트한다.
   제외는 폐기가 아니라 **보고**다(게이트 #21) — 파일명과 사유가 화면과 JSON 양쪽에 남는다.
6. **DO_NOT_CITE**: 서로 다른 모드를 섞은 풀링값(전환 풀링 중앙값, residency 풀링비)은
   JSON 안에서 키 이름이 `__DO_NOT_CITE`로 끝나고 `do_not_cite` 레지스트리에도 등재된다.

자기검사 + 변이 테스트 (교훈 #53)
---------------------------------
    python3 promotion_metrics.py --selftest            # 무변이 배터리 + 변이 3종
    python3 promotion_metrics.py --selftest-mutants    # 변이만
    python3 promotion_metrics.py                       # 실제 아티팩트 집계 (GPU 0)

변이 3종은 각각 (a) 착지 스트림 식별 (b) 매칭 (c) 건강도 게이트를 **되돌린다**.  각 변이는
**명명된 검사에서만** 실패해야 한다(extra 없음, raised 없음).  검사 본체는 언제나 무변이
파일에서 오므로 변이본이 자기 심사관을 약화시킬 수 없다.

★ 의도적으로 넣지 **않은** 검사: `s_upper == (d+2s)/2` 같은 **항등식 검사**(교훈 #9/#53).
  같은 코드가 양변을 계산하므로 어떤 변이에서도 실패하지 않는다 = 반증 불가능한 검사다.
  대신 그 식의 **입력**(셀 증분·매칭·착지 라벨)을 변이로 반증 가능하게 검사한다.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import statistics as st
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
GATE2 = os.path.join(HERE, "..", "p1_gates", "gate2")
DEFAULT_PATTERN = "g2_*_agnostic_rep*.holb.jsonl"
OUT_JSON = os.path.join(HERE, "PROMOTION_METRICS_2026-08-22.json")

SEED = 1
B_BOOT = 10000
MIN_MATCH_N = 5          # 매칭 셀당 최소 정상상태 참조 건수 (기본)
TLH_TOL = 0.01           # timeline_over_host 허용대
QUANTUM_MS = 1.024e-3    # cudaEventElapsedTime 의 관측 양자 (진단용, 문턱 아님)

GAP_STRICT = "strict"
GAP_AMBIGUOUS = "ambiguous"

FRAME = (
    "승격안 6문장 집계 도구. GPU 0 · 새 측정 0 · 새 결정량 0 · 새 판정어 0 · 성능 판정 0건. "
    "이 산출은 서술 통계이며, 인과 귀속은 audit_step0_2026-08-22/VERDICT.md 에서 이미 REFUTED 됐다."
)

CITATION_STOPS = [
    "0.910 ms(및 어떤 단일 셀 증분)를 '전환 기계 비용' 또는 그 상한으로 인용 금지 — 어떤 술어로도.",
    "s <= ... 는 **가법성 가정 하의 상한**이다. 인과 비용이 아니고 d(드레인+백엔드 스왑)와 분리되지 않는다.",
    "문장 2의 순서관계는 **비식별 논증**이다(전환 경계의 간극 증가를 파티션 변경에 귀속할 수 없다). "
    "'전환이 싸다' 또는 '자유롭게 전환해도 된다'로 읽지 말 것 — HE0 불변.",
    "모델 비교 금지: 이 20개에서 모델 == 노드 == job 완전 앨리어스(875346=gpu41=Granite x10, "
    "875344=gpu42=Zamba2 x10), 노드 2개.",
    "residency 수치를 정본 C2(2.36-2.91x 등)와 대조 금지 — 다른 캠페인·다른 격자(교훈 #31).",
    "관측된 전환은 green-ctx(74/34) <-> plain full-SM(108) 뿐 — green->green 전환 0건. "
    "'green-ctx 파티션 간 전환 비용을 쟀다'는 거짓.",
    "절대 수준(간극 ms)을 엔진 속성으로 인용 금지 — probe-off 대조 0건, 프로브 자기 비용 미분리.",
]

# ==========================================================================
# 중심추정량
# ==========================================================================


def med(xs):
    return st.median(xs) if xs else float("nan")


def _trimmed_mean(xs, frac):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(len(s) * frac)
    core = s[k:len(s) - k] or s
    return st.fmean(core)


CENTRALS = {
    "median": med,
    "trim10": lambda xs: _trimmed_mean(xs, 0.10),
    "trim20": lambda xs: _trimmed_mean(xs, 0.20),
}


# ==========================================================================
# 아티팩트 읽기
# ==========================================================================

HEALTH_COUNTERS = ("dropped_spans", "n_errors", "n_invalid_gaps",
                   "n_inconsistent_gaps", "dropped_events")


def read_artifact(path):
    """(gap 이벤트 목록[프로그램 순서], health 요약).  계측 채널은 읽기 전용."""
    gaps = []
    counters = {k: None for k in HEALTH_COUNTERS}
    health = {"has_summary": False, "n_summary": 0, "timeline_over_host": None}
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
            health["n_summary"] += 1
            for k in HEALTH_COUNTERS:
                v = d.get(k)
                if v is not None:
                    counters[k] = max(counters[k] or 0, int(v))
            t = d.get("timeline_over_host")
            if t is not None:
                # 요약이 여러 개다 -> 1 에서 **가장 먼** 값으로 게이트한다.
                cur = health["timeline_over_host"]
                if cur is None or abs(float(t) - 1.0) > abs(cur - 1.0):
                    health["timeline_over_host"] = float(t)
    health.update(counters)
    return gaps, health


# --- BEGIN repair-health ---
def health_verdict(h):
    """이 아티팩트를 집계에서 **제외**해야 할 사유 목록(빈 리스트 = 포함).

    가장 취약한 실패 모드는 `dropped_spans`(프로브 FIFO 포화, `holb_probe.py:368`)다 —
    그게 >0이면 decode 체인이 끊겨 **간극 부풀림과 `stream_key` 오라벨이 동시에** 난다.
    건강도를 확인할 수 없는 아티팩트도 fail-closed 로 제외한다.
    """
    if not h.get("has_summary"):
        return ["no holb_summary event -- health unverifiable (fail-closed)"]
    reasons = []
    for f in HEALTH_COUNTERS:
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
# --- END repair-health ---


# --- BEGIN repair-role ---
def role_map(gaps):
    """stream_key -> "PART" / "FULL".

    ★ 식별은 **관측량**으로 한다: 아티팩트 안에서 `decode_dur_ms` 중앙값이 **큰** 스트림이
    분할 파티션(PART, decode 34 SM), **작은** 쪽이 무분할(FULL, 108 SM)이다.  config의
    target(74/34)을 믿지 않고 realized 로 판정한다(Stage 0 교훈).
    스트림이 1개뿐이면 그 하나를 FULL 로 둔다(분할 상태 관측 0건).
    """
    dur = {}
    for d in gaps:
        v = d.get("decode_dur_ms")
        if v is None:
            continue
        dur.setdefault(d.get("stream_key"), []).append(float(v))
    meds = {k: med(v) for k, v in dur.items() if v}
    if not meds:
        return {}, {}
    slow = max(meds, key=lambda k: meds[k])
    roles = {k: ("PART" if k == slow else "FULL") for k in meds}
    if len(meds) == 1:
        roles = {k: "FULL" for k in meds}
    return roles, meds
# --- END repair-role ---


def boundaries(gaps, roles, seed_first=True):
    """연속 gap 이벤트 쌍 -> 경계 레코드.  착지 속성은 뒤쪽 이벤트에서 읽는다."""
    out = []
    prev = None
    for d in gaps:
        if d.get("gap_ms") is None:
            # gap_class=="first": 간극은 못 쟀지만 stream_key/decode_bs 는 유효하다.
            if seed_first and d.get("stream_key") is not None:
                prev = d
            continue
        if d.get("stream_key") is None:
            continue
        if prev is not None:
            out.append({
                "gap": float(d["gap_ms"]),
                "dur": float(d.get("decode_dur_ms") or float("nan")),
                "sw": d["stream_key"] != prev["stream_key"],
                "gc": d.get("gap_class"),
                "nof": int(d.get("n_other_fw") or 0),
                "ofw": float(d.get("other_stream_fw_ms") or 0.0),
                "bs": d.get("decode_bs"),
                "pbs": prev.get("decode_bs"),
                "role": roles.get(d["stream_key"], "FULL"),
                "phase": d.get("phase"),
            })
        prev = d
    return out


# ==========================================================================
# 부류 정의 (문장 2·3·4가 공유)
# ==========================================================================

def is_bs_up(b):
    return b["bs"] is not None and b["pbs"] is not None and b["bs"] > b["pbs"]


def is_bs_down(b):
    return b["bs"] is not None and b["pbs"] is not None and b["bs"] < b["pbs"]


CLASS_DESC = {
    "SW_to_PART": "전환 · 착지 = 분할 스트림 (승격안 서술: admission)",
    "SW_to_FULL": "전환 · 착지 = 무분할 스트림 (승격안 서술: completion+merge)",
    "NS_to_PART_bsup": "전환 아님 · decode_bs 증가 · 착지 = 분할 스트림 (인덱스 불변)",
    "NS_bsdown": "전환 아님 · decode_bs 감소 (요청 은퇴) -- 모형 밖 네 번째 부류",
}


def classify(strict):
    return {
        "SW_to_PART": [b for b in strict if b["sw"] and b["role"] == "PART"],
        "SW_to_FULL": [b for b in strict if b["sw"] and b["role"] == "FULL"],
        "NS_to_PART_bsup": [b for b in strict
                            if not b["sw"] and is_bs_up(b) and b["role"] == "PART"],
        "NS_bsdown": [b for b in strict if not b["sw"] and is_bs_down(b)],
    }


STEADY_DEFS = {
    # 정상상태 = 전환도 아니고 decode_bs 증가도 아닌 경계.
    "bs_equal": lambda b: (not b["sw"]) and b["bs"] == b["pbs"],
    "bs_not_increased": lambda b: (not b["sw"]) and not is_bs_up(b),
    "bs_equal_solo": lambda b: (not b["sw"]) and b["bs"] == b["pbs"] and b["nof"] == 0,
}


MATCH_KEYS = {
    # "착지 스트림 x decode_bs".  스트림은 아티팩트별 객체이므로 file 을 함께 잡는 것이
    # 더 엄격한 매칭이다(모델/노드/job 도 같이 고정된다).
    "file_role_bs": lambda b: (b["file"], b["role"], b["bs"]),
    "role_bs": lambda b: (b["role"], b["bs"]),
}


# --- BEGIN repair-match ---
def matched_increment(target, steady, keyfn, min_n=MIN_MATCH_N, central=med):
    """착지 스트림 x decode_bs 를 **매칭**한 증분.

    셀 = `keyfn(record)`.  각 셀의 기준선 = 그 셀의 **정상상태** 경계들의 중심값.
    참조가 `min_n` 미만인 셀은 **버린다**(그 셀의 target 경계도 함께 빠진다) — 매칭 없는
    비교를 조용히 섞지 않기 위해서다.  반환 `value` = 증분들의 중심값.
    """
    ref = {}
    for b in steady:
        ref.setdefault(keyfn(b), []).append(b["gap"])
    base = {k: central(v) for k, v in ref.items() if len(v) >= min_n}
    inc = [b["gap"] - base[keyfn(b)] for b in target if keyfn(b) in base]
    return {
        "n_in": len(target),
        "n_used": len(inc),
        "n_dropped_unmatched": len(target) - len(inc),
        "n_cells": len(base),
        "value": central(inc),
        "increments": inc,
    }


def bs_standardized_pair(part_pairs, full_pairs, min_n=MIN_MATCH_N, central=med):
    """decode_bs 구성을 맞춘 뒤의 (PART, FULL) 대표값.

    `part_pairs`/`full_pairs` = [(bs, decode_dur_ms), ...].  양쪽에 `min_n` 건 이상 있는 bs
    셀만 쓰고, 셀 가중치는 `min(n_PART, n_FULL)`(공통 지지 위의 직접 표준화)이다.
    ⇒ 두 팔이 **같은 bs 구성** 위에서 비교된다.
    """
    P, F = {}, {}
    for bs, v in part_pairs:
        P.setdefault(bs, []).append(v)
    for bs, v in full_pairs:
        F.setdefault(bs, []).append(v)
    cells = sorted(bs for bs in set(P) & set(F)
                   if len(P[bs]) >= min_n and len(F[bs]) >= min_n)
    w = {bs: min(len(P[bs]), len(F[bs])) for bs in cells}
    tw = sum(w.values())
    if not tw:
        return float("nan"), float("nan"), 0, 0
    sp = sum(w[bs] * central(P[bs]) for bs in cells) / tw
    sf = sum(w[bs] * central(F[bs]) for bs in cells) / tw
    return sp, sf, len(cells), tw
# --- END repair-match ---


def boot_ci(vals, stat=med, b=B_BOOT, seed=SEED):
    """1차 단위 = **아티팩트**(경계는 독립 반복이 아니다)."""
    vals = [v for v in vals if isinstance(v, float) and not math.isnan(v)]
    if len(vals) < 3:
        return [float("nan"), float("nan")], len(vals)
    rng = random.Random(seed)
    draws = sorted(stat([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(b))
    return [draws[int(0.025 * b)], draws[int(0.975 * b)]], len(vals)


def in_quanta(x, q=QUANTUM_MS, tol=1e-7, scale_ms=None):
    """(양자 개수, 격자 위인가, 격자에서 벗어난 양[양자]) -- **진단**이다.

    device 이벤트 간극은 이 격자에서 q=1.024 us 로 양자화돼 관측된다.  중앙값도 대개 격자
    위에 떨어지므로, 서로 다른 설계 선택이 **같은 양자 개수**를 내면 표기 자릿수가 그
    이상으로 의미를 갖지 않는다는 뜻이다.  어떤 게이트도 이 값을 쓰지 않는다.

    `tol` = float32 epsilon 수준(`gap_ms`는 cudaEventElapsedTime(float32) 출신).
    `scale_ms` = 잔차가 붙는 **입력의 크기**.  차(d+2s 처럼 큰 값들의 차)에서는 잔차가
    결과가 아니라 입력 크기에 비례하므로 호출자가 명시해야 한다.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan"), False, float("nan")
    n = x / q
    scale = (abs(scale_ms) if scale_ms is not None else abs(x)) / q
    off = abs(n - round(n))
    return n, off < max(tol, scale * tol), off


def model_cluster(fname):
    n = fname.lower()
    if "granite" in n:
        return "granite"
    if "zamba" in n:
        return "zamba2"
    return "unknown"


# ==========================================================================
# 아티팩트 1개 요약
# ==========================================================================

def summarize(path, seed_first=True):
    gaps, health = read_artifact(path)
    roles, key_meds = role_map(gaps)
    bnd = boundaries(gaps, roles, seed_first=seed_first)
    bnd_unseeded = boundaries(gaps, roles, seed_first=False)
    fname = os.path.basename(path)
    for b in bnd:
        b["file"] = fname
    strict = [b for b in bnd if b["gc"] == GAP_STRICT]
    cls = classify(strict)
    row = {
        "file": fname,
        "_path": path,
        "model_cluster": model_cluster(fname),
        "health": health,
        "n_gap_events": len(gaps),
        "n_boundary": len(bnd),
        "n_boundary_unseeded": len(bnd_unseeded),
        "n_switch": sum(1 for b in bnd if b["sw"]),
        "n_strict": len(strict),
        "n_strict_switch": sum(1 for b in strict if b["sw"]),
        "n_ambiguous": sum(1 for b in bnd if b["gc"] == GAP_AMBIGUOUS),
        "n_stream_keys": len(key_meds),
        "stream_key_dur_medians": sorted(round(v, 4) for v in key_meds.values()),
        "phase_values": sorted({str(b["phase"]) for b in bnd}),
        "decode_bs_max": max((b["bs"] for b in bnd if b["bs"] is not None), default=None),
        "n_decode_bs_gt15": sum(1 for b in bnd if (b["bs"] or 0) > 15),
        "strict_n_other_fw_pos": sum(1 for b in strict if b["nof"] > 0),
        # 문장 2 -- 부류별 중앙값
        "med": {k: med([b["gap"] for b in v]) for k, v in cls.items()},
        "n": {k: len(v) for k, v in cls.items()},
        "med_switch_pooled": med([b["gap"] for b in strict if b["sw"]]),
        # 문장 5 -- residency
        "residency": {},
        "_bnd": bnd,
        "_strict": strict,
    }
    # 문장 5: 착지 decode 의 지속시간
    P = [b for b in bnd if b["role"] == "PART"]
    F = [b for b in bnd if b["role"] == "FULL"]
    mp, mf = med([b["dur"] for b in P]), med([b["dur"] for b in F])
    sp, sf, ncell, wtot = bs_standardized_pair([(b["bs"], b["dur"]) for b in P],
                                               [(b["bs"], b["dur"]) for b in F])
    row["residency"] = {
        "n_land_PART": len(P), "n_land_FULL": len(F),
        "frac_land_PART": len(P) / len(bnd) if bnd else float("nan"),
        "med_dur_PART_ms": mp, "med_dur_FULL_ms": mf,
        "ratio": mp / mf if mf else float("nan"),
        "excess_ms_per_PART_step": mp - mf,
        "bs_matched_med_dur_PART_ms": sp, "bs_matched_med_dur_FULL_ms": sf,
        "bs_matched_ratio": sp / sf if sf else float("nan"),
        "bs_matched_n_cells": ncell, "bs_matched_weight": wtot,
        "mean_other_stream_fw_ms_PART": st.fmean([b["ofw"] for b in P]) if P else float("nan"),
        "mean_other_stream_fw_ms_FULL": st.fmean([b["ofw"] for b in F]) if F else float("nan"),
    }
    return row


# ==========================================================================
# 문장별 집계
# ==========================================================================

def sentence2(rows):
    """순서관계 (strict 층).  ★비식별 논증 -- 가법성 가정 불필요."""
    allstrict = [b for r in rows for b in r["_strict"]]
    cls = classify(allstrict)
    out = {
        "layer": "gap_class == strict",
        "reading": ("비식별 논증: 인덱스가 바뀌지 않은 경계가 전환 경계보다 크다면, "
                    "전환 경계의 간극 증가를 파티션 변경에 귀속할 수 없다. 인과 비용 추정 아님."),
        "pooled_median_ms": {k: med([b["gap"] for b in v]) for k, v in cls.items()
                             if k != "NS_bsdown"},
        "n": {k: len(v) for k, v in cls.items() if k != "NS_bsdown"},
        "per_artifact_median_ms": {
            k: [r["med"][k] for r in rows] for k in
            ("NS_to_PART_bsup", "SW_to_PART", "SW_to_FULL")},
        "per_artifact_n": {
            k: [r["n"][k] for r in rows] for k in
            ("NS_to_PART_bsup", "SW_to_PART", "SW_to_FULL")},
    }
    # ★ 모드별 순서관계: 혼합 중앙값을 쓰지 않고 아티팩트별 부호를 센다.
    for tgt in ("SW_to_PART", "SW_to_FULL"):
        sign, deltas = 0, []
        for r in rows:
            a, b = r["med"]["NS_to_PART_bsup"], r["med"][tgt]
            if math.isnan(a) or math.isnan(b):
                continue
            deltas.append(b - a)
            sign += 1 if a > b else 0
        ci, n = boot_ci(deltas)
        out[f"NS_gt_{tgt}"] = {
            "n_artifacts_holding": sign,
            "n_artifacts_evaluated": len(deltas),
            "paired_delta_ms_median": med(deltas),
            "paired_delta_ms_ci95_repboot": ci,
            "note": f"paired delta = median({tgt}) - median(NS_to_PART_bsup), 아티팩트별.",
        }
    # ★ 풀링 "전환" 중앙값 = 착지 PART/FULL 두 모드의 혼합 -> 인용 금지.
    out["switch_pooled_median_ms__DO_NOT_CITE"] = med(
        [b["gap"] for b in allstrict if b["sw"]])
    out["switch_pooled_n"] = sum(1 for b in allstrict if b["sw"])
    out["switch_pooled_per_artifact__DO_NOT_CITE"] = [r["med_switch_pooled"] for r in rows]
    out["why_do_not_cite"] = ("풀링 '전환' 중앙값은 착지 PART 와 착지 FULL 두 모드의 혼합이다"
                              "(모드가 서로 다른 중심을 가진다) -- 혼합 중앙값은 혼합비의 함수라 "
                              "어느 모드도 서술하지 않는다.")
    return out


def sentence3(rows, steady_def="bs_equal", match_key="file_role_bs",
              min_n=MIN_MATCH_N, central_name="median"):
    """가법 분해 (가법성 가정 하의 상한).  네 번째 부류(bs 감소)도 함께 낸다."""
    central = CENTRALS[central_name]
    keyfn = MATCH_KEYS[match_key]
    allstrict = [b for r in rows for b in r["_strict"]]
    steady = [b for b in allstrict if STEADY_DEFS[steady_def](b)]
    cls = classify(allstrict)
    cells = {k: matched_increment(v, steady, keyfn, min_n, central) for k, v in cls.items()}
    d2s = (cells["SW_to_PART"]["value"] + cells["SW_to_FULL"]["value"]
           - cells["NS_to_PART_bsup"]["value"])
    # 아티팩트별 (1차 단위 = 아티팩트).  ★집계 순서도 설계 선택이다 -- 두 대안을 함께 낸다:
    #   (i) 아티팩트별 s 상한의 중앙값   (ii) 아티팩트별 셀 값의 중앙값으로 만든 s 상한
    per, per_cells = [], {k: [] for k in ("SW_to_PART", "SW_to_FULL",
                                          "NS_to_PART_bsup", "NS_bsdown")}
    for r in rows:
        s_ = [b for b in r["_strict"] if STEADY_DEFS[steady_def](b)]
        c_ = classify(r["_strict"])
        v = {k: matched_increment(c_[k], s_, keyfn, min_n, central)["value"]
             for k in per_cells}
        for k in per_cells:
            per_cells[k].append(v[k])
        x = v["SW_to_PART"] + v["SW_to_FULL"] - v["NS_to_PART_bsup"]
        per.append(x / 2.0 if not math.isnan(x) else float("nan"))
    ci, n = boot_ci(per)
    cell_am = {k: med([x for x in v if not math.isnan(x)]) for k, v in per_cells.items()}
    d2s_am = (cell_am["SW_to_PART"] + cell_am["SW_to_FULL"] - cell_am["NS_to_PART_bsup"])
    return {
        "assumption": ("★가법성 가정 하에서만 식별된다. 3셀 4미지수이므로 d(드레인+백엔드 스왑, "
                       "양쪽 공통)와 s(파티션 인덱스 변경 자체)는 분리되지 않는다. "
                       "s <= (d+2s)/2 와 d <= d+2s 는 **상한**이지 비용 추정치가 아니다."),
        "config": {"steady_def": steady_def, "match_key": match_key,
                   "min_match_n": min_n, "central": central_name,
                   "steady_def_expr": {
                       "bs_equal": "전환 아님 AND decode_bs 불변",
                       "bs_not_increased": "전환 아님 AND decode_bs 증가 아님(감소 포함)",
                       "bs_equal_solo": "전환 아님 AND decode_bs 불변 AND n_other_fw==0"}[steady_def]},
        "n_steady": len(steady),
        "steady_median_ms": {
            "FULL": med([b["gap"] for b in steady if b["role"] == "FULL"]),
            "PART": med([b["gap"] for b in steady if b["role"] == "PART"])},
        "cells": {k: {"desc": CLASS_DESC[k], "n_in": v["n_in"], "n_used": v["n_used"],
                      "n_dropped_unmatched": v["n_dropped_unmatched"],
                      "n_cells": v["n_cells"], "matched_increment_ms": v["value"]}
                  for k, v in cells.items()},
        "d_plus_2s_ms": d2s,
        "s_upper_ms": d2s / 2.0,
        "d_upper_ms": d2s,
        "quantisation_diagnostic": {
            "quantum_ms": QUANTUM_MS,
            "cell_increment_in_quanta": {
                k: in_quanta(v["value"])[0] for k, v in cells.items()},
            "cell_increment_is_exact_multiple": {
                k: in_quanta(v["value"])[1] for k, v in cells.items()},
            "cell_increment_off_grid_quanta": {
                k: in_quanta(v["value"])[2] for k, v in cells.items()},
            "d_plus_2s_in_quanta": in_quanta(d2s)[0],
            # 차의 잔차는 **입력 크기**에 비례한다 -> scale 을 명시한다.
            "d_plus_2s_is_exact_multiple": in_quanta(
                d2s, scale_ms=sum(abs(cells[k]["value"]) for k in
                                  ("SW_to_PART", "SW_to_FULL", "NS_to_PART_bsup")))[1],
            "d_plus_2s_off_grid_quanta": in_quanta(d2s)[2],
            "note": ("간극은 1.024 us 격자 위에서 관측된다. 서로 다른 설계 선택이 같은 "
                     "양자 개수를 내면 그보다 잘게 적는 것은 의미가 없다. 이것은 진단이며 "
                     "어떤 게이트도 이 값을 쓰지 않는다.")},
        "s_upper_ms_per_artifact": per,
        "s_upper_ms_ci95_repboot": ci,
        "s_upper_ms_artifact_median": med(per),
        "s_upper_ms_per_artifact_min": min(per) if per else float("nan"),
        "s_upper_ms_per_artifact_max": max(per) if per else float("nan"),
        "n_artifacts_with_negative_s_upper": sum(1 for x in per
                                                 if not math.isnan(x) and x < 0),
        "cell_increment_ms_artifact_median": cell_am,
        "d_plus_2s_ms_from_artifact_median_cells": d2s_am,
        "s_upper_ms_from_artifact_median_cells": d2s_am / 2.0,
        "aggregation_note": ("집계 순서(경계 풀링 vs 아티팩트별 후 중앙값)도 설계 선택이다. "
                             "세 값을 모두 낸다: 풀링 s_upper · 아티팩트별 s_upper 의 중앙값 · "
                             "아티팩트별 셀 값의 중앙값으로 만든 s_upper."),
        "identity_note": ("d+2s = (SW_to_PART)+(SW_to_FULL)-(NS_to_PART_bsup) 는 **정의**다. "
                          "이 식 자체를 검사로 쓰지 않는다(항등식, 교훈 #9/#53) -- "
                          "검사는 입력(매칭·착지 라벨)에 건다."),
    }


def sentence4(s3):
    """생애주기 표 = 문장 3의 네 부류를 한 표로 (새 계산 없음)."""
    order = ("SW_to_PART", "SW_to_FULL", "NS_to_PART_bsup", "NS_bsdown")
    return {
        "note": ("문장 3과 **같은 매칭 증분**이다(재계산 없음). 네 번째 부류(decode_bs 감소, "
                 "요청 은퇴)는 3셀 가법 모형 **밖**의 효과 크기를 보여주기 위한 것이다."),
        "rows": [{"class": k, "desc": CLASS_DESC[k],
                  "n_used": s3["cells"][k]["n_used"],
                  "matched_increment_ms": s3["cells"][k]["matched_increment_ms"]}
                 for k in order],
    }


def sentence5(rows, s_upper):
    """residency.  ★공통 분모 비교까지 낸다."""
    per = []
    for r in rows:
        d = r["residency"]
        steps = r["n_boundary"]
        excess_per_step = d["excess_ms_per_PART_step"] * d["frac_land_PART"]
        sw_per_step = s_upper * (r["n_switch"] / steps) if steps else float("nan")
        per.append({
            "file": r["file"], "model_cluster": r["model_cluster"],
            "med_dur_FULL_ms": d["med_dur_FULL_ms"], "med_dur_PART_ms": d["med_dur_PART_ms"],
            "ratio": d["ratio"], "bs_matched_ratio": d["bs_matched_ratio"],
            "frac_land_PART": d["frac_land_PART"],
            "n_switch": r["n_switch"], "n_steps": steps,
            "switch_per_step": r["n_switch"] / steps if steps else float("nan"),
            "residency_excess_ms_per_step": excess_per_step,
            "switch_attributed_ms_per_step": sw_per_step,
            "ratio_residency_over_switch": (excess_per_step / sw_per_step
                                            if sw_per_step else float("nan")),
            "mean_other_stream_fw_ms_PART": d["mean_other_stream_fw_ms_PART"],
            "mean_other_stream_fw_ms_FULL": d["mean_other_stream_fw_ms_FULL"],
        })
    out = {"per_artifact": per,
           "common_denominator_definition": (
               "residency_excess_ms_per_step = (med_dur_PART - med_dur_FULL) * frac_land_PART ; "
               "switch_attributed_ms_per_step = s_upper * (n_switch / n_steps). "
               "두 항이 **같은 분모(decode step 1개)** 위에 있다. s_upper 는 문장 3의 "
               "가법성 상한이므로 이 비는 **하한 배수**다."),
           "s_upper_used_ms": s_upper,
           "caveat": ("이것은 decode-SM 민감도의 재진술이다. 새 레버 측정이 아니고, "
                      "정본 C2의 값(다른 모델·다른 캠페인)과 비교 금지."),
           }
    for k in ("ratio", "bs_matched_ratio", "ratio_residency_over_switch",
              "residency_excess_ms_per_step", "switch_attributed_ms_per_step",
              "mean_other_stream_fw_ms_PART", "mean_other_stream_fw_ms_FULL"):
        xs = [p[k] for p in per if not math.isnan(p[k])]
        out[k + "_range"] = [min(xs), max(xs)] if xs else [float("nan")] * 2
    # 모델 군집별 (모델==노드==job 앨리어스이므로 '모델 효과'로 읽으면 안 된다)
    out["by_model_cluster"] = {}
    for g in sorted({p["model_cluster"] for p in per}):
        sub = [p for p in per if p["model_cluster"] == g]
        out["by_model_cluster"][g] = {
            "n_artifacts": len(sub),
            "med_dur_FULL_ms_range": [min(p["med_dur_FULL_ms"] for p in sub),
                                      max(p["med_dur_FULL_ms"] for p in sub)],
            "med_dur_PART_ms_range": [min(p["med_dur_PART_ms"] for p in sub),
                                      max(p["med_dur_PART_ms"] for p in sub)],
            "ratio_range": [min(p["ratio"] for p in sub), max(p["ratio"] for p in sub)],
            "bs_matched_ratio_range": [min(p["bs_matched_ratio"] for p in sub),
                                       max(p["bs_matched_ratio"] for p in sub)],
            "ratio_residency_over_switch_range": [
                min(p["ratio_residency_over_switch"] for p in sub),
                max(p["ratio_residency_over_switch"] for p in sub)],
        }
    out["model_cluster_caveat"] = (
        "모델 == 노드 == job 완전 앨리어스(875346=gpu41=Granite x10 · 875344=gpu42=Zamba2 x10). "
        "군집 간 차이를 **모델 효과로 읽지 말 것**.")
    # ★ 풀링값 = 두 군집의 혼합 -> 인용 금지
    out["pooled_med_of_artifact_medians_FULL_ms__DO_NOT_CITE"] = med(
        [p["med_dur_FULL_ms"] for p in per])
    out["pooled_med_of_artifact_medians_PART_ms__DO_NOT_CITE"] = med(
        [p["med_dur_PART_ms"] for p in per])
    out["pooled_ratio_of_medians__DO_NOT_CITE"] = (
        med([p["med_dur_PART_ms"] for p in per]) / med([p["med_dur_FULL_ms"] for p in per]))
    out["pooled_bs_matched_ratio_of_medians__DO_NOT_CITE"] = med(
        [p["bs_matched_ratio"] for p in per])
    out["pooled_event_median_FULL_ms__DO_NOT_CITE"] = med(
        [b["dur"] for r in rows for b in r["_bnd"] if b["role"] == "FULL"])
    out["pooled_event_median_PART_ms__DO_NOT_CITE"] = med(
        [b["dur"] for r in rows for b in r["_bnd"] if b["role"] == "PART"])
    out["why_do_not_cite"] = (
        "20개는 두 군집(각 10개)의 혼합이고 두 군집의 중심이 다르다. 풀링 중앙값은 "
        "10번째/11번째 값의 평균 = **두 군집 경계에 걸린 값**이라 어느 군집도 서술하지 않는다.")
    return out


def sentence6(rows):
    allstrict = [b for r in rows for b in r["_strict"]]
    allb = [b for r in rows for b in r["_bnd"]]
    nof_pos = sum(1 for b in allstrict if b["nof"] > 0)
    phases = sorted({p for r in rows for p in r["phase_values"]})
    return {
        "strict_n_other_fw_pos": nof_pos,
        "strict_n": len(allstrict),
        "strict_frac_n_other_fw_pos": nof_pos / len(allstrict) if allstrict else float("nan"),
        "strict_definition": ("holb_probe.py:69-73 -- strict <=> 간극 내 **모든** 관측에서 "
                              "decode pending > 0. '다른 스트림 forward 없음'이 **아니다**."),
        "ambiguous_definition": "간극 내 **어느** 관측에서 decode pending == 0 (요청 도착 유휴).",
        "phase_values": phases,
        "phase_is_identity": len(phases) <= 1,
        "phase_note": ("holb_probe.py:469 가 모든 gap 이벤트를 리터럴 \"measure\"로 방출한다 "
                       "=> phase 필터는 항등식(0건 제거). '워밍업 제외'는 거짓 통제였다. "
                       "이 분석에 워밍업 제외는 **적용되어 있지 않다**."),
        "decode_bs_max": max((r["decode_bs_max"] for r in rows
                              if r["decode_bs_max"] is not None), default=None),
        "n_decode_bs_gt15": sum(r["n_decode_bs_gt15"] for r in rows),
        "decode_bs_scope_note": ("승격안 스코프의 'decode_bs 1-15'는 **거짓**이다: 관측 최댓값과 "
                                 ">15 이벤트 수를 그대로 적을 것."),
        "n_stream_keys_per_artifact": {r["file"]: r["n_stream_keys"] for r in rows},
        "all_artifacts_have_exactly_two_stream_keys":
            all(r["n_stream_keys"] == 2 for r in rows),
        "green_to_green_note": ("아티팩트당 stream_key 가 정확히 2개 = 관측된 전환은 "
                                "green-ctx(74/34) <-> plain full-SM(108) 뿐. green->green 전환 0건."),
        "n_boundary_seeded": sum(r["n_boundary"] for r in rows),
        "n_boundary_unseeded": sum(r["n_boundary_unseeded"] for r in rows),
        "n_switch": sum(r["n_switch"] for r in rows),
        "n_strict_switch": sum(r["n_strict_switch"] for r in rows),
        "n_ambiguous_boundary": sum(r["n_ambiguous"] for r in rows),
        "n_gap_events": sum(r["n_gap_events"] for r in rows),
        "gap_quantisation": {
            "quantum_ms": QUANTUM_MS,
            "n_boundaries_on_grid": sum(1 for b in allb if in_quanta(b["gap"])[1]),
            "n_boundaries": len(allb),
            "frac_on_grid": (sum(1 for b in allb if in_quanta(b["gap"])[1]) / len(allb)
                             if allb else float("nan")),
            "tolerance": "float32 eps 수준 (상대 1e-7)",
        },
        "first_event_rule": (
            "seeded(기본): gap_class=='first' 이벤트(gap_ms is None)는 체인의 씨앗으로만 쓰고 "
            "경계를 만들지 않는다 -> 경계 = (gap_ms 있는 이벤트 수). "
            "unseeded: first 이벤트를 버린다 -> 아티팩트당 경계 1개가 줄어든다. "
            "두 규약의 차이는 아티팩트당 정확히 1개(전부 정상상태 셀)이며 두 값 모두 보고한다."),
        "boundary_convention_delta_equals_n_artifacts": (
            sum(r["n_boundary"] for r in rows) - sum(r["n_boundary_unseeded"] for r in rows)
            == len(rows)),
    }


# ==========================================================================
# ★ 민감도 (차단 B1 의 본체: +-13% 산포를 스크립트가 직접 보여야 한다)
# ==========================================================================

SENS_AXES = [
    ("min_match_n", [5, 10, 20]),
    ("steady_def", ["bs_equal", "bs_not_increased", "bs_equal_solo"]),
    ("central", ["median", "trim10", "trim20"]),
    ("match_key", ["file_role_bs", "role_bs"]),
]

BASE_CFG = {"min_match_n": MIN_MATCH_N, "steady_def": "bs_equal",
            "central": "median", "match_key": "file_role_bs"}


def sensitivity(rows, rows_unseeded):
    base = sentence3(rows, BASE_CFG["steady_def"], BASE_CFG["match_key"],
                     BASE_CFG["min_match_n"], BASE_CFG["central"])
    s0 = base["s_upper_ms"]
    table = []

    def add(axis, value, cfg, s3):
        table.append({
            "axis": axis, "value": str(value), "config": dict(cfg),
            "n_steady": s3["n_steady"],
            "SW_to_PART_ms": s3["cells"]["SW_to_PART"]["matched_increment_ms"],
            "SW_to_FULL_ms": s3["cells"]["SW_to_FULL"]["matched_increment_ms"],
            "NS_to_PART_bsup_ms": s3["cells"]["NS_to_PART_bsup"]["matched_increment_ms"],
            "NS_bsdown_ms": s3["cells"]["NS_bsdown"]["matched_increment_ms"],
            "n_used": {k: s3["cells"][k]["n_used"] for k in
                       ("SW_to_PART", "SW_to_FULL", "NS_to_PART_bsup", "NS_bsdown")},
            "n_cells": s3["cells"]["SW_to_PART"]["n_cells"],
            "d_plus_2s_ms": s3["d_plus_2s_ms"],
            "s_upper_ms": s3["s_upper_ms"],
            "pct_vs_base": (s3["s_upper_ms"] - s0) / s0 * 100.0 if s0 else float("nan"),
        })

    add("(base)", "min5/bs_equal/median/file_role_bs", BASE_CFG, base)
    for axis, vals in SENS_AXES:
        for v in vals:
            if v == BASE_CFG[axis]:
                continue
            cfg = dict(BASE_CFG)
            cfg[axis] = v
            add(axis, v, cfg, sentence3(rows, cfg["steady_def"], cfg["match_key"],
                                        cfg["min_match_n"], cfg["central"]))
    # 집계 순서 축 (base 에서 이미 계산돼 있다 -- 재계산 없음)
    for lbl, cells_src, d2s_src in (
            ("per_artifact_cell_median", base["cell_increment_ms_artifact_median"],
             base["d_plus_2s_ms_from_artifact_median_cells"]),):
        cfg = dict(BASE_CFG)
        cfg["aggregation"] = lbl
        table.append({
            "axis": "aggregation", "value": lbl, "config": cfg,
            "n_steady": base["n_steady"],
            "SW_to_PART_ms": cells_src["SW_to_PART"],
            "SW_to_FULL_ms": cells_src["SW_to_FULL"],
            "NS_to_PART_bsup_ms": cells_src["NS_to_PART_bsup"],
            "NS_bsdown_ms": cells_src["NS_bsdown"],
            "n_used": {k: base["cells"][k]["n_used"] for k in
                       ("SW_to_PART", "SW_to_FULL", "NS_to_PART_bsup", "NS_bsdown")},
            "n_cells": base["cells"]["SW_to_PART"]["n_cells"],
            "d_plus_2s_ms": d2s_src, "s_upper_ms": d2s_src / 2.0,
            "pct_vs_base": ((d2s_src / 2.0 - s0) / s0 * 100.0 if s0 else float("nan")),
        })
    cfg = dict(BASE_CFG)
    cfg["aggregation"] = "per_artifact_s_median"
    table.append({
        "axis": "aggregation", "value": "per_artifact_s_median", "config": cfg,
        "n_steady": base["n_steady"],
        "SW_to_PART_ms": float("nan"), "SW_to_FULL_ms": float("nan"),
        "NS_to_PART_bsup_ms": float("nan"), "NS_bsdown_ms": float("nan"),
        "n_used": {k: base["cells"][k]["n_used"] for k in
                   ("SW_to_PART", "SW_to_FULL", "NS_to_PART_bsup", "NS_bsdown")},
        "n_cells": base["cells"]["SW_to_PART"]["n_cells"],
        "d_plus_2s_ms": base["s_upper_ms_artifact_median"] * 2.0,
        "s_upper_ms": base["s_upper_ms_artifact_median"],
        "pct_vs_base": ((base["s_upper_ms_artifact_median"] - s0) / s0 * 100.0
                        if s0 else float("nan")),
    })
    # 첫-이벤트 규약 축 (경계 정의 자체를 바꾼다)
    cfg = dict(BASE_CFG)
    cfg["first_event_rule"] = "unseeded"
    add("first_event_rule", "unseeded", cfg,
        sentence3(rows_unseeded, BASE_CFG["steady_def"], BASE_CFG["match_key"],
                  BASE_CFG["min_match_n"], BASE_CFG["central"]))

    def band(rows_):
        xs = [t["s_upper_ms"] for t in rows_ if not math.isnan(t["s_upper_ms"])]
        if not xs:
            nan = float("nan")
            return {"n_rows": 0, "min_ms": nan, "max_ms": nan,
                    "absolute_span_ms": nan, "span_pct_of_base_abs": nan,
                    "any_non_positive": False}
        lo, hi = min(xs), max(xs)
        return {
            "n_rows": len(xs), "min_ms": lo, "max_ms": hi,
            "absolute_span_ms": hi - lo,
            # ★ 부호가 바뀌면 비율 폭은 의미가 없다 -> base 절댓값 대비로 적는다.
            "span_pct_of_base_abs": ((hi - lo) / abs(s0) * 100.0
                                     if s0 else float("nan")),
            "any_non_positive": any(x <= 0 for x in xs),
        }

    med_family = [t for t in table if t["config"].get("central") == "median"]
    return {
        "base_config": BASE_CFG,
        "table": table,
        "band_all_rows": band(table),
        "band_median_estimator_only": band(med_family),
        "families_note": (
            "두 밴드를 따로 낸다. (i) `band_median_estimator_only` = 중심추정량을 중앙값으로 "
            "고정한 채 매칭 최소건수·정상상태 정의·매칭 키·첫-이벤트 규약만 바꾼 행들. "
            "(ii) `band_all_rows` = 여기에 절사평균 행까지 포함. 절사평균 행은 **추정량 자체를** "
            "바꾸므로 같은 밴드에 넣고 읽으면 안 된다 -- 그러나 숨기지도 않는다."),
        "reading": ("이 표는 **한 구현 안에서** 설계 선택을 하나씩 바꿨을 때 s 상한이 얼마나 "
                    "움직이는지만 보여준다. 어느 행이 옳다는 판정은 하지 않는다. "
                    "정본 표기 자릿수는 밴드 안에서 안정한 자리까지로 제한된다."),
    }


# ==========================================================================
# 전체 분석
# ==========================================================================

def self_sha256(path=None):
    p = path or os.path.abspath(__file__)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_do_not_cite(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if k.endswith("__DO_NOT_CITE"):
                out.append(p)
            out.extend(collect_do_not_cite(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(collect_do_not_cite(v, f"{prefix}[{i}]"))
    return out


def analyze(files):
    admitted, excluded = [], []
    for f in files:
        row = summarize(f, seed_first=True)
        reasons = health_verdict(row["health"])
        if reasons:
            excluded.append({"file": row["file"], "reasons": reasons,
                             "health": row["health"], "n_boundary": row["n_boundary"]})
        else:
            admitted.append(row)

    rep = {
        "tool": os.path.basename(os.path.abspath(__file__)),
        "self_sha256": self_sha256(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frame": FRAME,
        "citation_stops": CITATION_STOPS,
        "inputs": {
            "n_artifacts_seen": len(files),
            "n_artifacts_admitted": len(admitted),
            "admitted": [r["file"] for r in admitted],
            "excluded": excluded,
        },
        "conventions": {
            "boundary": "같은 아티팩트 안의 연속한 두 holb_gap 이벤트. 착지 속성은 뒤쪽 이벤트.",
            "landing_role": ("아티팩트별 decode_dur_ms 중앙값이 큰 stream_key = PART(분할, "
                             "target 74/34), 작은 쪽 = FULL(무분할 108). target 이 아니라 "
                             "realized 로 식별한다."),
            "layers": "축 W = gap_class(strict/ambiguous), 축 O = n_other_fw>0. 두 축은 별개다.",
            "health_gate": ("dropped_spans·n_errors·n_invalid_gaps·n_inconsistent_gaps·"
                            "dropped_events 는 요약 이벤트들의 최댓값, timeline_over_host 는 "
                            "1에서 가장 먼 값으로 게이트(허용 오차 1%). 요약 없으면 fail-closed."),
            "primary_unit": "아티팩트(rep). 경계는 독립 반복이 아니다 -- 부트스트랩 1차 단위 = 아티팩트.",
        },
    }
    if not admitted:
        rep["error"] = "no admitted artefacts"
        return rep

    rows_unseeded = [summarize(r["_path"], seed_first=False) for r in admitted]

    rep["health_observed"] = {r["file"]: {k: r["health"].get(k) for k in
                                          list(HEALTH_COUNTERS) + ["n_summary",
                                                                   "timeline_over_host"]}
                              for r in admitted}
    rep["scope_and_sentence6"] = sentence6(admitted)
    rep["sentence2_ordering"] = sentence2(admitted)
    s3 = sentence3(admitted, BASE_CFG["steady_def"], BASE_CFG["match_key"],
                   BASE_CFG["min_match_n"], BASE_CFG["central"])
    rep["sentence3_additive_upper_bound"] = s3
    rep["sentence4_lifecycle"] = sentence4(s3)
    rep["sentence5_residency"] = sentence5(admitted, s3["s_upper_ms"])
    rep["sensitivity"] = sensitivity(admitted, rows_unseeded)
    rep["per_artifact"] = [{k: v for k, v in r.items() if not k.startswith("_")}
                           for r in admitted]
    rep["do_not_cite"] = collect_do_not_cite(rep)
    return rep


# ==========================================================================
# 렌더링
# ==========================================================================

def _f(x, w=9, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ("{:>%d}" % w).format("--")
    return ("{:>%d.%df}" % (w, p)).format(x)


def render(rep):
    L = []
    A = L.append
    A("=" * 100)
    A("승격안 6문장 집계 (promotion_metrics.py) -- GPU 0 · 새 측정 0 · 판정어 0")
    A(f"self_sha256 = {rep['self_sha256']}")
    A("=" * 100)
    inp = rep["inputs"]
    A(f"아티팩트 {inp['n_artifacts_seen']}개 중 건강도 게이트 통과 {inp['n_artifacts_admitted']}개")
    if inp["excluded"]:
        A(f"EXCLUDED {len(inp['excluded'])}개 (폐기 아님 -- 사유 병기):")
        for x in inp["excluded"]:
            A(f"  - {x['file']}: {'; '.join(x['reasons'])}")
    else:
        A("EXCLUDED 0개")
    if "error" in rep:
        A(rep["error"])
        return "\n".join(L)

    s6 = rep["scope_and_sentence6"]
    A("")
    A("-- 스코프 사실 / 문장 6 " + "-" * 74)
    A(f"gap 이벤트 {s6['n_gap_events']} · 경계(seeded) {s6['n_boundary_seeded']} "
      f"· 경계(unseeded) {s6['n_boundary_unseeded']} · 전환 {s6['n_switch']} "
      f"· strict 경계 중 전환 {s6['n_strict_switch']} · ambiguous 경계 {s6['n_ambiguous_boundary']}")
    A(f"  첫-이벤트 규칙: {s6['first_event_rule']}")
    A(f"  두 규약 차 == 아티팩트 수: {s6['boundary_convention_delta_equals_n_artifacts']}")
    A(f"strict 층 n_other_fw>0 : {s6['strict_n_other_fw_pos']}/{s6['strict_n']} "
      f"= {s6['strict_frac_n_other_fw_pos']*100:.2f}%")
    A(f"phase 값 집합 {s6['phase_values']} · 항등식 여부 {s6['phase_is_identity']}")
    A(f"decode_bs 최댓값 {s6['decode_bs_max']} · >15 이벤트 {s6['n_decode_bs_gt15']}건  "
      f"(★승격안 스코프의 '1-15'는 거짓)")
    gq = s6["gap_quantisation"]
    A(f"간극 양자화(진단): {gq['n_boundaries_on_grid']}/{gq['n_boundaries']} "
      f"= {gq['frac_on_grid']*100:.2f}% 가 {gq['quantum_ms']*1000:.3f} us 의 정확한 배수")
    A(f"아티팩트당 stream_key 정확히 2개: {s6['all_artifacts_have_exactly_two_stream_keys']} "
      f"-- {s6['green_to_green_note']}")

    s2 = rep["sentence2_ordering"]
    A("")
    A("-- 문장 2 · 순서관계 (strict 층, 비식별 논증) " + "-" * 55)
    A(f"{'class':20s} {'n':>7s} {'pooled med(ms)':>15s}   설명")
    for k in ("NS_to_PART_bsup", "SW_to_PART", "SW_to_FULL"):
        A(f"{k:20s} {s2['n'][k]:7d} {_f(s2['pooled_median_ms'][k], 15, 4)}   {CLASS_DESC[k]}")
    for tgt in ("SW_to_PART", "SW_to_FULL"):
        d = s2[f"NS_gt_{tgt}"]
        A(f"  NS_to_PART_bsup > {tgt:11s}: {d['n_artifacts_holding']}/"
          f"{d['n_artifacts_evaluated']} 아티팩트에서 성립 · paired delta med "
          f"{_f(d['paired_delta_ms_median'],7,4)} ms  CI95(rep-boot) "
          f"[{d['paired_delta_ms_ci95_repboot'][0]:.4f}, {d['paired_delta_ms_ci95_repboot'][1]:.4f}]")
    A(f"  {'file':44s} {'NS->P|bs+':>10s} {'SW->PART':>9s} {'SW->FULL':>9s}   (아티팩트별 중앙값 ms)")
    pa = s2["per_artifact_median_ms"]
    for i, r in enumerate(rep["per_artifact"]):
        A(f"  {r['file'][:44]:44s} {_f(pa['NS_to_PART_bsup'][i],10,4)} "
          f"{_f(pa['SW_to_PART'][i],9,4)} {_f(pa['SW_to_FULL'][i],9,4)}")
    A(f"  [DO_NOT_CITE] 풀링 '전환' 중앙값 = "
      f"{_f(s2['switch_pooled_median_ms__DO_NOT_CITE'],7,4)} ms (n={s2['switch_pooled_n']}) "
      f"-- {s2['why_do_not_cite']}")

    s3 = rep["sentence3_additive_upper_bound"]
    A("")
    A("-- 문장 3 · 가법 분해 (★가법성 가정 하의 상한) " + "-" * 52)
    A(f"config {s3['config']}")
    A(f"정상상태 n={s3['n_steady']} · 중앙값 FULL {_f(s3['steady_median_ms']['FULL'],7,4)} "
      f"/ PART {_f(s3['steady_median_ms']['PART'],7,4)} ms")
    A(f"{'class':20s} {'n_in':>7s} {'n_used':>7s} {'dropped':>8s} {'cells':>6s} {'matched inc(ms)':>16s}")
    for k in ("SW_to_PART", "SW_to_FULL", "NS_to_PART_bsup", "NS_bsdown"):
        c = s3["cells"][k]
        A(f"{k:20s} {c['n_in']:7d} {c['n_used']:7d} {c['n_dropped_unmatched']:8d} "
          f"{c['n_cells']:6d} {_f(c['matched_increment_ms'],16,4)}")
    A(f"d + 2s = {s3['d_plus_2s_ms']:.5f} ms  =>  s <= {s3['s_upper_ms']:.5f} ms · "
      f"d <= {s3['d_upper_ms']:.5f} ms")
    A(f"  아티팩트별 s 상한 중앙 {_f(s3['s_upper_ms_artifact_median'],8,5)} ms · CI95(rep-boot) "
      f"[{s3['s_upper_ms_ci95_repboot'][0]:.5f}, {s3['s_upper_ms_ci95_repboot'][1]:.5f}] "
      f"· 아티팩트 범위 [{s3['s_upper_ms_per_artifact_min']:.5f}, "
      f"{s3['s_upper_ms_per_artifact_max']:.5f}] · 음수 아티팩트 "
      f"{s3['n_artifacts_with_negative_s_upper']}/20")
    q = s3["quantisation_diagnostic"]
    A(f"  진단 · 1.024 us 양자 격자: d+2s = {q['d_plus_2s_in_quanta']:.1f} 양자 "
      f"(정확한 배수 {q['d_plus_2s_is_exact_multiple']}) · 셀 증분(양자) "
      + ", ".join(f"{k}={v:.1f}" for k, v in q["cell_increment_in_quanta"].items()))
    A(f"  {s3['assumption']}")

    A("")
    A("-- 문장 4 · 생애주기 (문장 3과 같은 증분, 재계산 없음) " + "-" * 44)
    for r in rep["sentence4_lifecycle"]["rows"]:
        A(f"  {r['class']:20s} n={r['n_used']:6d}  {_f(r['matched_increment_ms'],8,4)} ms   {r['desc']}")

    s5 = rep["sentence5_residency"]
    A("")
    A("-- 문장 5 · residency " + "-" * 78)
    A(f"{'file':44s} {'FULL':>8s} {'PART':>8s} {'ratio':>6s} {'bsmat':>6s} "
      f"{'fracP':>6s} {'sw/step':>8s} {'exc/step':>9s} {'sw*s/step':>10s} {'x':>8s}")
    for p in s5["per_artifact"]:
        A(f"{p['file'][:44]:44s} {_f(p['med_dur_FULL_ms'],8,3)} {_f(p['med_dur_PART_ms'],8,3)} "
          f"{_f(p['ratio'],6,3)} {_f(p['bs_matched_ratio'],6,3)} {_f(p['frac_land_PART'],6,3)} "
          f"{_f(p['switch_per_step'],8,5)} {_f(p['residency_excess_ms_per_step'],9,4)} "
          f"{_f(p['switch_attributed_ms_per_step'],10,6)} {_f(p['ratio_residency_over_switch'],8,0)}")
    for g, d in s5["by_model_cluster"].items():
        A(f"  [{g}] n={d['n_artifacts']} FULL {d['med_dur_FULL_ms_range'][0]:.3f}-"
          f"{d['med_dur_FULL_ms_range'][1]:.3f} · PART {d['med_dur_PART_ms_range'][0]:.3f}-"
          f"{d['med_dur_PART_ms_range'][1]:.3f} · ratio {d['ratio_range'][0]:.3f}-"
          f"{d['ratio_range'][1]:.3f} · bs매칭비 {d['bs_matched_ratio_range'][0]:.3f}-"
          f"{d['bs_matched_ratio_range'][1]:.3f} · 배수 {d['ratio_residency_over_switch_range'][0]:.0f}-"
          f"{d['ratio_residency_over_switch_range'][1]:.0f}")
    A(f"  {s5['model_cluster_caveat']}")
    A(f"  [DO_NOT_CITE] 풀링 FULL {_f(s5['pooled_med_of_artifact_medians_FULL_ms__DO_NOT_CITE'],7,3)} "
      f"/ PART {_f(s5['pooled_med_of_artifact_medians_PART_ms__DO_NOT_CITE'],7,3)} ms · 비 "
      f"{_f(s5['pooled_ratio_of_medians__DO_NOT_CITE'],6,3)} · bs매칭 풀링비 "
      f"{_f(s5['pooled_bs_matched_ratio_of_medians__DO_NOT_CITE'],6,3)}")
    A(f"      {s5['why_do_not_cite']}")
    A(f"  진단 other_stream_fw_ms 평균: PART "
      f"{s5['mean_other_stream_fw_ms_PART_range'][0]:.2f}-{s5['mean_other_stream_fw_ms_PART_range'][1]:.2f}"
      f" ms vs FULL {s5['mean_other_stream_fw_ms_FULL_range'][0]:.2f}-"
      f"{s5['mean_other_stream_fw_ms_FULL_range'][1]:.2f} ms")
    A(f"  공통 분모 비교: 배수 범위 {s5['ratio_residency_over_switch_range'][0]:.0f}-"
      f"{s5['ratio_residency_over_switch_range'][1]:.0f} (s_upper={s5['s_upper_used_ms']:.5f} ms 사용)")
    A(f"  {s5['common_denominator_definition']}")
    A(f"  {s5['caveat']}")

    sn = rep["sensitivity"]
    A("")
    A("-- ★민감도 (차단 B1: s 상한이 설계 선택에 얼마나 흔들리는가) " + "-" * 37)
    A(f"{'axis':18s} {'value':22s} {'n_steady':>9s} {'SW->P':>8s} {'SW->F':>8s} "
      f"{'NS->P|bs+':>10s} {'NS|bs-':>8s} {'d+2s':>10s} {'s_upper':>10s} {'%vs base':>9s}")
    for t in sn["table"]:
        A(f"{t['axis']:18s} {t['value'][:22]:22s} {t['n_steady']:9d} "
          f"{_f(t['SW_to_PART_ms'],8,4)} {_f(t['SW_to_FULL_ms'],8,4)} "
          f"{_f(t['NS_to_PART_bsup_ms'],10,4)} {_f(t['NS_bsdown_ms'],8,4)} "
          f"{_f(t['d_plus_2s_ms'],10,6)} {_f(t['s_upper_ms'],10,6)} {_f(t['pct_vs_base'],9,1)}")
    for tag, key in (("중앙값 추정량 고정", "band_median_estimator_only"),
                     ("전 행(절사평균 포함)", "band_all_rows")):
        b = sn[key]
        A(f"  밴드[{tag}] n={b['n_rows']} · s_upper [{b['min_ms']:.6f}, {b['max_ms']:.6f}] ms "
          f"· 절대폭 {b['absolute_span_ms']:.6f} ms · base 절댓값 대비 "
          f"{b['span_pct_of_base_abs']:.1f}% · 0 이하 값 존재 {b['any_non_positive']}")
    A(f"  {sn['families_note']}")
    A(f"  {sn['reading']}")

    A("")
    A("-- 인용 금지 " + "-" * 86)
    for c in rep["citation_stops"]:
        A(f"  * {c}")
    A(f"  JSON 안의 DO_NOT_CITE 필드 {len(rep['do_not_cite'])}개: {rep['do_not_cite']}")
    return "\n".join(L)


# ==========================================================================
# 자기검사 + 변이 테스트 (교훈 #53)
# ==========================================================================

R1 = "R1 role_map assigns PART to the stream with the LONGER median decode_dur_ms"
R2 = "R2 residency reports the partitioned stream as the slower one (ratio > 1)"
R3 = "R3 sentence-2 landing labels follow the role map (SW_to_PART / SW_to_FULL counts)"
M1 = "M1 matched increment removes a decode_bs composition confound"
M2 = "M2 matching drops cells whose steady reference count is below min_n"
M3 = "M3 bs-standardised residency differs from the raw ratio under skewed bs composition"
H1 = "H1 dropped_spans>0 artefact is excluded (probe FIFO, the exposed failure mode)"
H2 = "H2 n_errors / n_invalid_gaps / n_inconsistent_gaps / timeline_over_host gate too"
H3 = "H3 an artefact with no holb_summary is excluded (fail-closed)"
H4 = "H4 exclusions are reported with file and reason, not silently discarded"
H5 = "H5 aggregates are computed from admitted artefacts only"
S1 = "S1 both first-event conventions are reported and differ by one boundary per artefact"
S2 = "S2 pooled mixture values carry a __DO_NOT_CITE key and appear in the registry"
S3 = "S3 the report records the sha256 of this file as it is on disk"

_FIX = []


def _rec(gap, sw_key, bs, pbs, role_dur, gc=GAP_STRICT, nof=0):
    """검사용 경계 레코드(합성).  matched_increment 는 dict 만 본다."""
    return {"gap": gap, "dur": role_dur, "sw": False, "gc": gc, "nof": nof, "ofw": 0.0,
            "bs": bs, "pbs": pbs, "role": sw_key, "file": "fx", "phase": "measure"}


def _write_fixture(path, specs, summary):
    """specs = [(gap_class, gap_ms, stream_key, decode_bs, decode_dur_ms, n_other_fw)]"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "holb_open", "phase": "startup"}) + "\n")
        for i, (gc, gap, key, bs, dur, nof) in enumerate(specs):
            fh.write(json.dumps({
                "event": "holb_gap", "phase": "measure", "gap_ms": gap, "gap_class": gc,
                "n_other_fw": nof, "other_stream_fw_ms": 0.0, "stream_key": key,
                "decode_bs": bs, "decode_fw_bs": bs, "decode_dur_ms": dur,
                "dropped_events": 0, "pend_min_in_gap": 1 if gc == GAP_STRICT else 0,
                "seq": i}) + "\n")
        if summary is not None:
            s = {"event": "holb_summary", "phase": "running", "dropped_spans": 0,
                 "n_errors": 0, "n_invalid_gaps": 0, "n_inconsistent_gaps": 0,
                 "timeline_over_host": 1.0002, "dropped_events": 0}
            s.update(summary)
            fh.write(json.dumps(s) + "\n")
    return path


def _fixtures():
    """합성 아티팩트.  GPU 0, 실제 캠페인 파일을 건드리지 않는다.

    스트림 "S"(decode_dur 10 ms) = 느림 = PART · 스트림 "f"(5 ms) = 빠름 = FULL.
    수열은 **셀 개수가 비대칭**이 되도록 짰다(SW_to_PART 5 vs SW_to_FULL 4) —
    착지 스트림 식별을 뒤집는 변이가 셀 개수 검사에서 반드시 드러나게 하기 위해서다.
    """
    if _FIX:
        return _FIX[0]
    td = tempfile.mkdtemp(prefix="promo_fix_")
    S, F = ("S", 10.0), ("f", 5.0)

    def ev(gap, who, bs, gc=GAP_STRICT):
        return (gc, gap, who[0], bs, who[1], 0)

    specs = [("first", None, "f", 1, 5.0, 0)]
    for _ in range(12):                       # 정상상태 (FULL, bs=1) x12
        specs.append(ev(0.30, F, 1))
    specs.append(ev(1.40, S, 1))              # SW_to_PART #1
    for _ in range(12):                       # 정상상태 (PART, bs=1) x12
        specs.append(ev(0.50, S, 1))
    for _ in range(6):
        specs.append(ev(1.80, S, 2))          # NS_to_PART_bsup x6 (bs 1->2)
        specs.append(ev(0.60, S, 2))          # 정상상태 (PART, bs=2)
        specs.append(ev(0.60, S, 2))          # 정상상태 (PART, bs=2)
        specs.append(ev(0.50, S, 1))          # NS_bsdown x6 (bs 2->1)
    for _ in range(4):                        # 교대 꼬리: SW_to_FULL x4, SW_to_PART x4
        specs.append(ev(0.90, F, 1))
        specs.append(ev(1.40, S, 1))

    fx = {"dir": td}
    fx["ok"] = _write_fixture(os.path.join(td, "g2_fx_ok.holb.jsonl"), specs, {})
    fx["ds"] = _write_fixture(os.path.join(td, "g2_fx_dropspans.holb.jsonl"), specs,
                              {"dropped_spans": 4})
    fx["err"] = _write_fixture(os.path.join(td, "g2_fx_unhealthy.holb.jsonl"), specs,
                               {"n_errors": 2, "n_invalid_gaps": 1,
                                "n_inconsistent_gaps": 3, "timeline_over_host": 1.4})
    fx["nosum"] = _write_fixture(os.path.join(td, "g2_fx_nosummary.holb.jsonl"), specs, None)
    _FIX.append(fx)
    return fx


def _checks(ns, fx):
    """R*/M*/H*/S* 배터리.  검사 본체는 언제나 무변이 파일에서 온다."""
    analyze_ = ns["analyze"]
    render_ = ns["render"]
    role_map_ = ns["role_map"]
    read_ = ns["read_artifact"]
    minc = ns["matched_increment"]
    bsstd = ns["bs_standardized_pair"]
    res = []

    def ck(name, fn):
        try:
            good = bool(fn())
        except Exception as e:  # noqa: BLE001
            res.append((name, "raised", repr(e)))
            return
        res.append((name, "ok" if good else "fail", ""))

    # ---- 착지 스트림 식별 --------------------------------------------------
    def r1():
        gaps, _ = read_(fx["ok"])
        roles, meds = role_map_(gaps)
        return roles.get("S") == "PART" and roles.get("f") == "FULL" and meds["S"] > meds["f"]

    def r2():
        rep = analyze_([fx["ok"]])
        p = rep["sentence5_residency"]["per_artifact"][0]
        return (p["med_dur_PART_ms"] == 10.0 and p["med_dur_FULL_ms"] == 5.0
                and abs(p["ratio"] - 2.0) < 1e-9)

    def r3():
        n = analyze_([fx["ok"]])["sentence2_ordering"]["n"]
        return (n["SW_to_PART"] == 5 and n["SW_to_FULL"] == 4
                and n["NS_to_PART_bsup"] == 6)

    # ---- 매칭 (단위 검사: 착지 라벨·건강도와 독립) --------------------------
    def m1():
        # target: bs1 x2 @1.0, bs2 x8 @2.0 ; steady: bs1 x10 @0.5, bs2 x10 @1.5
        tgt = ([_rec(1.0, "PART", 1, 1, 0.0)] * 2) + ([_rec(2.0, "PART", 2, 2, 0.0)] * 8)
        std = ([_rec(0.5, "PART", 1, 1, 0.0)] * 10) + ([_rec(1.5, "PART", 2, 2, 0.0)] * 10)
        keyfn = lambda b: (b["role"], b["bs"])  # noqa: E731
        got = minc(tgt, std, keyfn, 5, med)
        # 매칭하면 셀 안에서 항상 +0.5 ; 매칭을 없애면 풀링 기준선 1.0 -> 중앙값 +1.0
        return abs(got["value"] - 0.5) < 1e-9 and got["n_used"] == 10

    def m2():
        tgt = ([_rec(1.0, "PART", 1, 1, 0.0)] * 10) + ([_rec(9.0, "PART", 3, 3, 0.0)] * 3)
        std = ([_rec(0.5, "PART", 1, 1, 0.0)] * 10) + ([_rec(0.5, "PART", 3, 3, 0.0)] * 3)
        keyfn = lambda b: (b["role"], b["bs"])  # noqa: E731
        got = minc(tgt, std, keyfn, 5, med)
        return (got["n_in"] == 13 and got["n_used"] == 10
                and got["n_dropped_unmatched"] == 3 and got["n_cells"] == 1)

    def m3():
        part = [(1, 10.0)] * 10 + [(2, 20.0)] * 10
        full = [(1, 5.0)] * 10 + [(2, 10.0)] * 90
        sp, sf, ncell, w = bsstd(part, full, 5, med)
        raw = med([v for _, v in part]) / med([v for _, v in full])
        return (ncell == 2 and abs(sp / sf - 2.0) < 1e-9 and abs(raw - 1.5) < 1e-9)

    # ---- 건강도 -----------------------------------------------------------
    def h1():
        rep = analyze_([fx["ok"], fx["ds"]])
        ex = {x["file"]: "; ".join(x["reasons"]) for x in rep["inputs"]["excluded"]}
        return (rep["inputs"]["admitted"] == ["g2_fx_ok.holb.jsonl"]
                and set(ex) == {"g2_fx_dropspans.holb.jsonl"}
                and "dropped_spans=4" in ex["g2_fx_dropspans.holb.jsonl"])

    def h2():
        rep = analyze_([fx["ok"], fx["err"]])
        ex = {x["file"]: "; ".join(x["reasons"]) for x in rep["inputs"]["excluded"]}
        r = ex.get("g2_fx_unhealthy.holb.jsonl", "")
        return (rep["inputs"]["admitted"] == ["g2_fx_ok.holb.jsonl"]
                and all(k in r for k in ("n_errors=2", "n_invalid_gaps=1",
                                         "n_inconsistent_gaps=3", "timeline_over_host=")))

    def h3():
        rep = analyze_([fx["ok"], fx["nosum"]])
        ex = {x["file"]: "; ".join(x["reasons"]) for x in rep["inputs"]["excluded"]}
        return (rep["inputs"]["admitted"] == ["g2_fx_ok.holb.jsonl"]
                and "holb_summary" in ex.get("g2_fx_nosummary.holb.jsonl", ""))

    def h4():
        txt = render_(analyze_([fx["ok"], fx["ds"], fx["nosum"]]))
        return ("g2_fx_dropspans.holb.jsonl" in txt and "g2_fx_nosummary.holb.jsonl" in txt
                and "EXCLUDED" in txt and "dropped_spans=4" in txt)

    def h5():
        one = analyze_([fx["ok"]])["scope_and_sentence6"]["n_boundary_seeded"]
        two = analyze_([fx["ok"], fx["ds"]])
        return (two["inputs"]["n_artifacts_admitted"] == 1
                and two["scope_and_sentence6"]["n_boundary_seeded"] == one
                and len(two["sentence5_residency"]["per_artifact"]) == 1)

    # ---- 구조 (모든 변이에서 통과해야 하는 속성) ---------------------------
    def s1():
        rep = analyze_([fx["ok"], fx["nosum"]])
        s6 = rep["scope_and_sentence6"]
        return (s6["n_boundary_seeded"] - s6["n_boundary_unseeded"]
                == rep["inputs"]["n_artifacts_admitted"]
                and s6["boundary_convention_delta_equals_n_artifacts"] is True)

    def s2():
        rep = analyze_([fx["ok"]])
        reg = set(rep["do_not_cite"])
        need = {"sentence2_ordering.switch_pooled_median_ms__DO_NOT_CITE",
                "sentence5_residency.pooled_ratio_of_medians__DO_NOT_CITE",
                "sentence5_residency.pooled_bs_matched_ratio_of_medians__DO_NOT_CITE"}
        return need <= reg and len(reg) >= 6

    def s3():
        rep = analyze_([fx["ok"]])
        h = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
        return rep["self_sha256"] == h and len(h) == 64

    for name, fn in ((R1, r1), (R2, r2), (R3, r3), (M1, m1), (M2, m2), (M3, m3),
                     (H1, h1), (H2, h2), (H3, h3), (H4, h4), (H5, h5),
                     (S1, s1), (S2, s2), (S3, s3)):
        ck(name, fn)
    return res


# -- 변이본: 각 수리를 **되돌린** 소스 --------------------------------------
_MUT_ROLE = '''
def role_map(gaps):
    # 변이: 착지 스트림 식별을 뒤집는다 (짧은 쪽을 PART 로 부른다).
    dur = {}
    for d in gaps:
        v = d.get("decode_dur_ms")
        if v is None:
            continue
        dur.setdefault(d.get("stream_key"), []).append(float(v))
    meds = {k: med(v) for k, v in dur.items() if v}
    if not meds:
        return {}, {}
    fast = min(meds, key=lambda k: meds[k])
    roles = {k: ("PART" if k == fast else "FULL") for k in meds}
    if len(meds) == 1:
        roles = {k: "FULL" for k in meds}
    return roles, meds
'''

_MUT_MATCH = '''
def matched_increment(target, steady, keyfn, min_n=MIN_MATCH_N, central=med):
    # 변이: 매칭을 없앤다 -- 셀을 무시하고 하나의 풀링 기준선을 쓴다.
    base = central([b["gap"] for b in steady]) if steady else float("nan")
    inc = [b["gap"] - base for b in target]
    return {"n_in": len(target), "n_used": len(inc), "n_dropped_unmatched": 0,
            "n_cells": 1, "value": central(inc), "increments": inc}


def bs_standardized_pair(part_pairs, full_pairs, min_n=MIN_MATCH_N, central=med):
    # 변이: decode_bs 구성을 맞추지 않는다 -- 그냥 원 중앙값.
    p = [v for _, v in part_pairs]
    f = [v for _, v in full_pairs]
    return central(p), central(f), 0, len(p) + len(f)
'''

_MUT_HEALTH = '''
def health_verdict(h):
    # 변이: 건강도 게이트를 무력화한다 (텔레메트리 큐 카운터만 본다).
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
    "A_flip_landing_stream_identification": (_mut("repair-role", _MUT_ROLE), {R1, R2, R3}),
    "B_remove_matching":                    (_mut("repair-match", _MUT_MATCH), {M1, M2, M3}),
    "C_disable_health_gate":                (_mut("repair-health", _MUT_HEALTH),
                                             {H1, H2, H3, H4, H5}),
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
                    help=f"GATE2 아래 glob (기본: {DEFAULT_PATTERN})")
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

    pats = a.patterns or [DEFAULT_PATTERN]
    files = sorted(f for p in pats for f in glob.glob(os.path.join(a.gate2, p)))
    if not files:
        print("no artifacts matched")
        return 2
    rep = analyze(files)
    print(render(rep))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
