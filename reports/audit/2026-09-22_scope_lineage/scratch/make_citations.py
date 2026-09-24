#!/usr/bin/env python3
"""기술 집계 · 판정 아님.  citations.json 생성기.

`scripts/discipline/check_line_citations.py`의 `resolve`/`read_range`/
`fingerprint`를 그대로 import 해서(재구현 금지) 이번 감사가 인용한
`path:line-range`마다 {anchor, anchor_offset, sha, target}를 만든다.
GPU 지출 0.  읽기 전용.
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DISC = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..",
                                     "workspace", "engine-port", "scripts", "discipline"))
sys.path.insert(0, DISC)
import check_line_citations as CLC  # noqa: E402

DOC = "reports/audit/2026-09-22_scope_lineage/REPORT.md"

CITES = [
    # --- sticky semantics (설치 트리 == src 오버레이, byte-identical) ---
    ("src/multiplex/multiplexing_mixin.py", 388, 400),    # _init_sticky_partition: 기본값 OFF
    ("src/multiplex/multiplexing_mixin.py", 401, 428),    # 상호배제 거부 조건 4건 + 최소 분할 수
    ("src/multiplex/multiplexing_mixin.py", 430, 455),    # _sticky_fixed_idx 결정 + ENABLED 로그
    ("src/multiplex/multiplexing_mixin.py", 1169, 1182),  # 분기 조건 disjunct + fixed idx
    ("src/multiplex/multiplexing_mixin.py", 1198, 1211),  # decode-empty -> idx 0, elif 사문화
    ("src/multiplex/multiplexing_mixin.py", 1183, 1197),  # decode_bs 기반 drift (sticky OFF 경로)
    # --- admission ---
    ("src/multiplex/multiplexing_mixin.py", 1218, 1231),  # update_split_prefill_batch
    ("src/multiplex/multiplexing_mixin.py", 2045, 2070),  # _r2_admission_holds
    ("src/multiplex/controller.py", 88, 101),             # FixedPolicy.decide
    ("src/multiplex/controller.py", 320, 329),            # overload_streak -> admission_limited
    ("src/multiplex/dual_worker.py", 87, 97),             # begin/finish_admission = 스톱워치
    ("src/multiplex/dual_worker.py", 566, 574),           # observe_scheduler: active_batch 출처
    # --- upstream pristine (engine-port 미수정) ---
    ("sglang_engine_dev/python/sglang/srt/multiplex/pdmux_context.py", 54, 66),    # get_arch_constraints
    ("sglang_engine_dev/python/sglang/srt/multiplex/pdmux_context.py", 104, 137),  # initialize_stream_groups
    # --- hybrid forward_split_prefill (D-정정) ---
    ("src/models/zamba2.py", 752, 760),
    ("src/models/nemotron_h.py", 859, 867),
    ("src/models/falcon_h1.py", 503, 511),
    ("src/models/granitemoehybrid.py", 622, 630),
    # --- 집계기 ---
    ("results/residency_census_2026-09-21/residency_census.py", 96, 100),     # is_part
    ("results/residency_census_2026-09-21/residency_census.py", 428, 440),    # M0-M4 변이 arm
    ("results/r2_eval/e2_sticky_prereg/e2_realized_mix.py", 57, 66),          # _is_decode_busy
]


def main():
    out = {DOC: {}}
    problems = []
    for cited, lo, hi in CITES:
        path, why = CLC.resolve(cited)
        if path is None:
            problems.append(f"{cited}: {why}")
            continue
        chunk, total = CLC.read_range(path, lo, hi)
        if chunk is None:
            problems.append(f"{cited}:{lo}-{hi}: out of range (file has {total} lines)")
            continue
        sha, anchor, offset = CLC.fingerprint(chunk)
        key = f"{os.path.basename(cited)}:{lo}-{hi}"
        if key in out[DOC]:
            key = f"{cited}:{lo}-{hi}"
        out[DOC][key] = {
            "anchor": anchor, "anchor_offset": offset, "sha": sha,
            "target": os.path.relpath(path, CLC.PROJECT_ROOT),
        }
    dest = os.path.join(_HERE, "..", "citations.json")
    json.dump(out, open(dest, "w"), indent=1, sort_keys=True, ensure_ascii=False)
    print(f"wrote {os.path.normpath(dest)}  entries={len(out[DOC])}")
    for p in problems:
        print("  PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
