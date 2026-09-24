#!/usr/bin/env python3
"""기술 집계 · 판정 아님 (DESCRIPTIVE ONLY -- NOT A VERDICT).  GPU 지출 0.

질문 하나: **prefill이 실제로 층 단위로 쪼개져 발사되는가?**

`prefill_chunk_progress`(= `ScheduleBatch.split_index`, `dual_worker.py:571-573`)는
`forward_split_prefill`이 한 번 돌 때마다 `min(split_index + forward_count,
num_hidden_layers)`로 전진한다(`model_runner.py:2717-2727`).  그리고

    forward_count = max(1, split_forward_token_budget // extend_num_tokens)
                                   (`multiplexing_mixin.py` 4곳, 값은 config)

이므로 `extend_num_tokens <= budget / num_hidden_layers` 인 요청은 **한 번의 호출로
전 층을 돈다** = 층 단위 분할이 발화하지 않는다.  이 스크립트는 그것을 아카이브에서
직접 센다: prefill-active 스냅샷의 `split_index` 히스토그램.

★양성 대조가 내장돼 있다(교훈 232): 계측기가 중간 상태를 **해상할 수 있음**을 같은
집계 안에서 보여야 한다.  긴 컨텍스트 캠페인(`longctx_conflict`)이 4-층 사다리를
내면 계측기는 살아 있는 것이고, 짧은 컨텍스트 캠페인의 terminal-only는 계측 실패가
아니라 엔진 동작이다.

★한계: 스냅샷 격자 위에서만 본다.  `trace_forced` 열(PDMUX_TRACE_FORCE_PREFILL=1)이
켜진 캠페인은 prefill-in-flight 동기화마다 강제 방출되므로 격자 편향이 작다 --
그 캠페인의 결과를 우선한다.
"""
import argparse, glob, json, os, sys
from collections import Counter

DEFAULT_PATTERNS = [
    "s2_sticky/*_telemetry.jsonl",
    "sticky_smoke/*_telemetry.jsonl",
    "e1_traceforce/*.jsonl",
    "longctx_conflict/**/*telemetry*.jsonl",
    "r2_eval/**/*.jsonl",
]


def census(root, pattern):
    hist, forced, n, nfiles = Counter(), Counter(), 0, 0
    for p in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
        if os.path.getsize(p) < 1024:
            continue
        nfiles += 1
        with open(p, errors="replace") as fh:
            for line in fh:
                if '"runtime_snapshot"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("phase") == "startup":
                    continue
                if (rec.get("prefill_active_batch_size") or 0) <= 0:
                    continue
                n += 1
                hist[rec.get("prefill_chunk_progress")] += 1
                forced[rec.get("trace_forced")] += 1
    return {"files": nfiles, "prefill_active_snapshots": n,
            "trace_forced": {str(k): v for k, v in forced.items()},
            "split_index_hist": {str(k): v for k, v in
                                 sorted(hist.items(), key=lambda kv: (kv[0] is None, kv[0]))}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "workspace", "engine-port", "results")))
    ap.add_argument("--pattern", action="append", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    out = {}
    for pat in (args.pattern or DEFAULT_PATTERNS):
        r = census(args.results_root, pat)
        out[pat] = r
        if not r["prefill_active_snapshots"]:
            print(f"### {pat}: prefill-active 0"); continue
        n = r["prefill_active_snapshots"]
        term = max((int(k) for k in r["split_index_hist"] if k != "None"), default=None)
        print(f"### {pat}  files={r['files']}  prefill-active={n}  "
              f"trace_forced={r['trace_forced']}")
        for k, v in r["split_index_hist"].items():
            mark = "  <- terminal (= num_hidden_layers)" if k == str(term) else ""
            print(f"    split_index={k:>5}: {v:7d} ({100.0*v/n:6.2f}%){mark}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
