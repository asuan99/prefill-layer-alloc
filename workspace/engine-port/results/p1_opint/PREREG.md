# P1 운영점(cudagraph-ON) 대조 — 사전등록

작성: 2026-08-05 (experiment-runner). 제출 전 고정. 결과를 본 뒤 이 문서를 바꾸지 않는다
(방법론 게이트 #8).

## 동기 (요청 원문 요약)

`triage/p1_7_bench_one.sbatch:42`의 4-모델 캠페인은 전부 `--disable-cuda-graph`(비운영점)
로 측정됐다. fused의 死因은 TPOT > 60ms SLO 초과(Granite rate4 TPOT 61.21)인데, cudagraph
가 그 벽을 없앤다(`results/cudagraph_probe/cudagraph_results.md` Probe 1: plain TPOT
62.70→13.51ms, rate4 82.41→54.04ms=SLO 통과). 운영점(cudagraph-ON)에서 P1("PD 분리가
fused를 이긴다")이 축소/소멸하는지 직접 측정한다.

## 스코프

- **모델 2종**: `Zyphra/Zamba2-2.7B`(context-length 4096, attention-backend triton),
  `ibm-granite/granite-4.0-h-micro-base`(context-length 8192, attention-backend
  flashinfer). 두 backend 조합 모두 기존 4모델 no-cudagraph 캠페인에서 boot_ok=1로
  검증된 조합을 그대로 유지(§ "재사용 조건" 참조) — 이번에 새로 검증해야 하는 것은
  cudagraph-ON 하의 동일 조합뿐이다.
- **정책 2종**: `plain`(fused, `--enable-pdmux` 미사용) vs `agnostic`(pdmux v1,
  `--enable-pdmux --pdmux-config-path pdmux_a100_smoke.yml --chunked-prefill-size -1
  --disable-overlap-schedule`, 다른 env 변수 없음 — `PDMUX_FIXED_DECODE_SM_FILE`·
  `PDMUX_DUAL_WORKER`·`PDMUX_TRUE_DUAL_WORKER` 전부 미설정). layer-aware는 포함하지
  않는다(이미 4모델 서빙 실증으로 반증됨, `p1_7_layeraware_validation.txt`).
- **cudagraph**: ON. `--disable-cuda-graph` / `--disable-piecewise-cuda-graph` 둘 다
  **사용하지 않는다**. 이것이 원 캠페인 대비 바꾸는 **유일한 변수**다.
- **서버 플래그(원 캠페인과 동일, 고정)**: `--trust-remote-code --dtype bfloat16
  --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48`.
- **워크로드**: `random-ids`, `--random-input-len 2000 --random-output-len 96
  --random-range-ratio 1.0`, 120 prompts(채점 phase), `--warmup-requests 8`(각
  bench_serving 호출 내장 warmup, phase 분리).
- **SLO**: goodput@(TTFT ≤ 3.0s ∧ **request-level TPOT ≤ 60ms**), TPOT per-request
  = mean(itls[i])(원 캠페인·`bench_serving`의 `(latency-ttft)/(output_len-1)`과 동일 정의).
  원 캠페인과 동일 정의를 그대로 재사용(비교가능성 우선, CLAUDE.md 게이트 #4의 p95-ITL
  변형은 이번 비교에는 적용하지 않음 — 이유: 원 캠페인과의 직접 대조가 목적이므로 SLO
  정의를 바꾸면 "무엇이 바뀌어 death가 사라졌는가"가 혼입된다).

## 재사용 조건 (신규 위험)

Granite(`flashinfer` backend) + `--enable-pdmux`(green-ctx) + cudagraph 조합은 기존
`cudagraph_probe`에서 검증된 적이 없다(거긴 Zamba2 + triton backend만). 이 조합의 boot·
capture 성공 여부는 **capacity scan(Phase 0) 첫 boot에서 직접 확인**하고, 실패 시
즉시 triage하여 보고한다(성능 판정 아님, boot/capture 유효성 게이트).

## Phase 0 — 용량 스캔(채점 전, metric cliff 회피 — 방법론 게이트 #6)

- 각 (모델, 정책) 쌍마다 서버 1회 boot, rate 그리드 `{1,2,3,4,5,6,8,10,12}`,
  `num-prompts=60`(가벼운 스캔), `--seed 7`(고정, 정책 간 비교 목적 아니므로 pairing
  불필요), `--output-details`.
- 기록: completed/total, TTFT p50/p95, request_throughput, goodput.
- **사전등록 절벽(cliff) 플래그(기계적, 해석 아님)** — 채점 rate {2,3,4,6} 각각에 대해:
  - **flag A(포화)**: completed/total < 1.0 (요청 실패/드롭 존재)
  - **flag B(TTFT 배증)**: 그 rate의 TTFT p50 > 2 × (그리드에서 바로 아래 rate의 TTFT p50)
  - **flag C(SLO 근접)**: goodput / request_throughput < 0.95 (도착 대비 5%+ SLO 위반)
  - 세 플래그 중 하나라도 참이면 그 (모델,정책,rate) 셀은 "절벽 위"로 표시하고, 사전등록
    판정 규칙 적용 시 그 사실을 명시(제외하지는 않되 절벽 위임을 병기).

## Phase 1 — 채점 측정 (n=5, paired, 순서 무작위화)

- rate ∈ {2, 3, 4, 6}, n = **5 반복**(CLAUDE.md 기본 n≥5 충족, 요청 n≥4 상회).
- 반복 r(1..5)마다: 두 정책(plain, agnostic)의 실행 순서를 코인플립으로 무작위화하고
  로그에 남긴다(체계적 drift·thermal 편향 방지).
- 반복 r, rate R 셀마다 `--seed = 9000 + 100*r + R`로 고정 — **동일 반복 내 plain과
  agnostic은 완전히 동일한 trace(prompt 내용 + Poisson 도착열)를 공유**(paired design).
  반복마다 R을 그대로 시드에 넣으므로 같은 rate라도 반복 간 seed가 다르다(독립 draw).
- 정책마다 fresh server boot(반복마다 재부팅) → correctness gate(아래) → warmup →
  4개 rate 순차 측정 → kill. warm-up/correctness/benchmark phase 분리.
- **correctness gate**: `/generate`로 "The capital of France is..." 프롬프트,
  temperature 0, 응답에 "Paris" 포함 여부 확인. FAIL이면 그 (모델,정책,반복) 전체를
  스킵하고 로그에 明기(무효 데이터로 채점에서 제외).
- 출력: 반복·정책·rate별 raw JSONL(`--output-details`, ttfts/itls/input_lens/
  output_lens/errors 포함) → 사후 `p1op_analyze.py`로 TTFT p50/p95/p99, TPOT p50/p95,
  goodput mean±sd(n=5), paired diff(agnostic−plain, 같은 반복 내) 계산.

## 고정 변수 (한 번에 하나만 바꾼다 원칙)

cudagraph, backend, mem-fraction-static, max-running-requests, disable-radix-cache,
workload(in/out/range-ratio/num-prompts), rate grid — 전부 원 4모델 캠페인과 동일하게
고정. **바뀌는 것은 cudagraph ON 하나뿐.** GPU clock/power는 원 캠페인도 명시적으로
고정하지 않았으므로(공유 클러스터, root 권한 없음) 이번에도 동일하게 미고정 — 새로운
confound 아님(기존 캠페인과 동일 조건).

## 사전등록 판정 규칙 (고정, 결과 확인 후 변경 금지)

- **rate ≤ 4에서 goodput 차이(agnostic vs plain) < 3%** ⇒ P1은 **운영점에서 rate>4
  한정으로 축소**
- **rate 6에서도 차이 < 3%** ⇒ P1은 **비운영점 아티팩트로 전면 강등**
- **차이가 3% 이상이고 paired 95% CI(n=5, bootstrap 또는 t 기반)가 0을 배제** ⇒
  **P1 운영점 확인**
- 절벽 위로 플래그된 셀은 판정에서 신뢰도가 낮음을 별도 명기(제외 여부는
  result-analyst 판단으로 넘긴다 — 여기서는 존재만 보고).
- 3%는 프로젝트 헤드라인 하한(방법론 게이트 #3). 이 수치들은 결과를 본 뒤 바꾸지 않는다.

## 아티팩트 레이아웃

```
workspace/engine-port/results/p1_opint/
  PREREG.md                                  (본 문서)
  pdmux_a100_smoke.yml                       (triage/에서 그대로 복사, 미수정)
  p1op_run.sbatch                            (모델당 1 job, capacity scan + n=5 본측정)
  p1op_analyze.py                            (사후 분석: percentile/goodput/판정 대조)
  runtime_source_manifest_<tag>_<jobid>.sha256  (sync_engine_tree.sh manifest)
  p1op_<tag>_<jobid>.out / .err              (SLURM 표준 출력 — 루트 아님, 커밋 금지 규약은
                                               루트 .out/.err에만 해당하나 관례상 결과 dir에 보존)
  p1op_<tag>_capscan_<policy>_<jobid>.jsonl  (Phase 0 raw)
  p1op_<tag>_<policy>_rep<r>_<jobid>.jsonl   (Phase 1 raw, 4 rate lines per file)
  p1op_<tag>_srv_<policy>_{capscan|rep<r>}_<jobid>.log  (서버 로그, capture/error evidence)
```

`triage/` 아래 기존 4-모델 아티팩트는 수정하지 않는다(읽기 전용 참조만).
