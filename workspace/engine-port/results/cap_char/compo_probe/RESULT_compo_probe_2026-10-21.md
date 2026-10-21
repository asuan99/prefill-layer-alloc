# RESULT — 조성 변형 부팅 프로브 (Workspace bring-up, 2026-10-21, run `compo_probe_20261021T131621Z`)

> **지위: 비측정 부팅 프로브. 아래 모든 수치(가중치 GiB·KV 풀 토큰·슬롯·요청 시간·tel 레코드 수)는 인용 불가 프로브 값이다.** 성능 판정 0건. dummy 가중치 출력은 무의미하며(logprob이 상수에 가깝다) 정확성은 원본 실가중치 boot(real_orig)에서만 본다. 결정 24·26 근거(`results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`).
> 커밋 `170706171c202caf82db432f1a2491fb880c6fd4`(HEAD, dirty=False) · 엔진 ENGINE_DIGEST `99bbef91fe680b4d4b8a8fd0a08b8cc65426518c99baca0b3e18e4fa627f4325`(병합 후 기대값과 일치, `engine_digest.py digest --manifest`) · sync manifest sha256 `bbcac12a…d03b`(30 entries) · manual edits applied.
> 기판: pdmux-probe Workspace(A100-SXM4-80GB, UUID `GPU-7b1940f0-c94c-cbbc-6ecf-382c8c7da692`, 드라이버 580.178.04, host `main-wsp-7ix1cg4xl1jd-0`). 클럭 고정 불가(`meta/clk.csv` 기록).
> 설정: ctx 73728, mem-fraction 0.6, radix OFF, cap 128, chunk -1, cudagraph(decode) ON, PD-mux agnostic(`pdmux_agnostic_manual.yml`), bf16, flashinfer, seed 1.
> 원자료: `compo_probe_20261021T131621Z/`(SHA256SUMS 186개 전수 OK — 로컬 사본 `_vessl_fetch/compo_probe/`에서 검증). 10 MB 초과 6개(tel.jsonl 2, tokenizer.json 4)는 저장소 밖 `~/Experiments/KISTI/_vessl_fetch/compo_probe/compo_probe_20261021T131621Z/`, sha256은 `LARGE_FILES_NOT_IN_REPO.sha256`. `cache/`(JIT)는 제외.
> 실행 시간: 13:16:21–13:26:39Z(약 10.3분; 예산 96분 대비 짧다 — 부팅이 boot당 ≈1–2분). 전 단계 SUMMARY.md/summary.json 생성으로 완료 판정.

## 편차·재실행 (정직 기록)
- **시도 0(`compo_probe_20261021T131118Z`, 저장소의 `attempt0_FAILED_libnuma_*/`)**: 7개 boot 전부 FAIL(서버가 /health 전 사망). 원인 = `sgl_kernel` import 실패, `libnuma.so.1` 부재(이미지에 없음; 2026-10-04·10-20 프로브와 같은 편차). GPU 작업은 하지 않았다(TD는 PASS).
- `apt-get install libnuma1`(먼저 `apt-get update` 필요) 후 새 run으로 재실행했다. 그 사이 실수로 잘못 띄운 run `…T131543Z`는 설치 전이라 즉시 중단했고 버렸다(결과 없음). 시도 1이 본 결과다. 편차 기록: `COMPO_PROBE_DEVIATION.txt`.
- 코드 전달: 로컬 `git bundle HEAD` → `/data/r.bundle`(sha256 일치) → clone → `/opt/pdmux/prefill-layer-alloc` → `install_runtime.sh`.

## 단계 결과 (총 31개 stage 중 PASS 26 / FAIL 0 / SKIPPED 5 = a31_retry 5개, 필요 없음)
| 단계 | real_orig | a4 | a13 | a22 | a31 | a4_vanilla |
|---|---|---|---|---|---|---|
| BOOT | PASS | PASS | PASS | PASS | **PASS** (retry 불필요) | PASS |
| REQ(8 plain + logprob 1) | PASS | PASS | PASS | PASS | PASS | PASS |
| TEL | (OFF) | (OFF) | **PASS** | (OFF) | **PASS** | (OFF) |
| PRED | PASS | PASS | PASS | PASS | PASS | PASS |
| TD(teardown 후 0 MiB) | PASS | PASS | PASS | PASS | PASS | PASS |
PRE·VARIANTS: PASS. 요약상 boot 6 + TEL 2 + PRED 6 + REQ 6 모두 PASS, UNRESOLVED 0.

## 배너 대 variant.json [유도] 예측 (허용 ±2 %)
| boot | n_attn/n_mamba | 가중치 GiB 예측→배너 (오차) | k B/토큰 예측→배너 (오차) | s B/슬롯 예측→배너 |
|---|---|---|---|---|
| real_orig / a4 | 4/27 → 4/27 | 16.556→16.580 (+0.147 %) | 16384→16394 (+0.062 %) | 143,548,416→143,581,756 (+0.023 %) |
| a13 | 13/18 → 13/18 | 15.007→15.040 (+0.222 %) | 53248→53237 (−0.021 %) | 95,698,944→95,721,170 (+0.023 %) |
| a22 | 22/9 → 22/9 | 13.458→13.490 (+0.239 %) | 90112→90120 (+0.009 %) | 47,849,472→47,860,585 (+0.023 %) |
| a31 | 31/0 → 31/0 | 11.909→11.940 (+0.261 %) | 126976→126946 (−0.024 %) | 0→0 |
| a4_vanilla(PD-mux off) | 4/27 → 4/27 | 16.556→16.580 | 16384→16385 | 143,548,416→143,581,756 |
KV 풀(프로브 값): a4 886,804 / a13 419,117 / a22 334,083 / a31 298,407 토큰. mamba 슬롯 128(cap). cudagraph bs 최대 128, PD-mux stream group 4(vanilla는 없음).

## a31 (Mamba 0층)
- `--max-mamba-cache-size 128` 포함 원 argv 그대로 **부팅 성공**(`dummy_a31/launch_command.txt`). retry 경로(플래그 제거) 불필요. 배너의 s/슬롯 = 0, k = 126,946 B/토큰(예측 126,976). 8/8 요청 OK.

## dummy NaN 검사
- 모든 boot에서 logprob 요청 1건의 64개 토큰 logprob이 전부 유한(PRED·REQ 판정), `nan_logits`·`device_assert`·`oom`·`cuda_graph_capture_failed`·`eager_fallback` 시그니처 0(`status/BOOT-*.json`). 단 dummy 출력은 상수에 가까운 logprob(a31 예: −11.78 반복)이라 정확성 정보는 없다.

## TEL 단계 (결정 26)
| boot | span 쌍 | extend_tokens 8192 span_seq | decode_iteration | 공존 decode iter | 로그 실패 줄 |
|---|---|---|---|---|---|
| dummy_a13 | 9 | [9] | 1301 | 43 | 0 |
| dummy_a31 | 9 | [9] | 1269 | 34 | 0 |
- 조성 변형(혼합·Mamba 0층)에서도 `prefill_span_start/_end` 쌍이 같은 span_seq로 나오고 8192 토큰 prefill span이 decode 8개 공존 중에 기록된다(bs 최대 9). 위상 이벤트 코드 경로가 dummy 변형에서 막히지 않는다(존재 검사만; 값의 의미·정확성 판정 아님).

## J-H1·J-H2 설계에 영향을 주는 사실
- 변형 a13/a22/a31이 현 엔진(PD-mux agnostic, 운영점 cudagraph ON, chunk -1)에서 dummy 가중치로 부팅·요청 처리된다. 조성(s·k)은 배너에서 예측과 ±0.3 % 내로 재현된다 → 조성 축을 부팅 레벨 변수로 쓸 수 있다.
- T3 σ 이벤트(span 쌍)가 a13·a31에서 나온다 → J-H2의 T3 관측 경로 성립.
- KV 풀 크기가 조성에 따라 크게 달라진다(a4 887k → a31 298k 토큰, 같은 mem-fraction 0.6). 조성 비교 시 용량(C_mem)이 함께 바뀌므로 설계에서 별도 통제/보고 대상이다.
- 주의: 이 프로브는 요청 2048토큰·단일 조건이다. 장문(ctx 73728 근처)이나 부하 하의 부팅 안정성은 확인하지 않았다. dummy 가중치라 조성 효과의 성능·정확성 주장은 불가.

## Workspace
- `pdmux-probe` 회수 직후 `vesslctl workspace pause` → `workspace list`로 **standby** 확인(2026-10-21). `pdmux-build`도 standby.

이 Job이 채우는 COSTMODEL 칸: 미지정(결정 24 선행 부팅 프로브, 칸 채움 없음).
