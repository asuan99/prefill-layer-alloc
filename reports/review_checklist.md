# 검수 체크리스트 — v2 종결 전 사람-검증 항목

작성일: 2026-06-17 · 대상: prefill-layer-alloc v2 (Zamba2-1.2B/2.7B, Falcon-H1-1.5B/3B · A100-SXM4-80GB)
용도: **현재 실행 중인 widened sweep(batch/decode {1..512}, E3/E5 zamba2 PENDING: job 770986/770988)을 기다리는 동안** 사람이 직접 확인해야 할 항목.
원칙: 게이트(`adjudicate.py`)는 보조이고 **최종 판정은 사람**(v2_report §6). 아래는 "닫기/발표 전에 내 눈으로 봐야 하는 것" 목록이며, 각 항목에 위험·확인위치·통과기준을 적었다.

판정 범례: **[P0]** 결론을 뒤집을 수 있음(필수) · **[P1]** 결론 강도/정직성에 영향 · **[P2]** 정리/위생.

---

## A. 결론을 떠받치는 핵심 주장 (P0)

### A1. [P0] 헤드라인 음성 주장 — "공간 분할은 동시실행을 한 번도 못 이김"
- **주장:** `max(two_stream/green_ctx)` = 0.987(zamba)/0.996(falcon) → Green Context가 단 한 셀도 못 이김 (v2_report §4.6-D, §7.1).
- **위험:** 이 값이 *전체* E5 매트릭스에서 재산출된 것인지, 아니면 일부/구버전 run의 잔재인지. widened decode sweep(→512) 후에도 1.0 미만이 유지되는지.
- **확인:** `results_v2/e5/serving_coexec_*.csv`에서 backend별 `concurrent_ms`로 ratio를 직접 재계산. zamba2_1.2b CSV는 6/17 13:03 갱신됨(widened) — 나머지 3모델(6/15)과 셀 수가 일치하는지 대조.
- **✅ 통과 (widened 검증 §1.1, 2026-06-18):** 의미있는 승리 0. 단 db1–2 저배치에서 **노이즈 수준(≤0.4%, ≤0.002ms) 동률 3셀**(falcon3b/zamba1.2b/zamba2.7b 각 1)이 ratio≥1.0 → **주장 문구를 "한 번도 못 이김"에서 "의미있게는 못 이김"으로 정정**(전 보고서 반영 완료). 고배치선 green_ctx 최대 1.9× 패.

### A2. [P0] microbench 한계 — E5는 "진짜 prefill"이 아님
- **주장:** "분할 무용"은 1576셀에서 일관 (v2_report §6).
- **위험:** E5의 prefill은 **chunk 1개·batch=1·scan-only**, decode는 **step 1개**, n_measure=20. 실제 prefill엔 GEMM(in/out_proj)·다중 chunk가 포함되고, E1이 보였듯 **고batch에선 GEMM이 prefill을 지배**(scan share 20~40%로 붕괴). GEMM-heavy 진짜 prefill에서 overlap·분할 결론이 그대로인지 미검증.
- **확인:** `--prefill-layer ssm_full`(report §6이 지목한 점검 경로)로 E5 일부 셀 재측정. 적어도 pf=ssm×dec=ssm 최고 셀과 db8/db256 양 끝.
- **통과:** ssm_full 모드에서도 (a) green_ctx가 two_stream 못 이김, (b) overlap이 prefill 크기로 스케일, (c) window가 db↑에서 닫힘 — 세 정성 결론 유지. **이게 음성 결론의 최대 위협. 닫기 전 반드시 1회.**

### A3. [P0] G1 술어 mis-specification (상위 결함) + 비대칭 robustness (부차)
- **상위 결함(술어):** G1이 "attn-ssm **sat_sm 비대칭이 있는가**"를 gate 술어로 쓴 것 자체가 category error다. sat_sm 비대칭은 *solo 커널의 occupancy 속성(a)*이고, partition 이득을 결정하는 건 *공유 하 동적 co-schedule이 회수 못 하는 slack(b)*이며 둘은 독립. `g1_verdict.json`의 *"SSM frees SMs for attention → proceed to E4"*가 (a)→(b) 비약. → [중단 보고서 §3.1](project_closure_report.md) 참조. **이 비판을 반영해도 헤드라인(분할 무용)은 불변·강화** — 올바른 술어(b)는 E4/E5(`two_stream` vs `green_ctx`)가 이미 음성 측정했기 때문.
- **확인(술어):** 보고서·verdict가 "비대칭 present → 분할 후보"를 *valid 추론*인 양 서술하는 곳이 남았는지 grep. 모두 "비대칭은 lever 아님(서술적 사실), 이득 판정은 E4/E5의 (b)" 로 정정됐는지.
- **부차(robustness):** matched-granularity(`reports/matched_granularity_zamba2.md`)에서 zamba 20셀 중 **MAINTAINED 11 / REDUCED 7 / ELIMINATED 2**, full-seq→chunked로 attn sat_sm 최대 **+40 SM** 이동 → 비대칭 일부는 granularity 아티팩트(audit §3 confound). falcon_h1·nemotron_h도 같은지.
- **통과:** (1) "비대칭→분할" 비약 서술 0건, (2) 비대칭의 robustness 여부와 무관하게 헤드라인 유지 명기. *비대칭을 "실재"로 단정 금지 — 단 §3.1에 따라 이건 §4.3 sub-claim 강도 문제일 뿐 결론 불변.*

### A4. [P0] BW 절대%에 의존한 주장 0건
- **위험:** `ssm_scan_bytes`는 recurrent state 트래픽 누락(과소), `attn_bytes`는 L2-캐시 KV 과대계상 → attn에서 BW% >100% 발생, **절대 BW% 비신뢰** (v2_report §3, audit §4).
- **확인:** 보고서·그림·verdict의 모든 메커니즘 문장이 절대 BW%가 아니라 **latency 기반 sat_sm + E0 grid 예측**에 근거하는지 grep. `attn_mechanism: "bw_or_compute_limited(util-hint;abs%unreliable)"`처럼 라벨에 불확실성이 박혀 있는지 확인.
- **통과:** "SSM is BW-bound at X%" 같은 절대% 단정 0건. ssm 메커니즘은 verdict에서 대부분 `indeterminate`/`grid_limited`로 정직하게 남아 있어야 함.

### A5. [P0] 선행연구 충돌 — green_ctx를 *SLO 지표*로 재평가
- **사실:** E5의 `green_ctx@f`는 prefill/decode 스트림 사이 SM 분할(= MuxWise·Bullet, ASPLOS'26이 단일 LLM에서 이득 본 *같은 메커니즘*). 우리 헤드라인 "분할이 동시실행을 못 이김"은 **throughput 지표(`two_stream/green_ctx`)로만** 판정됐다 — 그런데 분할이 이기는 metric은 **decode tail latency / SLO**다(closure §5.1).
- **위험:** throughput만 보고 "분할 무용"이라 적으면 *선행연구가 이미 반증한 일반화*를 주장하게 됨. green_ctx가 throughput은 지면서 decode 지연은 *줄였을* 수 있고, 그게 바로 분할의 존재 이유.
- **확인:** E5 CSV의 **`decode_inflation_pct`를 backend별(two_stream vs green_ctx@f)로 비교.** green_ctx가 decode_inflation을 유의하게 낮추는 셀이 있는지. 있으면 microbench 안에서도 SLO-이득 신호가 잡힌 것.
- **✅ 통과 (widened §1.3 + real-prefill, 2026-06-18) — 단 반전 주의:** green_ctx(prefill-우대 f)는 decode를 *보호*가 아니라 **starve**(inflation 64–83%, real prefill·7B서 66–101% vs two_stream 1–10%). 즉 우리 분할은 thro승·decode-latency *양 축 모두* 열등.
- **decode-보호형 측정 코드 구현됨(2026-06-18, `--decode-protect` + `analyze_slo_partition.py`, 설계 §4.5):** decode에 E3 floor SM 예약. **단 SLO 분석기가 이미 시사 — microbench에선 two_stream이 *양 축 우세***(decode_inflation 3.6–5.3% + throughput 1.23–1.29×)라 protective 분할이 비집을 SLO 격차가 없음. → **진짜 SLO 이득(MuxWise/Bullet)은 sustained-load/queue 현상**이고 단일셀 microbench로 재현 불가. `--decode-protect`는 *microbench엔 격차 없음을 확정*하는 용도, 결정적 검증은 **queue 시뮬레이터(별도 build)**. **"분할은 decode도 못 지킨다"로 일반화 금지 — 큐 부하 미검증.**

---

## B. 데이터 무결성 (P1)

### B1. [P1] widened run의 batch=512 chunk=full 실패
- **사실:** 현재 widened E1(job 770376)에서 `batch=512 chunk=full FAILED: CUDA illegal memory access`. CSV는 120행 기록.
- **확인:** 실패 셀이 0/NaN으로 기록돼 평균/포화점을 오염시키지 않는지. 인접 셀(chunk 256/512 @b512)이 정상값인지. illegal access가 그 1셀 한정인지(이후 프로세스 오염 없음 — subprocess 격리 §3 의도).
- **통과:** 실패 셀은 status≠ok로 제외. 다른 셀 정상. **단순 극한-batch OOM/limit이면 보고만, 커널 버그면 E3/E5 b512에도 영향 가능 → PENDING 잡 결과도 같은 셀 점검.**

### B2. [P1] 수정된 미커밋 CSV가 결론을 바꾸지 않았는지
- **사실:** `git status`에 e1/e2 zamba2_1.2b CSV 3개 modified(widened 재실행 산출).
- **확인:** widened 값이 기존 narrow run 대비 G0/G1 판정을 바꿨는지(`adjudicate.py` 재실행). e1 로그상 `max scan_share=82.7%`로 G0=OK 유지, e2 640행 정상 — 일관해 보이나 verdict JSON diff 확인.
- **통과:** 판정 불변이면 커밋. 바뀌면 v2_report 표 갱신 후 커밋.

### B3. [P1] matched-granularity CSV가 실측인지
- **위험:** audit "측정 블로커"에 따르면 작업 머신(A100 PCIe + 깨진 mamba_ssm)에선 신규 측정 불가. matched_granularity 보고서(zamba/falcon/nemotron)가 **SXM4 실측**인지, 코드만 돌린 빈 산출인지.
- **확인:** `results/stage1_v2/{ssm,attn}_chunked_*_a100_sxm4_80gb.csv` 존재·device 태그·행수. 보고서 "schema diff 0 ✓"가 실측 행 기반인지.
- **통과:** 실측 CSV가 SXM4 태그로 존재하고 보고서 표 수치의 출처임.

### B4. [P2] deprecated/미통일 경로가 v2 결론에 안 섞였는지
- **확인:** `run_ssm_two_pass_sweep.py`(deprecated), `a100_40gb` 격자([11,22,..], audit §1), 미완 Phase 4.1/4.2 grid-literal — 이들 중 어느 것도 v2_report 표/그림의 데이터 출처가 아닌지.
- **통과:** v2 결론은 전부 `results_v2/{e0..e5}` + SXM4 태그에서만 옴.

---

## C. 범위·정직성 (P0/P1)

### C1. [P0] "결정적 다음 단계"인 7B 회귀가 **미실행**임을 명시
- **사실:** `shared/configs/models.yaml:88` — "the 7B entries above are kept but **unused**". v2_report §1·§7.4는 7B 회귀를 "결정적(decisive) 다음 단계"라 부르지만 **한 번도 돌리지 않았다.** 모든 결론은 1.2~3B(SLM)뿐.
- **확인:** v2_report의 모든 결론 문장에 "SLM/A100 한정" 스코프가 붙어 있는지. §7.4를 "미완의 결정적 테스트"로 명확히 표기.
- **통과:** "hybrid serving에서 SM 분할 불필요"를 **모델 크기 무관 결론으로 쓰지 않음.** 닫기 보고서/논문은 "SLM에서 기각, 7B 미검증"으로 정직하게 한정. (→ `additional_value_paths.md` Path 1)

### C2. [P1] 양성 결과(prefill+decode overlap)의 재현성
- **주장:** overlap ~2.04×(db8) → 1.15×(db256), 분할 없이 co-schedule로 공짜 (v2_report §4.6-C, §7.2). 이게 프로젝트의 유일한 양성 산출.
- **확인:** widened decode sweep(→512)에서 window가 db256 이후로도 단조 감소하는지(1.15→? @db512). dec=ssm이 context 1k→16k 평탄 ~2.0× 유지인지.
- **통과:** 양성 결과가 widened 격자에서 재현. (이 결과가 닫기 보고서의 "회수 가능 산출"과 추가가치 Path 2의 토대.)

---

## D. 검수 진행 순서 (권장)

1. **PENDING 잡 완료 대기** (770986 E3, 770988 E5 zamba2) → 완료 즉시 B1(b512 실패)·A1(ratio 재산출)·C2(window) 동시 점검.
2. `adjudicate.py` 재실행 → B2(판정 불변 확인) → 불변이면 modified CSV 커밋.
3. A3(matched-granularity robustness) + B3(실측 여부) — 이미 산출된 보고서 검토만으로 가능, GPU 불필요.
4. A4(BW 절대% 의존 0건) + C1(7B 미실행 스코프) — 문서 grep, GPU 불필요.
5. **A2(ssm_full 진짜-prefill 검증)** — 유일하게 신규 SXM4 측정 필요. 닫기/발표 직전 1회. *이 항목 결과가 "닫기" vs "추가 진행" 분기의 마지막 입력.*

> A1·A2·A3·A4·A5·C1 = P0 6건이 전부 통과해야 v2를 "음성 결과로 종결" 또는 "발표"로 확정 가능. 하나라도 흔들리면 `additional_value_paths.md`의 해당 경로로 전환. (A5는 선행연구 MuxWise/Bullet과의 충돌 처리 — 헤드라인 scope 한정이 핵심.)
