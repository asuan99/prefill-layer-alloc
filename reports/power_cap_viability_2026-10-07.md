# GPU power cap — 개입여지 평가 (2026-10-07)

**문서 등급: 평가(비정본).** 정본 위계(`PROJECT_STATUS.md` > `reports/paper/` > `CONSENSUS.md`)를 바꾸지 않는다.
새 판정 0건, GPU 지출 0 h. 근거 자료 = `workspace/engine-port/results/power_cap/RESULT_power_cap_passive_2026-10-07.md`(기술적,
claims-auditor 감사 결과는 §5-1에 반영) · 선행연구 검색(venue-strategist, 2026-10-07 웹) · 기판 문서.

## 0. 한 줄 결론

**지금 개입할 수 있는 곳은 "관측 재분석"뿐이다.** 현재 기판(VESSL)에서는 power cap을 **설정할 수 없고 관측만 된다**(`-pl/-lgc` 권한 거부).
관측 재분석(0 GPU-h, claims-auditor 감사 반영)의 핵심은 하나다 — **기존 VESSL 데이터에서 cap 중 SM 클럭 하락은 −3…−4 %뿐(최저 ~1250 MHz)이라
power cap 처치의 효과를 알려 주는 데이터가 없다.** 부수적으로 에너지/요청 지표는 쓸 수 있으나 SLO cliff에 민감하고, 전력 ≠ 에너지(정규화 교훈),
KISTI deconf 분할 arm은 cap에 닿지 않았다(캠페인별 기술). **처치가 없으므로 어느 것도 power cap의 효과가 아니다.**
power cap이 연구 축이 되려면 `nvidia-smi -pl`이 되는 기판이 먼저 필요하고, 그 뒤에도 일반 기제("최적 공간 분할은 주파수 의존")는
PEACE(SoCC'26)가 이미 출판했으므로 남는 공백은 **"단일 GPU 동거 PD-mux × chip-wide cap"의 LLM판**뿐이다 — 메인트랙급이 아니라
워크숍/short-paper급이다.

## 1. 질문 · 범위 · 결정

- 질문(사용자, 2026-10-07): `prefill-layer-alloc`와 병행해 GPU power cap의 가용성을 판단하고 싶다 — 어디에 개입여지가 있는가.
- 범위(사용자 결정): 평가 문서 + 수동 재분석(CPU) + rung C 사전등록 초안. `-pl` 가능한 기판은 **아직 없음**.
- 이 문서가 하지 않는 것: 정책 주장, 정본 수정, GPU 실행.

## 2. 기판 제약

| 항목 | 상태 | 근거 |
|---|---|---|
| VESSL `nvidia-smi -pl / -lgc / -pm` | **불가**(권한 거부; `--privileged`·host device 불가) | `VESSL_A100_CONTEXT.md` §4·§4-1, `handoff-report/vessl_cloud_setup_plan_2026-09-24.md:268-269` |
| VESSL power/clock **관측** | 가능, 모든 Job이 100 ms `meta/clk.csv` 기록 | `scripts/vessl/job_entry.sh:155-157` |
| KISTI SLURM | 계정 차단(2026-09-15~) — root 여부와 무관하게 현재 실행 불가 | `[[slurm-account-blocker]]` |
| 로컬 5060 Ti | 서빙 실험 금지(`sgl_kernel` sm120 빌드 부재·16 GB·게이트 1) | `CLAUDE.md` |
| 필요 권한(활성 노브) | root `nvidia-smi -pl`, `-pm 1`, `power.min/max_limit` 읽기, 설정→읽기 검증, 기본값 복원 | 사전등록 초안 §2 |

### 2-1. VESSL 안에서 power cap을 설정할 수 있는 경로 (2026-10-07 조사, docs.cloud.vessl.ai·vesslctl 2026.09.22-01)

| 경로 | 가능? | 근거 |
|---|---|---|
| Workspace / Job 컨테이너(현재 경로) | **불가 — 실측 확정**(2026-10-07 Job `job-x3lzellrkm8z`: root지만 CapEff에 CAP_SYS_ADMIN 없음, no-op `-pl 400`·NVML set 모두 `Insufficient Permissions`; 결과 `results/power_cap/probe_job-x3lzellrkm8z/`) | `vesslctl job/workspace create`에 privileged·capability·securityContext 플래그 없음(`-f` JSON도 동일 필드); 문서 비교표 "Container privileges: No — you're already inside a container", "no host or kernel access"; 약관 §1 "no root access to the host OS or kernel". `-pl`은 CAP_SYS_ADMIN + `/dev/nvidiactl` root가 필요(NVIDIA container-toolkit 문서) |
| Reserved capacity(A100/H100 term) · 요청형 GPU(H200/B200…) | 불가 | 같은 컨테이너 모델, 권한 동일 |
| **VM Cluster**(`docs.cloud.vessl.ai/vm-clusters/*`) | **가능(유일)** — "Full root access on each VM", "run any container, including `--privileged`, with full capabilities and device mounts", 드라이버·커널 모듈 교체 가능 | 단 **GPU passthrough VM**(bare metal 아님) · **H100/H200/B200/B300만, A100 없음** · 노드당 8 GPU, 단일 노드부터 · **sales 견적·선불 reserved 계약**(크레딧 결제 불가, 기간 협의) · 계약 종료 시 데이터 즉시 삭제 |
| VESSL 지원팀에 컨테이너 capability 예외 요청 | 미확인 | 문서상 근거 없음 — 문의로만 확인 가능 |

VM Cluster로 가면 기판이 **H100(sm90)** 으로 바뀐다: 기존 A100 격자·λ\*는 전부 무효(게이트 1), green-context 입도(코드 상수 8 SM)·λ0 재앵커가 선행돼야 하고,
8 GPU 노드를 1-GPU 실험에 쓰면 7 GPU가 놀거나(비용) 8 arm을 병렬로 돌리되 같은 HGX 보드의 열·전력 결합이 새 교락이 된다.
passthrough VM에서 `nvidia-smi -pl`이 실제로 먹는지는 **계약 첫날 B0 프로브로 확인**해야 한다(일반적으로 passthrough 게스트 root에서 동작하나 VESSL 문서는 언급 없음).
컨테이너 경로는 2026-10-07 프로브로 닫혔다(위 표 1행). 같은 프로브로 이 A100의 설정 가능 범위 100–400 W(NVML 제약 일치)도 실측됐다.

### 2-2. 에너지 측정(관측)은 VESSL에서 가능 — 2026-10-08 프로브

Job `job-iwzlsxq6yu8t`: NVML 보드 누적 에너지 카운터(mJ)가 컨테이너에서 권한 없이 읽히고 ~100 ms마다 갱신된다. 5–20 s window에서
`power.draw` 100 ms 적분과 −2.0…+1.7 % 안에서 일치(부하 20 s −0.6 %; **정상 부하에서만** — 주기 부하에서는 §2-3 aliasing 편향). 호스트 RAPL은 컨테이너에 없음(CPU·DRAM 에너지 불가).
⇒ `job_entry.sh`가 이후 모든 Job에서 `meta/energy.csv`를 기록하도록 추가했고, `pc_energy.py`는 그 카운터를 우선 사용한다.
power cap을 **처치**로 쓰는 실험은 여전히 불가하지만, 에너지를 기존 노브(SM 분할·sticky·rate)의 **결과 지표**로 보고하는 것은 지금부터 가능하다
(GPU 보드 전체 값, 파티션 분리 불가, J/good은 SLO cliff 주의 — §5-1). 결과: `results/power_cap/probe_job-iwzlsxq6yu8t/`.

### 2-3. 에너지 측정의 시간 해상도 — 2026-10-08 센서 대역폭 프로브 2회

Job `job-absefkt9e88x`(v1)·`job-tkgxusf12n0v`(v2), 사각파 부하 + 3 ms 폴링:
- 폴링 power(`power.draw`)와 에너지 카운터는 **~100 ms마다 갱신**. 카운터는 갱신 사이를 연속 적분해 구간 총량이 정확(블록 평균 기준값), 폴링 스냅샷 적분은 aliasing으로 −27…−33 %(50 ms 주기).
- **NVML 버퍼 전력 샘플(`nvmlDeviceGetSamples`)은 ~20 ms 간격**, 계단 응답 12–34 ms, 블록 평균 편향 없음(0.98–1.02). 100 ms 주기 63 %, 50 ms 33 % 깊이로 보이고 20 ms 이하는 안 보인다.
  권한이 필요 없다 — v1의 "100 ms 하드웨어 한계" 판정은 폴링 값 한정으로 정정.
- ⇒ prefill/decode-only 구간 전력 차이는 시간 영역에서 직접 관측 가능, decode step(12–19 ms) 단위는 귀속 모델(20 ms 샘플 구간 × telemetry step 수 회귀) 필요.
  이후 Job은 `meta/energy.csv`(카운터)와 `meta/power_samples.csv`(20 ms 샘플)를 모두 기록. 결과: `results/power_cap/sensor_bw/RESULT_sensor_bandwidth_2026-10-08.md`, 그림 `results/power_cap/sensor_bw/figures/`.

### 2-4. 파티션별 전력 관측 — 2026-10-08 합성 부하 프로브

Job `job-78xs9733hcjy`(green context (92,16)/(64,44)/(16,92), 드라이버 smCount 일치, prefill-like GEMM·decode-like GEMV를 혼자/함께, 20 ms 버퍼 샘플):
- 혼자 돌 때 파티션 전력은 SM 수를 따라 분명히 구분된다(P 92→16 SM: 316→79 W, D 16→92 SM: 108→277 W). decode 위 prefill 200 ms 토글은 20 ms 샘플에서 계단으로 보인다(+218/+96/+40 W).
- 함께 돌면 보드가 **400 W 한계에 붙어** 가산성이 0.63–0.87로 깨진다 — 동시 실행 중 보드 전력을 파티션별로 나누는 것은 식별 불가, 귀속은 배제 설계에서만 성립.
  기본 cap이 이미 두 파티션을 묶는 공유 자원으로 작동한다(합성 포화 부하 기준, 클럭 미기록이라 처리량 0.83–0.93× 하락의 원인은 미분리).
결과: `results/power_cap/partition_power/RESULT_partition_power_2026-10-08.md`, 그림 `results/power_cap/partition_power/figures/`.

후보 기판은 **사용자 결정**: 베어메탈/루트 VM 대여(A100-SXM4 400 W면 VESSL 격자와 "같은 하드웨어"지만 **다른 기판**이므로 λ\*·수치 혼합 금지),
또는 KISTI 계정 복구. 어느 쪽이든 B0 프로브(사전등록 초안 §2)를 먼저 통과해야 한다.

## 3. 이미 있는 것 (코드·데이터)

- 기록: `clk.csv`(timestamp, clocks.sm, power.draw, temperature, throttle reasons; 100 ms), `substrate.json`(power.limit 400 W, max SM 1410 MHz).
- 분석기: bench window별 SW-power-cap(bit 0x4) 노출 %·SM MHz — `results/n3_capacity/n3_analyze.py:593-619`,
  `results/r2_eval/lambda0_vessl/analysis_2026-10-04/lam0v_descriptive.py:113-191`, `results/r2_eval/e2_sticky_vessl/analysis_2026-10-06/e2v_r2_result.py:115-160`.
- 실측 노출(VESSL, D44 arm): shape A 27 %, shape B 49 %, N3 r3 rate 0.10→0.77에서 9→46 %. 규칙 LVR-2/VC-2/N3C-16 = **arm Δ를 cap에 귀속 금지**.
- KISTI(2026-07~08) 5 s `*_clocks.csv` 800파일(타임스탬프 없음) — 이번에 처음 집계됨(RESULT §2).
- **없던 것**: 에너지 지표(J, J/req, J/token) → 이번에 `pc_energy.py`로 추가(보조 지표). 관련연구의 power/DVFS 축 → 이번 §4.

## 4. 선행연구 (2026-10-07 웹 검색; "미검증"은 1차 페이지로 확인 못 한 항목)

| 연구 | 장소 | 기제 | 수준 | 단일 GPU PD 동거? |
|---|---|---|---|---|
| Splitwise (+HHcap) | ISCA'24 | PD 분리 머신, decode 머신 power cap | 클러스터 | 아니오 |
| Towards Greener LLMs | EMC2@ASPLOS'24 | phase별 주파수·병렬·배치 에너지 노브 | 머신 | 아니오 |
| DynamoLLM | HPCA'25 | 인스턴스 수·병렬도·GPU 주파수 재구성 | 클러스터 | 아니오 |
| TAPAS | ASPLOS'25 | 열·전력 인지 스케줄링 | 데이터센터 | 아니오 |
| μ-Serve | ATC'24 | 모델별 주파수 | 클러스터 | 아니오 |
| throttLL'eM | HPCA'25(미검증) | iteration 예측 → 주파수·인스턴스 축소 | GPU/인스턴스 | fused |
| GreenLLM | arXiv 2508.16449 | phase별 주파수(분리 풀) | 분리 풀 | 아니오 |
| Phase-Decoupled Power Control | arXiv 2609.11133 | prefill 클럭 창 + decode 보정 cap | 분리 레인 | 아니오 |
| Illusion of Power Capping in LLM Decode | arXiv 2605.11999 | decode는 cap에 안 닿음(137–300 W/700 W), **clock locking > capping** | GPU | phase 개별 |
| **PEACE** | **SoCC'26** | MPS 동거 + cap: **최적 공간 분할이 주파수 의존**(10/90↔90/10), 도달 주파수는 SM 할당 의존 | GPU | 일반 워크로드(LLM PD 아님) |
| arXiv 2405.03838 | — | SM 분할·배치·cap 공동 최적화(HPC/ML) | 노드 | 아니오 |
| ECLIP | ISLPED'25 | AMD CU 마스킹 동거 에너지 | GPU | cap/DVFS 축 없음 |
| arXiv 2507.12442 | — | Transformer/Mamba-2/Falcon-H1 end-to-end 에너지 | 모델 | 서빙·phase·cap 없음 |

- **공백**: green context 기반 단일 GPU PD-mux × chip-wide cap은 미검증. hybrid(SSM) 모델의 phase별·cap 하 전력은 미보고(얇음).
- **반대 증거**: 2605.11999 — decode는 cap에 닿지 않는다 ⇒ cap은 "power 분할"이 아니라 **prefill TTFT 노브**일 가능성. RESULT §3에서
  N3 shape B의 0x4가 prefill 동거((64,44))와 거의 겹치는 관측도 같은 방향(단, 그 arm 한정·동거=실현 분할 교락).
- 리뷰어 요구(메인트랙): cap 격자 ≥ 4 × split 격자 × n ≥ 4 × 변화 trace, 기본 cap의 최적 split을 고정한 null, **clock-locking 베이스라인**,
  tokens/J@SLO-goodput, per-partition 전력 귀속(NVML 불가 → Nsight/DCGM), GPU 2세대. 현실적 장소: HotCarbon/EMC2/ISLPED/IGSC/MLSys 워크숍/SoCC short.

## 5. 개입점 판정

| # | 개입점 | 성립 조건 | 지금 | 주장 가능 범위 | 상태(2026-10-07) |
|---|---|---|---|---|---|
| ① | **에너지/요청·J/token 보조 지표** | 기존 clk.csv 적분 | CPU | 기술만 | **구현·실행 완료**(RESULT §1). J/good은 cliff 민감 |
| ② | **전력 vs SM 분할** 캠페인별 기술(deconf 분할 arm은 cap 미도달, np·P92만 도달) | KISTI per-split + VESSL 노출 | CPU | 기술만, 캠페인 간 비교·귀속 금지 | **집계 완료**(RESULT §2) |
| ③ | entanglement의 **클럭 채널**(prefill→cap→clock↓→decode ITL↑) | N3 PEV=1 ±50 ms 정렬 | CPU | 탐색적 | **미해결** — 기존 데이터의 클럭 하락이 −3 %뿐이고 동거=실현 분할 교락(RESULT §3). 낮은 cap은 미측정 |
| ④ | 운영 가이드(기여 3)에 **cap 인덱스** 추가 | `-pl` 기판 + rung C | 기판 후 | — | 사전등록 **초안**만 |
| ⑤ | split×cap 격자(최적 static split이 cap 따라 이동) | rung C 통과 + n ≥ 4 격자 | 가장 뒤 | — | PEACE가 일반판 선점 — PD-mux판만 남음 |

### 5-1. RESULT에서 인용 가능한 것 (claims-auditor 감사 반영; `results/power_cap/AUDIT_RESULT_passive_2026-10-07.md`)

- **핵심**: 기존 VESSL 데이터의 cap 중 SM 클럭 하락은 −3…−4 %(1250 MHz 아래 없음) ⇒ power cap 처치를 알려 주는 데이터가 없다, `-pl` 기판이 필요하다.
- N3 r3 한 Job: J/req는 0.10→0.77 req/s에서 약 2.7× 감소(TIMING window 규약), 0x4 노출 9→46 % — 부하 내생적이고 rate는 실행 순서와 교락(N3C-16).
- **J/good은 SLO cliff에 노출**되고 토큰당 에너지는 분모 정의에 따라 순위가 바뀐다(정성적 주의).
- E2 shape A 같은 Job·seed 쌍: 창 평균 전력과 창 에너지/완료요청이 **반대 부호**(8/8) — **전력 ≠ 에너지**라는 정규화 교훈으로만.
- KISTI s8 deconf P16 decode-축 arm은 near-cap 0.00, np·s8p P92 prefill-축 arm은 0.4–0.8 — 캠페인별 기술만(캠페인 간 비교·작업량 정규화 없음).
- bs1 (0,108) 전체-GPU decode: SM −2.8 % 동안 step 중앙값 +0.2 %(n=1,035, 전부 0x4 전환 인접) — 3 % 아래이나 "변화 없음" 아님; 다른 decode 상태로 일반화 금지.
- ③(클럭 채널)에 대한 판정: 이 arm에서 동거 플래그 = 실현 분할((64,44) vs (0,108))이라 SM 수와 동거를 분리할 수 없다 → **미해결**(탐색적 수치만).

### 5-2. 금지 문안 (이 평가서와 그 하류 전부; RESULT §5 전체 승계)

"power cap이 arm Δ의 원인" · "sticky ON이 에너지를 절감/비효율" · "SM 분할이 전력을 분할한다(power partition)" · "cap은 decode에 무해/클럭 채널 없음" ·
"동거 간섭 1.5–1.6×"(SM 수 교락) · "전력은 실현 분할의 증인" · "prefill 축이 decode 축보다 전력에 민감"(캠페인 간) · "cap은 prefill이 돌 때 걸린다"(일반 명제) ·
J/* 수준의 Job·기판 간 비교 · rate 순 추세를 cap 효과로 · "최적 split이 cap 따라 이동한다"(측정 전) · "최초"(교훈 260) · <3 % Δ headline.

## 6. 사다리

| rung | 내용 | GPU-h | 상태 |
|---|---|---|---|
| **A** 수동 재분석 | `pc_energy.py`·`pc_split_power.py`·`pc_itl_throttle.py`(selftest 3/3 PASS, 회귀 대조 Δ0) | 0 | **완료**, 감사 §7 |
| **B** 활성 노브 타당성 | B0 프로브(권한·min/max·읽기 검증·복원), `PDMUX_POWER_CAP_W` runner 허용목록, `ABORT_POWER_CAP`, cleanup 복원 | ≈0.1(스모크) | 기판 대기 |
| **C** 최소 사전등록 캠페인 | Nemotron-H, D44 sticky-OFF B1 고정, cap {400,300,250,200} W × seed 4, shape A/B 각 1 Job, 고정 부하(B 0.30 / A 1.80 req/s), (선택) clock-locking arm | B 1.47 h(worst 1.67) / A 1.40 h(1.59, 상수 미측정) [+0.3 h lock] | **초안** `results/power_cap/PREREG_PC_CAP_REV0_DRAFT_2026-10-07.md` |
| **D** split×cap 격자 | {D16,D24,D44,D64}×{400,300,200}×{A,B}×4 seed ≈ 96 유닛(점격자 하한 ≈ 5.5 h/shape; capacity-first면 ~3× N3) | ≥ 11 h | C 결과로 격자 축소 후 |

rung C의 예보(초안 §9): shape B에서 cap↓ → TTFT p95 단조 상승, 300 W에서 ITL |Δ| < 3 %, 250/200 W는 SM MHz < 1200일 때만 ITL 변화,
J/req 감소는 goodput cliff가 먼저 막는다. 예보가 깨져도 정책 주장은 생기지 않는다.

## 7. 감사 (claims-auditor, RESULT 대상, 2026-10-07)

치명 0건. 문구 수정 다수(§0(3) 철회·E2 문장 2곳 삭제·s0 "증인" 철회·"동거 채널" 교락 표기·금지 목록 9건 추가) — 전부 RESULT 감사 반영판과 §5-1/§5-2에 반영.
등재본: `workspace/engine-port/results/power_cap/AUDIT_RESULT_passive_2026-10-07.md`.
감사가 제시한 §3 해결 실험 = rung C 그 자체(`-pl` 기판, shape B D44 운영점, 400/300/250 W, n ≥ 4 교차, PHASE_EVENTS=1, 실현 분할×bs 층화 step 시간, paired CI) — 사전등록 초안 §4·§6에 반영됨.

## 8. 사용자 결정 목록

1. **기판**: `-pl` 가능한 호스트를 확보할 것인가 — VESSL 안에서는 **VM Cluster(H100, 8 GPU/노드, 선불 계약)** 뿐(§2-1); 밖에서는 루트 VM(GCP a2-ultragpu-1g 등 A100 80GB 1장) 또는 KISTI 복구. 없으면 ④⑤는 가설로 남고 이 트랙은 ①②③의 보조 지표로만 유지된다.
   (컨테이너 경로 확정 프로브는 2026-10-07 실행 완료 — 불가.)
2. **우선순위**: rung C(≈3 GPU-h)를 N3/N2 사다리(게이트 #6 미닫힘)보다 앞세울 것인가. Nemotron-first 규칙과는 충돌하지 않으나 GPU 시간을 나눈다. 권고: **N3/N2 뒤**.
3. **보조 지표 채택**: J/req·J/out_tok을 N3/N2 RESULT의 기술 열로 병기할 것인가(판정 입력 아님). 권고: 병기(`pc_energy.py` 재사용, 비용 0).
4. **clock-locking arm** 포함 여부(rung C +0.3 h): 선행 2605.11999가 베이스라인으로 요구 — 권고: 기판이 `-lgc`를 허용하면 포함.
5. **venue**: 이 축은 워크숍급(HotCarbon/EMC2/ISLPED)으로만 계획할 것인가 — 권고: 본 논문 기여 3(운영 가이드)의 부록 수준으로 두고 별도 투고는 rung D 이후 판단.

## 9. 파일

- 코드·표: `workspace/engine-port/results/power_cap/analysis_2026-10-07/{pc_common,pc_energy,pc_split_power,pc_itl_throttle}.py`, `out/pc_*.{md,json}`
- RESULT: `workspace/engine-port/results/power_cap/RESULT_power_cap_passive_2026-10-07.md`
- 사전등록 초안: `workspace/engine-port/results/power_cap/PREREG_PC_CAP_REV0_DRAFT_2026-10-07.md`
- 메모리: `[[power-cap-track]]`
