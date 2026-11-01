# PREREG STOCK-BASE rev0 (DRAFT, 미봉인) — 기본 서빙 구조(stock SGLang · stock vLLM) 대 static PD-mux, VESSL·Nano (2026-11-01)

> 지위: **초안.** 봉인 전 (i) 사용자 결정 §13 (ii) 하네스 감사 1회(규칙층 + 파이프라인 모집단 층) (iii) 선행 작업 §11 완료가 필요하다.
> 결정 62(2026-11-01, "진행") + 2026-11-01 사용자 "진행 그리고 관련 수치들에 대한 확보"로 착수. 계획 원문 `PLAN_stock_baseline_metrics_2026-11-01.md` §3, Phase A 결과 `RESULT_stock_baseline_phaseA_2026-11-01.md`.
> 2026-11-01 개정 규약(experiment-runner "PREREG 필수 항"·"부하 위치"·"승계 상수"·"처치 발생 횟수"; claims-auditor 파이프라인 모집단 층)을 이 문서가 처음 적용한다.
> 라벨: [측정]/[유도]/[가정]/[코드]/[결정]/[제안].

## 0. 답하는 질문 목록 (규약 필수 항 1 — 예산이 모자라면 arm이 아니라 여기서 줄이고 적는다)

| # | 질문 | 답을 주는 블록 | 라벨 출력 |
|---|---|---|---|
| Q1 | VESSL·Nano·W1에서 stock SGLang(기본값)의 용량 λ\*(버스트·지속)은 얼마인가 | (가) 사다리 + 지속 셀 | 수치(기술값) |
| Q2 | 같은 Job·같은 절대 rate(용량 아래)에서 static PD-mux(D44)의 꼬리 SLO goodput이 stock SGLang과 다른가 (부호·크기) | (나) 정상상태 셀 ×2 + 변화 trace | `PM_ABOVE_SG` / `EQUIV_WITHIN_3PCT` / `PM_BELOW_SG` / `UNRESOLVED_INTERVAL` |
| Q3 | 같은 조건에서 stock vLLM 대 stock SGLang은 어떤가 (대조군이 약하다는 반박의 방어) | (나) | `VL_vs_SG` 같은 라벨 집합 |
| Q4 | static PD-mux 대 stock vLLM | (나) | `PM_vs_VL` |
| Q5 | mixed-chunk(SG-mix)가 stock SGLang의 ITL 꼬리를 바꾸는가 | (나) | `MIX_vs_SG` |
| Q6 | stock SGLang의 r_ser(L)·r_mix가 A5 예보(`R_SER_FORECAST_NANO.csv`)와 맞는가 | 계측 boot | 예보 대비 잔차(HIT/MISS 띠는 §8) |
| Q7 | W3(장문)에서 Q2·Q3의 부호가 유지되는가 | (나) W3 정상상태 셀 1개 | 같은 라벨, W3 접미 |

**이 Job이 답하지 않는 것(§10 잃는 범위에 재기재)**: 조율한 fused(chunk·overlap·mixed 튜닝) · Llama-8B 참조점 · 동적 PD-mux · Nano에서 static PD-mux의 best D(D44는 다리 arm, 아래 §2) · 과부하 구간.

## 1. 기판·트리 [결정/코드]
- VESSL A100 SXM 80GB ×1, 이미지 2종: SGLang = `PDMUX_VESSL_IMAGE`(pristine SGLang v0.5.10, digest 고정) · vLLM = **새 이미지**(§11-1, pristine vLLM, 엔진 코드 미포함). 두 이미지 모두 공개이며 PREREG·결과·엔진 패치를 담지 않는다.
- SGLang arm은 `install_runtime.sh`로 현재 main 트리 설치. `SG-stock`·`SG-mix`는 PD-mux 경로 미사용(`PDMUX_ENGINE_MODE` 미설정, `--enable-pdmux` 없음). `PM-ref`는 비sticky static D44(결정 49).
- 엔진 digest는 계측 플래그(§5)가 들어간 트리로 1개 고정(봉인 시 `ENGINE_DIGEST.txt`).
- vLLM 버전: **0.31.0**(pip index 2026-11-01 최신 stable; 결정 62 ① "제출 시점 최신 stable") — 봉인 시점에 더 새 stable이 있으면 그것으로 갱신하고 봉인 문서에 적는다. attention 백엔드·`CUDAGraphMode`·chunked prefill 상태는 서버 로그에서 채취해 결과에 병기(기본값 그대로, 튜닝 0).

## 2. arm [결정 62 + 승계 상수 규약]

| arm | 엔진 | 설정 | 역할 |
|---|---|---|---|
| `SG-stock` | SGLang v0.5.10 | 기본값: chunked prefill 8192, overlap ON, mixed OFF, flashinfer, cudagraph ON, radix OFF(공통) | 기본 구조 baseline |
| `SG-rep` | 〃 | `SG-stock` 복제 boot | 플라시보(boot 효과 1회 추출) |
| `SG-mix` | 〃 | + `--enable-mixed-chunk` | Q5 |
| `VL-stock` | vLLM 0.31.0 | 기본값(무패치), prefix caching OFF(radix OFF 대응), cudagraph 기본 | Q3·Q4 |
| `PM-ref` | SGLang + PD-mux | static D44 비sticky, budget 65536, cap 48 | **다리 arm** — CAP/TUNE/implfix 수치와 잇는 arm |

**승계 상수 공시**: `PM-ref`의 D44·cap 48·budget 65536은 KISTI 승계값이며 Nano에서 재결정되지 않았다(R Job: D34–64 goodput 고원 1.5% 안, 순위 미분해). 따라서 Q2·Q4의 라벨은 **"static PD-mux(D44 구성)"** 범위로만 성립하고 "static PD-mux 최적" 범위가 아니다. 최적 D 재결정은 CAP′(HE0 셀 재정의 결정 뒤)의 몫이다. [제안] 예산이 허용하면 `PM-ref34`(D34) 1 arm을 (나)에 추가해 고원 안 두 점으로 범위를 넓힌다 — §13 결정 4.

공통: `--context-length 20480`·`--max-prefill-tokens 16384`(정책 하네스 S7 승계; W3 7k 평균이 들어감) · max_running 48(vLLM `--max-num-seqs 48`) · 출력 길이 trace 고정 · 같은 체크포인트 경로 `/data/hf`.

## 3. 워크로드·셀 [결정 44·49, 부하 위치 규약]
- W1 ShareGPT(정책 하네스 `ph_spec` 정의·seed 그대로), W3 arXiv-Sum(7k 평균, `W3_MAX_NEW` 1024). SLO: W1 TTFT 3000 ms ∧ 요청 내부 ITL p95 60 ms; W3 TTFT = 3 × floor(같은 Job 첫 boot에서 측정, N3 규칙).
- **(가) 용량 사다리** — arm 4개(SG-stock·SG-mix·VL-stock·PM-ref) × W1 10단(5.516 × 1.15^k, k=0..9 — CAP1-R 격자 승계, Nano PD-mux 용량 7.29가 격자 안에 있음을 확인한 격자) × 120요청 × seed {201, 202}. λ\*_arm = attainment ≥ 0.90이 앞에서부터 끊기지 않는 최대 rung(`pick`와 같은 규칙). **동률 규칙 등록**: 동률이면 낮은 rate(보수). 격자 최상단이 통과하면 `LADDER_TOP_PASS`(λ\* 미확정, 한 번의 확장 재실행 등록: 격자 ×1.15^k, k=10..13).
- **지속 셀**: 각 arm λ\*_arm × 600 s × seed 201 → 완료 처리율(120 s 창별)·admission 속도 분리 보고(CAP2 감사 (3) 규약). (a) "둘 다 보고" [제안, §13 결정 1].
- **(나) 판정 셀** — 공통 절대 rate. λ\*_ref = min over arm {λ\*_arm(버스트)}. 정상상태 셀 R₁ = 0.5·λ\*_ref, R₂ = 0.8·λ\*_ref, 각 200요청 × seed {1,2,3,4}. 변화 trace: LO = 0.2·λ\*_ref, HI = 0.8·λ\*_ref (HI/LO 4, 결정 44), 구간당 200요청·drain·3라운드 × seed {1,2,3,4}. **부하 위치 예보**: HI에서 best arm attainment 0.6–0.85 [유도, CAP1-R의 7.29 → 8.39 사이 0.7대 하락에서]; 측정 후 HI attainment가 전 arm에서 > 0.9이거나 best arm < 0.5면 `CELL_OUT_OF_BAND` — 라벨은 내되 크기 인용 금지(규약 차단형 게이트 4가 자동 적용).
- W3: 정상상태 셀 R₃ = 0.8·λ\*_W3(SG-stock 기준, 사다리 6단 0.1×1.5^k) × 96요청 × seed {1,2,3,4}, arm 4개. W3 사다리는 SG-stock 1 arm만(예산; §10).

## 4. 판정 규칙 (데이터 전 등록; 정책 하네스 `pi_analyze` 규약 승계)
- 쌍: seed-level paired ln(A/B) goodput, t(3) 95% CI. `EQUIV_WITHIN_3PCT`는 |mean| < ln1.03 ∧ CI ⊂ (−ln1.03, +ln1.03); `A_ABOVE_B`는 CI 하한 > 0 ∧ mean ≥ ln1.03; 반대는 `A_BELOW_B`; 그 외 `UNRESOLVED_INTERVAL`.
- **검출력 사전 계산(규약 필수 항 2)**: 플라시보(SG-rep vs SG-stock) sd(ln) 예보 ≤ 0.006 [유도: H Job static 플라시보 0.0016, P Job 0.0054]. n=4·t(3)=3.182에서 EQUIV 가능 조건 sd < 0.0186, 반폭 = 3.182·sd/2 → sd 0.006이면 ±1.0%. 플라시보 sd가 0.0186을 넘으면 그 셀의 EQUIV는 불가하고 `UNRESOLVED_INTERVAL`이 정상 결과다(공시). 엔진 간 쌍(SG vs VL)은 같은 trace·seed라도 다른 프로세스라 sd가 플라시보보다 클 수 있다 — 예보 ≤ 0.02 [가정], 미달이면 seed 8로 재실행(등록 1회, 같은 Job 구조).
- 플라시보 무효 규칙(K3 단일): SG-rep vs SG-stock이 n<4 또는 |mean| ≥ ln1.03 또는 CI가 0 밖 → 그 셀 전부 `BOOT_NOISE`.
- 유효성 사슬: ① 같은 seed trace hash 동일(두 엔진 클라이언트 동일 하네스) ② cliff(TTFT가 SLO ±5% 안 > 10%) 양 arm 어디에도 없음 ③ 스탬프(Job·GPU UUID 동일) ④ 플라시보 유효 ⑤ 정의 일치 smoke 통과(§5-0) ⑥ 요청 실패 0.
- 지표: goodput(정본 술어) headline, 병기 TTFT/TPOT/ITL p50/p95/p99 · req/s · out tok/s · E2E(요청별 send/first/finish 시각 기록 — 새 측정은 처음부터) · 요청 내부 ITL p95 중앙.

## 5. 계측 boot (측정 arm과 분리, C5) + 축약 입력 필터 (규약 필수 항 3)
- 5-0 **정의 일치 smoke**(첫 boot, 측정 아님): 고정 요청 20개를 SG-stock·VL-stock에 같은 클라이언트(`bench_serving` 호환, vLLM은 `--backend vllm`)로 보내 클라이언트 TTFT − 서버 로그 first-token 시각 차의 중앙값이 두 엔진에서 ±5 ms 안인지. 실패 → `CLIENT_DEF_MISMATCH`, (나) 엔진 간 쌍 전부 INVALID(fail-closed).
- 5-1 SG-stock 계측 boot: `PDMUX_PROBE_EXTEND_TIME=1`(§11-3, 기본 OFF) + decode_probe, R₂ 부하 60요청 seed 301 → r_ser(L)·r_mix. **입력 필터**: `t ∈ [t_load_start, t_load_end]`(클라이언트 첫 요청 송신 ~ 마지막 완료) ∧ `role=pc` ∧ warmup·smoke 요청 ID 제외. 필터 밖 레코드가 섞이면 r_ser p50이 바뀌는 **변이 테스트** 등록(교훈 53).
- 5-2 VL-stock 계측 boot: 무패치 원칙 → `/start_profile`~`/stop_profile` torch profiler 창(같은 부하), step별 GPU 시간 + 토큰 수로 r_mix. 관측자 비용은 계측 boot 안에서만 공시.
- 5-3 PM-ref 계측 boot: implfix `pc_reduce` 정의 재사용하되 **부하 창 필터를 적용한 수정판**(봉인 코드 수정 아님 — 새 함수, 변이 테스트 포함).
- 라벨별 입력 필터 표(감사자용): `λ*` ← (가) run 클라이언트 요청만 · `goodput` ← (나) 측정 boot 클라이언트 요청만(계측 boot 제외) · `r_ser/r_mix` ← 5-1/5-2 부하 창 · `플라시보` ← SG-rep 측정 boot.

## 6. 처치 발생 횟수 (규약)
- Q5(SG-mix): MIXED forward가 run당 ≥ 50회여야 유효(`MIX_NOT_FIRED` 아니면). 예보 [유도]: W1 R₂에서 prefill 배치 ≈ 요청 수 200 → 충족.
- Q6: 계측 boot에서 단일 요청 EXTEND span ≥ 30개(A5 예보 격자 L ∈ {512, 2k, 8k} 각 ≥ 10).

## 7. 예산·Job 분할 [유도, 시간 모형 `pi_spec` 승계: boot 6 min, 120요청 셀 ≈1.6 min, he0 run 7.5 min]
- (가) 사다리: 4 arm × 10단 × 2 seed × 1.6 = 128 min + 지속 4 × 10 = 40 + boot 4×6 + smoke/계측 3 boot ≈ 215 min → **Job 1(≈3.6 h)**, 하드캡 ×1.25.
- (나) 판정: 5 arm(+rep) × [정상상태 2 × 4 seed × 2.5 min + trace 4 × 7.5] × = 5 × 50 = 250 min + W3 4 × 4 × 5 = 80 + boot ≈ 360 min → **Job 2·Job 3 분할**(arm을 Job에 교차 배치하지 않고 **셀을 나눈다**: Job 2 = 정상상태 R₁·R₂ 전 arm, Job 3 = 변화 trace + W3 전 arm). 한 쌍의 arm은 항상 같은 Job·같은 GPU(스탬프 검사).
- 사용률 계산·`$/h`는 dry-run 출력 그대로(과금 서술은 보고에 넣지 않는다).

## 8. 사전 예측 (봉인 대상; 채점만, 종합 판정 없음)
- P-1 λ\*_SG-stock(W1, 버스트) ∈ [6, 10] req/s [유도: PD-mux D44 7.29와 M3 J1 A LO req/s 동률에서]. P-2 λ\*_VL-stock / λ\*_SG-stock ∈ [0.8, 1.25]. P-3 R₂에서 PM-ref vs SG-stock goodput ln 비 부호 **+**(KISTI 방향) — 단 M3 J1 shape B 모양(PM TTFT ≫)이 n≥4에서 재현되면 **−** [두 방향 모두 거짓이 될 수 있음 → N3 아님]. P-4 r_ser(L=2k, SG-stock) 예보 = `R_SER_FORECAST_NANO.csv` 해당 셀 ±30%. P-5 SG-mix ITL p95 < SG-stock ITL p95(정지가 MIXED 안으로 옮겨감).

## 9. 허용/금지 문언
- 허용 형식: "VESSL A100 ×1 · Nano-9B-v2 · 이 Job에서, 셀 R(절대 rate x) arm A 대 B의 seed 4쌍 goodput ln 비는 평균 m, 95% CI [lo, hi], 라벨 L(유효성 사슬 통과/사유)."
- 금지: "PD-mux가 vLLM을 이긴다/진다"(범위 없는 일반화) · "최강 baseline" (조율 arm 없음) · KISTI 행과 섞은 비교 · 계측 boot 수치를 goodput 근거로 · D44 결과를 "static PD-mux 최적" 으로 · 과부하 셀(CELL_OUT_OF_BAND) 크기 인용 · A5 예보를 측정값처럼.

## 10. 잃는 범위
조율 fused · Llama-8B · 동적 PD-mux · best D 재결정 · W3 사다리(SG-stock 외 arm) · 과부하 구간 · prefix caching ON · TP>1.

## 11. 선행 작업 (GPU 전; 전부 끝나야 봉인)
1. **vLLM 이미지**: `env/docker/Dockerfile.vllm-runtime`(pristine vLLM 0.31.0 + 같은 base) 작성·빌드·push, digest를 `vessl.env`에 `PDMUX_VESSL_IMAGE_VLLM`으로. Workspace bring-up 1회(측정 아님, **비용 행위 → 사용자 승인**): Nano 체크포인트 부팅 + 고정 프롬프트 출력 비교 가드 + 선택된 백엔드/CUDAGraphMode 로그 채취.
2. 요청별 send/first-token/finish 시각을 쓰는 클라이언트 경로 확인(`bench_serving` 호환, 두 엔진 공통).
3. `decode_probe` EXTEND 시간 플래그(engine-porter, 진행 중) + CPU 회귀 + 변이 테스트 → digest 갱신.
4. 부하 창 필터 축약 함수 + 변이 테스트(5-1·5-3).
5. A5 예보 파일 sha256 봉인.
6. 하네스 감사 1회(규칙층 반전 시험 + 파이프라인 모집단 층 표).

## 12. 이 Job이 채우는 COSTMODEL 칸
CM-6(대기열·TTFT: stock 구조의 정지 r_ser(L) 실측, 예보 대비 잔차) · CM-1(T_dec 점 추가, stock 경로) · CM-2(r_ser 분자 = EXTEND GPU 시간) · 확장 열 "엔진 대조" 신설 후보.

## 13. 사용자 결정 대기 (봉인 전)
1. 용량 기준량 보고 방식: 둘 다(권고) / 버스트만.
2. vLLM 0.31.0 고정(권고) / 봉인 시점 재확인.
3. 판정 셀 rate를 λ\*_ref = min_arm 기준 공통 절대 rate로(권고) / arm별 자기 λ\* 상대 rate.
4. `PM-ref34` 추가(권고: 고원 안 두 점) / D44만.
5. Job 분할 3개(권고) / 2개(W3 제외).
6. 선행 1(vLLM 이미지 + Workspace bring-up 1회) 승인.
