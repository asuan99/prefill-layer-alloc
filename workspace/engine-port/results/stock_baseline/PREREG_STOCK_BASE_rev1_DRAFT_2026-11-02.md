# PREREG STOCK-BASE rev1 (DRAFT, 미봉인; rev0 2026-11-01 → rev1 2026-11-02) — 기본 서빙 구조(stock SGLang · stock vLLM) 대 static PD-mux, VESSL·Nano (2026-11-01)

> 지위: **초안 rev1.** §13 결정은 전부 닫혔다. 봉인 전 남은 것 = 하네스 감사 1회(규칙층 + 파이프라인 모집단 층) + bring-up(A100 용량).
> **rev0 → rev1 변경(2026-11-02, 사용자 "권고대로 진행")**: ① W1 변화 trace 제거(고정 설정끼리의 비교라 컨트롤러 반응을 볼 대상이 없음; 게이트 2는 동적 arm 비교에 한정 — §3·§10) ② vLLM은 같은 공개 이미지 안 별도 venv(`sglang-vllm-runtime@sha256:a5f9bf0f…`, SGLang 쪽 바이트 불변 검증 완료, Public 확인 2026-11-02)로 같은 Job에서 boot ③ 계획 대체 6건 등록(§5·§3, 아래 "대체 목록") ④ 시간 기준 확인을 2단계로(§5-0) ⑤ 용어: "기준 용량" = 7개 arm 중 가장 낮은 지속 용량(min), "부하 창" = 클라이언트가 요청을 보내기 시작해 마지막 완료까지의 시간 구간 — 사용자 승인. 코드명(D-n·O-n·J1b·R₁ 등)은 문서에서 쓰지 않는다; 코드 식별자는 하네스 안에서만.
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
| Q7 | W3(장문)에서 Q2·Q3의 부호가 유지되는가 | 네 번째 Job W3 정상상태 셀 | 같은 라벨, W3 접미 |
| Q8 | 수정 트리·플래그 OFF(`SG-off`)가 pristine stock(`SG-stock`)과 같은가 — implfix C6의 GPU 검증 | (나) | `PATCH_OFF_vs_STOCK` ∈ {EQUIV_WITHIN_3PCT, …} |

**이 Job이 답하지 않는 것(§10 잃는 범위에 재기재)**: 조율한 fused(chunk·overlap·mixed 튜닝) · Llama-8B 참조점 · 동적 PD-mux · Nano에서 static PD-mux의 best D(D44는 다리 arm, 아래 §2) · 과부하 구간.

## 1. 기판·트리 [결정/코드]
- VESSL A100 SXM 80GB ×1, **이미지 1종**(rev1): `PDMUX_VESSL_IMAGE_SBASE` = `ghcr.io/asuan99/sglang-vllm-runtime@sha256:a5f9bf0f…` — 기존 공개 `sglang-runtime@1e0d335b…`에 `/opt/vllm-python`·`/opt/vllm-venv`(손대지 않은 vLLM 0.31.0, venv freeze sha `adff2ff0…`)만 더한 것. SGLang 쪽 파일 84,274개 해시가 기존 이미지와 동일(`BRINGUP_vllm_0.31.0_2026-11-01.md`). 공개 이미지, PREREG·결과·엔진 패치 없음. boot마다 두 freeze sha를 검사.
- ★**트리 구분(2026-11-01 사용자 지적으로 정정)**: 공개 이미지는 pristine v0.5.10이지만 Job 시작 시 `install_runtime.sh`가 PD-mux 수정 트리(14파일, +2623/−50)를 editable 설치한다. 따라서 **지금까지의 모든 "fused/OFF" arm(M3 J1 fused 포함)은 "수정 트리·플래그 OFF"였다.** 이 PREREG에서는 boot 단위로 트리를 고른다(`PDMUX_ENGINE_TREE=pristine|patched`, 선행 작업 §11-7):
  - `SG-stock`·`SG-mix` = **pristine 트리**(install 전 보존본, `PDMUX_*` env 0개, 마커 가드) — 진짜 stock.
  - `SG-off` = **수정 트리·플래그 OFF**(새 arm) — M3 J1 fused·implfix C6("OFF = main"은 CPU golden만)·CAP/TUNE OFF 참조와 잇는 다리. `SG-off vs SG-stock`이 처음으로 GPU에서 "수정 트리 OFF = stock인가"를 잰다(라벨 `PATCH_OFF_vs_STOCK`, Q8).
  - `PM-ref` = 수정 트리, 비sticky static D44(결정 49).
  - 각 boot 메타(`tree_<arm>.json`)에 `engine_tree`·`engine_tree_manifest_sha256`·`engine_tree_ref_sha256`·`engine_tree_ref_overridden`·`engine_tree_markers_found`·`engine_tree_sglang_init`·`engine_tree_root`를 기록하고 분석기 스탬프 검사(⑤)에 포함: SG-stock/rep/mix는 `pristine`·manifest == ref `1bd84d88…`·overridden false·markers `[]`·argv에 `--enable-pdmux` 없음; SG-off/PM은 `patched`·마커 3개·같은 Job의 patched manifest 동일. 30항목 ENGINE_DIGEST(`2836112b…`)와 트리 전체 manifest를 둘 다 기록.
- 엔진 digest는 계측 플래그(§5)가 들어간 트리로 1개 고정(봉인 시 `ENGINE_DIGEST.txt`).
- vLLM 버전: **0.31.0**(pip index 2026-11-01 최신 stable; 결정 62 ① "제출 시점 최신 stable") — 봉인 시점에 더 새 stable이 있으면 그것으로 갱신하고 봉인 문서에 적는다. attention 백엔드·`CUDAGraphMode`·chunked prefill 상태는 서버 로그에서 채취해 결과에 병기(기본값 그대로, 튜닝 0).

## 2. arm [결정 62 + 승계 상수 규약]

| arm | 엔진 | 설정 | 역할 |
|---|---|---|---|
| `SG-stock` | SGLang v0.5.10 | 기본값: chunked prefill 8192, overlap ON, mixed OFF, flashinfer, cudagraph ON, radix OFF(공통) | 기본 구조 baseline |
| `SG-rep` | 〃 | `SG-stock` 복제 boot | 플라시보(boot 효과 1회 추출) |
| `SG-mix` | 〃 | + `--enable-mixed-chunk` | Q5 |
| `VL-stock` | vLLM 0.31.0 | 기본값(무패치), prefix caching OFF(radix OFF 대응), cudagraph 기본 | Q3·Q4 |
| `SG-off` | SGLang 수정 트리 | 플래그 OFF(기존 모든 fused/OFF arm과 같은 트리) | Q8 다리 arm (§1 트리 구분) |
| `PM-ref` | SGLang 수정 트리 + PD-mux | static D44 비sticky, budget 65536, cap 48 | **다리 arm** — CAP/TUNE/implfix 수치와 잇는 arm |
| `PM-ref34` | 〃 | static D34 비sticky(실현 (74,34) B3a PASS, 입도 2 SM) | 고원 안 두 번째 점(결정 ④) |

**승계 상수 공시**: `PM-ref`의 D44·cap 48·budget 65536은 KISTI 승계값이며 Nano에서 재결정되지 않았다(R Job: D34–64 goodput 고원 1.5% 안, 순위 미분해). 따라서 Q2·Q4의 라벨은 **"static PD-mux(D44 구성)"** 범위로만 성립하고 "static PD-mux 최적" 범위가 아니다. 최적 D 재결정은 CAP′(HE0 셀 재정의 결정 뒤)의 몫이다. 결정 ④(2026-11-01): `PM-ref34` 포함 — Q2·Q4 라벨은 D34·D44 두 점으로 보고하며 "고원 안 두 점" 범위.

공통: **`--disable-piecewise-cuda-graph`**(v0.5.10 NemotronH 기본 ON이나 이 이미지에서 트리 무관 부팅 실패 — `multigpu/RESULT_m2g_c1395d0_2026-10-12.md` `typing.Union`; "기본값"에서 유일하게 바꾸는 플래그, 공시) · `--context-length 20480`·`--max-prefill-tokens 16384`(정책 하네스 S7 승계; W3 7k 평균이 들어감) · max_running 48(vLLM `--max-num-seqs 48`) · 출력 길이 trace 고정 · 같은 체크포인트 경로 `/data/hf`.

## 3. 워크로드·셀 [결정 44·49, 부하 위치 규약]
- W1 ShareGPT(정책 하네스 `ph_spec` 정의·seed 그대로), W3 arXiv-Sum(7k 평균, `W3_MAX_NEW` 1024). SLO: W1 TTFT 3000 ms ∧ 요청 내부 ITL p95 60 ms; W3 TTFT = 3 × floor(같은 Job 첫 boot에서 측정, N3 규칙).
- **(가) 용량 사다리** — arm 4개(SG-stock·SG-mix·VL-stock·PM-ref) × W1 10단(5.516 × 1.15^k, k=0..9 — CAP1-R 격자 승계, Nano PD-mux 용량 7.29가 격자 안에 있음을 확인한 격자) × 120요청 × seed {201, 202}. λ\*_arm = attainment ≥ 0.90이 앞에서부터 끊기지 않는 최대 rung(`pick`와 같은 규칙). **동률 규칙 등록**: 동률이면 낮은 rate(보수). 격자 최상단이 통과하면 `LADDER_TOP_PASS`(λ\* 미확정, 한 번의 확장 재실행 등록: 격자 ×1.15^k, k=10..13).
- **지속 셀**(결정 ①): 각 arm {λ\*_arm(버스트), 0.8·λ\*_arm} × 600 s × seed 201 → 완료 처리율(120 s 창별)·admission 속도 분리 보고(CAP2 감사 (3) 규약). λ\*_arm(지속) = 두 점 중 "120 s 창 완료율이 도착률의 ≥ 0.95이고 마지막 창 TTFT p50이 첫 창의 ≤ 2배"를 만족하는 최대 rate; 둘 다 미달이면 0.5·λ\*_arm 1점 추가(등록 1회). 버스트·지속 둘 다 보고.
- **(나) 판정 셀** — 공통 절대 rate(결정 ③). 기준 용량 = 7개 arm 중 가장 낮은 **지속** 용량(결정 ①; 지속 셀이 두 번째 Job에서 나오므로 판정 셀 rate는 그 회수 뒤 확정). 정상상태 셀 두 개: 기준 용량의 50%와 80%, 각 200요청 × seed {1,2,3,4}. **변화 trace(LO↔HI 반복)는 쓰지 않는다**(rev1: 7개 arm 전부 고정 설정이라 부하 변화에 반응할 컨트롤러가 없고, 저부하·고부하는 두 정상상태 셀이 덮는다; 시간 모형상 trace가 네 번째 Job의 76%를 차지했다). **부하 위치 예보**: 80% 셀에서 best arm attainment 0.6–0.85 [유도]; 측정 후 전 arm > 0.9이거나 best < 0.5면 `CELL_OUT_OF_BAND` — 라벨은 내되 크기 인용 금지.
- **Job 간 기판 눈금(refcell)**: Job 2·3·(1b) 시작과 끝에 `SG-stock`을 λ\*_ref(지속)에서 120요청·seed 201로 1회씩 돌려 attainment·TTFT p50·throttle 비율을 기록한다. Job 1의 같은 점과 attainment 차가 **10%p 이상**이면 그 Job의 셀에 `SUBSTRATE_SHIFT`를 붙인다 — 라벨은 내되 Job 1의 용량 수치·다른 Job의 goodput 크기와 섞지 않는다. 판정 셀의 `CELL_OUT_OF_BAND` 검사는 Job별로 적용한다(GPU UUID가 Job마다 다르므로, 같은 절대 rate가 다른 상대 부하가 될 수 있다).
- W3(장문): 네 번째 Job. 시작 boot에서 floor(단일 요청 TTFT 중앙) 측정 → SG-stock W3 용량 사다리(0.1×1.5^k 6단 × 48요청 × seed 2) → 정상상태 셀 = SG-stock W3 용량의 80% × 96요청 × seed {1,2,3,4}, arm 4개(SG-stock·SG-rep·PM-ref·VL-stock). 사다리는 SG-stock 1 arm만(§10).

## 4. 판정 규칙 (데이터 전 등록; 정책 하네스 `pi_analyze` 규약 승계)
- 쌍: seed-level paired ln(A/B) goodput, t(3) 95% CI. `EQUIV_WITHIN_3PCT`는 |mean| < ln1.03 ∧ CI ⊂ (−ln1.03, +ln1.03); `A_ABOVE_B`는 CI 하한 > 0 ∧ mean ≥ ln1.03; 반대는 `A_BELOW_B`; 그 외 `UNRESOLVED_INTERVAL`.
- **검출력 사전 계산(규약 필수 항 2)**: 플라시보(SG-rep vs SG-stock) sd(ln) 예보 ≤ 0.006 [유도: H Job static 플라시보 0.0016, P Job 0.0054]. n=4·t(3)=3.182에서 EQUIV 가능 조건 sd < 0.0186, 반폭 = 3.182·sd/2 → sd 0.006이면 ±1.0%. 플라시보 sd가 0.0186을 넘으면 그 셀의 EQUIV는 불가하고 `UNRESOLVED_INTERVAL`이 정상 결과다(공시). 엔진 간 쌍(SG vs VL)은 같은 trace·seed라도 다른 프로세스라 sd가 플라시보보다 클 수 있다 — 예보 ≤ 0.02 [가정], 미달이면 seed 8로 재실행(등록 1회, 같은 Job 구조).
- 플라시보 무효 규칙(K3 단일): SG-rep vs SG-stock이 n<4 또는 |mean| ≥ ln1.03 또는 CI가 0 밖 → 그 셀 전부 `BOOT_NOISE`.
- 유효성 사슬: ① 같은 seed trace hash 동일(두 엔진 클라이언트 동일 하네스) ② cliff(TTFT가 SLO ±5% 안 > 10%) 양 arm 어디에도 없음 ③ 스탬프(Job·GPU UUID 동일) ④ 플라시보 유효 ⑤ 정의 일치 smoke 통과(§5-0) ⑥ 요청 실패 0.
- 지표: goodput(정본 술어) headline, 병기 TTFT/TPOT/ITL p50/p95/p99 · req/s · out tok/s · E2E(요청별 send/first/finish 시각 기록 — 새 측정은 처음부터) · 요청 내부 ITL p95 중앙.

## 5. 계측 boot (측정 arm과 분리, C5) + 축약 입력 필터 (규약 필수 항 3)
- 5-0 **시간 기준 확인(2단계)**. 목적: SGLang `/generate` 파서와 vLLM `/v1/completions` 파서가 "첫 토큰"을 같은 사건에서 찍는지. 서버는 요청별 첫 토큰 시각을 로그에 남기지 않으므로(SGLang은 `--enable-metrics` 시 `sglang:time_to_first_token_seconds` 히스토그램만, vLLM은 기본 `/metrics` 히스토그램만) 서버 쪽 값은 히스토그램 합계·개수로만 얻는다. (a) **하나씩 20요청**(첫 boot, 측정 아님): 요청 하나가 끝날 때마다 히스토그램 합계 증가분(개수 증가 1인 것만) = 그 요청의 서버 TTFT; 클라이언트 TTFT와의 차(오프셋)가 엔진마다 작은 상수(HTTP 왕복 수 ms)이고 분산이 작은지 본다. 두 엔진의 상수가 같을 필요는 없다(서버 쪽 시작점이 다름: SGLang은 tokenizer manager 수신, vLLM은 프론트엔드 도착). (b) **동시 요청 상태**(rev1 추가): 50% 셀 부하(200요청, seed 301)에서 SGLang은 `--enable-metrics`를 켠 별도 boot(계측 역할, 세 번째 Job), vLLM은 50% 측정 boot 자체의 run 전후 `/metrics` 합계·개수로, run 단위 클라이언트 TTFT 평균 − 서버 TTFT 평균을 (a)의 오프셋과 비교. 차이 > 20 ms 또는 2배면 `TTFT_OFFSET_SHIFT` → 엔진 간 쌍 전부 `INVALID/UNRESOLVED:TTFT_OFFSET_SHIFT`(fail-closed). (a) 실패는 `CLIENT_DEF_MISMATCH`, 같은 처리. 비교는 끝까지 클라이언트 TTFT로만 한다.
- 5-1 SG-stock 계측 boot: `PDMUX_PROBE_EXTEND_TIME=1`(§11-3 **완료, CPU**: `decode_probe` `extend_forward` 레코드 — `extend_gpu_ms`·`extend_tokens`·`n_reqs`·`stream_key`·`seq`·`decode_iter_seq`; 순수 EXTEND만, MIXED·SPLIT_PREFILL 제외; `pacing_ratios(records, min_bs=5, t_window)`가 r_ser·r_mix를 냄, T_dec = bs별 순수 DECODE 중앙값; OFF 골든 26레코드 바이트 동일, 변이 12종 검출; 엔진 digest `65cbb49d…` → `2836112b…`) + decode_probe, R₂ 부하 60요청 seed 301. **입력 필터**: forward 레코드에는 요청 ID가 없으므로 **시간 창만** — `t_window = [t_load_start, t_load_end]`(프로브 `perf_counter` 시계; boot 레코드의 `t_realtime`/`t_host`로 클라이언트 송신·완료 시각을 사상), warmup·smoke는 창 밖에 두어 제외. 창 밖 레코드를 섞으면 r_ser p50이 바뀌는 **변이 테스트** 등록(교훈 53). GPU 미검증 항목(봉인 전 하네스 감사 공시): EXTEND 이벤트 실측·overlap 경로의 단일 forward stream(`stream_key` 상수)·ON boot 출력 불변·관측자 비용.
- 5-2 VL-stock 계측 boot: 무패치 원칙 → `/start_profile`~`/stop_profile` torch profiler 창(같은 부하), step별 GPU 시간 + 토큰 수로 r_mix. 관측자 비용은 계측 boot 안에서만 공시.
- 5-3 PM-ref 계측 boot: implfix `pc_reduce` 정의 재사용하되 **부하 창 필터를 적용한 수정판**(봉인 코드 수정 아님 — 새 함수, 변이 테스트 포함).
- 라벨별 입력 필터 표(감사자용): `λ*` ← (가) run 클라이언트 요청만 · `goodput` ← (나) 측정 boot 클라이언트 요청만(계측 boot 제외) · `r_ser/r_mix` ← 5-1/5-2 부하 창 · `플라시보` ← SG-rep 측정 boot.

### 계획 대체 목록 (rev1 등록, 2026-11-02 승인) — 초안 문장이 전제한 것이 시스템에 없어 자리를 옮긴 6건
1. SG-stock의 prefill/decode 시간비(Q6) 타이머는 우리 패치 코드라 손대지 않은 SGLang에 없다 → 타이머 측정 boot만 패치 트리·PD-mux OFF로 돌린다. Q6 결과는 "패치 트리·OFF" 범위. goodput 측정 boot는 손대지 않은 SGLang 그대로.
2. 서버 로그에 요청별 첫 토큰 시각이 없다 → 5-0의 히스토그램 증가분 방식(위).
3. 타이머 측정 boot의 부하(기준 용량의 80%)는 두 번째 Job 뒤에 정해진다 → 세 번째 Job으로.
4. Job 시작·끝 기준 셀을 "지속 용량"에서 돌리려 했으나 첫 번째 Job에는 그 점이 없다 → SG-stock 버스트 사다리 rung(첫 번째 Job에 trace hash까지 같은 셀이 실재)에서 돌린다. 없으면 `SUBSTRATE_UNCHECKED`.
5. 지속 판정 "완료 ≥ 0.95 × 명목 도착"은 푸아송 도착 변동(창당 ±5%)만으로 건강한 서버가 탈락 → 분모를 **그 창에 실제로 도착한 수**로(명목값 병기). 두 번째 조건(마지막 창 TTFT 중앙 ≤ 첫 창의 2배) 유지.
6. W3 floor·사다리의 Job 배정이 빠져 있었다 → 네 번째 Job.

## 6. 처치 발생 횟수 (규약)
- Q5(SG-mix): MIXED forward가 run당 ≥ 50회여야 유효(`MIX_NOT_FIRED` 아니면). 예보 [유도]: W1 R₂에서 prefill 배치 ≈ 요청 수 200 → 충족.
- Q6: 계측 boot에서 단일 요청 EXTEND span ≥ 30개(A5 예보 격자 L ∈ {512, 2k, 8k} 각 ≥ 10).

## 7. 예산·Job 분할 [유도, 시간 모형 `sb_spec`; 기준 용량 3.0 req/s 가정]
Job 4개(각 ≲4 h 규약, 하드캡 = 추정 × 1.25): **첫 번째** 시간 기준 확인(a) + 용량 사다리(5 arm × 10단 × seed 2; PM-ref34는 CAP1-R D34 용량 6.343 승계, 기술값) ≈ 160 min · **두 번째** 지속 셀(5 arm × {100%, 80%} × 600 s) + 기준 셀 시작·끝 ≈ 166 min · **세 번째** 정상상태 50%·80% 셀(7 arm × seed 4, ABBA) + 타이머 측정 boot + 시간 기준 확인(b) ≈ 216 min · **네 번째** W3 ≈ 211 min. rev0의 변화 trace(≈659 min)는 제거. 각 쌍의 두 arm은 항상 같은 Job·같은 GPU. 사용률·`$/h`는 dry-run 출력 그대로(과금 서술은 보고에 넣지 않는다).

## 8. 사전 예측 (봉인 대상; 채점만, 종합 판정 없음)
- P-1 λ\*_SG-stock(W1, 버스트) ∈ [6, 10] req/s [유도: PD-mux D44 7.29와 M3 J1 A LO req/s 동률에서]. P-2 λ\*_VL-stock / λ\*_SG-stock ∈ [0.8, 1.25]. P-3 R₂에서 PM-ref vs SG-stock goodput ln 비 부호 **+**(KISTI 방향) — 단 M3 J1 shape B 모양(PM TTFT ≫)이 n≥4에서 재현되면 **−** [두 방향 모두 거짓이 될 수 있음 → N3 아님]. P-4 r_ser(L=2k, SG-stock) 예보 = `R_SER_FORECAST_NANO.csv` 해당 셀 ±30%. P-5 SG-mix ITL p95 < SG-stock ITL p95(정지가 MIXED 안으로 옮겨감).

## 9. 허용/금지 문언
- 허용 형식: "VESSL A100 ×1 · Nano-9B-v2 · 이 Job에서, 셀 R(절대 rate x) arm A 대 B의 seed 4쌍 goodput ln 비는 평균 m, 95% CI [lo, hi], 라벨 L(유효성 사슬 통과/사유)."
- 금지: "PD-mux가 vLLM을 이긴다/진다"(범위 없는 일반화) · "최강 baseline" (조율 arm 없음) · KISTI 행과 섞은 비교 · 계측 boot 수치를 goodput 근거로 · D44 결과를 "static PD-mux 최적" 으로 · 과부하 셀(CELL_OUT_OF_BAND) 크기 인용 · A5 예보를 측정값처럼.

## 10. 잃는 범위
조율 fused · Llama-8B · 동적 PD-mux · best D 재결정 · W3 사다리(SG-stock 외 arm) · 과부하 구간 · **부하 변화(LO↔HI) 반응** · prefix caching ON · TP>1.

## 11. 선행 작업 (GPU 전; 전부 끝나야 봉인)
1. **vLLM 이미지**: `env/docker/Dockerfile.vllm-runtime`(pristine vLLM 0.31.0 + 같은 base) 작성·빌드·push, digest를 `vessl.env`에 `PDMUX_VESSL_IMAGE_VLLM`으로. Workspace bring-up 1회(측정 아님, **비용 행위 → 사용자 승인**): Nano 체크포인트 부팅 + 고정 프롬프트 출력 비교 가드 + 선택된 백엔드/CUDAGraphMode 로그 채취.
2. 요청별 send/first-token/finish 시각을 쓰는 클라이언트 경로 확인(`bench_serving` 호환, 두 엔진 공통).
3. `decode_probe` EXTEND 시간 플래그(engine-porter, 진행 중) + CPU 회귀 + 변이 테스트 → digest 갱신.
4. 부하 창 필터 축약 함수 + 변이 테스트(5-1·5-3).
5. A5 예보 파일 sha256 봉인.
6. 하네스 감사 1회(규칙층 반전 시험 + 파이프라인 모집단 층 표).
7. ~~pristine 트리 boot 모드~~ **완료(CPU, 2026-11-01)**: `install_runtime.sh` step 0이 이미지 트리를 `sglang_pristine/`에 보존(커밋된 참조 manifest `env/sglang_v0.5.10_python_tree.sha256` 2257항목·sha `1bd84d88…`와 바이트 대조, 마커 가드), 서버 boot는 `engine_tree.py boot --stamp <json> -- <server cmd>`로 `PDMUX_ENGINE_TREE=pristine|patched` 선택(pristine은 `PDMUX_*` 전부 제거·`--enable-pdmux` 거부·`find_spec` 해석 위치 검사). 테스트 `test_engine_tree` 21 + `test_job_entry_cpu.sh` 29 통과. **GPU 이미지에서 미검증**(pip -e 잔여물로 보존 FAIL 가능 → `PDMUX_REQUIRE_PRISTINE=1`로 fail-closed). 진단 표는 부록 D.

## 12. 이 Job이 채우는 COSTMODEL 칸
CM-6(대기열·TTFT: stock 구조의 정지 r_ser(L) 실측, 예보 대비 잔차) · CM-1(T_dec 점 추가, stock 경로) · CM-2(r_ser 분자 = EXTEND GPU 시간) · 확장 열 "엔진 대조" 신설 후보.

## 13. 사용자 결정 대기 (봉인 전)
1. 용량 기준량 보고 방식: 둘 다(권고) / 버스트만.
2. vLLM 0.31.0 고정(권고) / 봉인 시점 재확인.
3. 판정 셀 rate를 λ\*_ref = min_arm 기준 공통 절대 rate로(권고) / arm별 자기 λ\* 상대 rate.
4. `PM-ref34` 추가(권고: 고원 안 두 점) / D44만.
5. Job 분할 3개(권고) / 2개(W3 제외).
6. 선행 1(vLLM 이미지 + Workspace bring-up 1회) 승인. → **승인·진행 중(2026-11-01)**
7. `SG-off` arm 추가(권고: Q8은 implfix·M3 J1 수치의 범위를 정하는 데 필요) / 미추가.

**rev0 → rev1 전 추가 결정(2026-11-02, `REPORT_sb_harness_design_2026-11-01.md` §3·§4)**: ⑧ vLLM 배치 — (a) 별도 Job / **(b′) `sglang-runtime` + `/opt/vllm-venv` 새 공개 이미지(권고)** / (b) Job 안 격리 venv / (c) 비권고 — J1b 전 결정 필수 · ⑨ J3 시간 ≈670 min(상한 초과) — J3a(W1 trace, ≈8.6 h → 또 분할) / trace 라운드 3→2(게이트 2 이탈 공시) / W3를 J4로 분리 / 조합 — 권고: **W3를 J4로 분리 + W1 trace를 arm 절반씩 J3a·J3b(쌍은 같은 Job 유지)** · ⑩ 하네스 편차 D-1…D-6·해소 O-2…O-14 승인(권고: 전부 승인, D-1·O-10은 범위 문구로 공시).

확정(2026-11-01, 사용자 "진행" = 권고 채택): ① 둘 다 보고·판정 셀 rate는 **지속 기준** · ② vLLM 0.31.0 · ③ 공통 절대 rate · ④ `PM-ref34` 추가 · ⑤ Job 3분할 · ⑥ 승인(Workspace create 재시도 허용) · ⑦ `SG-off` 추가. → §13은 닫혔고 남은 봉인 조건은 §11 선행 작업 + 하네스 감사 1회.

## 부록 D — 수정 트리에서 플래그 OFF일 때 upstream과 달라지는 실행 경로 (engine-porter 진단, 2026-11-01, 코드 판독·GPU 미측정)

기준: `git archive v0.5.10` + HEAD `install_runtime.sh` → 수정 파일 15개(+3071/−55), 새 파일 11개. 줄 번호는 수정 트리.

| 파일 | OFF에서 발화 | 성격 | 근거 줄 |
|---|---|---|---|
| `layers/attention/mamba/causal_conv1d_triton.py` | **발화, 커널 수준** — int64 캐스트 2곳. triton conv 경로를 타는 모든 Mamba2 prefill(NemotronH xBC 포함). int32 범위에서 값 동일, 컴파일된 커널이 다름 → 성능 미측정 | GPU | 69, 77 |
| `models/nemotron_h.py` | 발화 — 매 forward env 3–4회 읽기·`current_stream()`·층별 `nullcontext()`; env 없으면 같은 층 순서·같은 커널. decode cudagraph에서는 capture 때만 | host | 645–737, 608, 859 |
| `models/granitemoehybrid.py` | 같은 패턴 | host | 455–497 |
| `distributed/parallel_state.py` | 매 `get_tp_group()`에서 `ContextVar.get()`(값 항상 False) | host | 1439–1454 |
| `managers/scheduler.py` | 프로브 import·`_holb is not None` 검사 2회/run_batch(env 없으면 None) | host | 189, 292, 440, 458, 2708, 2762, 3539, 3671, 3681 |
| `model_executor/model_runner.py` | ngram `getattr` 동치, Zamba2 isinstance, coord 메서드는 pdmux 전용 | 없음 | 2407, 1862 |
| `utils/hf_transformers_utils.py`·`configs/__init__.py` | import 시 mamba2/zamba2 전역 등록 | 없음(모델 종류 한정) | 122, 142–145 |
| `layers/attention/triton_backend.py` | `--attention-backend triton` + mamba 계열만(flashinfer 미발화) | 조건부 | 98 |
| `memory_pool.py`·`model_runner_kv_cache_mixin.py`·`server_args.py` | 순수 SSM / `Mamba2ForCausalLM`만 | 조건부 | 1420, 184, 1919 |
| `falcon_h1.py`·`multiplexing_mixin.py`·`pdmux_context.py` | OFF 변화 없음(새 메서드·상수·pdmux 전용) | 없음 | — |

**함의**: "수정 트리 OFF = upstream"은 **커널 수준에서 성립하지 않는다**(causal_conv1d int64). implfix C6의 CPU golden은 프로브 None 경로만 덮는다. 이 차이를 GPU에서 처음 재는 것이 Q8(`SG-off vs SG-stock`)이며, 그 전까지 M3 J1 fused·CAP/TUNE/EVAL OFF 참조·KISTI `plain`은 전부 "수정 트리·OFF" 범위다. 그 밖의 공시: 두 트리는 site-packages(sgl_kernel·flashinfer·triton·torch·py3.14 shim)를 공유한다 — "pristine"은 SGLang 파이썬 소스 한정; triton/flashinfer JIT 캐시 공유(정확성 영향 없음, warmup 순서 차이만); pip 메타데이터는 수정 트리를 가리키므로 선택은 PYTHONPATH 우선순위 + boot 시 해석 위치 검사에 의존.
