# AMD(Attention–Mamba Disaggregation) 설계 논의 — PD 분리 + 층 종류별 디바이스 분리의 오류 지점 · 의의 · 기대 효과 (2026-10-11, rev2)

> **지위**: 설계 기준선(`reports/system_design_anchor_2026-09-30.md`)의 부속 **논의 문서**(병행 트랙 제안).
> **측정 아님** — GPU 0 · 새 성능 판정 0 · Claim 등급 변경 0 · 엔진 코드 변경 0. 숫자는 전부 출처 라벨을
> 달았고([CANON]/[YAML]/[WORK]/[triage]/[CHAR]/[BOOT]/[ASSUME]), 정본이 인용 금지한 열은 쓰지 않았다.
> **rev2**: rev1에 대한 claims-auditor 감사(`reports/audit/2026-10-11_amd_design/AUDIT_amd_design_memo_rev1_2026-10-11.md`,
> `MIXED`, 死因 4건 D1–D4)를 전부 반영했다 — 반영 대조표는 §10. rev1의 "1차 결론"(ITL 비용 > TTFT 절감)·"§1-24 stall
> 소멸"·"모델-양극 ≈5% vs ≈75%"·"B의 ITL 경감"은 **철회**한다.
> 1차 계산표: `workspace/engine-port/results/amd_track/amd_firstorder_calc_2026-10-11.md`(재생 스크립트 동봉, 커널 시간
> 0건, **가정 하 가산 비용이지 상한도 예측도 아님**). 그림: `reports/figures/amd_firstorder_2026-10-11.png`.
> 사용자 결정 3건(2026-10-11): 산출 = 메모+계산 부록, 모델 = Nemotron-H 주·Zamba2 대극, 기판 = NVLink·PCIe 병기.

## 0. 요약

1. **전제 수정**: "AMD는 현존 기법이 아니다"는 틀렸다. NVIDIA/Harvard **SQD**(arXiv 2609.13134, 2026-09)가 decode를
   full-attention 층 vs Mamba+FFN 층으로 이미 분리했고, 그 실험 자체가 **8×B200 homogeneous GPU 프록시**(decode 풀을
   MPS로 16/148 SM 제한)에서 **full-attention을 prefill과 같은 GPU에 동거**시킨 형태다. 따라서 "homogeneous GPU"도
   "PD 결합 형태"도 남는 공백이 아니다. 남는 차별점은 **(iii) SLO goodput(TTFT≤SLO ∧ token-ITL p95≤SLO) 판정과
   (iv) 이 프로젝트가 정본화한 entanglement(admission 사슬)가 디바이스 경계로 어떻게 옮겨 가는지를 측정하는 것**뿐이다(§1).
2. **정본 §1-4의 entanglement는 decode SM 굶김이 admission 사슬을 거쳐 TTFT로 전파되는 것**이다(TTFT 차이의 95%+가
   공유 대기열에 있다는 것은 HE0 trace·Zamba2-2.7B·cap 48 범위의 산술). 층을 다른 GPU로 옮겨도 사슬은 **P→D 핸드오프와
   B-GPU의 state pool로 자리만 옮긴다**(§4-7) — 없어지는 것이 아니라 측정 대상이 바뀐다.
3. **AFD의 이득 원천(MoE sparsity → FFN 배치 집계)이 dense hybrid에는 없다.** 남는 후보는 footprint 이종성(KV는
   L-스케일, SSM state는 B-스케일·L-불변·**작지 않음**)을 두 풀로 나누는 것인데, 이것도 **메모리가 배치를 묶는 ctx
   영역에서만** 이득 후보가 되고(§4-1), 그 영역 밖에서는 1-GPU `max_running_requests` 상향과 구별되지 않는다(§5).
4. **L-B(사용자 구상: GPU-A = prefill + decode-attention, GPU-B = decode mamba+MLP)는 per-token ITL을 줄일 수 없다**
   (attention·mamba 직렬 의존, attention 시간은 A로 옮겨질 뿐). crossing(attention 호출당 왕복 2회: Nemotron-H 8회/step,
   Zamba2-2.7B 18회)이 ITL에 더해지고, A에 prefill과 decode-attention을 동거시키므로 **정본 §1-24의 monolithic-prefill
   stall을 다시 들여올 수 있다**(미측정, L-B의 1순위 위험, §4-2). PD-disagg 대조군 (i)는 이 stall을 없앤다.
5. **1차 계산은 결론이 아니라 뒤집힘 조건의 지도다**(§6.2). crossing은 step당 `2×(attention 호출 수)×(L0 + B·hidden·2B/BW)`를
   ITL 쪽에 더하고, L-B가 아끼는 것은 TTFT 쪽 KV 전송 1회다. 둘을 같은 ms로 비교한 R은 SLO 의미가 없으며, ctx 4096·출력
   256·NVLink·L0 65 µs[ASSUME]에서는 R≈0.002–0.017이지만 **ctx≥8k·짧은 출력·PCIe·작은 L0에서는 R>1로 뒤집힌다**(Zamba2-2.7B
   PCIe·10 µs·ctx 8192·출력 64 → 3.17; Nemotron-H PCIe·10 µs·ctx 16k·출력 64 → 1.02). 어느 축이 이기는지는 **V4 직접
   측정만** 답한다.
6. per-type decode 조성(attention 비중)은 모델 상수가 아니라 **(bs, ctx)의 함수**이고 저장소 자료는 전부 eager·n=1·미감사
   이며 원 파일이 조성 해석을 금지한다 → 이 문서는 **크기를 인용하지 않고** "같은 조건에서 MHA인 Zamba2의 attention 비중이
   GQA인 Nemotron-H보다 크다"는 방향만 가설로 쓴다(§2.2, §6.1).

## 1. 선행연구 — AMD가 서 있는 자리

| 선행 | 분리 축 | 이득 원천 | 기판 | AMD와의 관계 |
|---|---|---|---|---|
| **MegaScale-Infer** (arXiv 2504.02263) | decode attention(DP 복제) ∥ MoE-FFN(EP) | **MoE sparsity**로 memory-bound화한 FFN에 배치 집계 → compute-bound 복귀 | homogeneous/heterogeneous GPU | ping-pong 조건 `T_a≈T_e`, `T_c<T_f`, `m≥2(1+T_c/T_f)`(≥3 micro-batch). **per-token latency는 줄지 않고 늘 수 있다**(constraint 3, §7.2). dense 모델은 Fig.1(a)에서 FFN이 max batch에서 이미 포화 → **dense에 utilization 논거 없음**. prefill 이득 없음(e2e 최대 1.18×)[원문 확인 2026-10-11] |
| **Step-3** (arXiv 2507.19427) | AFD + MFA | MoE 321B | H800/H20/A800/Ascend | 동일 계열 |
| **AIC++ "How Far Can Disaggregation Go?"** (arXiv 2605.28302) | token mixer(attention **+ Mamba**) ∥ MoE-FFN | 설계공간 시뮬레이터(AIConfigurator+AstraSim) | 128×B200 모델링 | ★**Mamba를 attention 풀에 둔다**("attention/mixer cost") — attention↔Mamba 분리가 아님. aggregated가 throughput 대부분, AFD는 **latency 축**·memory cap 초과 long-context에서만 승[원문 확인] |
| ★**SQD** (arXiv 2609.13134, NVIDIA/Harvard) | **quadratic(full attention) ∥ subquadratic(Mamba/linear/SWA)+FFN** | SRAM-only ASIC의 ridge(8.2 FLOP/B)에 decode가 가깝다 → **tokens/J**·고정 전력 하 latency | **8×B200 프록시(decode 풀 MPS 16/148 SM)** + Rubin+LPX 분석 모델 | **가장 가까운 선행이며 기전·배치·기판 동질성을 이미 포함**: full-attention 층은 **prefill과 같은 GPU에 동거**(KV 공유), Nemotron 3 Ultra 12/108 dense층 → crossing은 attention 층에서만. SGLang 수정판 layer-split transport, one-sided put 64.6–130.7 µs[원문 확인]. homogeneous GPU에서의 **SLO goodput 이득은 주장하지 않음** |
| **DUET** (arXiv 2603.15530) | prefill 패키지 ∥ decode 패키지(하드웨어) | 위상 전용 마이크로아키텍처 | 칩렛 설계 | attention·Mamba는 **같은 패키지**(runtime-configurable). Nemotron-H-56B roofline: 두 층 operational intensity **유사** |
| **Lamina** (2405.01814) · **Adrenaline** (2503.20552) | dense 모델 decode attention 분리/오프로딩 | attention은 memory-bound → 메모리 디바이스/prefill 인스턴스로 | 이종 | Lamina: "층마다 통신 → 최적화 네트워크 스택 없이는 불가". ★Nemotron-H처럼 attention이 4/52층이면 AMD는 **KV-attention 오프로딩 계열로 퇴화** |
| vLLM hybrid SSM P/D disagg (2026-04-21, v0.20) · SGLang PD-disagg | prefill ∥ decode (Mamba state 전송) | — | — | 대조군 (i)의 공개 구현. vLLM AFD plugin(2026-07-23)은 DeepSeek MoE 한정 |

**AMD의 차별점(좁다)**: SLO goodput 판정과 admission 사슬의 이동 측정. 기전·배치·homogeneous 기판·PD 결합 형태는 SQD
실험에 포함돼 있다. 저장소 자체 판정도 대면해야 한다 — `reports/spatial_decoupling_design_review_2026-07-25.md` §1은
"§1-20 spatial-decoupling을 **독립 연구 기여로 추진하는 것은 비권장**(협공)"이고 "KV O(L) vs state O(1)은 **정의상 자명한
관찰**"이라 판정했다. §5의 "두 binder 분리"는 바로 그 자명한 framing이므로, 이 트랙의 가치는 framing이 아니라 **측정**
(사슬이 어디로 갔는가)에만 있다.

가정 목록(사실 아님): SQD 코드는 비공개로 가정 · VESSL 멀티-GPU 노드의 interconnect 미확인 · L0 65/130 µs는 B200 값 이식.

## 2. 정본·저장소 근거(재도출 아님)

### 2.1 AMD 판단에 직접 걸리는 정본 결론

| 정본 | 내용 | AMD에 주는 함의 |
|---|---|---|
| §1-4 entanglement (`reports/CONSENSUS.md` §1-4) | running batch·KV 공유 → decode 굶김→ITL↑→batch 정체→admission 차단→TTFT 폭발. 2026-10-02 스코프: TTFT 차이 95%+는 **공유 대기열**(산술 CONFIRMED; HE0 trace·Zamba2-2.7B·cap 48 범위), **cap 48 귀속은 NOT-YET-SUPPORTED**(E-1 미실행) | entanglement = decode SM 굶김이 admission 사슬을 거쳐 TTFT로 전파. 디바이스 분리는 SM 경합을 없애지만 사슬은 P→D 핸드오프로 이동(§4-7) |
| §1-24 | ITL 꼬리 = monolithic prefill이 decode를 stall(pdmux 하 chunked prefill assert 금지) | **prefill과 decode 경로를 완전히 분리한 레이아웃(대조군 (i), L-A의 제3 GPU prefill)에서만 사라진다.** L-B는 A에 prefill과 decode-attention을 동거시키므로 같은 기전을 다시 들여올 수 있다(미측정) |
| §1-3 layer-type 全形態 死 + `PROJECT_STATUS.md` "철회된 가설" 1번("attention/SSM layer별 static resource partition이 보편적으로 유리하다") | 죽은 것은 단일 GPU·한 forward 안 sub-step green-ctx 재분할(19 window, sync 직렬화, cudagraph 불가, TPOT 42→124/85 ms)과 **offline 고정 per-type 분할**(§1-15), 그리고 4모델 agnostic 4/4 승 | **AMD는 정본이 철회한 가설(층 종류별 static 자원 분할)과 같은 형식을 디바이스 경계로 옮긴 것이다.** 단일 GPU 반증은 디바이스 간 경우를 시험하지 않았으므로 "모순 아님"이 아니라 **"미시험"**이다. A의 green-ctx decode-attention 파티션은 층 종류별 SM 할당이므로 anchor §9("layer-type 런타임 정책 재개 금지")와의 관계를 명시한다: 런타임 정책이 아니라 정적 디바이스 배치이되, 철회 가설 1번의 디바이스 판이다. `workspace/characterization/reports/v2_report.md` §5[CHAR]의 "attention/SSM cannot be disaggregated(활성화 결합)"는 같은 GPU 안 진술이고 디바이스 간에서는 그 결합이 crossing 비용으로 나타난다(§6.2) |
| C2 (`PROJECT_STATUS.md` 8B 측정 노트, CONFIRMED scoped) | decode ITL SM16→92 2.36–2.91×, 4 arm 모델-무관; 원인 = weight-sweep 68–94%(B≈9–12·L≈1–1.5k). "hybrid는 KV 적다" 거짓(per-seq Ha8 601 > M8 260 > Hs8 117 > T8 70 MiB) | SSM state는 L-불변이지만 **작지 않다**. C2 수치의 2-GPU 이식 금지(attention-only decode는 별도 축) |
| §1-20 oracle | 108 SM 초과 가상 결합 구성(n=1–2, 절벽 위) → **"진짜 여유는 디바이스 사이" 인용 금지**(2026-10-02) | AMD 동기로 쓰지 않는다 |
| `reports/spatial_decoupling_design_review_2026-07-25.md` | §1: 2-GPU disaggregation 독립 기여 **비권장**(협공), constant-vs-linear는 자명한 관찰. §2: (a) 2 GPU = 자원 2배 → 가설 검정 아님 (b) SGLang v0.5.10 mooncake 경로가 conv+temporal state 전송 지원(`mooncake/conn.py` `_send_mamba_state`), **mooncake/nixl 미설치·RDMA NIC 필요** (c) `enable_pdmux`는 엔진 assert로 `disaggregation_mode`·chunked prefill과 **상호배타**(`server_args.py:6125-6134`) | **iso-GPU 대조군 필수**; 대조군 (i)는 엔진 내장 경로; L-B의 A 공존 스케줄러(pdmux+전송)는 새로 짜야 함 |
| HE0 (§1-7, 부호 CONFIRMED scoped; 범위 = A100·Zamba2-2.7B·ShareGPT 3↔12·cap 48·cudagraph ON·반응형 컨트롤러 3종) | 동적 제어가 best-static(d44)을 못 넘음(HI phase −6.8~−6.9%, 60 ms cliff 증폭 제외) | AMD는 "정적 배치 축 하나 추가"로만 서술, HE0 재논쟁 금지 |
| `reports/dual_worker_design.md`, `reports/r2_decoupling_review_2026-07-24.md` | true-dual = 1 프로세스·1 CUDA 컨텍스트·weights/pool/partition 공유; 전송·multi-GPU 코드 0 | AMD는 H-Architecture의 디바이스 확장판이지만 구현 출발점은 0 |

### 2.2 모델 구조·state 사실

| 모델 | 층 구성 | hidden | KV(K+V) B/token | mamba state/req(ssm+conv) | decode attn 비중 자료(미감사, 크기 인용 금지) | 출처 |
|---|---|---:|---:|---:|---|---|
| Nemotron-H-8B | 52 = M24 / attn **4** / MLP 24 | 4096 | 16,384 (GQA 8×128) | 97.5 MiB(분석식 97.46, 배너/49 97.38) | 짧은 ctx(≈240 이하)·bs32·eager·첫 arm: attn 1.165 / step 22.5 ms; ctx 5097: attn 1.796 / 22.1 ms — **ctx에 따라 움직이고**, attention은 ctx 5100에서 16 SM일 때 +43%(SM-평탄은 짧은 ctx 한정) | [YAML]; `triage/p1_4_nh_smsens_827583.txt`, `p1_4_nh_ctxsens_827617.txt`, `triage/notes.md:139`[triage] |
| Nemotron-Nano-9B-v2 | 56 = attn 4 / M27 / MLP 25 | 4480 | 16 KiB cell | **136.90 MiB [CANON]**(ssm+conv 합) | — | `PROJECT_STATUS.md` ~2290 |
| Zamba2-2.7B | mixer 54개(그중 9개 위치에 공유 attn+MLP 블록도 호출, 호출마다 별도 KV) | 2560 | **184,320**(MHA 32×160) | 69.2 MiB | bs32·ctx4096·108 SM: per_attn 4.14×9 vs per_mamba 0.20×54(깨끗한 셀이나 원 파일이 조성 해석 금지); bs8·ctx4092: attn 9.4 vs mamba 24.4 ms — **bs에 따라 뒤집힘** | `results/r0c/P5_GATES_BATCH1_2026-08-05.md` Table A; `triage/p1_4_zb_ctxsens_831012.txt`[WORK/triage, eager·n_indep=1] |
| Zamba2-7B | 81 mixer, 13개 위치에 공유 블록 | 3584 | 372,736(MHA 32×224) | 145.1 MiB | — | [YAML] |
| Granite-4.0-h-micro | 40 = 36 mamba + 4 attn, dense | 2048 | 8,192 | 72.9 MiB(배너/49) | **측정 무효(host-bound 의심)**: per-attn이 ctx 64→4000 평탄, per-mamba가 16→108 SM 평탄 | boot smoke[WORK]; `triage/p1_7_gr_smsens_832611.txt` |
| Falcon-H1-3B/7B | 32/44층 **매 층 attn∥mamba(병렬)** | 2560/3072 | — | ≈131 MiB(3B 배너/49) | per-type 비중 없음 | [YAML]; roster |

- per-type decode 조성은 **(bs, ctx)의 함수**이고 자료는 전부 eager·n_indep=1·claims-auditor 미통과·원 파일이 조성 해석 금지
  (`P5_GATES_BATCH1_2026-08-05.md:3-7` "no composition … claim is made or permitted"). 이 문서는 백분율을 인용하지 않고,
  "같은 (bs, ctx)에서 MHA인 Zamba2의 attention 비중이 GQA인 Nemotron-H보다 크다"는 **방향만 가설**로 쓴다. Zamba2
  ctx16k attn 비중(§1-5 인용 금지)·r0c (256,full) 셀(가장 오염된 셀)은 쓰지 않았다.
- 참고: prefill Zamba2-2.7B는 attn ∝ L^1.916, mamba ∝ L^0.954, per-call 교차 ≈2.75k(정본 rev2 스코프) — L-B는 prefill을
  분할하지 않으므로 이 축은 AMD와 무관하다(§3).

### 2.3 엔진·도구 사실

- multi-GPU·전송 코드 **0**(job script 전부 `gpu:1`, torchrun/NCCL/`--tp-size`/`--disaggregation*` 0건).
- 재사용 가능: `forward_split_prefill`(4모델 전부, `hidden_states`를 창 사이로 carry — **AMD crossing 경계와 같은 지점**),
  `la_coord_windows`(Zamba2만), `benchmarks/pdmux_eval/analyze.py`(goodput·paired bootstrap·round 합산), telemetry
  `runtime_snapshot`/`PDMUX_PHASE_EVENTS`, residency census, N2 Job-PROFILE(SM×batch×ctx, 미봉인). per-layer 타이머는
  eager 전용·NemotronH accumulator 미수리.
- `PDMUX_MODEL_PROFILE` 인스턴스 없음(스키마만). nsys/%smid 산출에 per-layer-type 분해 없음.
- 부팅 배너 "Mamba Cache … ssm_state size"의 단위는 GiB(`memory_pool.py:74`), 슬롯 수는 size+1 = 49(`memory_pool.py:255,269`).

## 3. 구성 정의

```
L-A  "AMD-decode"      GPU-A: decode attention 층 + KV        GPU-B: decode mamba+MLP + state     prefill: 제3 GPU(또는 A)
L-B  "PD+AMD"(SQD형,   GPU-A: prefill(전 층, weights 복제)     GPU-B: decode mamba+MLP(state 상주)
      사용자 구상)              + decode-attention(KV 상주,      ← hidden 왕복 2회 / attention 호출 →
                               작은 green-ctx 파티션)
대조군(iso-GPU, 2장)   (i) PD-disagg 1P+1D(KV+state 전송, SGLang 내장)  (ii) 2× PD-mux DP 복제(decode-heavy static)
                       (iii) TP=2 PD-mux(동작 미검증)                   (iv) 1-GPU PD-mux에서 max_running_requests 상향(용량 손잡이 대조)
```

- crossing/step = 2 × attention 호출 수: Nemotron-H **8**, Granite **8**, Zamba2-2.7B **18**, Zamba2-7B **26**,
  Falcon-H1은 매 층 두 분기 합산이 필요해 **정의 불가**(제외).
- L-B에서 KV는 A를 떠나지 않는다(전송 0). mamba state는 prefill 종료 시 **A→B 1회** 전송(또는 prefill의 mamba 층을 B에서
  돌리면 전송 0이지만 B에 prefill 동거가 생긴다 — 변형 L-B′, §4-2).
- ★**2 GPU vs 1 GPU 비교는 금지**(자원 2배). 모든 판정은 2 GPU iso 대조군 안에서만. (iv)는 "B 풀 수 손잡이"가 AMD 고유인지
  가르는 대조군이다(§5).

## 4. 오류 가능 지점

1. **이득 원천 부재(dense hybrid·homogeneous)**. MegaScale 논거(sparsity → 배치 집계)가 없다. 남는 후보 (a) footprint
   이종성: KV(L-스케일)는 A, state(B-스케일·L-불변)는 B로 분리해 각각 사이징; (b) C2가 보인 weight-sweep 지배 구간(68–94%)
   에서 B-GPU 배치 확대로 weight 상각 — 단 DP 복제도 같은 배치를 주므로 (a)(b) 모두 **메모리가 배치를 묶을 때만** 유효.
   80 GB A100에서 Nemotron-H 48 요청의 state는 4.6 GiB, KV(ctx4k)는 3 GiB — Nemotron-H ctx4k에서는 메모리가 배치를 묶지
   않는다(cap 48은 config 상수). 반면 Zamba2-2.7B ctx16k는 48 요청 KV가 48×2.88 GB ≈ 135 GiB로 80 GB를 넘어 **묶는다**.
   ⇒ 이득 후보 영역 = **KV가 묶는 긴 ctx × MHA 모델**뿐이고, 그 밖에서는 (a)(b) 모두 공집합이며 "B 풀 수 손잡이"는 1-GPU
   `max_running_requests` 상향(대조군 (iv))과 구별되지 않는다.
2. **L-B는 A에 PD 동거를 다시 만든다 — 1순위 위험.** Nemotron-H에서 attention GPU는 decode step의 작은 몫만 일하므로
   (크기 인용 금지, §2.2) 유휴를 메우려면 prefill을 얹어야 하고(SQD 배치) 그러면 정본 §1-24의 monolithic-prefill stall이
   A에서 재현될 수 있다. decode step은 attention 호출마다 A를 기다리므로 A의 stall이 **decode 전체로 번진다**. 게다가
   `enable_pdmux`와 `disaggregation_mode`는 엔진 assert로 상호배타라 A의 공존 스케줄러는 새로 짜야 하고 stall 회피
   보장이 없다. 대조군 (i)는 stall을 없애므로 비교 방향이 L-B에 불리하다. 변형 L-B′(prefill의 mamba 층도 B에서)는 B에
   prefill 동거를 만들어 entanglement를 B에서 재현한다.
3. **L-B는 per-token ITL을 줄일 수 없다.** attention·mamba 층은 직렬 의존이고 attention 시간은 A로 옮겨질 뿐 사라지지
   않는다. micro-batch 파이프라이닝이 없으면 L-B ITL ≥ PD-disagg decode ITL + crossing; 파이프라이닝(MegaScale m≥3)을
   넣으면 per-token latency가 늘어난다. 가능한 이득은 B의 **배치 용량**뿐이고 그것도 §4-1 조건에서만.
4. **crossing 지연과 cudagraph 운영점**. 정본 운영점은 cudagraph ON(TPOT 41→12 ms, `results/cudagraph_probe`). step
   중간의 디바이스 간 통신을 그래프 안에 캡처하지 못하면 decode가 eager로 떨어진다 — layer-aware를 죽인 기전("sub-step
   경계 = 실행비용", §1-15)의 디바이스 판이다. 이 프로젝트의 goodput은 ITL p95 **cliff(55–62 ms)** 에 민감(§1-7)하므로
   ITL 증가는 goodput에 비선형으로 반영된다. 계산(§6.2, [ASSUME]): L0 65 µs에서 step당 +0.53 ms(Nemotron-H)·+1.19 ms
   (Zamba2-2.7B), 10 µs에서 +0.09/+0.20 ms — 모델 공통 환산 눈금 12 ms 대비 0.7–10%. **캡처 불가는 단독 NO-GO**(§7).
5. **Falcon-H1(병렬 hybrid)은 AMD ill-posed**. 매 층 attn∥mamba를 합산하므로 층 단위 분리가 정의되지 않는다 —
   "layer-aware ≡ agnostic"(`triage/notes.md` P1.7) 판정의 거울.
6. **Zamba2 공유 attention 블록**: 호출 9/13회(crossing 18/26), MHA KV가 state의 **10×**(ctx4k 2.7B: KV 720 MiB vs state
   69 MiB) → **A의 KV 용량이 새 binder**(L-B의 KV는 여전히 A 한 장에 묶인다 — PD-disagg의 D GPU 대비 용량 이득은
   미제시), A에서 decode-attention이 prefill과 경합한다.
7. **admission 사슬 이동**. B의 state pool(`max_mamba_cache_size = max_running_requests`, CONSENSUS §1 행 23)이 binder →
   A가 prefill을 마친 요청을 B가 못 받으면 backlog → TTFT. 정본 §1-4의 "cap 48 귀속 NOT-YET-SUPPORTED"가 그대로
   미해결이며, AMD는 이 질문을 없애지 않고 디바이스 경계로 옮긴다.
8. **P→D state 전송은 안 없어진다**(Nemotron-H 97.5 MiB/req, Nano 136.9 MiB; NVLink 실효 200 GB/s 0.5 ms, PCIe 25 GB/s
   4.1 ms[ASSUME]). L-B가 없애는 것은 KV 전송뿐 — Nemotron-H ctx4k 64 MiB(0.34 ms), Zamba2-2.7B ctx4k 720 MiB(3.8 ms).
9. **iso-GPU 공정성·게이트 1**. 2-GPU 수치는 1-GPU 정본과 섞지 않는다. VESSL 멀티-GPU 노드의 interconnect(NVLink vs
   PCIe)와 RDMA NIC(mooncake 요구) 가용성은 **미확인**(`handoff-report/vessl_operating_model_2026-10-02.md`는 A100 SXM ×1만).
10. **엔진 공백**. layer-split transport·multi-GPU·state 전송 코드 0, mooncake 미설치. 구현 비용은 중단 사유가 아니라는
    사용자 결정(`handoff-report/session_handoff_2026-09-28.md:120`, `session_handoff_2026-10-02.md:20`)은 유효하나 correctness
    gate가 선행한다.
11. **측정 공백**. per-layer-type 조성이 전부 eager·n=1·미감사·조성 해석 금지이고 cudagraph ON per-layer 프로파일은 0건 →
    §6의 기대 효과는 전부 **가설**.
12. **novelty 협소**. SQD(기전·배치·homogeneous 프록시·PD 동거 포함)·Lamina/Adrenaline(attention 오프로딩)·AIC++(mixer/FFN)
    와의 차별점은 SLO goodput 판정과 admission 사슬 이동 측정뿐. 저장소 자체 판정(spatial review §1 "독립 기여 비권장")도
    같은 방향이다.

## 5. 의의(가설로 강등 — 무엇이 남는가)

- **[가설] 두 capacity binder의 분리**. KV(attention 층만, L-스케일)와 SSM state(B-스케일·L-불변·작지 않음)를 다른 풀에
  두면 각 풀을 독립 사이징할 수 있다 — Transformer AFD에는 없는 hybrid 구조(FFN은 stateless). 단 (1) 이 관찰은 spatial
  review §1이 "정의상 자명"이라 판정한 것이고, (2) 용량 이득은 **KV가 배치를 묶는 ctx 영역에서만** 가능하며(§4-1), (3) 그
  밖에서는 1-GPU `max_running_requests` 상향(대조군 (iv))과 구별되지 않는다. 가설로만 남긴다.
- **[가설] entanglement를 사이징 손잡이로**. 정본 §1-4의 admission 사슬을 "B 풀 수 × state/req"라는 자원으로 바꾼다.
  측정 가능한 질문: "B 풀을 k배 키우면 §1-4 사슬의 TTFT 성분이 얼마나 줄고 crossing의 ITL 성분이 얼마나 느는가" — 두
  성분이 같은 goodput 지표 위에서 교환된다. 이 질문이 AMD 고유인지는 대조군 (iv)가 가른다.
- **prefill-layer-alloc과의 결합 지점**(§9): A-GPU의 작은 decode-attention 파티션은 이 프로젝트의 SM 분할 기전(green-ctx,
  realized pin 검증, 변화 trace, goodput 정의)을 그대로 쓴다. attention-only decode의 SM 민감도는 **미측정**(C2는 전 모델
  decode)이라 새 프로파일 축이 필요하다.
- **4모델 roster가 자연 대조군**: interleaved(Nemotron-H·Granite·Zamba2) vs parallel(Falcon-H1), GQA(Nemotron-H·Granite)
  vs MHA(Zamba2). "AMD 효과 = KV/state 비 × attention 비중"(인자 2개)은 **모델 ≥3의 hold-out**에서만 판별된다(단일·2모델
  비교는 재매개화, anchor §7 + CONSENSUS §3 항목9 追記).

## 6. 현재 결과 기반 기대 효과(전부 가설 · 미측정)

### 6.1 모델별 방향

| 모델 | attention 비중(방향만) | KV vs state(ctx4k) | L-B 기대(방향) |
|---|---|---|---|
| **Nemotron-H / Nano** | GQA·4/52층 — 같은 조건에서 Zamba2보다 작다(가설) | state(97–137 MiB) ≫ KV(64 MiB) | A가 거의 prefill 전용 → §4-2 PD 동거 재현 위험이 그대로; KV 전송 절감 작음(0.34 ms/req, 1회), crossing 8회/step이 ITL에 가산. 용량 이득 영역(KV가 묶는 ctx) 없음 |
| **Zamba2-2.7B/7B** | MHA·9/13 호출 — 같은 조건에서 더 크다(가설), bs·ctx에 따라 뒤집힘 | KV(720 MiB) ≫ state(69 MiB) | KV 전송 절감 큼(3.8 ms/req)이나 crossing 18/26회(1.2–1.7 ms/step @65 µs) + A에서 decode-attention과 prefill 경합. ITL은 줄지 않는다(§4-3). 유일한 이득 후보 = ctx≥16k에서 KV가 배치를 묶는 영역(§4-1) |
| Granite-4-h-micro | 자료 무효(host-bound 의심) | state(73 MiB) > KV(32 MiB) | Nemotron-H와 같은 방향(가설) |
| Falcon-H1 | 정의 불가 | — | 제외 |

- §1-24 stall은 **대조군 (i)·L-A(제3 GPU prefill)에서만** 사라진다. L-B에 대해서는 미측정 위험(§4-2). AMD 효과로 귀속 금지.
- AMD 판정은 **워크로드 (bs, ctx) 분포(U×E 조성, `cost_model_design_space_2026-10-01.md` §1.0)** 를 고정하지 않으면
  성립하지 않는다 — anchor §7 타겟(긴 문서 요약, W2/W4 = 8K/64)이 1차 조건이다.

### 6.2 1차 계산(구조 상수 × 대역폭·L0 가정, 커널 시간 0건 — 가산 비용이지 상한·예측 아님)

`workspace/engine-port/results/amd_track/amd_firstorder_calc_2026-10-11.md` 요약(B=32, L=4096, NVLink 실효 200 GB/s):

| 모델 | crossing/step | 가산 지연/step @L0 10 / 65 / 130 µs[ASSUME] | ÷12 ms 눈금(모델 공통 환산) | KV 절감/req | state 전송/req |
|---|---:|---|---|---:|---:|
| Nemotron-H-8B | 8 | 0.090 / 0.530 / 1.050 ms | 0.8 / 4.4 / 8.8% | 0.34 ms | 0.51 ms |
| Zamba2-2.7B | 18 | 0.195 / 1.185 / 2.355 ms | 1.6 / 9.9 / 19.6% | 3.78 ms | 0.36 ms |
| Zamba2-7B | 26 | 0.290 / 1.720 / 3.410 ms | 2.4 / 14.3 / 28.4% | 7.63 ms | 0.76 ms |
| Granite micro | 8 | 0.085 / 0.525 / 1.045 ms | 0.7 / 4.4 / 8.7% | 0.17 ms | 0.38 ms |

- L0 65/130 µs는 SQD의 **B200** one-sided put 실측(64.6–130.7 µs)을 A100 가정으로 이식한 값이고, 12 ms 눈금은 cudagraph ON
  Zamba2-2.7B rate2 TPOT을 **모델 공통 환산**에 쓴 것이지 모델별 TPOT이 아니다. PCIe(실효 25 GB/s)면 바이트 항과 전송 열이
  8배(계산표 §1–2).
- **wall-time 합산 비율 R의 뒤집힘 격자**(계산표 §3): R = KV 절감 ÷ (출력 토큰 × step당 crossing). ctx 4096·출력 256·
  NVLink·65 µs에서 R≈0.002–0.017이지만 **ctx≥8k·짧은 출력·PCIe·작은 L0에서 R>1**: Zamba2-2.7B PCIe·10 µs·ctx 8192·출력 64
  → 3.17; PCIe·65 µs·ctx 16k·출력 64 → 1.47; Zamba2-7B NVLink·10 µs·ctx 16k·출력 64 → 1.65; Nemotron-H PCIe·10 µs·ctx 16k·
  출력 64 → 1.02. **R은 SLO 의미가 없다**(TTFT 쪽 1회와 ITL 쪽 매 step을 같은 ms로 더한 것) — ITL 축은 "÷눈금" 열, TTFT
  축은 전송 열이 각각의 가산 비용이고, 두 축이 goodput 위에서 어떻게 교환되는지는 **V4 직접 측정만** 답한다.
- 그림: `reports/figures/amd_firstorder_2026-10-11.png` — 좌: 모델별 step당 crossing 가산 지연 vs L0(기판 2종, L0가 결정적
  = V1 실측 대상), 우: ctx별 KV vs state 바이트(어느 모델·ctx에서 어느 쪽이 binder인지).

## 7. 검증 사다리(제안 — 실행은 사용자 결정, GPU 0, V1부터 결정 규칙은 사전등록 + 규칙층 감사 선행)

| 단계 | 질문 | 조건 | NO-GO(결과로 기록) |
|---|---|---|---|
| **V0** 기판 프로브 | VESSL 멀티-GPU 노드 존재? interconnect(NVLink/PCIe) p2p 대역폭·L0 실측, RDMA NIC(mooncake) 유무, substrate.json·clk 기록 | Workspace 프로브 | 멀티-GPU 노드 없음 → 트랙 보류 |
| **V1** crossing 마이크로벤치 | 같은 Job·A100 2장·cudagraph ON: hidden B×d bf16 왕복 L0·BW, **그래프 캡처 가능 여부**, B∈{8,32,48}, n≥4 | 사전등록(L0 문턱은 실측값을 §6.2 격자에 넣어 재계산하는 입력이지 B200 이식값이 아님) | **캡처 불가 → 단독 NO-GO**(L-B는 비운영점) |
| **V2** PD-disagg 대조군 bring-up | SGLang 내장 disagg(mooncake), KV+state 전송 실측, Nemotron-H·Zamba2 | correctness gate | 전송 실측이 계산표와 어긋나면 계산표 갱신(판정 아님) |
| **V2′** A 동거 stall 검사 | A에서 monolithic prefill(2–8k)+decode-attention 동시, §1-24 방식 stall probe(A_all/A_free), 대조 (i) | 같은 Job | stall 재현 → L-B 1순위 위험 확정 |
| **V3** L-B 프로토타입 | `forward_split_prefill` 경계 재사용한 layer-split transport, A의 green-ctx decode-attention 파티션 | correctness gate = 고정 seed greedy 토큰 일치 + logit 허용오차(byte-동일은 커널·reduction 순서로 도달 불가일 수 있음 — 기존 PD-mux arm 간 도달 가능성 먼저 확인) | 토큰 불일치 |
| **V4** iso-GPU goodput 비교 | 2-GPU λ\* 먼저 측정(게이트 6, cliff 밖) → anchor §7 타겟 W2/W4 trace, L-B vs (i)(ii)(iii)(iv), n≥4, 짝지은 CI, 한 비교의 arm은 한 Job 안 | 캠페인급; 효과 분해 가설(§5)은 모델 ≥3일 때만 | 3% 미만 → headline 아님 |

조성 재측정(Nemotron-H·Zamba2 같은 (bs, ctx) 격자, whole-module 버킷, 무작위 arm 순서, host-bound 판정 사전등록, n≥3)이
끝나기 전에는 per-type 백분율을 어디에도 인용하지 않는다.

## 8. 금지 서술(이 문서와 후속 문서에 적용)

"AMD는 현존 기법이 아니다"(SQD) · "homogeneous GPU/PD 결합이 남는 공백"(SQD 실험에 포함) · "진짜 여유는 디바이스 사이에
있다"(§1-20 금지) · "hybrid는 KV가 작다"(C2 반증) · "layer-type 정책 부활"(§1-3·anchor §9) · "HE0 재논쟁" · C2/λ\*/Diff B
수치의 2-GPU 이식 · Zamba2 ctx16k attn 비중(§1-5 금지) · **per-type decode 백분율 일체**(조성 재측정 전) · λ0 시간가중 열
(E2C-8′) · "2-GPU가 1-GPU를 이긴다" · §6.2 계산표를 "상한"·"예측"·"결론"으로 인용(가정 하 가산 비용) · §1-24 stall 제거를
AMD 효과로 귀속 · "L-B가 ITL을 줄인다" · "B 풀 수 손잡이는 AMD 고유"(대조군 (iv) 전).

## 9. prefill-layer-alloc과의 연결점(무엇이 이전되고 무엇이 안 되는가)

| 이전됨 | 이전 안 됨 |
|---|---|
| goodput 정의(TTFT≤SLO ∧ token-ITL p95≤SLO)·paired bootstrap·round 합산(`analyze.py`) | C2 수치(전 모델 decode 등량곡선 — attention-only decode는 별도 축) |
| 변화 trace·n≥4·cliff 회피·realized pin 검증·한 Job 안 arm 규약 | prefill Diff B(L-B는 prefill을 안 나눔) |
| telemetry(`runtime_snapshot`, `PDMUX_PHASE_EVENTS`)·residency census·§1-24 stall probe 방식 | λ\* 격자·rate 3↔12 trace(1-GPU Zamba2-2.7B 용량값, 2-GPU에서 재측정) |
| `forward_split_prefill` 창 경계 = crossing 경계; `la_coord_windows` 분해 | layer-aware sim 이득(fidelity ladder: sim=상한) |
| entanglement 기전(§1-4)을 **측정 대상**으로: 사슬이 어디로 갔는가 | HE0 판정(1-GPU 동적 제어 한정) |
| 4모델 roster(interleaved vs parallel, GQA vs MHA) = hold-out 구조(≥3모델일 때) | 1-GPU 운영 권고(decode-heavy static)의 2-GPU 외삽 |

## 10. rev1 → rev2 감사 반영 대조표

| 감사 항목 | rev1 | rev2 |
|---|---|---|
| D1 §0-5 "1차 결론" | R로 "ITL 비용 > TTFT 절감" 결론 | 철회. §0-5·§6.2를 뒤집힘 조건 지도로 교체, 계산표 §3 격자 추가 |
| D2 §1-24 stall 소멸 | "어떤 2-GPU 레이아웃에서나 사라짐" | (i)·L-A에서만. L-B는 재도입 위험(§4-2 1순위), V2′ 추가 |
| D3 모델-양극 백분율 | ≈5% / ≈75% / ≈7%, ctx256 역전 | 전부 삭제, 방향만 가설, Granite 무효, (256,full) 셀 미사용 |
| D4 "B의 ITL 가벼워짐" | 주장 | 철회, §4-3 신설 |
| entanglement 오인용·mamba pool·§1-3 범위·철회 가설 1번·§1-20 숫자·HE0 범위·포인터 4건·C2 행 비약 | — | §2.1 전면 수정 |
| 계산기 (i)–(viii) | /48, ±8%, Nano conv 이중계상, "상한", B200 L0를 "실측", 그림 공식 중복 | /49·0.5%·변이 대조, conv=0, "가산 비용", [ASSUME], calc 함수 호출 |
| novelty·spatial review 비권장 | (i)(ii) 공백 주장 | (iii)(iv)만, 비권장·자명 판정 대면(§1, §5) |
| 사다리 (a)–(f) | 연언 NO-GO, 규칙층 감사 V3 전, rate 3↔12, byte-동일 | 캡처 불가 단독 NO-GO, V1부터 사전등록, 2-GPU λ\* 먼저·W2/W4, 토큰 일치+logit 허용오차, V0에 RDMA, (iv) 추가 |
| 논리 공백(§4-1 공집합 범위·손잡이 고유성·A KV 묶임) | — | §4-1·§4-6·§5 가설 강등·대조군 (iv) |

## 부록 A. 출처 라벨 범례

[CANON] `PROJECT_STATUS.md`/`reports/CONSENSUS.md`/`reports/`(paper 제외) · [YAML] `workspace/shared/configs/models.yaml`
(config_status verified) · [WORK] `workspace/engine-port/results/**` 미감사 작업 파일 · [triage] `workspace/engine-port/triage/`
· [CHAR] `workspace/characterization/`(활성 소스, 정본 아님) · [BOOT] 서버 부팅 배너 역산(GiB, /49) · [ASSUME] 대역폭·지연 가정
· [원문 확인] 2026-10-11 alphaXiv 본문으로 메인 세션이 확인한 선행연구 수치.
