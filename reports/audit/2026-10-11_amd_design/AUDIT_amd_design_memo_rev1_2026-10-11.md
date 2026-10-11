# claims-auditor 감사 — `reports/amd_design_discussion_2026-10-11.md` rev1 (2026-10-11, 읽기 전용, GPU 0)

> 산출 주체: claims-auditor 서브에이전트(메인 세션이 보고문을 그대로 보존). 대상 = 메모 rev1 + 계산기 rev1 +
> 그림 스크립트. **총평 `MIXED`, 死因 4건(D1–D4)·비치명 다수.** 처방은 같은 날 메모 rev2·계산기 rev2에 반영했다
> (반영 대조표는 메모 rev2 §10). 이 파일은 rev1 본문을 기준으로 한 감사 원문이므로 줄 번호는 rev1 기준이다.
> 감사자 자신이 명시한 한계: 웹 접근 없음(선행연구 수치는 내부 정합만 검사). 메인 세션은 선행연구 원문을 같은 날
> alphaXiv 본문으로 확인했다(MegaScale e2e 1.18×·Fig.1(a), SQD 64.6–130.7 µs·16/148 SM, AIC++ Mamba 배치).

## 총평: `MIXED`

메모는 "논의 문서"를 표방하지만 실제로는 세 가지를 몰래 들여온다: (1) 근거가 지지하지 않는 "1차 결론",
(2) 정본 §1-24에 대한 틀린 함의, (3) 원 파일이 조성 해석을 명시적으로 금지한 미감사 수치를 "방향"이라는 이름으로
6회 이상 인용. `check_citation_stops.py --file`은 4개 파일 모두 위반 0건(숫자 패턴만 잡음). `C --selftest` PASS,
렌더 결과는 표와 byte-동일.

## 死因

**D1. §0-5 "ITL 축 비용 > TTFT 축 절감이 1차 결론" — `REFUTED`(근거로서).** R은 "goodput과 무관"(§6.2)이라
적어 놓고 §0-5가 그 R로 결론을 냄(내부 모순). 결론은 고른 점(ctx 4096, OUT_TOKENS=256[미검증], NVLink, L0 65 µs)의
아티팩트. anchor §7 타겟(긴 문서 요약, W2/W4 = 8K/64)에서는 R>1로 뒤집힘(B=32): Zamba2-2.7B PCIe·10 µs·ctx 8192·
출력 64 → R=3.17; PCIe·65 µs·ctx 16k·출력 64 → 1.47; Zamba2-7B NVLink·10 µs·ctx 16k·출력 64 → 1.65; Nemotron-H
PCIe·10 µs·ctx 16k·출력 64 → 1.02. 메모는 뒤집힘 조건을 적지 않음. 처방: §0-5 교체 + 계산표 §3에 뒤집힘 격자.

**D2. "§1-24 stall은 어떤 2-GPU 레이아웃에서나 사라진다" — L-B에 대해 `REFUTED`.** §1-24 stall은 PD-mux 아래
monolithic prefill이 decode를 멈추는 현상(chunked prefill assert 금지). L-B는 GPU-A에 prefill(전 층)+decode-attention을
동거시키므로 정확히 그 구조이고, decode step이 attention 호출마다 A를 기다리므로 A의 stall이 decode 전체로 번진다.
`enable_pdmux`와 `disaggregation_mode`는 엔진 assert로 **상호배타**(`server_args.py:6125-6134`) → A의 공존 스케줄러는
새로 짜야 하고 stall 회피 보장 없음. 대조군 (i) PD-disagg는 stall을 없애지만 L-B는 다시 들여옴. 처방: 해당 문장 교체,
§4에 1순위 위험으로 추가.

**D3. "모델-양극(≈5% vs ≈75%)" — 크기·"양극"은 `NOT-YET-SUPPORTED`, 방향은 `PLAUSIBLE`.** (a) 전사는 맞음
(827583: 14.129/7.217/1.165, 16 SM 37.127/15.086/1.085; r0c (4096,full) per_attn 4.1406·per_mamba 0.1995). (b) "≈75%"는
분모 미명시(attn+mamba 77.6% / 버킷 합 72.8% / fwd_ms 71.6%). (c) 조건이 다름: Nemotron 5%는 ctx≈240 이하·eager·첫
arm(warm-up 편향), Zamba2는 ctx 4096·bs 32. 조건 맞춘 자료가 크기를 반박: Nemotron ctx 5097 → 8.1%
(`triage/p1_4_nh_ctxsens_827617.txt`), Zamba2 bs 8·ctx 4092 → attn 9.4 vs mamba 24.4 ≈ 28%(`triage/p1_4_zb_ctxsens_831012.txt`)
— 조성은 (bs, ctx)의 함수. (d) 원 파일 `results/r0c/P5_GATES_BATCH1_2026-08-05.md:3-7`이 "UNAUDITED … No knee, no
composition, no performance/policy claim is made or permitted"라 명시. (4096,full)은 깨끗한 셀이나 "ctx256 역전"에 쓴
(256,full)은 **가장 오염된 셀**(per_mamba 2.78× 부풀림). (e) "크기는 인용하지 않는다"고 쓰고 6곳 인용. (f) Nemotron
"attn SM-평탄"은 짧은 ctx 한정(ctx 5100에서 16 SM +43%, `triage/notes.md:139`). (g) Granite ≈7%는 측정 실패 징후
(per-attn ctx 64→4000 평탄, per-mamba 16→108 SM 평탄: host-bound 의심). 처방: 백분율 전부 삭제, 방향(같은 조건에서
MHA Zamba2의 attention 비중 > GQA Nemotron-H)만 가설로, "ctx256 역전" 삭제, Granite 비중 열 "측정 무효".

**D4. "B의 mamba-only ITL은 가벼워짐" — `REFUTED`.** ITL은 토큰 하나가 forward 전체를 통과하는 시간이고 attention·
mamba는 직렬 의존. attention 시간은 A로 옮겨질 뿐 사라지지 않으므로 micro-batch 파이프라이닝 없이는 L-B ITL ≥ PD-disagg
decode ITL + crossing. 파이프라이닝을 넣으면 MegaScale처럼 per-token latency 증가. 처방: "L-B는 per-token ITL을 줄일 수
없다(직렬 의존). 가능한 이득은 B의 배치 용량뿐이고 §4-1에 따라 메모리가 배치를 묶을 때만."

## 영역별 판정(요지)

1. **정본 충실도** — CONFIRMED: §1-2 41→12 ms · §1-3/§1-15 19 windows/sync/cudagraph/42→124/85 · §1-7 HI −6.8~−6.9%·
   cliff 55–62 ms · C2 2.36–2.91×/68–94%/B≈9–12·L≈1–1.5k/601>260>117>70 MiB · spatial review §2 (a)(b)(c) · dual_worker
   공유 상태 · Nano 136.90 MiB. **오인용(비치명)**: "entanglement는 SM 경합이 아니라 admission 사슬" → 정본 사슬은
   decode **굶김(SM)**에서 시작, 95%+는 HE0 trace·Zamba2-2.7B·cap 48 범위 산술 · "running batch·KV/**mamba pool** 공유"
   → mamba pool 삭제(cap 48 귀속 NOT-YET-SUPPORTED 우회) · §1-3 "모순 아님" → PROJECT_STATUS:9129 철회 가설 1번
   ("attention/SSM layer별 static resource partition이 보편적으로 유리하다")과 대면해 "미시험"으로 · §1-20 숫자(+16%·116)
   caveat 없이 재현 → 숫자 삭제 · HE0 범위 누락 · 포인터 4건(§3 항목23→§1 행 23; anchor §8→§7+CONSENSUS §3 항목9 追記;
   anchor §1.0→cost_model §1.0; "anchor 결정 ②"→session_handoff_2026-09-28.md:120/2026-10-02.md:20) · C2 행
   "B-GPU binder는 state pool" 비약(§4-1과 모순) → 삭제.
2. **미감사 수치 승격** → D3.
3. **계산기** — CONFIRMED: KV 공식·Zamba2 호출별 KV(`zamba2.py:223-235`)·7B 81 mamba·crossing 2×호출. **결함(비치명)**:
   (i) 배너 단위 GiB(`memory_pool.py:74`)·슬롯 size+1=49(:255,269) → /48은 +2.08% off-by-one(49면 전 모델 0.01% 내 일치)
   (ii) selftest ±8%는 GiB/GB(7.4%)·+1 슬롯을 못 가림(교훈 9/53) → 0.5% (iii) Nano 136.90은 ssm+conv 합인데 conv 이중계상
   (138.8) (iv) "상한" 오라벨 → "가정 하 가산 비용(경계 아님)" (v) L0 65/130 µs는 B200 이식인데 "실측"이라 표기, TPOT 12 ms
   눈금을 Nemotron-H에도 적용 → [ASSUME] 라벨·"모델 공통 환산" (vi) R은 항등식 아님, 면책은 충분하나 §0-5가 무시(D1)
   (vii) 그림 스크립트가 공식 인라인 재구현 → calc 함수 호출 (viii) Zamba2 공유 블록 concat[hidden, original] — L0 지배라
   수치 영향 무시, 주석.
4. **선행연구(내부 정합)** — SQD 행 자체가 "8×B200 프록시(MPS 16/148 SM)"와 "full-attention은 prefill과 동거"를 적고
   있으므로 homogeneous GPU·PD 결합 형태는 SQD 실험 안에 이미 있음 → novelty 좌표 (i)(ii) 불성립, 남는 것은 (iii) 지표·
   (iv) 측정 대상. spatial review §1의 "2-GPU disaggregation 독립 기여 비권장(협공)"·"constant vs linear 자명" 판정과
   대면 필요(§5 "두 binder 분리"가 그 자명 framing). "SQD 코드 비공개(가정)"는 가정 목록으로.
5. **게이트 준수** — HE0 재논쟁 없음·기판 혼합 없음·λ0/E2C-8′/ctx16k 미사용(CONFIRMED). layer-type 런타임 정책 재개는
   아님(PLAUSIBLE 조건부: A의 green-ctx decode-attention 파티션은 층 종류별 SM 할당이므로 철회 가설 1번과의 관계 명시).
   사다리: (a) V1 NO-GO가 연언 → 캡처 불가 단독 NO-GO, 65 µs 문턱은 B200 이식이라 자유 표면 (b) V1·V2 결정 규칙이 규칙층
   감사보다 앞 → V1부터 사전등록 (c) V4 rate 3↔12는 1-GPU Zamba2-2.7B 용량값 → 2-GPU λ\* 먼저(게이트 6), anchor §7 타겟
   W2/W4 (d) "한 비교 arm은 한 Job"·substrate.json/clk 기록 누락 (e) 모델 2개로 인자 2개 가설 판별 불가(재매개화) (f) V3
   "출력 byte-동일"은 도달 불가일 수 있음 → 고정 seed greedy 토큰 일치+logit 허용오차, 기존 PD-mux arm 간 도달 가능성 먼저;
   V2 mooncake는 RDMA NIC 필요(spatial review §2) → V0에 추가; TP=2+pdmux 동작 미검증.
6. **논리 공백** — §4-1(b) "현 격자에서 공집합"은 Nemotron ctx4k만 참(Zamba2-2.7B ctx16k: 48×2.88 GB=135 GiB>80 GB) ·
   메모리가 안 묶는 영역에선 (a) 독립 사이징도 공집합 · cap 48이 config 상수면 "B 풀 수 손잡이"는 1-GPU
   `max_running_requests` 상향으로도 존재 → §5 의의는 AMD 고유 아님 → 가설로 강등 + 대조군 (iv) "1-GPU PD-mux,
   max_running 상향" 추가 · L-B의 KV는 여전히 A 한 장에 묶임(용량 이득 미제시).

## 확정에 필요한 실험(사전등록 대상, 실행은 사용자 결정)

1. 같은 Job·A100 2장·cudagraph ON에서 crossing 마이크로벤치(hidden B×d bf16 왕복 L0·캡처 가능 여부, B∈{8,32,48}, n≥4);
   캡처 불가 시 그 자체로 정지. 2. L-B의 A 동거 stall 검사(monolithic prefill 2–8k + decode-attention 동시, §1-24 방식
   A_all/A_free, 대조 (i)). 3. 조성 재측정(Nemotron-H·Zamba2 같은 (bs, ctx) 격자, whole-module 버킷, 무작위 arm 순서,
   host-bound 판정 사전등록, n≥3) — 그 전까지 per-type 백분율 인용 금지. 4. V4: 2-GPU λ\* 먼저, iso-GPU (i)(ii)(iii)+(iv),
   W2/W4 trace, n≥4, paired CI, 3% 게이트; 효과 분해 가설은 3모델 이상 hold-out일 때만.

## 死因 / 비치명

死因(4): D1 · D2 · D3 · D4. 비치명(주요): 배너 /48 off-by-one·무력한 selftest · Nano conv 이중계상 · "상한" 오라벨 ·
B200 L0 이식·단일 TPOT 눈금 · entanglement 오인용 · §1-3 범위 축소·철회 가설 미대면 · §1-20 숫자 재현 · HE0 범위 누락 ·
포인터 4건 · C2 행 비약 · SQD novelty 과장 · spatial review 비권장 미대면 · V1 연언 NO-GO·규칙층 감사 위치 · 2-GPU λ\*
미측정 · 2모델 인자 판별 불가 · byte-동일 게이트 도달성 · mooncake RDMA 확인 · Granite 비중 무효 · Nemotron "SM-평탄"
짧은 ctx 한정 · 그림 공식 중복. 반증 실패(계산상 CONFIRMED): KV·state 공식, Zamba2 호출별 KV, 81 mamba, crossing 횟수,
전사값. **성능 함의는 위 실험 전에는 어느 것도 CONFIRMED가 아니다.**
