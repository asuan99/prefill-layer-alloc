# 사전등록 — **Nemotron-H 층 타입 계측 이식** (`_zt` → `nemotron_h`)

2026-08-28 · ★**규칙층 초안 — 미감사**(게이트 #34 **1단계 대기**) · GPU **0** · 미제출 ·
새 성능 판정 **0건** · 정본 편집 0건

> 금지: *"이식이 규칙층을 통과했다"* · *"Nemotron Diff A를 쟀다"*.
> ★게이트 #34는 **규칙 먼저, 하네스(=이식 코드) 나중**이다. 이 문서가 통과하기 전에 코드를 쓰지 않는다.

## 0. 왜 이식 외에 길이 없는가 — 전제가 실측으로 확정됐다
`../../../deprecated_v2/README.md` §1: **두 모델 계열의 동작 가능 백엔드가 서로소**다
(Nemotron-H+triton = 부팅 거부 `server_args.py:1959` / Zamba2+flashinfer = 스케줄러 사망,
`illegal memory access`, **2/2 재현**, job 896776).
⇒ *"모델 고정, 백엔드만 변경"* 셀이 **없다** ⇒ 백엔드 효과와 모델 효과는 **분리 불가**
⇒ **정본 triton Diff A를 Nemotron arm으로 이전할 수 없고, 그 arm에서 직접 재는 수밖에 없다.**

## 1. 이식 대상 — 훅 지점은 Zamba2보다 깨끗하다
`nemotron_h.py:554` `ALL_DECODER_LAYER_TYPES`가 층 타입마다 **독립 클래스**를 준다:

| pattern 문자 | 클래스 | 라인 | 버킷 |
|---|---|---|---|
| `*` | `NemotronHAttentionDecoderLayer` | :510 | `attn` |
| `M` | `NemotronHMambaDecoderLayer` | :377 | `mamba` |
| `-` | `NemotronHMLPDecoderLayer` | :293 | `mlp` |
| (미사용) | `NemotronHMoEDecoderLayer` | :341 | `moe` — Nano-9B엔 없음, **버킷만 예약** |

Zamba2는 하나의 합성 블록 **안에서** `_zt_begin/_zt_end`로 구간을 갈랐다(`zamba2.py:339-356`).
Nemotron은 **클래스 경계 = 층 타입 경계**이므로 각 `forward` 진입/이탈에 한 쌍만 두면 된다.
⇒ Zamba2의 **버킷 비대칭 결함이 구조적으로 재발하지 않는다**(§2-a).

## 2. 재발시키지 말아야 할 정본 등재 결함 2종
정본이 2026-08-04에 등재한 계측 결함(`CONSENSUS.md` §1-3 정정 · `layertype_dynamic_*`):

**(a) 버킷 비대칭** — Zamba2의 `_zt('attn')`은 RadixAttention **코어만**, `_zt('mamba')`는 mixer **전체**를
쟀다. ⇒ Nemotron 이식은 **층 전체(forward 진입~이탈)**로 **양쪽 동일**하게 잰다.
검증: 버킷 합 / 총 forward 시간 = `closure`를 보고하고 **≥0.95를 요건으로 등록**한다.

**(b) 누산기 미리셋** — `_zt_acc`가 리셋되지 않고 하네스가 `tail -1`을 취해 **모든 보고값이 영구히
cold start를 포함**했다. ⇒ 현행 zamba2는 이미 **BLOCK 필드**(`b_per_*`, 매 emit 리셋)로 수리했다.
Nemotron 이식은 ★**BLOCK 필드만 방출하고 legacy 러닝평균 필드를 아예 만들지 않는다.**
(이 세션이 legacy 필드를 잘못 읽었다가 정정한 이력이 있다 — 필드를 없애면 그 실수가 불가능해진다.)

**(c) 신설 — 태그 fail-loud 승계**: 태그는 **`ZNPT2`**(Zamba의 `ZBPT2`와 다르다). 모델이 다르면
양이 다르므로 **Zamba 파서가 Nemotron 로그를 조용히 읽는 일이 없어야 한다.**

## 3. ★결정적 등록 — 이것은 정본 Diff A와 **같은 양이 아니다**
| | Zamba2(정본) | Nemotron(이식) |
|---|---|---|
| `attn` 버킷 내용 | 합성 블록의 attn 구간(adapter·LoRA·o_proj 포함) | **attention 디코더 층 전체** |
| `mlp` 위치 | 블록 **안** | **독립 층**(56층 중 25층) |
| 층 수 | attn 9 / mamba 54 | attn **4** / mamba **27** / mlp **25** |
| 백엔드 | triton | **flashinfer**(강제) |

⇒ **`Diff A_nemotron`은 새 기호**이며 정본 `Diff A`와 **수치 비교 금지**다.
비교 가능한 것은 **부호와 L-의존 방향**뿐이다(*"L이 늘면 열리는가"*).
★그리고 **네 축이 동시에 다르다**(버킷 내용·mlp 위치·층 수·백엔드) — confound #10이지만
§0에 의해 **분리 불가능**하므로, 이 사전등록은 그것을 **해소하지 않고 등록**한다.

## 4. 정확성 게이트 (engine-porter, 측정보다 먼저)
1. **계측 OFF에서 출력 byte-identical** — `SGLANG_NEMO_TIMING` 미설정 시 생성 토큰이 이식 전과 동일.
2. **OFF 경로 무비용** — 훅이 `if _ZT["on"]:`로 단락되어 OFF에서 분기 외 연산 0.
3. `closure ≥ 0.95`(§2-a) — 미달이면 **버킷 정의가 층을 못 덮은 것**이므로 측정 중단.
4. `hybrid_override_pattern`에서 센 층 수와 버킷별 관측 층 수가 **일치**(attn 4 / mamba 27 / mlp 25).
5. **cudagraph-ON에서 통과** — 운영점이다. 캡처가 깨지면 이식 실패로 본다.
6. CPU 회귀 통과 · `sync_engine_tree.sh` manifest 갱신.

## 5. 결정 규칙 (측정 후)
| 라벨 | 조건 |
|---|---|
| `DIFFA_OPENS` | `Diff A_nemotron`(steady, BLOCK 필드)이 L 2000→8000에서 **단조 증가** ∧ 증가폭 ≥1.5× |
| `DIFFA_FLAT` | 같은 구간 변화가 ±20% 이내 |
| `DIFFA_INCONCLUSIVE` | 그 외 |
| `CLOSURE_FAIL` | `closure < 0.95` — 측정 실패(게이트 #21), 판정 아님 |
| `INSTRUMENTATION_ABSENT` | `ZNPT2` 0줄 — ★job 896776에서 실제로 겪은 실패 모드 |

★**성능·정책 주장 0건.** 이 측정은 R2′의 계수를 **그 arm에서 재는 것**이지 정책 비교가 아니다.

## 6. 설계층 도달가능성 (게이트: `presubmit.py`)
이식 후 `design_reachability.py` spec을 작성해 **`DISCRIMINATING`을 확인한 뒤** 본 측정을 제출한다.
`NOTHING_PURCHASABLE`·`SINGLE_LABEL_FORCED`·`RESTRICTIONS_INERT`면 제출하지 않는다.

## 7. 닫지 못하는 것
1. **백엔드 효과 분리** — §0에 의해 원리적으로 불가.
2. **정본 Diff A와의 연속성** — §3에 의해 수치 비교 불가. 정본 21× 진폭·교차점 ≈3k는 **이전되지 않는다.**
3. **R2′ 자체** — 계수를 재는 것이지 *"강등 길이 예측이 goodput을 올린다"*를 검정하지 않는다.
4. **Diff B** — 이 사전등록은 **비용비(Diff A)만** 다룬다. SM 민감도비는 SM 스윕이 따로 필요하다.

## 8. 금지 문장
*"이식이 규칙층을 통과했다"* · *"Nemotron Diff A가 정본보다 크다/작다"*(수치 비교 금지) ·
*"백엔드가 Diff A를 X% 바꾼다"*(분리 불가) · *"정본 교차점 ≈3k가 이 arm에서도 성립한다"* ·
*"`boot_ok=1`이므로 측정됐다"*(job 896776이 그 상태로 죽었다).

## 9. 1단계 감사에 묻는 것 — 단 하나
> **§3의 "같은 양이 아니다" 등록이 충분한가, 아니면 이 이식은 비교 불가능한 것을 비교하려는 것인가?**
합격 기준: ① §2의 결함 3종 방지책이 실효인가(특히 legacy 필드 **제거**가 과한지) ② §4 정확성 게이트가
이식 실패를 잡을 수 있는가 ③ §5 결정 규칙에 항등식·도달불가 라벨이 있는가 ④ §7이 정직한가.
