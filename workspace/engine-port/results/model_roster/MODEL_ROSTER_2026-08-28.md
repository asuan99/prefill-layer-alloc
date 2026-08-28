# 모델 로스터 — 비교연구 후보와 **엔진 지원 필터**

2026-08-28 · GPU 0 · **성능 판정 0건** · 정본 아님(작업 목록)

## 0. 왜 이 문서가 있나
TC1 P3(job 896689/896690)가 *"Qwen2.5-3B는 ShareGPT에서 컨트롤러 결정 지점을 만들지 않는다"* 로
죽으면서, **긴 프롬프트 + 크기 정합**이 동시에 필요해졌다. 그 조건으로 모델을 고르려니
**어떤 모델이 이 엔진에서 도는지가 저장소 어디에도 적혀 있지 않았다.** 이 문서가 그것이다.

★**이 문서는 성능을 비교하지 않는다.** 크기·구조·컨텍스트·지원 여부만 적는다.

## 1. 하드 필터 — `sglang_engine_dev` v0.5.10 트리가 지원하는 hybrid 아키텍처
`sglang/srt/models/` 직접 열람(2026-08-28):

```
zamba2 · nemotron_h (+_mtp) · granitemoehybrid · falcon_h1 · qwen3_next · jet_nemotron · lfm2 (_moe) · mamba2
```

★**`jamba.py` 없음 · `BambaForCausalLM` 없음.** Jamba·Bamba 계열은 **이 엔진에서 돌지 않는다** —
쓰려면 engine-porter가 모델 지원을 이식하고 **correctness gate를 따로 통과**해야 한다.
탐색 결과가 아니라 **차단 조건**이므로 맨 앞에 적는다.

## 2. 로스터 (파라미터는 safetensors 실측, bf16 기준)

| 모델 | 클래스 | 실측 params | ctx | 구조 | 로컬 | 역할 |
|---|---|---|---|---|---|---|
| **NVIDIA-Nemotron-Nano-9B-v2-Base** | `nemotron_h` ✅ | ~9B | **128K** | **interleaved** · Mamba-2 다수 + attn 4층 | 2026-08-28 수신 | ★**긴 ctx hybrid arm 1순위** |
| **Qwen/Qwen2.5-7B** | `qwen2` ✅ | **7.62B** | 131K | pure Transformer | ✅ | ★**대조 arm**(정본 `T8`) |
| nvidia/Nemotron-H-8B-Base-8K | `nemotron_h` ✅ | **8.10B** | **8192** | interleaved · 52층 중 attn 4층(≈8%) | ✅ | 정본 `Hs8` · 8K 상한 대안 |
| Zyphra/Zamba2-2.7B | `zamba2` ✅ | **2.69B** | **4096** | interleaved(shared-attn) | ✅ | ★정본 정책 arm(HE0 전체) |
| Zyphra/Zamba2-7B-Instruct | `zamba2` ✅ | **7.41B** | **4096** | interleaved | ✅ | 정본 `Ha8` · 긴 ctx 불가 |
| **tiiuae/Falcon-H1-7B-Instruct** | `falcon_h1` ✅ | **7.59B** | **262K** | ★**블록 내 병렬**(모든 층이 attn+SSM 혼합) | ✅ | ★**§5 참조 — 배제 아님, 보류** |
| tiiuae/Falcon-H1-3B-Base | `falcon_h1` ✅ | 3.15B | 131K | 블록 내 병렬 | ✅ | 정본 4모델 arm |
| ibm-granite/granite-4.0-h-micro-base | `granitemoehybrid` ✅ | 3.19B | 131K | interleaved(Mamba:attn ≈ 9:1) | ✅ | 정본 4모델 arm |
| ibm-granite/granite-4.0-h-tiny | `granitemoehybrid` ✅ | 7B total / **1B active** | 128K | interleaved + **MoE** | ✗ | ★§4-2 — 크기 정합 함정 |
| mistralai/Mamba-Codestral-7B | `mamba2` ✅ | 7.29B | — | **pure SSM** | ✅ | 정본 `M8` 음성대조 |
| Qwen/Qwen2.5-3B | `qwen2` ✅ | 3.09B | 32K | pure Transformer | ✅ | TC1 P3에서 사용(死) |

## 3. 구조 분류 — 이 프로젝트에서 왜 중요한가
이 프로젝트의 창립 축은 **층 타입**이다. 그래서 hybrid를 한 덩어리로 묶으면 안 된다.

- **(A) interleaved / 층 교대** — attn 층과 Mamba 층이 **깊이 방향으로 분리**돼 있다.
  Zamba2 · Nemotron-H · Nemotron-Nano · Granite-4.0-h-*.
  ⇒ *층 타입*이라는 개념이 성립하는 유일한 부류이고, `la_coord_windows()`·Diff A/B가 정의되는 곳이다.
- **(B) 블록 내 병렬** — **모든 층**이 attn과 SSM을 함께 갖는다. Falcon-H1.
  ⇒ 층 타입 축이 **정의상 없다**. 정본이 S0에서 *"Falcon-H1은 단일 타입이라 정의상 layer-aware ≡ agnostic"*
  이라 적은 것이 이 뜻이다.
- **(C) pure SSM** — Mamba-Codestral. 음성대조.
- **(D) pure Transformer** — Qwen2.5-*. 양성대조.
- ★**직교 축: dense vs MoE.** 이건 (A)–(D)와 별개이고 **크기 정합을 깨뜨린다**(§4-2).

## 4. 지금 쓰지 않는 것과 사유
1. **Jamba · Bamba** — ★**엔진 미지원**(§1). 성능·적합성 문제가 아니라 **못 돈다**.
   쓰려면 모델 이식 + correctness gate가 선행한다. **다음 세션이 이 줄을 근거로 재탐색하지 말 것.**
2. **granite-4.0-h-tiny** — 7B **total / 1B active MoE**. dense Qwen2.5-7B와 붙이면
   *"크기를 맞췄다"* 가 **total로도 active로도 성립하지 않는다** ⇒ C2b를 죽인 것과 같은 부류의 교락
   (파라미터·형상이 동시에 다름). **dense hybrid가 필요하다.**
3. **Zamba2 전 계열** — ctx **4096**(1.2B/2.7B/7B 전부 확인). **긴 프롬프트 설계에 쓸 수 없다.**
   ★정본 정책 결과(HE0·얽힘·positioning)가 전부 여기 있으므로, 긴 ctx로 가는 순간
   **정본 정책 결과와의 직접 연결이 끊긴다** — 그 대가를 설계에 명시해야 한다.

## 5. ★Falcon-H1 — 배제가 아니라 **보류**, 그리고 어디에 쓸 것인가
사용자 지시(2026-08-28): *"완전히 배제시키는 게 아니라 다른 비교연구에서 쓸 수 있도록 리스트업."*

**보류 사유**: 구조 분류 (B). 층 타입 축이 정의되지 않으므로 *"hybrid의 층 구성이 정책을 바꾸는가"* 를
묻는 실험에서는 **처치가 정의되지 않는다.**

**그러나 다음 질문들에는 오히려 (B)가 필요하다** — 향후 비교연구 후보로 등재한다:
| 질문 | Falcon-H1의 역할 |
|---|---|
| *"hybrid의 이득이 층 분리에서 오나, 단지 SSM이 섞여서 오나"* | ★**(A) vs (B) 대조** — Falcon-H1-7B ↔ Nemotron-Nano-9B가 그 축을 만든다 |
| *"prefill 강등 길이가 층 조성으로 예측되나"*(R2′) | (B)는 층 조성이 **균일** ⇒ 예측식의 **퇴화 케이스**(음성대조) |
| *"긴 컨텍스트에서 SM 민감도"* | ctx **262K**로 로스터 최장 — 상한 탐색용 |
| C2 4-arm 확장 | 이미 Falcon-H1-3B가 정본 4모델 arm에 있어 **크기 축 확장**이 자연스럽다 |

⇒ **Falcon-H1-7B(7.59B)는 Qwen2.5-7B(7.62B)와 크기비 1.00×로 로스터 최적합**이다.
층 타입 축이 필요 없는 비교에서는 **1순위 후보**다.

## 6. 이 문서가 하지 않는 것
성능 비교·정책 판정·arm 선택의 확정. 모델 선택은 각 캠페인의 사전등록에서 하고,
이 문서는 **선택지와 그 제약**만 제공한다. 실측 파라미터는 safetensors 총 바이트 ÷ 2(bf16)이며
config의 공칭값과 다를 수 있다.
