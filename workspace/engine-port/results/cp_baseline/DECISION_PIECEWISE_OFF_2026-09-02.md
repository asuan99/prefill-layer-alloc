# 설계 결정 — **piecewise CUDA graph는 전 arm에서 끈다**

2026-09-02 · 사용자 지시 · GPU 0(이 문서 자체) · 성능 판정 **0건** · 등급 변경 **0건**

> **사용자 지시(원문)**: *"piecewise CUDA graph 캡처 크래시가 발생한다면 해당 부분은 굳이
> 사용하지 않아도 괜찮아. 그 부분은 prefill에서 cuda graph의 활용을 위한 기술인데, 기술상
> 문제가 있다면 그렇게 수행하지 않는 걸로 실험들을 진행할 것"*

## 1. 근거가 된 관측

모델 부팅 스모크(job **900731**)에서 `--chunked-prefill-size 2048` arm이 두 모델에서 부팅에
실패했고, 원인은 chunked prefill 비호환이 **아니라** piecewise 캡처 크래시였다:

```
Piecewise CUDA Graph failed with error:
  'typing.Union' object has no attribute '__module__' and no __dict__ for setting new attributes
To work around this error, add --disable-piecewise-cuda-graph to your launch command.
```

- 발생: `tiiuae/Falcon-H1-3B-Base`(triton·flashinfer 양쪽) · `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`(flashinfer)
- **미발생**: `Zyphra/Zamba2-2.7B`(job **900680**에서 `cps 2048` + `piecewise_cuda_graph_max_tokens=2048`으로 정상 부팅) · `ibm-granite/granite-4.0-h-micro-base`(4/4 셀 부팅)

⇒ **모델 × 환경 성질**이며, 엔진 자신이 우회책을 출력한다.

## 2. 결정

**모든 실험의 모든 arm에서 `--disable-piecewise-cuda-graph`를 명시적으로 넘긴다.**
`decode` CUDA graph는 **그대로 ON**이다(이 플래그는 prefill piecewise 전용).

★**`cps = -1`(pdmux arm)이 이미 캡처 목록을 비운다는 사실에 기대지 않는다.** F3 관측이 그것을
보였지만(job 900053, 캡처 목록 len 0), *"설정이 그럴 것이다"* 라는 추론과 *"플래그로 그렇게
지정했다"* 는 다르다. 두 arm 모두 **같은 축에서 명시적으로 동일**하게 만든다.

## 3. ★이 결정이 닫는 것 — F3 교락

CP-0 3회차 감사 死因 **F3** / 4회차 **G4**의 내용은 *"`fused_mono`가 단일 레버가 아니다 —
`cps`가 prefill 예산과 piecewise CUDA graph 캡처 범위를 **함께** 정한다"* 였다.

piecewise를 **전 arm에서 끄면 `cps`는 다시 단일 레버**(prefill 예산)가 된다.
이는 G1 프로브가 이미 채택했던 단계 규약(*"cps가 유일한 레버가 되도록 전 arm에서 piecewise를
끈다"*)과 같고, 이제 **트랙 전체의 기본값**이 된다.

`PREREG_CP2_2026-09-01.md` §1이 등록한 **4중 강제 번들이 3중으로 줄어든다**:

| | 이전(CP-2 rev1) | 이 결정 이후 |
|---|---|---|
| chunked prefill | ON / OFF | ON / OFF (**남음**) |
| SM 분할 | 없음 / green ctx | 없음 / green ctx (**남음**) |
| overlap schedule | ON / OFF(강제) | ON / OFF(강제) (**남음**) |
| **piecewise CUDA graph** | **캡처됨 / len 0** | **양쪽 OFF — 소거** |

★단 **번들이 사라진 것은 아니다.** 남은 셋은 여전히 엔진이 강제하므로 §1의
positivity-violation 논증과 "기전 귀속 불가" 결론은 **그대로 유효**하다.

## 4. ★대가 — 반드시 동반할 스코프

piecewise CUDA graph는 **prefill 최적화**다. `cps > 0`인 arm은 원래 이 최적화를 쓸 수 있고
(Zamba2에서는 실제로 썼다), `cps = -1`인 pdmux arm은 구조적으로 못 쓴다. 따라서 이 결정은
**chunked-prefill 가족에게서 하나의 이점을 제거하는 방향**이다.

⇒ 결과 문서는 반드시 다음을 동반한다:

> 이 비교는 **piecewise prefill CUDA graph 없이** 수행됐다. 그 최적화는 이 환경의 장문
> hybrid 모델(Falcon-H1 · Nemotron-H 계열)에서 **캡처가 크래시하여 사용할 수 없고**, 모델 간
> 비교 가능성을 위해 전 arm에서 껐다. chunked-prefill 가족이 piecewise를 쓸 수 있는 환경에서는
> 결과가 달라질 수 있다.

**금지 문장**: *"piecewise를 껐으므로 공정한 비교다"* — 공정해진 축은 하나이고, 동시에
CP 가족의 가용 최적화 하나를 뺀 조건이다. 둘 다 적는다.

## 5. 기존 산출물에 대한 영향

| 산출물 | piecewise 상태 | 판정 |
|---|---|---|
| V-probe(900411·900423–900436, Zamba2) | `fused_default` arm이 **ON**(`piecewise_cuda_graph_max_tokens=8192`) | 무처치 **분산** 측정이므로 값 자체는 유효. 단 이 결정 이후 조건과 다름 ⇒ **분산을 새 조건으로 이월할 때 재측정 필요** |
| W1 스모크(900680, Zamba2 `cp2048`) | **ON**(`=2048`) | 배관 확인 목적이므로 무해. **rate 위치 추정치는 재측정 대상** |
| 모델 부팅 스모크 1차(900731) | cp arm **ON** → 크래시 | 이 결정의 **근거**로만 사용 |
| 모델 부팅 스모크 2차(900752) | cp arm **OFF** | 이 결정을 이미 반영 |
| G1 프로브(900067) | 전 arm **OFF** | 이 결정과 **일치**(선례) |

## 6. 하네스 반영

- `model_boot_smoke.sbatch` — cp arm에 적용됨(job 900752). ★**pdmux arm에도 명시적으로 추가**(§2).
- `w1_probe.sbatch` — 전 arm에 추가.
- 이후 신설되는 모든 arm 정의는 이 플래그를 **공통 플래그**에 둔다(arm별 분기 아님) —
  분기에 두면 새 arm이 추가될 때 빠진다(`PROJECT_STATUS` "방법론 게이트" #66의 형태).
