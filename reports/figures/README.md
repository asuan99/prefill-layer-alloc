# `reports/figures/` — 실험 수치 시각화

작성 2026-08-05. **상태 요약이며 새 판정을 만들지 않는다** — 모든 값은 정본에서
전사했고, 충돌 시 정본이 우선한다.

- 정본 위계: `PROJECT_STATUS.md` > `reports/paper/` > `reports/CONSENSUS.md` > 기타
- 대시보드: `dashboard.html` (자기완결 단일 파일, 한국어 서술 + 영문 figure)
- 논문용: `fig01..fig11 *.pdf` (벡터) · `*.light.png` / `*.dark.png` (200 dpi)

## 파이프라인

```bash
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc/reports/figures

python extract_raw.py       # 캠페인 아티팩트 → raw_extracted.json
python make_figures.py      # → fig*.pdf + fig*.{light,dark}.png
python make_dashboard.py    # → dashboard.html (그림을 data URI로 내장)
```

`make_figures.py --only 6` 으로 한 장만 다시 그릴 수 있다.

## 파일

| 파일 | 역할 |
|---|---|
| `canon.py` | **정본 수치 전사** — 모든 항목이 `src`(출처 문서)와 `grade`(증거 등급)를 함께 들고 있다. 파생 계산 없음 |
| `extract_raw.py` | 원자료 파싱(P1 4-모델 goodput, r0c decode-knee). 읽기 전용 |
| `style.py` | dataviz 레퍼런스 팔레트 + rcParams. light/dark 양쪽 |
| `make_figures.py` | figure 11종 |
| `make_dashboard.py` | 단일 파일 HTML |
| `raw_extracted.json` | 파싱 산출물(생성물) |

## figure 목록

| # | 내용 | 등급 |
|---|---|---|
| 1 | 연구 arc — 3축 12단계 kill-chain | 서빙 (space 축만 열림) |
| 2 | P1 — PD 분리가 fused를 이긴다 | **스코프 축소** (no-cudagraph, n=1/셀) |
| 3 | layer-aware 死 — TPOT 42→124 ms + 모델별 goodput 붕괴 | 서빙 |
| 4 | HE0 — 동적이 best-static을 못 넘음(관대·tight 양쪽) | 서빙, n≥4 |
| 5 | 왜 동적이 쫓아갈 대상이 없나 — LO/HI 최적 비충돌 | 서빙 |
| 6 | C2 — 운영점 decode-SM 레버 | **CONFIRMED (scoped)** |
| 7 | prefill 축 거울상 + 여기서 반증된 추론 | **미감사, 정본 인용 금지** |
| 8 | Diff B 철회 — 보고 vs steady vs 정책 단위 | **계측 결함, 인용 금지** |
| 9 | 메트릭 절벽 — 3% 섭동이 2× 신호로 | 서빙 |
| 10 | sticky partition — 2.6× 축 모순의 해소 | 서빙, 사전등록 n=8 |
| 11 | 증거 사다리 + 철회 원장 15건 | — |
| 12 | 동적 컨트롤러 해부(신호/규칙/死因) + 실제로 앉은 자리 | 서빙, n≥4 |

figure 12의 컨트롤러 사양은 **정본이 아니라 엔진 소스가 출처**다 —
`workspace/engine-port/src/multiplex/multiplexing_mixin.py`의
`_slo_decide_idx`(L807, v7b) · `_slo_decide_idx_binding`(L726, Step E/F) ·
`_slo_feasible`(L681, Step G 게이트). 판정 수치는 `CONSENSUS.md` §1-8·§1-10·§1-12·§1-17.

## 인용 규율 (그림에도 적용된다)

- **C2**: 구간으로만. `SM16→SM108` 금지(분할 자체가 없는 셀). 등량곡선이므로 **레버 존재만**
  확립하고 정책 이득이 아니며 HE0를 되살리지 않는다. 2.36–2.91×는 **끝점 비**이고
  국소 탄력도가 4× 다르므로 **D=44 이상 구간 적용 금지**. 다른 격자로 이전 금지.
- **fig 7·8**: claims-auditor 미통과 / 계측 결함 확인. 정본(`PROJECT_STATUS.md`,
  `CONSENSUS.md`, `CLAIM_EVIDENCE_MATRIX.md`) 인용 불가.
- **P4 magnitude**: 방향만 인용. `d16 vs d24 = 5.9×(6.240)`은 폐기 벤치의 최량 rep이라
  금지 — 정본은 `5.282 ± 1.302`.
- **P7 magnitude**: `92 SM인데 TTFT 7.24 s`는 폐기 벤치 n=1이라 금지, 방향만 인용.

## 알려진 정본 불일치 1건

`research_arc.md` S10은 변화-trace d16 goodput을 **2.817**로, 2026-08-04 종합 문서
2건(`layertype_dynamic_NEGATIVE` §2.1 · `POSITIVE` §2.3)은 **2.846**으로 적는다.
figure 4와 대시보드는 **2.846**(최신·중복 기재)을 쓰고 그 사실을 그림 안에 명시했다.
순위·판정에는 영향이 없다. doc-steward 확인 대상.

## 커버리지 메모 1건

P1의 4-모델 주장 중 **Zamba2의 fused arm은 이 clean-async 하네스(`p1_7_bench_one`)에
없다** — 존재하는 Zamba2 fused 런은 폐기된 GIL-client 계열(`p1_4_zb_serving_*`)이다.
figure 2는 fused arm이 확인되는 3모델(Nemotron-H / Granite / Falcon-H1)만 그리고
캡션에 이 사실을 적었다.

## 스타일 근거

`dataviz` 스킬의 레퍼런스 팔레트를 그대로 쓴다. 카테고리 슬롯 1–4를 고정 순서로 사용하며,
**adjacent pairlist**(막대·선 차트의 올바른 목록)로 검증기를 돌려 양 모드 통과를 확인했다:

```
light  CVD ΔE 9.1 (protan) / normal-vision 22.9   [contrast WARN → relief rule 적용]
dark   CVD ΔE 8.4 (protan) / normal-vision 19.8
```

light 모드에서 2개 슬롯이 3:1 미만이라 **relief rule**을 적용했다 — 모든 다계열 차트가
직접 라벨을 달고, 대시보드에 표 뷰(`수치 보기`)를 함께 싣는다. all-pairs 목록에서는
4슬롯이 실패하므로(yellow↔orange) 그 목록이 필요한 차트 형태는 쓰지 않았다.
