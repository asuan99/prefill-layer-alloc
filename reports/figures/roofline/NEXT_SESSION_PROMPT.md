# 다음 세션 전달용 프롬프트 (2026-10-25, main 트랙)

아래 블록을 새 세션 첫 메시지로 붙여 넣는다.

```
catch-up main 을 먼저 실행해 줘. 그 다음 아래만 해.

배경(재도출 금지): 직전 main 세션이 `prefill-layer-alloc/reports/figures/roofline/`에 "attention vs Mamba2 roofline·스케일링" 그림 묶음을 만들었다.
수식 단일 출처는 `workspace/engine-port/scripts/costmodel/analytic_terms.py`(analytic-v1)이고, 그림은 전부 [유도] 이상 하한 + J1 FLOOR.json 관측 대조다.
먼저 `reports/figures/roofline/README.md`를 읽어라(각 파일의 용도·재현·한계·정정 이력·인용 규칙이 있다). 새 판정은 0건이고 감사는 아직 안 받았다.

할 일(우선순위 순):
1) 감사: claims-auditor로 README의 "말하는 것" 5개 문장과 `prefill_observed_vs_ideal.py`(관측 2.06~2.28배, 상수항 약 7.2 ms)를 반증 시도.
   특히 (a) 이상 바이트 모델이 하한이라는 가정, (b) 관측 floor와 이상 시간의 분모 정의 차이(opchar Mamba 비중 66~71% 대 이상 50~54% 불일치의 원인), (c) LM head 포함 여부가 비율에 주는 영향.
   감사는 최대 2회, 그 뒤 사용자 결정으로 넘긴다.
2) 정본 수식 쪽 제안 검토(결정 아님): 수식 문서 §8의 "층 종류별 효율 η"에 prefill 실효 η≈0.47~0.49, 상수항 ≈7 ms를 첫 관측점으로 올릴지.
   올린다면 ANALYTIC_COST_FORMULAS.md 변경 규칙(코드·문서 동시 + 버전 증가 + render_formula_tables.py + 축 지도 재발행)을 따른다.
3) T1 데이터가 생기면(아직 없음) `python3 reports/figures/roofline/t1_compare.py --data cells.csv`로 decode 대조를 그려 README에 추가. cells.csv 열: bs,ctx,t_s(S_eng 중앙값, 초).

하지 말 것: ★128k 프로브 TTFT(`cap_char/probe_128k`)를 정본·논문에 인용(기술값, 인용 금지) · "decode는 memory-bound라 SM 둔감" 서술(정본 X-7 반증)
· GPU Job 제출(smoke는 하네스층 감사·미복구 설정 제약 점 결정·사용자 승인이 선행) · 정본(PROJECT_STATUS·CONSENSUS·COSTMODEL·MEMORY) 직접 수정(promote 경로만).
```

## 이 세션의 맥락 한 줄
사용자 질문 흐름: "Mamba가 compute-bound인 건 in_proj 때문인가?" → roofline/R/스케일링 그림 → "관측값은?" → J1 대조 → "비용 모델의 이점과 식의 빈틈" → 32k·64k 보류 점 → T1 비교 틀. 미해결 질문은 decode의 SM 의존성(η(SM))과 동거 간섭 ρ.
