# 감사 판정 r1 — 변화 trace 대표성 검토 (CHANGE_TRACE_VALIDITY_2026-10-26)

- 대상: `reports/hymux_review/CHANGE_TRACE_VALIDITY_2026-10-26.md` (미감사 검토) · 감사자: claims-auditor(읽기 전용) · 일자 2026-10-27 · 회차 1/2
- 결론: 이 문서는 HE0를 반증하지도 범위를 새로 좁히지도 않는다. 정본 CONSENSUS §1-7이 이미 "이 trace 범위 한정"이다. 새로 더한 것은 그 trace의 시간 척도 명시뿐 → HE0 부호·등급·본문 불변, 범위 단서 한 문장만 promote 가.

## 판정 표

| 주장 | 판정 | 근거·confound |
|---|---|---|
| (i-a) 변화 trace 구조: 구간당 200요청, LO ≈67 s / HI ≈17 s, 구간 사이 drain, 3라운드 | CONFIRMED | `results/slo_sched/sharegpt_vary_bench.sbatch:35,66-75`(NP=200, ROUNDS=3). 정본 출처 = CONSENSUS job 태그 `sgptv_852341`·`856889`·`859005`, §1-32 `sgptv{Lo,Hi}_*_L3H12`, :2090 "정본 변화-trace 하네스" |
| (i-b) 실 trace는 느리게 변한다(Azure code 4× 변화에 수 시간) | PLAUSIBLE(조건부), 인용 불가 | 외부 공개 데이터 집계이나 스크립트가 휘발성 scratchpad에만 있었고 데이터 해시 없음(→ 스크립트는 `reports/hymux_review/change_trace_stats_scripts_2026-10-26/`로 보관, 데이터 sha256은 여전히 없음). 문서 자체의 BurstGPT 1분 peak/mean 68.5가 "실 trace = 느림" 일반화의 반례 → "trace 계열에 따라 다르다" |
| (i-c) drain 때문에 변화 trace가 실제 버스트보다 쉽다 | NOT-YET-SUPPORTED | 방향 미측정. §1-6 비대칭·§1-13(LO split 무관심)로 보면 HI backlog 이월 시 decode-heavy static이 더 유리할 가능성 |
| (ii) 그래서 HE0 범위가 좁아진다, 느린 변화 영역 증거 0 | 새 결론으로는 REFUTED · 범위 명문화는 PLAUSIBLE | §1-7에 범위 한정 기존. 전환비용 0 per-phase static oracle(drain 있는 phase 단위) = quasi-static 극한: §1-13, §1-19 +2.1%(n 1–2), §1-32 +1.67%. 열린 것(중간 rate·혼합, 다른 기판·모델, §3 항목57)은 기존 등록 범위 |
| (iii-a) 실 trace 재생 1종 추가 | PLAUSIBLE(조건부), HE0 문안엔 불필요 | 시간 척도 보존 재생 = 수 시간×arm×n; 압축하면 문제 변수 파괴. Nano엔 동적 arm 없어 판정량 = 창별 static oracle(H-D와 같은 estimand). 창·thinning·배율 자유 표면 |
| (iii-b) 정상 상태 격자를 static 권고의 1차 근거로 | PLAUSIBLE(조건부) | static 설정 주장에는 타당(V-O 형태). "static ≥ 동적" 정책 비교 1차 근거로 쓰면 게이트 2 개정 = 사용자 결정. 창 간 가산성(backlog 이월 없음) 가정 공시 필요 |

적용 confound: (11) 기판 혼합 — Nano/VESSL로 KISTI Zamba2-2.7B HE0 확인·반박 불가(별도 claim) · (6) 정상 상태 격자 static 순위 ≠ 동적 제어기 행동 · (5) 격자 rate는 cliff 회피(게이트 2가 폐기한 r8이 cliff rate).

## HE0 범위 문장 (DIAG_lockstep과 결합)

허용(문자 그대로):
> "HE0는 KISTI A100·Zamba2-2.7B·cudagraph-ON에서, ShareGPT rate 3↔12 교대(구간당 200요청, 도착 약 67 s/17 s, 구간 사이 drain, 3라운드)와 같은 계열의 mix-스윙 trace로 관측한 *달성된* reactive 동적 제어 계열의 결과다. 분–시간 척도의 실 trace, 중간 rate·혼합, 다른 기판·모델로의 일반화는 측정되지 않았고, HE0 run에는 phase 계측이 없어 lockstep 정지의 기여는 판단할 수 없다."

선택 부기(기술값): "같은 범위에서 전환 비용 0의 per-phase static oracle은 best static 대비 약 +2% 이하였다(§1-19 n 1–2, §1-32)."

금지:
- "HE0는 실 워크로드·일주기 변동에서도 성립한다" / "변화 trace는 실 trace를 모사한다"
- "느린 변화에서는 동적이 이긴다" / "HE0는 느린 변화에서 무효다"
- "실 trace는 수 시간 척도로만 변한다"
- "drain 때문에 변화 trace는 동적(또는 static)에 유리하다"
- "HE0는 lockstep의 산물이다" / "lockstep 정지는 HE0와 무관하다"
- "Nano/VESSL 결과가 HE0를 확인했다(또는 반박했다)"
- "HE0 범위가 이번에 새로 좁아졌다"

## promote 가부
- 위 허용 문장을 §1-7 범위 단서로: 가(부호·등급·본문 불변)
- 공개 trace 통계 수치(5.85–11.3×, 2.69× 등): 불가 — 데이터 sha256 보관 전
- 게이트 2 개정: 불가 — 사용자 결정

## 다음 Job(PLAN_V rev4) 최소 arm 제안
1. 결정 규칙 추가 없음. H-D 깨끗한 시작 phase-oracle(120 s·600 s, D {16,24,34,44,64}, q=1.0, seed 8 교차적합)이 quasi-static 극한 estimand. 시간 척도 공시 필요: "HE0 phase 17/67 s, H-D 120/600 s, 둘 다 실 trace 시간 단위 변화보다 짧다."
2. 실 trace 재생은 1차 Job에 넣지 않음 권고. 넣는다면 판정 없는 기술값 블록 R-1: Azure 2024 code 연속 1시간 창 1개, 시간 압축 없음(배율 1), thinning만, 창·thinning률 데이터 전 봉인 · arm = V-O G_best D, D̂, D64(같은 Job·GPU, seed 블록 내 순서 교대) · n = thinning seed 4 · thinning률 = 10분 피크가 1.0×λ*_best-static, 10분 창별 `CLIFF_ADJACENT` 비율 보고 · goodput = Σgood/Σduration · 10분 창별 static oracle H, TTFT/ITL p50/p95/p99 · 비용 [유도] ≈13–15 GPU-h · caveat: "R-1은 Nano/VESSL 한 창의 static 기술값이며 HE0(KISTI) 범위를 넓히거나 좁히지 않는다."
3. R-1 포함 여부는 결정 38에 따라 V-O 결과 회수 뒤 사용자 결정.
