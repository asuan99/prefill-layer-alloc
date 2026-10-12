# 규칙층·하네스층 경량 델타 감사 — c1395d0(경로 수리)·f3da641(PREREG rev7, fused piecewise OFF, G4 게이팅), 2026-10-12

> 감사자: claims-auditor(읽기 전용, GPU 0). 요지를 메인 세션이 저장(caveat 문안은 원문 그대로).

## 0. 결론
| 층 | 등급 | 死因 | 재봉인 / M2g 재제출 |
|---|---|---|---|
| 규칙층(rev7) | `GO-with-caveats` | 0 | 가능 |
| 하네스층 | `GO-with-caveats` | 0 | 필수 수정 **F-1** 반영 뒤 재봉인 → dry-run → 사용자 승인 → 제출 |

## 1. 필수 수정
- **F-1**: `mg_const.py:15` `PREREG_REV = 6` → 7, `mg_job_common.sh:72` "(rev6)" → "(rev7)". (메인 세션 반영 완료)

## 2. 검토 요지
- (1) resolve→abspath: 의미 변경은 results/ 쪽 4모듈에만(수리 대상), scripts/ 5개는 동일. g5a `parents[5]`는 Job `/opt/pdmux` 실증. 2회차 Job이 실제 레이아웃에서 모든 경로를 맞게 소비(G3 stress 경로, G5(a) PASS, boot 21/21 health).
- (2) rev7: 비-fused arm은 chunked −1 → piecewise 토큰 목록 빔 → "capture size is not set"으로 러너 미생성(2회차 21 boot + g5a boot 전부 확인, piecewise begin 0). 플래그 소비처 7곳 전수 — A100 sm80·NemotronH n_groups=8·mem-fraction 고정에서 전부 비활성 → MGP-41. G\_def의 "운영 기본값 대비" 해석은 거짓이 됨 → MGP-40.
- (3) G4 게이팅 정상. 2회차 산출물 표본: G0(ii)·G0(iii)·G3·G3b·G5(b)·contam·G4(healthy) 실질 PASS, 남은 공허 PASS 경로 미발견. 새 발견: (a) G4 nsys 프로브 미실행(rc=1, 3.1 s, server.log 공백, nsys 부재 추정) → MGH-24, (b) agnostic (54,54) 행은 G1·G5(b)에서 미피복 → MGH-23.

## 3. 반전 시험 요지
fused 플래그 적용 범위·잔여 piecewise 차이·G4 게이팅·2회차 PASS 재사용 여부: 판정 변화 없음. (참고) G5(c)(iii) 분모 hi↔lo에서 TP2 라벨 반전(0.9989% ↔ 1.0090%) — 분모는 봉인 코드에서 hi 고정, cliff caveat.

## 4. caveat (문자 그대로 승계)
- **MGP-40**: "rev7의 `fused`는 SGLang 업스트림 기본값이 아니라 piecewise CUDA graph를 끈 vanilla다(이 이미지의 Python 3.14에서 piecewise ON은 기동 불가, M2g 2회차 7/7). G\_def를 '운영 기본값 대비 PD-mux 이득'으로 쓰지 않는다 — piecewise가 동작하는 환경의 fused 대비 값은 미측정이다."
- **MGP-41**: "fused와 fused\_sym·PD-mux arm의 `disable_piecewise_cuda_graph` 값은 다르다(True vs False). 'piecewise 노브 해당 없음'은 그 플래그의 소비처 7곳이 A100(sm80)·NemotronH(n\_groups=8)·`--mem-fraction-static` 고정에서 전부 비활성이라는 코드 독해와 2회차 비-fused 21 boot 로그에 근거한다. 모델·GPU·메모리 규약이 바뀌면 이 문장은 성립하지 않는다."
- **MGH-22**(실행 전 예보): "2회차 비-fused 3 arm만으로도 TP4·TP8의 arm 간 `max_total_num_tokens` 상대차는 1.38%·1.33%(>1%)이므로, rev7 재실행에서 TP4·TP8은 fused 값과 무관하게 `MEMORY_MISMATCH`로 예보된다(M3 G 인용 금지, 측정은 진행). TP2는 0.9989%로 문턱까지 약 70 토큰 여유뿐이고 fused 실측값이 판정을 정한다 — TP2 G5(c) PASS를 '메모리 조건 동등'의 강한 증거로 쓰지 않는다." (1% 규칙이 비구속 KV 풀에 걸려 있음: TP4 필요 1.59 M 토큰 vs 가용 13.7 M. 규칙 변경은 사후 변경 — 사용자 결정·델타 감사·사후 변경 공시 필요. 재제출은 막지 않음.)
- **MGH-23**: "M2g의 G1·G5(b) PASS는 agnostic의 (74,34) 분할 행만 덮는다(2회차 N1·DP에서 (54,54) 실현 0 snapshot, G1 동시 32 < 임계 36). (54,54) 행은 TP2 G3b(생존성만)에서만 실현됐다. '다중 GPU agnostic 출력 정확성 PASS'를 (54,54) 행까지 확장하지 않는다."
- **MGH-24**: "2회차 G4 nsys 프로브(TP2·TP8 × d44)는 rc=1, 3.1 s, server.log 공백으로 실행되지 않았다. all-reduce 커널 이름 기술치는 존재하지 않으며 RESULT 문서의 G4 'PASS'는 캡처 성공만 뜻한다."
- **MGH-25**: "레이아웃 회귀 테스트는 results/ 쪽 4개 모듈만 Job 레이아웃으로 실행하고, 정적 가드는 `Path(__file__).resolve()` 리터럴만 검사한다. 'Job 레이아웃 경로 결함 0'은 2회차 Job의 실제 소비로 확인된 범위까지만 쓴다."
- **MGH-26**: "G0(ii) scorer는 공통 거부 문자열만 대조한다. 2회차 5/5 PASS가 env별 거부라는 사실은 로그의 `Refused because:` 줄을 수동으로 확인한 결과이지 scorer가 판정한 것이 아니다."

## 5. 권고(차단 아님)
fused 실현값(piecewise.json disable=True, PIECEWISE_BEGIN 0) 기록·대조, G0(ii) env별 reason 대조, 정적 가드 정규식화, 제출 전 `nsys --version` 확인(없으면 nsys 블록 NOT_RUN 공시), 예산 MGH-18 승계.
