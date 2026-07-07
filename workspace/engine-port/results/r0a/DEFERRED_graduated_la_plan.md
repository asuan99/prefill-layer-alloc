# DEFERRED work order — 22:00 KST 2026-07-06 이후 실행 (지금은 미실행)

사용자 지시: 아래 (A)보고서 정정 + (B)추가 실험은 **2026-07-06 22:00 KST 이후에만** 진행.
발단: 사용자 지적 — 테스트된 layer-aware가 실제 아이디어(per-type graduated 할당)가 아닌
**조잡한 이진 근사(sensitive=full / insensitive=floor 16)** 이고, agnostic의 over-provisioning
(GPU 저활용)을 보고서가 서술 못 함. 확인: green-ctx는 임의 SM 지원(`create_greenctx_stream_by_value`,
zamba2.py:334) → binary는 메커니즘 제약이 아니라 정책 선택. graduated는 ~10줄.

## (A) 보고서 정정 (sm_policy_report.html v3 → v4, 원본 _v3로 보존)
1. §01 la 카드 + §06 표 + §07: layer-aware를 **"per-type graduated 할당(각 타입을 그 knee에 맞추고 잉여만 환원)"**
   으로 재프레이밍. **테스트된 정책은 그 2-level(full/16) 근사임**을 명시(strawman 경고).
2. §04/§06/§07 어딘가에 **agnostic over-provisioning** 명시: 균일 ~54 SM이 low-knee 타입엔 낭비 →
   GPU 저활용. 단 그 잉여의 **회수 가능성**이 관건(→ (D) granularity).
3. §07 (D) 문구 **"layer-aware ≤ agnostic at ANY implementation"** →
   **"측정된 모든 구성(binary full/16 + floor sweep)에서 열위 + granularity 논증; graduated per-type는 미측정"**
   으로 약화. (D)는 논증이지 graduated 측정 아님을 정직히.
4. green-ctx 임의-SM 지원 각주(binary = 정책 선택, 메커니즘 아님).
5. r_series_status.md에 이 후속(R0b) 링크 + 판정 추가.

## (B) 실험 (신규 산출은 results/r0b/ 이하; dev-tree 수정은 src/ 미러 + reversible)
**B1. per-type serving-batch SM knee 측정** — 진짜 잉여가 있나?
   - `_fixed` 핀(agnostic-style)으로 decode SM ∈ {16,24,32,40,54,72,108} 스윕,
     Zamba2 in3600/out32 (+옵션 in2000/out96) 서빙 batch에서 mamba vs attn(hybrid) decode 기여/TPOT.
   - 판정: knee ≪ 54인 타입 존재? (SGLANG_ZAMBA_TIMING로 per-type 누적 이미 계측됨 — 활용)
**B2. graduated-la 구현** (~10줄, env-gated, 하위호환):
   - zamba2.py `_tgt`에 per-type SM 맵 env 추가 (예: `PDMUX_LA_SM_MAP="mamba:54,attn:96"` 파싱 →
     타입별 `_get_gctx_decode_stream(N_type)`; 기본=현행 binary). nemotron_h/granite도 동형.
   - 미러: src/ 사본 + env/dev_tree_edits.md 기록.
**B3. release-attn(protect-mamba) 이진 변종**도 sanity 포인트로(이전 논의).
**B4. 측정**: agnostic vs graduated-la(+변종) — in3600/out32 clean async(`sglang.bench_serving`),
   num-prompts 120, rates 1-6 sub-saturation, 2 reps. TTFT/TPOT p50/p99 + goodput@SLO → CSV(results/r0b/).
**B5. 판정**:
   - graduated가 agnostic **우위** → ★은퇴 결론 **불완전**. 즉시 사람 보고(자동 재작성 금지).
   - graduated ≈/< agnostic(예상, (D) 정합) → 은퇴 **강화**; (A) 정정에 실측 근거 추가.

## 제약 (R-series와 동일)
clean async만·sub-saturation·ncu/nsys 금지·CUDA graph/py env/sglang 버전 불변·
기존 결과 CSV/로그 불변. dev-tree 수정은 reversible + src/ 미러. 신규 산출 results/r0b/.
정본 보고서 자동 재작성 금지(B5 우위 시 사람 게이트).
