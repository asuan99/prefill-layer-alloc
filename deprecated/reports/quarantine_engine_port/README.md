# deprecated_reports — 이력 보존용

여기 문서들은 **당시 시점의 기록**이며 **현재 결론과 충돌할 수 있다.**
현재 확정/철회 상태의 정본은 **`../../../reports/CONSENSUS.md`** 이다. 인용은 CONSENSUS를 우선하라.

| 문서 | 시점 | 왜 deprecated |
|---|---|---|
| `p0_triage.md` | 07-02 | 초기 트리아지. 이후 전 결론이 대체 |
| `zamba2_port_plan.md` / `zamba2_troubleshooting.md` | 07-03 | 포팅기. 완료됨 |
| `p1_4_layer_aware_평가_kr.md` | 07-05 | layer-aware 평가 — 트랙 死((B,L) knee) |
| `p1_3_nemotronh_layer_aware.md` | 07-06 | 모델별 layer-aware — 트랙 死 |
| `p1_7_hybrid_generalization_kr.md` | 07-06 | hybrid 일반화 — 이후 실측이 대체 |
| `sm_policy_report_v2.html` | 07-06 | `../../../reports/sm_policy_report.html`이 대체 |
| `r_series_status.md` | 07-07 | R-시리즈 상태 — 핸드오프가 대체 |

⚠️ 특히 이 문서들의 **layer-aware "生" 서술**과 **저부하 synthetic 기반 "최적=d16" 서술**은
`CONSENSUS.md` §1-3·§1-5에서 **반증/정정**되었다.

★**2026-07-24**: `session_handoff_2026-07-07.md`·`session_handoff_2026-07-13.md`·
`session_handoff_2026-07-15.md`는 세션 핸드오프 전용 아카이브
[`../../../handoff-report/`](../../../handoff-report/)로 이동했다(다른 quarantine
문서와 분리 관리). 07-15의 "부분 반증 + 대체됨" 주석(cudagraph 최적
static=d16·실전=d16 고정은 저-decode-부하 synthetic 아티팩트, 실 trace서 d16
최하위·최적은 d44, 전체 흐름은 research_arc.md가 대체)도 함께
[`handoff-report/README.md`](../../../handoff-report/README.md)로 이관 보존했다.
