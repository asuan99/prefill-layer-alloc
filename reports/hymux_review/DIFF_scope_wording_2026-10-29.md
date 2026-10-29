# 범위 정정 문구 diff — 적용 전 (2026-10-29)

> 지위: **적용 전 diff**(결정 57 ③). 정본은 고치지 않았다. promote가 2차(마지막) 감사 뒤 적용한다. §6 등급 변경 0, 새 판정 0.
> 기준: main HEAD `eef6eaff`. 입력: `PROPOSAL_scope_wording_impl_defects_2026-10-29.md`(15행), 감사 r1 `workspace/engine-port/results/impl_defects/AUDIT_scope_loop_dependence_r1_2026-10-29.md`.
> 파일: `DIFF_scope_wording_2026-10-29.patch`(repo 루트에서 `git apply`; CONSENSUS·COSTMODEL), `DIFF_scope_wording_2026-10-29_CLAUDE_md.patch`(작업 루트 `~/Experiments/KISTI`의 `CLAUDE.md`, repo 밖이라 `patch -p1`/`git apply`를 그 디렉터리에서). 둘 다 `--check` 통과.
> 편집 방식: 전부 **줄 안 편집**(같은 줄에 구절 추가·교체). 새 줄 0 → **모든 파일의 줄 번호 이동 0**. 구 문안은 가능한 한 남기고(철회 가시성) 마커 "(2026-10-29 범위 정정, 감사 r1 #N)"를 붙였다.

## 줄 번호 재확인
제안서가 적은 줄 번호는 HEAD에서 전부 그대로 맞았다(CONSENSUS :3666-3667·3680·3682·3683·3685·3686·3698·3711·3714-3719·3722·3727·3729·8782·8785·8786, COSTMODEL :21·39·50-54·73·82-86). 이동: CONSENSUS 0, COSTMODEL 0, CLAUDE.md 0.
제안서의 "§1-17 :3717"(positioning)과 ":3722"(tight 재튜닝)는 둘 다 §1-17 번호를 쓴다(원장의 중복 번호). 줄 번호로 구분했다.

## 행 → hunk 맵
| # | 유형 | 편집 위치 |
|---|---|---|
| 1 | 교체 | CONSENSUS :3666(제목 문장, 구 문안은 괄호로 보존), :3667 첫 절 |
| 2 | 교체 | CONSENSUS :3682 §1-3 제목 셀 뒤 범위 절; :3667; CLAUDE.md :39 |
| 3 | 교체 | CONSENSUS :3719 §1-15 "오직 whole-phase floor" 뒤 범위 절 |
| 4 | 추가 | CONSENSUS :3680 §1-1 제목, :8786 Claim G 등급 셀 뒤(등급어 그대로); CLAUDE.md :38 |
| 5 | 추가 | CONSENSUS :3686 §1-7 제목, :3722 §1-17 제목; CLAUDE.md :41 |
| 6 | 교체 | CONSENSUS :3717 §1-17 positioning 뒤; COSTMODEL :73; CLAUDE.md :42 |
| 7 | 추가 | CONSENSUS :3683 §1-4 제목, :8782 Claim C 등급 셀 뒤 |
| 8 | 추가 | CONSENSUS :3685 §1-6 끝, :3729 실전 권고 끝; CLAUDE.md :44 |
| 9 | 추가 | CONSENSUS :3711 §1-13 "LO는 무관심" 뒤 |
| 10 | 교체 | CONSENSUS :3698 §1-24(제목 뒤 포인터 + 본문 "★고칠 수 없다" 앞에 정정 문장, 구 문안 "구 문안:"으로 보존); COSTMODEL :50, :86 |
| 11 | 교체(뒤 붙임) | COSTMODEL :21(CM-4 셀 뒤), :54(허용 문안 뒤 붙여 쓸 문장) |
| 12 | 추가 | CONSENSUS :3727 §1-38 제목; COSTMODEL :52 |
| 13 | 교체 | COSTMODEL :82 답 1 |
| 14 | 교체 | COSTMODEL :39 CM-2, :83 답 2 |
| 15 | 추가 | CONSENSUS :3714·3715·3716(§1-20·19·18 제목), :8785 Claim F 등급 셀 뒤 |

감사 r1 18행(Phase 0 문서 §2.4 문구)은 제안서 15행에 없어 이 diff 범위 밖이다.

## 판단·주의
- 교체 행도 정본 줄 번호 보존을 위해 **구 문안을 지우지 않고 범위 한정 절을 덧붙이거나**(#2·#3·#6·#10) 구 구절을 괄호로 남겼다(#1). 문안을 통째 교체(제안서 그대로)하려면 promote에서 조정. 구 "全형태 死"·"고칠 수 없다" 라벨이 아직 줄에 남아 있는 곳은 범위 절이 같은 셀에 바로 붙어 있다.
- #1은 제안서 문안을 그대로 쓰되 :3667의 "layer-type 기반 정책은 全형태 死"는 "층 경계 SM 전환 계열은 … 死"로 좁혔다(제안서 #1은 :3667 전체를 한 문장으로 교체하라고 했으나 동적 제어 절은 남겨야 해서 첫 절만 교체).
- #11 :54는 "허용 문안" 인용 블록을 건드리지 않으려고 **인용 뒤에** 붙여 쓸 문장을 지시하는 형태로 넣었다.
- #12 CM-4 :51(존재 확인 인용 줄)에는 넣지 않고 :52(크기 미분리 줄)에 넣었다.
- #14 CM-2 :39 "아래 열린 질문"은 :43을 가리킨다(줄 번호를 박지 않아 이동에 강함).
- #8 CLAUDE.md :44는 원 bullet 뒤에 이어 붙였다. "MuxWise upstream 공개 코드(PR #10692)와 이 루프의 대응은 미대조" 원칙(제안서 §0-3)은 본문에 넣지 않았다 — 새 줄이 필요해서. 필요하면 promote가 CONSENSUS §0 근처에 별도 한 줄로 추가.
- 못 둔 것: tracked CLAUDE.md 사본 — `tools/claude/CLAUDE.md`는 **없다**(추적되는 것은 agents/skills뿐). 작업 루트 CLAUDE.md만 패치. 그래서 `sync_claude_tools.sh --import`는 이 패치에 필요 없다. COSTMODEL.md가 CLAUDE.md "현재 결론"을 인용하는 곳은 없다.
- PROJECT_STATUS.md는 이 작업 대상이 아니다(지시). 제안서 §0 원칙의 원장 최상단 5줄 항목은 promote가 쓴다.

## MEMORY.md 색인 줄 변경 제안 (텍스트, 미적용)
- `[SLO-aware 스케줄링 track]` 줄: "HE0 확정: 동적은 best-static 못 넘음, 권고 = peak decode 기준 decode-heavy static" → "HE0: SGLang 공개 PD-mux 루프(비sticky·입장 시점 결정·진행 중 drain)에서 동적은 best-static 못 넘음, 권고 = peak decode 기준 decode-heavy static(루프 한정, Nano 재결정, 2026-10-29 범위 정정)".
- `[engine-port P0 트리아지]` 줄: "layer-aware 全형태 서빙 반증(TPOT 42→124ms)" → "층 경계 SM 전환형 layer-type 정책 서빙 반증(KISTI·Zamba2·no-cudagraph, 공개 루프 한정; span 크기·박자·handoff·preemption 미시험)".
- `[prefill-layer-alloc 결정 시점]` 줄: "layer-type 공간분할 死(1.2B–7B)" → "층 경계 SM 전환형 layer-type 공간분할 死(1.2B–7B, 공개 루프·no-cudagraph 한정)".
- `[main 트랙 …]` 줄에 한 구절: "정본 범위 정정(루프 구조 의존, 감사 r1 2026-10-29) 적용".
- 150자 예산 유지. 번호·날짜는 promote 시점에 확정.
