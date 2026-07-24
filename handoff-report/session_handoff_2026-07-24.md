# Session handoff — 2026-07-24

## 이번 세션 요약

사용자가 "단일-GPU 동적 제어 트랙 완전 종결"에 대한 논의 재개를 요청. 새 측정은
없었고, **종결 선언의 유효 경계를 claims-auditor로 적대적 검증 → "완전 종결"이
overclaim(문서 간 불일치)임을 확정 → 정본 스코프 축소 + 세 재오픈 벡터를 하나의
HE0-reopen 게이트로 설계**했다. 서빙 실험은 돌리지 않았다(설계·문서 델타 세션).

## 결정·측정 (측정 없음, 판정만)

- **"완전 종결"은 최상위 정본에 없다.** PROJECT_STATUS.md는 "single-worker dynamic이
  best static을 이긴다"만 철회했고 Claim D/E는 명시적으로 "미검증"으로 열어둠. "완전
  종결" 표현은 CONSENSUS §5-7에만 있었고 PROJECT_STATUS·§1-20과 모순 → **overclaim**.
- claims-auditor 축별 판정:
  - `single-worker · SM-split · reactive 동적 死`(관대 §1-7 n≥4 + tight §1-17 n=4) =
    **CONFIRMED, 반증 실패**. 이 범위는 견고 — 재오픈 대상 아님.
  - "동적 전체" 종결(비-SM lever 포함) = **REFUTED**. admission/KV-aware는 구현조차
    안 됨. 死因(§1-4 얽힘 = 공유 running-batch/KV)을 직접 치는 lever 미시도.
  - §1-13 구조 논증의 substrate-independence = **REFUTED**. §1-20(oracle 분해)이
    스스로 반박: 진짜 headroom +16%인데 `92 prefill+24 decode=116 SM > 108 = coupling
    tax`라 **단일-GPU서만 닫힘, decoupled substrate엔 열림**.
  - 충돌 regime 워크로드(escape hatch) = **PLAUSIBLE**. 배제 아니라 underpowered
    탐침(§1-18 n=2, §1-19 n=1~2 overload-only, n≥4 게이트 위반), long-context 혼합
    (H_L4) 미측정.
  - R2 decoupled 동적 = **범위 밖**. Claim D 서빙 측정 0건(`results/r2_eval/` 비어있음).
  - 남은 confound: escape-hatch 실험에 #3 small-n + #5 metric-cliff LIVE.
- **engine-porter admission-control lever 스코핑**(구현 아님, read-only 조사):
  - admission throttle 골격이 **이미 존재**: `multiplexing_mixin.py:758` `r2_admission_
    limited` early-return. 단 (a) SM-split 정책에 커플링(`controller.py:174-183`
    `CoarseGrainedController.stabilize`만 세팅), (b) coarse all-or-nothing, (c)
    **재평가 경로 잠복 starvation 버그**(`split_prefill_batch` in-flight일 때만
    재평가 → drain 후 stale True latch, clear 경로 없음 → waiting_queue 영구 기아).
  - 관측 신호 전부 계측됨(`_r2_runtime_snapshot` running/kv occupancy·queue depth·
    ITL p95; `_r2_cache_occupancy`는 R2 정책 없이도 호출 가능).
  - SM-split과 **구조적 직교 가능**(admission `:758` vs stream_idx `:714-753` 무관).
  - **cudagraph 호환**(get_new_batch_prefill 호출 여부만 바꿈) = 죽은 layer-switching과
    달리 운영점 호환 = 긍정 신호. **R2·long-context 모델교체 불필요**.
  - **경계(구현 없이 확정)**: preventive-only(포화 후 역전 불가, 조기 발동=TTFT 선희생)
    + 부분 lever(자원-점유 축만, compute-side 단일스레드/SM 타임슬라이스 커플링은 R2 몫).

## 코드·문서 변경

- **`reports/CONSENSUS.md`** (미커밋, +8/-2): doc-steward가 §5-7 "완전 종결"을
  `single-worker · SM-split · reactive 한정 死`로 스코프 축소. 철회 표현은 지우지 않고
  ★정정(2026-07-24) 가시 표기. **새 항목 §5-8**: 종결되지 않은 3개 하위 트랙 명시
  ((a) dual-worker/decoupled Claim D, (b) 비-SM lever admission/KV-aware, (c) 충돌
  regime 워크로드). §1-7·§1-17 견고함은 유지.
- **메모리 `slo-aware-scheduling-track.md`** (repo 밖, doc-steward가 갱신): 동일
  "완전 종결" 문구 정정 + ★★정정(2026-07-24) 문단으로 3개 열린 하위 트랙 병기.
- 코드 변경 없음. admission lever는 스코핑만(구현 미착수).

## 열린 항목 / 다음 세션 시작점

이번 세션 산출물 = **HE0-reopen 게이트 설계**(아직 EXPERIMENT_ROADMAP에 미배선 —
사용자 승인 대기 중). 브랜치, 싼→비쌈 순, 각자 kill 조건:

1. **G2.0-pre** (엔진작업 0): 기존 short-ctx disjoint(§1-19)를 n≥4·cliff 회피로
   재측정. Kill: 단일 static 지배 유지 → 충돌 갈래 short-ctx 근거 소멸.
2. **V3 admission-control** (엔진작업 有, R2·모델교체 無):
   - **선결 correctness gate**(perf 전 필수): starvation 버그 수정(admission decide를
     `split_prefill_batch` 무관하게 매 iteration + "running 비면 무조건 admit" 불변식),
     `PDMUX_ADMISSION_POLICY`를 SM-split과 직교 분리, CPU-only 회귀 5종(단조성/
     no-deadlock/직교성/히스테리시스/chunked-req 안전).
   - observer<3% → 2×2 {admission on/off}×{SM 고정 best-static}, coupled, 변화 trace,
     n≥4. Kill: admission-only가 best-static+3% paired CI 못 넘음 → 자원-점유 축만으론
     부족, 남은 희망을 compute-side(V1)로 좁힘.
3. **V1 decoupled** (R2 Claim D 선결): P1(observer<3%)→P2(true-dual fixed>legacy
   fixed, throughput regression<3%). 먼저 decoupled-fixed-oracle vs single-GPU
   best-static으로 §1-20 +16%가 실엔진서 열리는지 → 열리면 동적 판정.
4. **G2.0-full + G2.1** (최고비용, HE0 반전 유일 지점): Zamba2-2.7B ctx4096 모델교체
   + 용량 선측정. static-스윕으로 충돌 존재 반증 먼저 → 충돌 실재 시에만 decoupled
   위 동적 vs best-static.

의존: ①·② 병렬 가능(독립) · ③ R2 D 선결 · ④ ③+모델교체 선결. 벡터1+2 joint(사용자
강조)=③→④, 벡터3(②)은 독립 조기 갈래.

**다음 세션 첫 액션 후보**: (a) 게이트를 EXPERIMENT_ROADMAP.md에 배선(doc-steward),
(b) ①/② 병렬 착수 — ①은 sbatch(experiment-runner), ②는 admission 버그수정+correctness
gate(engine-porter).

## 미완·주의

- **CONSENSUS.md 미커밋** — 커밋 여부 사용자 확인 필요(원격 URL에 토큰 평문 노출 →
  push 금지, git-committer 위임 권장).
- **HE0-reopen 게이트 미배선** — 설계만, 정본 반영 전 사용자 승인 대기.
- **admission starvation 버그는 기존 코드에 잠복**(벡터3 착수 시 반드시 선수정). 아직
  serving에서 발현 관측된 건 아님(coarse defer가 R2 정책+overload에서만 발동).
- 성능 주장 전무 — 이번 세션은 경계 재정립·설계뿐. 모든 재오픈 브랜치는 서빙 실증·
  n≥4·paired CI 전에는 "미검증".
