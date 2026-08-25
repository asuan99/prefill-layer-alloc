# 사전등록 — **E-A**: cap ↔ mamba pool ↔ KV 예산을 **배너로 실측**한다

2026-08-25 · **서빙 0 · 요청 0 · 성능 판정 구조적으로 불가**(부팅 후 배너만 읽고 죽인다) ·
근거: `audit_mechanism_narrative_2026-08-25/VERDICT.md` §⑤ **E-A**.
정본 규칙: 이 문서. 결과가 이 문서와 다르면 **이 문서가 이긴다**.

---

## 0. 왜 이것을 사는가

정본 §3-23(2026-08-02)과 감사 B3가 등재한 사실 — `--disable-radix-cache`에서
`max_mamba_cache_size = max_running_requests // dp`(`model_runner_kv_cache_mixin.py:223-230`)이고
KV 예산은 `total_rest_memory − mamba_state_memory`(`:250-255`) — 는 **코드로 강제되는 항등식**이다.
★**그러나 저장소에 hybrid 모델을 서로 다른 *명시* cap으로 부팅한 기록이 0건**이라(감사 주장 2c),
*"cap을 내리면 KV 예산이 는다"* 는 **관측이 아니다.**

⇒ E-A는 그 항등식을 **실측으로 바꾼다.** 그리고 `estimated` 클램프(`:861-874`)로 **cap이 조용히
잘리는지**(라벨 ≠ 실현, Stage 0 D108 계열)도 드러난다. **이것이 ③(손잡이 순화)의 전제다.**

★**이것이 하지 않는 일**: *"cap이 무는가"* 에 답하지 않는다(감사 B1로 점 술어 무효). B3를 **닫지도**
않는다 — confound를 **유도된 것에서 측정된 것으로** 바꿀 뿐이다.

## 1. 셀 (4 부팅, 서빙 0)

| 셀 | `--max-running-requests` | `--max-mamba-cache-size` | 역할 |
|---|---|---|---|
| `c24` | 24 | (미지정) | |
| `c48` | 48 | (미지정) | 기준 — 기존 g16 telemetry와 같은 값 |
| `c96` | 96 | (미지정) | |
| ★`c48m96` | 48 | **96 명시** | ★**대조** — KV 예산을 움직이는 것이 **cap인가 mamba pool인가** |

**통제**: 같은 노드·같은 커밋·`Zyphra/Zamba2-2.7B`·`--context-length 4096`·`--attention-backend triton`·
`--disable-radix-cache`·`--mem-fraction-static 0.82`·`--enable-pdmux --pdmux-config-path pdmux_d44.yml`·
`--chunked-prefill-size -1`·`--disable-overlap-schedule`. **셀 순서 고정**(c24 → c48 → c96 → c48m96),
셀마다 **서버 완전 종료 후** 다음 부팅.

## 2. 읽는 값 (배너 3줄 + 1)

각 셀에서 서버 로그로부터:
- `max_mamba_cache_size` (엔진이 **확정한** mamba 슬롯 수)
- `max_total_num_tokens` (attention KV 토큰 예산)
- `max_running_requests` (★엔진이 **실현한** cap — CLI 인자가 아니다, 감사 B7 / 교훈 *"pin은 target 아닌 realized로 검증"*)
- (부수) mamba conv/ssm GB

## 3. ★ 결정 규칙 — **실행 전에 등록한다**

`c ∈ {24, 48, 96}`(명시 mamba 미지정 셀)에 대해:

```
P1 (파생)      := 모든 c 에서  max_mamba_cache_size == c
P2 (반대 방향)  := max_total_num_tokens 가 c 에 대해 STRICTLY DECREASING  (c24 > c48 > c96)
P3 (실현=요청)  := 모든 c 에서  배너 max_running_requests == c
C1 (대조)      := c48m96 의 max_mamba_cache_size == 96  AND
                  |max_total_num_tokens(c48m96) − max_total_num_tokens(c96)| / max_total_num_tokens(c96) < 0.01
```

| 라벨 | 조건 | 뜻 |
|---|---|---|
| `ARITHMETIC_CONFIRMED` | P1 ∧ P2 ∧ P3 ∧ C1 | 항등식이 **실측됐다**. KV 예산을 움직이는 것은 **mamba pool 크기**이고 cap은 그 입력이다 |
| ★`KV_BUDGET_NOT_OPPOSED` | ¬P2 | **"반대 방향" 주장이 반증된다** — 사용자 설명과 감사 판정 양쪽을 정정해야 한다 |
| ★`CAP_SILENTLY_CLAMPED` | ¬P3 | **라벨 ≠ 실현.** cap arm 실험 전체가 이 위에 서 있었으므로 **정본 경고 사안** |
| `DERIVATION_BROKEN` | ¬P1 | `// dp` 파생이 이 설정에서 성립 안 함 |
| ★`CONTROL_DISAGREES` | P1∧P2∧P3 ∧ ¬C1 | KV 예산이 cap을 따라 움직이지만 **mamba pool을 경유하지 않는다** ⇒ 기전 서술이 틀렸다 |
| `MEASUREMENT_ABSENT` | 어느 셀이든 배너 부재/부팅 실패 | 판정 안 함(게이트 #21) |

★**부수 관측(판정 아님, 반드시 기록)**: `c96`에서 mamba 메모리가 96 × ≈68.9 MB ≈ 6.6 GB이므로
`rest_memory`가 줄어 **부팅이 거부될 수 있다**(`MemoryPoolConfig.__post_init__`). 그러면 그 셀은
`MEASUREMENT_ABSENT`이고 **그 자체가 감사 주장 3-(3)의 확증**이다 — 별도로 기록한다.

## 4. 예산

부팅 4회 × ≈33 s(Zamba2-2.7B 실측 n=10: 평균 33.3 s, 중앙값 32.8 s) + 종료 여유
⇒ `--time=00:15:00`, **등록가 = 벽시계 상한 0.25 GPU-hr**(비exclusive `--gres=gpu:1`).
실측 예상 ≈0.02 GPU-hr는 **참고치**.

## 5. 아티팩트

`nsl_ea_banner_<jobid>.json` **단일 정본**(게이트 #56). 필드: 셀별 3값 + 실현 cap + mamba GB +
`git_head` · 노드 · driver/CUDA/torch · 라벨 · P1–P3·C1 각각의 참거짓 · 각 셀 부팅 시간.

## 6. ★ 쓰면 안 되는 문장

- ✗ *"cap이 문다 / 안 문다"* — **감사 B1로 점 술어 무효.** 이 실험은 그것에 답하지 않는다.
- ✗ *"E-A가 B3를 닫았다"* — confound를 **측정된 것으로 바꿀 뿐** 제거하지 않는다. ③은 여전히 필요하다.
- ✗ *"NSL-1이 admission 축을 쟀다"* · *"cap 축이 무력함이 확인됐다"* · *"rev3이 규칙층을 통과했다"*.
- ✗ **어떤 지연·처리량·goodput 문장도** — 요청이 0건이다.
- ✗ *"cap을 올리면 크래시한다"* — 감사 주장 3-(3)이 반증했다. 부팅 거부 또는 조용한 클램프가 먼저다.
- **승계**: HE0 · gate #13/#16 · switch-cost "닫았다" · C2 인용정지 (a)(b) · job 870301 인용정지. `[CS-OK]` — 이 줄은 인용정지를 **선언**하는 것이지 그 수치를 인용하지 않는다(교훈 #54의 자기 인용 형태: 정지를 선언하는 문서가 그 정지에 걸린다).
