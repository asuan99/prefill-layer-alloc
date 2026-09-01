# 철회 — `spec_cp0_arm_reqinactive.json` (2026-08-28)


> ⚠️ **날짜 정정**: 파일명/본문 날짜 2026-08-28은 오기 — 실제 작성/실행일 **2026-09-01**. 상세: `DATE_CORRECTION_NOTE.md`.

**철회 사유**: 이 spec의 제약 `req_channel ∈ {inactive, unmeasured}`와 그 `why` 문자열
(*"structural, not rare … no prompt this arm could split survives the filter"*)이 **거짓 기전
주장**에 근거했다. 3회차 감사 F2, 메인 세션이 코드로 독립 확인:

```
schedule_policy.py:802-842
    elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
        ...non-chunked...
    else:
        trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
        req.set_extend_input_len(trunc_len)      # <- 절단 = chunking
```

절단 기준은 **남은 배치 예산**(`rem_chunk_tokens`)이지 요청 길이가 아니다. ⇒ `cps 4096`에서도
backlog가 예산을 소진시키면 **짧은 요청이 쪼개진다.** `req_channel=inactive`는 증거로 고정할 수
없다.

**파급(재도출)**:
- 이 spec의 인증서 `reach_cp0_arm_reqinactive.json`은 `ARM_DISTINCT_BOTH`·
  `ARM_DISTINCT_REQUEST_ONLY`를 `unreachable_by_design`으로 배제했다 — **HI backlog에서 가장
  일어나기 쉬운 두 라벨**을. 그 배제는 무효다.
- ★같은 spec 안의 `priors`가 이미 자기모순이었다: *"a 4096-token batch cap needs about 12 queued
  requests … reachable under HI backlog"*. 엔진에서 **cap이 무는 방식이 곧 요청을 자르는 것**이므로
  `batch_budget=binds` ∧ `req_channel=inactive`는 동시에 성립할 수 없다. prior가 맞고 restriction이
  틀렸다.
- `measure_served_population.py`의 필드명 `n_requests_split_by_cps` → **`n_prompts_longer_than_cps`**로
  정정하고(계산하던 것이 그것이다) `budget_exhaustions_if_fully_backlogged`를 병기했다
  (512→133 · 1024→66 · 2048→33 · **4096→16**). 앞의 수는 **하한**이다.

**대체하지 않는다.** `cps 4096`에는 이제 **증거에 근거한 제약이 없다**. 제약 없는 spec을 다시
등록하지 않는 이유: 빈 `restrict`는 도구의 `RESTRICTIONS_INERT` 검사를 **구조적으로 회피**한다
(`design_reachability.py`: `if restrict and _inert(...)`) — 3회차 감사 L1이 실증했다. 따라서
`cps 4096`의 도달가능성은 **전체 격자**이고, 도구 자신의 문구대로 그런 실행은
*"could not have failed"*이므로 인증서를 발급하지 않는다. 이 사실을 사전등록 본문에 적는다.

**유지되는 것**: `spec_cp0_arm_reqactive.json`(cps 512/1024/2048). 그 arm들에는 **cps보다 긴
프롬프트가 51/10/4개** 있고, cps보다 긴 프롬프트는 예산이 비어 있어도 **혼자서 한 배치에 못 들어가
반드시 쪼개진다** ⇒ `req_channel=active`는 배치 기전과 **무관하게** 과결정된다. `why` 문자열을
그 논거로 교체했다.
