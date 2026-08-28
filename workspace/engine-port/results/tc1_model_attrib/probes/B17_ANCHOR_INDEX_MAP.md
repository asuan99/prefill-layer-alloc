# B17 — `PDMUX_SLO_ANCHOR_IDX` 인덱스 사상 (코드 사실, GPU 0)

2026-08-28 · TC1 rev2 §10 선행 · **성능 판정 0건**

감사 B17: *"`lff_bench.sbatch:47` 주석은 `idx2=d24`인데 `pdmux_slo.yml`의 division 리스트는
d16부터 5개다(오프셋 1). 정수로 옮기는 사상이 미등록이다."*

## 사상 — 추론이 아니라 코드에서

`multiplexing_mixin.py:328-333` (sticky 대상 선택부, 축자):

```
            # Index 0 is the plain prefill-only group and the last index is the
            # plain unpartitioned group; neither is a green-context division,
            # so neither is a legal sticky target.
            divisions = [
                index for index in matches if 1 <= index <= self.real_sm_group_num - 2
            ]
```

`results/slo_sched/pdmux_slo.yml`: `sm_group_num: 7`, `manual_divisions` 5개.

| idx | 그룹 | `(prefill, decode)` | 라벨 |
|---|---|---|---|
| 0 | prefill-only | (108, 0) | — |
| **1** | division | (92, 16) | **d16** |
| **2** | division | (84, 24) | **d24** ← 하네스 기본 anchor |
| **3** | division | (74, 34) | **d34** |
| **4** | division | (64, 44) | **d44** |
| **5** | division | (54, 54) | **d54** |
| 6 | unpartitioned | (0, 108) | — |

`_slo_decide_idx`의 `_lo, _hi = 1, real_sm_group_num - 2` = **1..5** 와 정확히 일치한다.

⇒ ★**`PDMUX_SLO_ANCHOR_IDX=4` = d44.** 하네스 기본 `2` = d24 (주석과 일치).

## 이 문서가 말하지 않는 것
성능·정책 주장 없음. 다른 `pdmux_*.yml`(division 개수가 다름)에 이 표를 이식하지 말 것 —
사상은 `sm_group_num`과 `manual_divisions` 길이에 의존한다.
