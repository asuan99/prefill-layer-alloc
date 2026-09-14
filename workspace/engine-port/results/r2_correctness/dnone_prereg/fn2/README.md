# `fn2/` — F-n2 (피크 축) 재현 스크립트 (job 908534)

`FN2_PEAK_ANALYSIS_908534_2026-09-14.md`의 표·수치를 만든 스크립트다. result-analyst가 작성하고
메인 세션이 저장소로 옮기면서 **경로를 자기 위치 기준(`Path(__file__).parent`)으로 고쳤다**
(원본은 세션 스크래치 절대경로를 박아 놔서 저장소에서는 재현 불가였다).

## 재현
```bash
cd ../../job_908534 && python3 ../dnone_prereg/fn2/table.py     # §4 표 그대로
```

## ★provenance 한계 (반드시 병기)
- **1차 입력 `job_908534/tel_*.jsonl`은 git 밖이다**(`workspace/engine-port/.gitignore`의 `*.jsonl`)
  ⇒ **fresh clone은 원자료에서 재도출할 수 없다.** 저장소에 남은 것은 파생 JSON 3개
  (`epochs.json` = epoch별 피크, `batches.json` = 배치열, `rows.json`)이며 **이 셋이 저장소 안의
  provenance 정본**이다. `table.py`는 앞 둘만 읽으므로 표는 저장소만으로 재생성된다.
- **`recs.json`(15.7 MB, 스냅샷 원시 추출)은 저장소에서 제거했다** — (i) `epochs.json`이 그 요약이고
  `table.py`는 그것만 쓴다, (ii) 15.7 MB 단일라인 JSON이라 `check_citation_stops.py`가 **금지 수치의
  숫자 부분열**을 raw 타임스탬프 안에서 4건 오검출했다(예: `2885533.041271976` ⊃ `33.04`, `2885413.711833206` ⊃ `13.711` — 이 두 언급은 오검출 사례를 설명하기 위한 것이고 측정값 인용이 아니다) [CS-OK]. ★**그 4건은 전수 확인된 false positive**이며, 그럼에도
  **체커가 숫자 경계 없이 부분열을 매칭한다는 도구 결함은 남아 있다**(다음 세션 열린 항목 —
  수리 시 "실제 금지 인용은 여전히 잡힌다"는 양성대조가 필수다).
- `extract.py`는 그 제거된 캐시를 만드는 단계이므로 **git 밖 입력(`tel_*.jsonl`)이 있어야 돌아간다**.
