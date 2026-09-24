# GPU 대여 전/직후 체크리스트 — 기판 판정 (2026-09-18)

이 문서는 `migration_plan_local_dev_remote_gpu_2026-09-18.md` §2("GPU 선택 기준")를 **검증
가능한 형태**로 푼 것이다. 짝 문서: 같은 §2 · `session_handoff_2026-09-17.md`(이양 체크포인트).

> **지위**: 이 문서는 **새 성능 판정 0건 · 새 측정 0건 · GPU 지출 0**이다. 내용은 (a) upstream
> SGLang v0.5.10 소스의 1차 확인과 (b) 저장소 안 설정 파일의 확인, 두 가지뿐이다. 어떤
> 연구 결론·Claim 등급·게이트도 이 문서로 바뀌지 않는다.
>
> **기판 주의(CLAUDE.md 방법론 게이트 연장)**: 옮긴 기판의 수치를 옛 기판(A100 108 SM)
> 결론과 섞지 않는다. 아래 어떤 항목도 "재측정해도 된다/안 해도 된다"를 **판정하지 않는다** —
> 그건 사용자 결정 + 규칙층 감사 사항이다.

## 0. ★선행 확인에서 드러난 소스 사실 (1차, upstream 소스 직접 확인)

2026-09-18 로컬에서 upstream `v0.5.10` 태그의 두 파일을 받아 직접 읽었다(GPU 0).

**(가) `python/sglang/srt/multiplex/pdmux_context.py`**

```python
def get_arch_constraints(compute_capability):
    major, minor = compute_capability
    if major == 6:   return 1, 1      # min_per_part, multiple
    elif major == 7: return 2, 2
    elif major == 8: return 4, 2
    elif major == 9 and minor >= 0: return 8, 8
    else: raise ValueError(f"Unsupported compute capability: {major}.{minor}")
```

**(나) ★`get_arch_constraints`는 `manual_divisions`를 쓰면 호출되지 않는다.**
같은 파일 `initialize_stream_groups`:

```python
if config.manual_divisions:
    divisions = [(p, d) for p, d, _ in config.manual_divisions]   # ← arch 분기 미경유
else:
    divisions = divide_sm(total_sm_count, torch.cuda.get_device_capability(device), ...)
```

`get_arch_constraints`는 `divide_sm` 안에서만 불린다. 그리고 **이 프로젝트는 전적으로
`manual_divisions` 경로를 쓴다** — `benchmarks/configs/pdmux_r2.yml`(`sm_group_num: 6`,
`[92,16,0]/[84,24,0]/[74,34,0]/[64,44,0]`), `results/stage0_xctrl/pdmux_d{16,44,92}.yml`
(`sm_group_num: 3`) 등 확인된 설정 전부가 `manual_divisions`를 명시한다.

★**스코프 정정(2026-09-22, 코드 근거 감사 `reports/audit/2026-09-22_scope_lineage/
REPORT.md` §6 I-1, doc-steward 등재 2026-09-24, 사용자 승인 2026-09-24)**: 위
"전적으로"는 grep 기반 확인이었고 거짓이다. `yaml.safe_load` 전수 파싱 결과 현행
`*pdmux*.yml` **55/59**만 `manual_divisions`를 쓴다. 나머지 4개(byte-identical
`pdmux_a100_smoke.yml`, sha256 `8e991318…`)는 이 키가 **없어** 자동 격자(`divide_sm`)
경로로 가며(주석이 "No manual_divisions -> `divide_sm()`"라 적어 grep이 오검출),
이 config를 쓰는 job script 27개에 **P1.7 4모델·P1-opint(873944/873945)·
`p1_gates/gate2` HOLB**가 있다. 아래 "우리 실행 경로에서 그 분기는 발화하지 않는다"는
**`manual_divisions` 계열(E1/S2/λ0/E2) 한정**으로 좁아진다 — A100(major 8)에서는
자동 격자 캠페인 결과도 안 바뀌므로 §0(나)의 실질 결론은 불변. 상세
`CONSENSUS.md` §3 항목280.

⇒ **따라서 "major 10+에서 `ValueError`로 거부된다"는 자동 격자 경로의 사실이며, 우리
실행 경로에서 그 분기는 발화하지 않는다.** 우리 경로에서 실제로 제약을 강제하는 것은
`sgl_kernel.spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)`
= CUDA green context API다.

**이것은 정본(`CLAUDE.md` "환경 / 실행")과 어긋나는 기술 서술이다 — doc-steward 회부
대상**(§5). 단 **로컬 RTX 5060 Ti에서 실험하지 않는다는 결론은 바뀌지 않는다**: 정본이
든 나머지 두 사유(`sgl_kernel`에 sm120 빌드 없음 · 16 GB로 9B 모델 불가)가 그대로
유효하고, 무엇보다 **게이트 1(기판이 다르면 수치가 주장 근거가 못 된다)**이 독립적으로
막는다. 바뀌는 것은 "왜 안 되는가"의 정확도뿐이다.

**(다) 드라이버 요구는 확증됨.** `sgl-kernel/python/sgl_kernel/spatial.py`의 import 실패
메시지: `"Failed to load sgl_kernel.spatial_ops extension. Ensure CUDA Driver >= 12.4"`.

**(라) `divide_sm`의 자동 격자는 어차피 우리 격자를 만들지 못한다.** 필터가
`x >= total_sms - x and total_sms - x >= 16`(x=prefill SM) 이라 A100 108에서
prefill ≥ 54만 나온다 ⇒ decode ≤ 54. 우리 격자의 D92·D108은 자동 경로로는 생성 불가.
`manual_divisions`를 쓰는 이유가 여기 있다(이 사실 자체는 재확인이며 새 판정이 아니다).

## 1. 후보 기판 판정표

| 기판 | compute cap | 총 SM | 판정 | 근거·전제 |
|---|---|---:|---|---|
| **A100 80GB 1장** | 8.0 | 108 | ★**직접 비교 가능** | 지금까지의 **모든** 측정 기판. 격자 (92,16)/(84,24)/(74,34)/(64,44)·λ\*·E2 등록이 그대로 성립 |
| A100 **40GB** | 8.0 | 108 | ⚠**같은 기판 아님** | SM 수는 같으나 KV 예산이 달라진다(`--mem-fraction-static 0.82`·`--max-running-requests 48` 전제). 용량·λ\*는 **재측정 대상** |
| L40S / RTX 40 | 8.9 | 142 / — | ⚠ 총 SM 다름 | arch 분기는 같은 major 8이나 **총 SM이 달라 격자가 통째로 바뀐다** |
| **H100 / H200** | 9.0 | 132 | ⚠**격자 재설계 + λ0 재측정 전제** | §4 참조. E2 사전등록은 **그대로 성립하지 않음**(미결 상태이지 성능 판정 아님) |
| **Blackwell** (B200·RTX 50) | 10.x / 12.x | — | ⚠**미지**(단순 "불가"가 아님) | §0(나)에 따라 `manual_divisions` 경로는 arch 분기를 우회한다. 실제 관건은 **`sgl_kernel`에 해당 SM 아키텍처 빌드가 있는가 + green context가 동작하는가**이며 **미실측**이다 |
| MIG / vGPU 분할 인스턴스 | — | — | ❌**불가** | green context가 GPU 전체를 전제 |

## 2. 계약 **전** 확인 (업체 스펙·문의로 답이 나오는 것)

각 항목은 "예/아니오"로 답이 떨어지게 썼다.

1. **GPU 모델과 메모리**가 정확히 무엇인가? (`A100 80GB PCIe/SXM` / `H100 80GB` / …)
   → **A100 80GB면 §4를 건너뛴다.** 40GB면 같은 기판이 아니다(표 참조).
2. **1장 전체를 점유**하는가? **MIG·vGPU·분할 인스턴스가 아닌가?** → 분할이면 탈락.
3. **NVIDIA 드라이버 버전**은? CUDA 드라이버 **≥ 12.4**인가? → 미만이면 탈락(§0(다)).
4. **디스크 여유 ≥ 100 GB**인가? (모델 Nano-9B-v2 ≈ 17 GB + 실행당 telemetry 수백 MB–수 GB;
   참고로 누적 결과는 41 GiB/5,210 파일까지 자랐다)
5. **인스턴스가 반납·재기동되면 디스크가 날아가는가?** → 날아간다면 §3-6(회수 규약)이
   더 엄격해진다: **실행 종료 = rsync 회수 + sha256 완료**까지가 한 작업.
6. **과금 단위**(시간·분)와 **단가**는? → E2 등록 예산 1.803 GPU-h(최악 2.338, 하드 캡 3.0)에
   곱해 둔다. 누적 실적 ≈462 GPU-h가 전체 비용 감각의 출발점.
7. **ssh + rsync 아웃바운드**가 되는가? (Claude Code는 로컬에서 돌고 원격은 ssh로만 부린다)
8. HuggingFace·PyPI **아웃바운드 네트워크**가 열려 있는가? (부트스트랩이 모델·패키지를 받는다)

## 3. 접속 **직후** 기판 검증 순서 (GPU를 쓰지만 실험은 아님 — 배관 확인)

순서대로, 앞이 실패하면 뒤로 가지 않는다. **B3까지 통과하기 전에는 어떤 캠페인도 제출하지 않는다.**

- **B0. 기계 사실 채취** — `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv`,
  `nvidia-smi -L`, `df -h`. 출력을 그대로 결과 디렉터리에 남긴다(나중에 기판 귀속의 1차 근거가 된다).
- **B1. 환경 재구성** — `migration_plan...md` §3-3 순서: torch/CUDA → SGLang **v0.5.10 소스** →
  `sync_engine_tree.sh` → `env/devtree_manual_edits.patch` → 모델·trace.
  패키지 고정값 `env/venv_packages_2026-09-17.txt`, py3.14 shim `env/sitecustomize.py`.
- **B2. CPU 회귀** — `python -m unittest discover -s workspace/engine-port/tests -v`.
  로컬에서 통과하는 것과 **같은 결과**여야 한다.
- **B3. ★SM 입도 probe (이 기판이 우리 격자를 실제로 줄 수 있는가)** — 가장 중요한 단계.
  총 SM을 읽고, **쓰려는 분할값 각각에 대해 green context 생성이 성공하는지, 그리고 실제로
  부여된 SM이 요청값과 같은지** 확인한다:

  ```python
  from sgl_kernel import spatial
  print("total_sm =", spatial.get_sm_available(0))
  for p, d in [(92,16), (84,24), (74,34), (64,44)]:      # A100 격자. 다른 기판이면 후보를 바꿔라
      try:
          s = spatial.create_greenctx_stream_by_value(p, d, 0)
          print(p, d, "OK", s)
      except Exception as e:
          print(p, d, "FAIL", type(e).__name__, e)
  ```

  ★**"생성 성공"을 "요청한 SM 수를 받았다"로 읽지 마라.** 이 프로젝트는 pin을 **target이
  아니라 realized로 검증**하라는 교훈을 두 번 값비싸게 배웠다(Stage 0 D108 앵커가 실은
  16 SM이었던 건, λ0에서 λ\*(A)의 D44 점유가 실은 소수였던 건). 저장소에 이미 있는
  **`src/multiplex/green_readout.py`**와 `%smid` 계열 probe(`results/bcg_probe/`)가 그
  확인 수단이다 — 새로 만들지 말고 그것들을 이 기판에서 먼저 돌려라.
- **B4. 부팅 스모크** — 엔진이 실제로 뜨고 짧은 요청이 왕복하는지. `triage/pdmux_a100_smoke.yml`
  계열이 선례다(기판이 A100이 아니면 이 yml은 그대로 못 쓴다 — §4).
- **B5. 회수 경로 실증** — 더미 결과 하나를 `results/<campaign>/`에 만들고 로컬로 rsync +
  sha256 대조까지 **끝까지 한 번** 돌려 본다. 진짜 캠페인이 끝난 뒤에 회수가 처음이면 늦다.

## 4. H100/H200을 고르는 경우 — 재측정 범위 산정

"A100이 아니면 얼마나 일이 늘어나는가"에 대한 **범위 산정**이다(비용 추정용, 판정 아님).

| 항목 | A100(현재) | H100에서 어떻게 되나 |
|---|---|---|
| 총 SM | 108 | **132** ⇒ 모든 분할값의 합이 바뀐다 |
| 격자 | (92,16)(84,24)(74,34)(64,44) | **통째로 재설계**. 합이 132여야 하고 입도 제약을 통과해야 한다 |
| 입도 | major 8 → `(4, 2)` | upstream은 major 9 → `(8, 8)`이라 적는다. ★단 §0(나)에 따라 **우리 경로(`manual_divisions` 계열)는 이 상수를 타지 않는다** ⇒ 실제 입도는 **B3로 실측해야 한다**(코드 상수를 실측 대신 인용하지 말 것). ★**스코프 정정(2026-09-22 감사 I-1, 2026-09-24 등재)**: 이 프로젝트의 **자동 격자** 캠페인(P1.7·P1-opint·`p1_gates/gate2`)을 다른 기판에서 재실행하면 이 상수가 **직접 적용**된다 — H100이면 major 9 `(8,8)` 상수를 실측 없이 인용하지 말고 여전히 B3로 확인할 것 |
| λ\* | λ\*(A)≈3.05 req/s · λ\*(B)≈0.696 req/s (job 908623, A100) | **재측정 필수**(λ0 트랙 재실행 ≈1.39 GPU-h가 선례 비용) |
| 용량·SLO 임계 | A100 기준 | 재측정. goodput은 metric cliff가 있으므로 **용량을 먼저** 측정한다 |
| **E2 사전등록** | 20 boot / 1.803 GPU-h 등록·감사 5회 GO | ★**그대로 성립하지 않는다.** 격자·λ\*가 등록의 일부다 ⇒ 재등록 + 규칙층 재감사 + **새 OVERRIDE**(OVERRIDE 문서 §은 "설계나 예산이 바뀌면 새 OVERRIDE"라고 명시) |
| 과거 결과와의 비교 | — | ❌ **섞지 마라.** 기존 결론(HE0·layer-type 死·정책 순위)은 A100 기판에서 확정된 것이고, 새 기판 수치로 그것을 보강도 반박도 **자동으로는** 할 수 없다 |

⇒ **H100 선택의 실질 비용 = (λ0 재측정 ≈1.4 GPU-h) + (격자 재설계·재등록·감사 회차) +
(E2 새 OVERRIDE) + (기존 결론과의 비교 가능성 상실).** A100 80GB가 구해지면 이 전부가 0이다.
**A100 80GB 1장을 최우선으로 찾을 것을 권고한다**(이건 운영 권고이지 연구 판정이 아니다).

## 5. doc-steward 회부 사항 (정본 정정 후보 1건)

★**추가 정정(2026-09-22, 코드 근거 감사 `reports/audit/2026-09-22_scope_lineage/
REPORT.md` §6 I-1, doc-steward 등재 2026-09-24, 사용자 승인 2026-09-24)**: 아래
회부 당시 이 문서 자신이 "우리 실행 경로(`manual_divisions`)"를 전부라고 전제했으나
`yaml.safe_load` 전수 파싱으로 **55/59**임이 드러났다(자동 격자 경로로 가는 4개는
byte-identical `pdmux_a100_smoke.yml`). 아래 정정 방향 제안은 **여전히 유효**하나
"이 프로젝트가 전적으로 쓰는 경로"라는 전제 문구는 "`manual_divisions` 계열(E1/S2/
λ0/E2)"로 좁혀 읽어야 한다. A100(major 8) 결과 자체는 바뀌지 않는다. 상세
`CONSENSUS.md` §3 항목280, `PROJECT_STATUS.md` 2026-09-24 정정 배너.

`CLAUDE.md` "환경 / 실행" 항목은 로컬 GPU 거부 사유를 이렇게 적는다:

> "PD-mux 경로가 **엔진 단계에서 거부된다**(dev tree `pdmux_context.py:get_arch_constraints`는
> major 6–9만 지원, major 10+ → `ValueError`)"

§0(나)의 1차 확인에 따르면 이 인과는 **우리 실행 경로(`manual_divisions`)에서는 발화하지
않는다**. 정정 방향 제안(문구는 doc-steward가 정한다):
- `get_arch_constraints`의 major 제한은 **자동 격자(`divide_sm`) 경로**의 사실이라고 스코프를 붙인다.
- 로컬 거부의 **실효 사유**는 (i) `sgl_kernel`에 sm120 빌드 없음 (ii) 16 GB로 9B 모델 불가
  (iii) **게이트 1**(기판이 다르면 수치가 주장 근거가 못 됨)로 유지한다 — 결론은 불변.
- §4의 "H100은 8 SM 단위"도 같은 스코프가 붙어야 한다(코드 상수이지 우리 경로의 실측 아님).

## 6. 근거 목록 (1차/2차 구분)

**1차(이 세션에서 직접 읽음)**
- upstream **태그 `v0.5.10`** `python/sglang/srt/multiplex/pdmux_context.py` — `get_arch_constraints`,
  `divide_sm`, `initialize_stream_groups`(manual_divisions 분기).
- upstream **태그 `v0.5.10`** `sgl-kernel/python/sgl_kernel/spatial.py` — 드라이버 ≥ 12.4 메시지.
  ★**조달 경로 명시**(2026-09-18 규칙층 감사 Q4의 지적 반영): 두 파일은
  `curl -sSL https://raw.githubusercontent.com/sgl-project/sglang/v0.5.10/<path>`로 받았고
  세션 scratchpad에 `pdmux_v0510.py`(5,126 B)·`spatial.py`(1,808 B)로 남아 있다. 저장소에는
  dev tree 사본이 없어 **저장소 안에서는 재현 불가**이며, scratchpad는 세션 종료 시 사라진다.
  재확인이 필요하면 같은 URL을 다시 받아라. 감사자는 이 태그 사본에 접근하지 못해 대신
  로컬 conda 환경의 **설치 wheel `sglang-0.5.10.post1`**로 대조했고(구조 결론 일치),
  추가로 `0.5.3rc0` 소스와도 대조해 `get_arch_constraints`·`divide_sm`·`manual_divisions`
  분기가 **바이트 동일**함을 확인했다. ⇒ **태그와 `post1`의 바이트 동일성 자체는
  `UNDETERMINED`**이나, 세 버전에서 구조가 같으므로 §0 결론에 실질 위험은 없다.
- 저장소: `benchmarks/configs/pdmux_r2.yml`, `results/stage0_xctrl/pdmux_d44.yml`,
  `scripts/bootstrap/sync_engine_tree.sh`(install 대상 8파일에 `pdmux_context.py` 없음),
  `env/devtree_manual_edits.patch`(대상 5파일에 `pdmux_context.py` 없음)
  ⇒ **우리 실행 트리의 `pdmux_context.py`는 upstream 원본 그대로**이므로 위 1차 확인이 우리 런타임에 유효하다.
- `scripts/r2_eval/engine_bench_runner.sh:83` — `--mem-fraction-static "${PDMUX_MEM_FRACTION:-0.82}"`.

**2차(문서를 인용한 것, 재확인 안 함)**
- λ\*(A)≈3.05 / λ\*(B)≈0.696 req/s, job 908623 실지출 1.391 GPU-h — λ0 판정서.
- E2 예산 1.803 / 2.338 / 3.0 GPU-h, 20 boot — `OVERRIDE_E2_SUBMIT_2026-09-15.md` §5.
- 누적 ≈462 GPU-h, 모델 ≈17 GB, 결과 41 GiB — `session_handoff_2026-09-17.md`.

**미실측(이 문서가 답하지 않는 것)**
- Blackwell에서 `manual_divisions` 경로가 실제로 동작하는지 — **미지**(§1 표).
- H100의 실제 green context 입도 — B3로만 확정된다.
- 대여 인스턴스의 성능 재현성(공유 호스트·클럭 정책) — 접속 후 B0/B3 이후 문제.
