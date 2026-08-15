# engine-port environment and reproduction handoff

연구 결론과 다음 실험의 정본은
[`../../PROJECT_STATUS.md`](../../PROJECT_STATUS.md)와
[`reports/paper/`](../../reports/paper)다. 이 문서는 환경 복구만 다룬다.

## Runtime

- editable tree: `/scratch/ehmoon/whlee/sglang_engine_dev/python`
- virtualenv: `/scratch/ehmoon/whlee/sglang_engine_venv`
- SGLang base: v0.5.10
- cluster stack: PyTorch 2.9.1+cu130, CUDA 13, A100 108 SM
- Zamba2 serving: bfloat16, Triton attention backend

실행 전에 tracked PD-mux source와 thread-local role patch를 적용하고 hash
manifest를 만든다.

```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh
python -m unittest discover -s workspace/engine-port/tests -v
```

CPU 회귀는 **노드 구분 없이** 돌려도 된다 — 로그인 노드에서 **140 tests OK
/ 약 73초**(대부분이 `import sglang.srt.managers.scheduler` 70.6초, 2026-08-11
세션 측정). "컴퓨트 노드에서 돌려야 한다"는 재현되지 않았다; 과거 관측된
7분 stall은 **Lustre 콜드 캐시로 추정**되며 코드 사실로 확정된 것은 아니다.
별개로 `results/p1_gates/gate2/g2s_run.sbatch:91`은 **이미 컴퓨트 노드에서**
전체 회귀를 차단 게이트로 돌린다(실패 시 `exit 5`) — 이 sbatch를 제출하면
로그인 노드 실행과 별개로 컴퓨트 노드에서도 같은 회귀가 다시 통과해야 한다.

`sync_engine_tree.sh`는 dual runtime, mixin, profile, controller, telemetry와
`parallel_state.py` thread-local role patch를 설치한다. 실험 manifest에는
생성된 SHA-256 파일을 보존한다.

## 하드웨어 노드 이질성 (2026-08-14 신설)

★**로그인 노드 `glogin01`과 컴퓨트 노드(`amd_a100nv_8` 파티션, gpu36–43 등)는
GPU SKU가 다르다** — `glogin01`은 **A100 80GB PCIe**(1935 GB/s), 컴퓨트
노드는 **A100 80GB SXM4**(2039 GB/s, 400W TDP+NVLink)다. 컴퓨트 노드끼리는
동질(`scontrol show node gpu36 gpu40 gpu43` 전부 `AvailableFeatures=
A100-80GB_8,hwperf`). 발견 경위: `workspace/engine-port/results/
smsplit_realized/PREREG_SMSPLIT_REALIZED_2026-08-14.md`(addendum 1·2)가
두 노드에서 `torch.cuda.get_device_name()`을 각각 기록해 차이를 드러냈고,
이 오식별이 `results/s8_scaleup/TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`에
실제로 들어가 있었음을 §11 addendum으로 정정했다(사양 BW 1935→2039 GB/s,
성능 판정은 불변).

**실무 규칙**: green-context 생성 등 **드라이버 층만 건드리는 프로브는
SLURM 할당 없이 `glogin01`에서 검증 가능**하다(비용 0) — 단 **하드웨어
사양(대역폭·클럭·SM 개수 등)에 의존하는 판정은 반드시 컴퓨트 노드에서
재확인**하라. `nvidia-smi -q` 등 하드웨어 사양 인용은 **측정이 실제로
실행된 노드**에서 읽어 job 아티팩트에 기록해야 한다 — 로그인 노드에서
읽은 값을 컴퓨트 노드 캠페인 문서에 쓰지 말 것(`reports/CONSENSUS.md`
§3 항목46 추가 사례, `PROJECT_STATUS.md` "방법론 게이트" #32 参照).

## Git push 이력 — 운영 사실 (2026-08-14 신설, 같은 날 재조사로 프레이밍 정정)

이 절은 **운영 사실**만 기록한다(연구 결론 아님, `reports/CONSENSUS.md`에는
등재하지 않는다).

★★**정정(2026-08-14, 같은 날 2차 — 전체 reflog 재조사)**: 이 절의 초판과
2026-08-13 handoff §5-1이 함께 만든 "미승인/출처 불명 push" 프레이밍은
**오도였다**. `origin/main` reflog(`git reflog show --date=iso
origin/main`, 165줄)를 전량 재확인한 결과 **163/165개 항목이 `update by
push`**이고, 나머지 2개는 최초 `fetch: storing head`(2026-05-07)와
`pull --tags`(2026-07-22) 뿐이다 — 즉 **push 이력은 이 저장소 시작
시점(2026-05-07)부터 죽 있었다**, 2026-08-07 이후에 새로 나타난 패턴이
아니다. 초판이 인용 범위로 삼았던 2026-07-27~2026-08-14 구간만 봐도
**push 23건이 19일 중 10개 날짜에 분산**돼 있고(07-27×3·07-28×3·
07-29×1·07-31×5·08-02×2·08-03×3·08-07×1·08-10×1·08-11×3·08-14×1)
각 활동일은 1–5건 — "그날그날은 1–5회, 매일은 아님"이 정확한 표현이다
(★"내내 하루 1–5회"라는 표현은 활동이 없던 9개 날짜를 가려 과잉
단순화였다). ⇒ **이 저장소에서 push는 이상 동작이 아니라 확립된
일상 워크플로**다. 2026-08-11 핸드오프의 "미승인/출처 불명" 서술도
같은 오독(정상 워크플로를 이상 신호로 잘못 표시)일 가능성이 높다 —
단 이 판단은 doc-steward의 문서 정합 확인이지 별건 재조사가 아니므로
handoff 문서 자체는 여기서 소급 수정하지 않는다.

**아래는 초판이 기록한 개별 커밋·시각 사실**(이 값들 자체는 재확인
결과 변하지 않았다 — 프레이밍만 정정됐다): 2026-08-07~2026-08-14 구간
`update by push` 6건 — `2a92938`(08-07 13:14:52) · `558c467`(08-10
11:45:26) · `8f7e536`(08-11 01:22:55) · `4dc6191`(08-11 02:42:20
커밋/10:08:07 push) · `87f6038`(08-11 15:01:10 커밋/15:44:59 push) ·
`de501fd`(08-14 18:38:09 커밋/18:45:19 push). 6개 커밋 전부 author가
`asuan99`(git user, gitStatus와 일치)이고 커밋 메시지가 그 시점 세션의
실제 작업(Gate 1-b/Gate 2-S harness/G1-c/traffic-roofline diagnostic/
E-1 prereg)과 일치한다. `4dc6191`(02:42 커밋/10:08 push)의 커밋↔push
시각 격차는 **병렬 Claude 세션(`whlee-f0`·`whlee-aa`, `/tmp/claude-<uid>/`
프로세스 트리 존재로 확인)이 설명 가능한 가설**이지만 어느 세션이
실제로 push했는지 저장소 기록만으로는 판별 불가 — 위 재프레이밍이
이 가설 자체를 반증하진 않는다(그저 "이상 징후"로 볼 근거가 약해졌을
뿐). 2026-08-14 세션(이 문서를 갱신하는 세션 포함) 에이전트 로그엔
`git push` 실행 0건, git 훅/cron/VSCode 자동 push 설정도 없음을
직접 확인했다(`.git/hooks/` `.sample` 외 없음, `crontab -l` 무등록,
두 `settings.json` 모두 grep 무매치).

**남는 실질 사안은 하나뿐이다**: ★★**원격 URL에 GitHub PAT가 평문
노출**돼 있다(`.git/config`의 `[remote "origin"] url =
https://<user>:<PAT>@github.com/...`, `credential.helper = store`로
로컬에도 캐시됨). **PAT 값 자체는 이 문서를 포함 어떤 문서에도 절대
쓰지 않는다.** push 빈도·출처는 더 이상 조사 대상이 아니다(확립된
워크플로로 재분류) — 남는 조치는 PAT 노출 해소뿐이며 이는 doc-steward
소관 밖(engine-porter/운영자 조치)이다.

## Runtime modes

| 환경 변수 | 의미 |
|---|---|
| unset | legacy baseline |
| `PDMUX_DUAL_WORKER=1` | R1 observer 재현 전용; architecture separation 아님 |
| `PDMUX_TRUE_DUAL_WORKER=1` | 두 host issue thread를 사용하는 R2 experimental path |
| `PDMUX_R2_POLICY=fixed\|generic\|hybrid` | safe-boundary controller 선택 |
| `PDMUX_MODEL_PROFILE=<json>` | Hybrid policy의 versioned offline profile |
| `PDMUX_TELEMETRY_PATH=<path>` | 모든 arm에 같은 async telemetry |
| `PDMUX_RUN_ID`, `PDMUX_WORKLOAD_ID` | trace 식별자 |

True dual은 thread-local role capability가 없으면 fail-fast한다. GPU correctness
gate를 통과하기 전에는 기본값을 켜지 않는다.

## SLURM `--comment` (제출 필수)

2026-08-12 18:00 KST 이후 뉴론은 응용 분야·프로그램명이 없는 job을 **제출 거부**한다.
모든 job script는 `#SBATCH` 블록 안(첫 실행 라인 위)에 아래를 갖는다:

```bash
#SBATCH --comment="field=efficientai;appl=pytorch"
```

`field=efficientai`(Efficient & Scalable AI Systems) · `appl=pytorch`(SGLang은
`showappl` 목록에 없어 PyTorch 기반으로 신고 — `vllm`은 다른 엔진이라 쓰지 않는다).
인터랙티브는 `salloc ... --comment="field=efficientai;appl=pytorch"`.
**CLI에서 `--comment`를 덮어쓰지 말 것** — script directive가 있어도 CLI 인자가
이를 무효화해 제출이 거부된다. **새로 만드는 job script도 같은 형식**을 따른다.
저장소 내 job 제출 경로 124개(`.sbatch` 97 + `.sh` 26 + `.tmp` 1 + CLI 4경로
`submit.sh`/`run_pipeline.sh`/`submit_size_sweep.sh`/`interact_scheduler.sh`)는
2026-08-14 커밋 `31b3e96`으로 전부 적용됐다. 검사·정정(`86179ec`, 내용 기반
스캔이라 `.sh`로 저장된 job script도 잡는다):

```bash
python3 workspace/engine-port/scripts/bootstrap/check_sbatch_comment.py [--fix]
```

역사적 `PREREG_*.md`가 옛 형식(`--comment=pytorch`)을 그대로 인용하는 것은
그 시점 사전등록 기록을 보존하기 위한 **의도적 조치**이며 손대지 않는다.

## R2 campaign

```bash
workspace/engine-port/scripts/r2_eval/generate_campaign.sh
PDMUX_CAMPAIGN=workspace/engine-port/results/r2_eval/campaign.json \
  sbatch --array=0-<N-1> workspace/engine-port/scripts/r2_eval/r2_eval.sbatch
```

기본 runner는 CUDA Graph를 켠 SGLang server와 immutable SSE trace load
generator를 사용한다. 필요하면 `PDMUX_R2_RUNNER` 또는
`PDMUX_ENGINE_BENCH_RUNNER`로 pinned adapter를 교체한다. 각 array entry는
immutable run record, source hash, telemetry를 별도 디렉터리에 남긴다.
과거 manual dev-tree edit 목록은
[`env/dev_tree_edits.md`](env/dev_tree_edits.md)에 역사 기록으로 유지한다.
