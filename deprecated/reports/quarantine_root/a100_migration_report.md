# A100 이관 마이그레이션 보고서

**작성일**: 2026-05-02
**대상**: prefill-layer-alloc 프로젝트 (RTX 5060 Ti → NVIDIA A100)
**분석 범위**: Stage 1~3 전체 코드베이스 (src/, stage1_sm_scaling/, stage2_overhead/, stage3_hm_eval/, configs/)

---

## 1. 심각도별 분류

| 심각도 | 건수 | 설명 |
| --- | --- | --- |
| 🔴 Critical | 5 | 실험 결과가 틀리거나 스크립트 실패 |
| 🟡 Warning | 5 | 환경 설정 또는 사전 확인 필요 |
| 🟢 Non-issue | 7 | 런타임 자동 감지, 수정 불필요 |

---

## 2. 🔴 Critical — 즉시 수정 필요

### C-1. `plot_srm.py` GPU 상수 하드코딩

**파일**: `stage1_sm_scaling/plot_srm.py:57–60`

```python
# 현재 — RTX 5060 Ti 전용 상수
GPU_NAME        = "RTX 5060 Ti"
PEAK_TFLOPS_FP16 = 200.0    # A100: 312.0
PEAK_BW_GBS      = 288.0    # A100 40GB: 1555.0 / A100 80GB: 2000.0
RIDGE_FULL       = PEAK_TFLOPS_FP16 * 1e3 / PEAK_BW_GBS  # 694 → 201 (A100 40GB)
```

**영향**:

- SRM Roofline의 BW ceiling 기울기가 ~5.4배 낮게 그려짐
- Ridge point가 694 FLOPs/B(RTX) → 201 FLOPs/B(A100 40GB)로 바뀌어야 하는데 반영 안 됨
- `plot_srm_bound_analysis()`의 efficiency 지표 전체 오계산

**수정 방법**:

```python
# stage1_sm_scaling/plot_srm.py에 --hw-config 인자 추가
def load_srm_hw_config(device_key: str) -> tuple[str, float, float]:
    cfg_path = Path(__file__).parent.parent / "configs" / "hardware.yaml"
    with open(cfg_path) as f:
        hw = yaml.safe_load(f)
    cfg = hw[device_key]
    return cfg["name"], cfg["compute_fp16_tflops"], cfg["memory_bw_GBs"]

parser.add_argument("--hw-config", default="a100_40gb")
GPU_NAME, PEAK_TFLOPS_FP16, PEAK_BW_GBS = load_srm_hw_config(args.hw_config)
RIDGE_FULL = PEAK_TFLOPS_FP16 * 1e3 / PEAK_BW_GBS
```

---

### C-2. `BandwidthEstimator._query_theoretical_bw()` 속성 누락 버그

**파일**: `src/profiling/metrics.py:159–177`

```python
def _query_theoretical_bw(self) -> float:
    props = torch.cuda.get_device_properties(self.device_id)
    mem_clock_hz = props.memory_clock_rate * 1e3   # ← AttributeError 발생
    bus_width_bits = props.memory_bus_width          # ← AttributeError 발생
    ...
    except Exception:
        return 1000.0   # 항상 여기로 폴백
```

**현재 상태**: `torch.cuda.get_device_properties()`에는 `memory_clock_rate`와 `memory_bus_width` 속성이 없음 (현재 PyTorch 버전 기준). RTX 5060 Ti에서도 이미 발생 중 — Stage 1 결과 CSV의 `theoretical_bw_GBs` 열이 1000.0으로 기록되어 있는 이유.

**A100에서의 추가 영향**:

- A100 실제 BW = 1555 GB/s인데 1000 GB/s로 기록 → `bw_utilization_pct`가 155% 초과하는 비정상값 출력
- `--device auto` 시 `memory_bw_GBs: None`이 LayerRunner에 전달되어 이 경로가 반드시 통과됨

**수정**: `--device a100_40gb`를 명시적으로 지정해 `hardware.yaml`의 정확한 값(1555 GB/s)을 직접 사용

```bash
# 올바른 실행 방법
python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --device a100_40gb

# 피할 것 — BW가 1000 GB/s fallback으로 기록됨
python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --device auto
```

---

### C-3. `results/` 디렉토리 오염 — 멀티-디바이스 glob 충돌

**파일**: `stage2_overhead/compute_decision_matrix.py:55,101,145`

```python
csv_files = list(stage1_dir.glob("ssm_scaling_*.csv"))          # stage1 전체 읽음
json_files = list(stage2_dir.glob("smctrl_overhead_*.json"))     # stage2 전체 읽음
for f in stage2_dir.glob("layer_latency_*.csv"):                 # stage2 전체 읽음
```

**문제**: glob 패턴이 device 필터링을 하지 않음. `results/stage1/`에 RTX 파일과 A100 파일이 공존하면 두 기기의 saturation SM 수가 같은 dict 키 `(model, seq_len, batch_size)`로 충돌 → 후자가 전자를 덮어씀.

**실제 충돌 예시**:

```text
ssm_scaling_zamba2_geforce_rtx_5060_ti.csv   # max_sm=36
ssm_scaling_zamba2_nvidia_a100_40gb.csv       # max_sm=108
```

두 파일을 동시에 읽으면 `saturation[(zamba2, 1024, 4)]`가 어느 기기 기준인지 보장 불가.

**수정 방법**:

```bash
# 방법 A: 디바이스별 results 분리
results/stage1_rtx5060ti/
results/stage1_a100/

# 방법 B: compute_decision_matrix.py에 --device 필터 인자 추가
--device-tag a100_40gb  # glob을 ssm_scaling_*_a100_40gb.csv로 제한
```

---

### C-4. Stage 2 `layer_latency_*.csv` — A100 미측정 상태로 decision matrix 구성

**파일**: `stage2_overhead/measure_layer_latency.py`, `compute_decision_matrix.py`

현재 `results/stage2/`의 `layer_latency_*.csv`는 Stage 1 RTX 5060 Ti 데이터에서 full-SM 행을 추출해 생성된 것 (A100에서 실제로 측정하지 않음). A100에서는 다음 두 가지가 달라짐:

| 항목 | RTX 5060 Ti | A100 40GB |
| --- | --- | --- |
| full SM | 36 SMs | 108 SMs |
| SSM 1024, bs=4 latency | ~8.3ms | 예상 ~1–2ms (BW 3.5×) |
| smctrl overhead | ~4.7μs (MPS fallback) | 예상 ~2–10μs (libsmctrl) |
| overhead_ratio | 0.06% | 미측정 (policy 판단 불가) |

**필수 행동**: A100에서 Stage 2 측정을 처음부터 재실행

```bash
# A100에서 순서대로 실행
python stage2_overhead/measure_smctrl_latency.py --device a100_40gb
python stage2_overhead/measure_layer_latency.py --model zamba2 --device a100_40gb
python stage2_overhead/measure_layer_latency.py --model falcon_h1 --device a100_40gb
python stage2_overhead/compute_decision_matrix.py \
    --stage1-dir results/stage1 --stage2-dir results/stage2
```

> Stage 2 결과 없이 Stage 3를 실행하면 `should_run_policy_c()`가 RTX 기반 decision_matrix.json을 읽어 Policy C 실행 여부를 잘못 판단할 수 있음.

---

### C-5. `hardware.yaml` RTX 5060 Ti 스펙 오기재 (기존 버그)

**파일**: `configs/hardware.yaml`

```yaml
# 현재 — 잘못된 값
rtx_5060ti:
  memory_bw_GBs: 448        # 실제: 288 GB/s (GDDR7 192-bit @ 9000 MT/s)
  compute_fp16_tflops: 55   # 실제: ~200 TFLOPS (FP16 Tensor Core, Blackwell 4세대)
```

A100으로 이관하면 이 항목은 더 이상 활성 경로에 없지만, `plot_srm.py` 하드코딩(200 TFLOPS, 288 GB/s)이 yaml보다 정확한 아이러니한 상황. yaml 정정:

```yaml
rtx_5060ti:
  memory_bw_GBs: 288
  compute_fp16_tflops: 200
```

A100 값은 이미 정확:

```yaml
a100_40gb:
  memory_bw_GBs: 1555       # HBM2e 5120-bit ✓
  compute_fp16_tflops: 312  # ✓
```

---

## 3. 🟡 Warning — 환경 설정 및 사전 확인 필요

### W-1. libsmctrl.so A100 환경에서 재빌드 필수

RTX 5060 Ti에서는 libsmctrl이 로드 실패(커널 버전 불일치)로 MPS fallback만 동작해 SM 제한이 실질적으로 무효였음. **A100에서는 libsmctrl이 정상 작동**해야 실험 전체가 유의미해짐.

```bash
# A100 머신에서 빌드
git clone https://github.com/msr-fiddle/libsmctrl
cd libsmctrl && make
export LIBSMCTRL_PATH=/absolute/path/libsmctrl.so
```

빌드 성공 여부 확인:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from src.smctrl.libsmctrl_wrapper import SMController
c = SMController()
print('Backend:', c.get_backend_name())   # 'libsmctrl' 이어야 함
print('SM control works:', c.verify_sm_control())
"
```

**A100 TPC 구조**: GA100 = 108 SM / 54 TPC, `sm_per_tpc=2` → libsmctrl_wrapper 기본값과 일치, **코드 수정 불필요**.

---

### W-2. Policy SM 비율 상수 — A100 Stage 1 결과 후 재조정 권장

**파일**: `stage3_hm_eval/policy_layer_wise.py:28–30`, `policy_step_adaptive.py:26–28`

```python
# 현재 — RTX 5060 Ti saturation 데이터 없이 경험적으로 설정
SSM_PREFILL_RATIO = 0.70   # SSM에 70% SM 할당
ATTN_PREFILL_RATIO = 0.40
MODEL_PREFILL_RATIOS = {"zamba2": 0.40, "falcon_h1": 0.70}
```

RTX 5060 Ti에서 SM 제한이 작동하지 않아 이 비율들의 효과를 실측한 적이 없음. A100에서 Stage 1 완료 후 `compute_decision_matrix.py`의 `free_sm_fraction` 결과를 보고 재설정 권장.

---

### W-3. `ncu` 프로파일링 권한 — 데이터센터 환경

**파일**: `stage1_sm_scaling/run_ncu_profile.py`

A100이 있는 데이터센터/클러스터 환경에서는 `NVreg_RestrictProfilingToAdminUsers=1`이 기본으로 설정된 경우가 많음.

```bash
# 권한 확인
cat /proc/sys/kernel/perf_event_paranoid   # 2 이하이어야 함

# ncu 실행 시 오류 발생하면:
sudo ncu --metrics ... python ...
# 또는 관리자에게 요청:
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo modprobe -r nvidia && sudo modprobe nvidia
```

---

### W-4. `sm_sweep_steps` 불일치 — `--device auto` vs yaml preset

**파일**: `stage1_sm_scaling/run_ssm_prefill_sweep.py:55–62`

```python
# --device auto 시 runtime 계산
compute_sm_steps(108, n_steps=8)  →  [14, 27, 41, 54, 68, 81, 95, 108]

# --device a100_40gb 시 yaml preset
sm_sweep_steps: [11, 22, 33, 44, 54, 65, 87, 108]
```

두 결과 CSV의 `sm_count` 열 값이 달라 `compute_decision_matrix.py`의 saturation 감지 알고리즘이 다른 결과를 줄 수 있음. 재현 가능성을 위해 `--device a100_40gb`를 항상 명시적으로 지정.

---

### W-5. `_run_decode_step` seq_len=1 병렬 스캔 — 실제 decode와 다름

**파일**: `stage3_hm_eval/run_concurrent_eval.py:221–233`

```python
def _run_decode_step(...):
    runner.run_ssm_layer(
        seq_len=1,   # ← 병렬 스캔 모드로 실행됨 (recurrent 모드가 아님)
        ...
    )
```

실제 SSM decode는 recurrent 모드(O(1) per step)로 실행되며 병렬 스캔보다 훨씬 빠름. 현재 구현은 seq_len=1짜리 병렬 스캔을 사용해 decode TPOT이 과대 추정됨.

현재 데이터에서는 TPOT ≈ 0.8–0.9ms로 SLO(50ms)와 크게 떨어져 있어 결론에 영향 없음. 단, A100에서 더 큰 batch로 실험하거나 SLO를 타이트하게 설정할 경우 recurrent 모드 구현 고려.

---

## 4. 🟢 Non-issue — 런타임 자동 적응

이하 항목들은 A100에서 별도 수정 없이 정상 동작.

| 항목 | 파일 | 근거 |
| --- | --- | --- |
| SM 총 개수 | `run_*_sweep.py` | `props.multi_processor_count`: 108 자동 감지 |
| GPU 이름 / 파일 태그 | `run_ssm_prefill_sweep.py:device_tag()` | `torch.cuda.get_device_name()` → `nvidia_a100_40gb` |
| NCU 메트릭 선택 | `src/profiling/ncu_runner.py:142–149` | `sm_major=8` (Ampere) → Legacy 메트릭셋 자동 선택 |
| Max warps/SM | `ncu_runner.py:75` | `max_threads_per_multi_processor // 32`: A100 = 2048/32 = **64** |
| TPC mask 계산 | `libsmctrl_wrapper.py:132–145` | `sm_per_tpc=2` 기본값이 A100(GA100)과 일치 |
| 대역폭 BW 공식 | `metrics.py:163–175` | HBM2e 처리 코드 존재 (단, 속성 조회 버그로 `--device` 명시 필요) |
| Policy 인터페이스 | `stage3_hm_eval/policy_*.py` | SM 비율은 GPU 무관한 fraction, 절대값은 smctrl이 변환 |

---

## 5. A100에서 새롭게 유효해지는 기능

### 5-1. libsmctrl SM 제한 실제 작동

RTX 5060 Ti에서 SM 제한이 전혀 작동하지 않아 Stage 1 SRM 분석, Stage 2 overhead 측정, Stage 3 Policy 비교 전체가 SM-agnostic 상태였음. A100에서 libsmctrl이 정상 작동하면:

- Stage 1: `sm_count`가 진정한 독립변수 → SRM curve가 실제 shape를 보임
- Stage 2: smctrl overhead를 libsmctrl ioctl 기준으로 측정 가능 (μs 단위 정밀도)
- Stage 3: Policy A/B/C 간 실질적인 TPOT·TTFT 차이 확인 가능

### 5-2. GreenContext API 활성화 (Stage 4)

RTX 5060 Ti에서 `cuGreenCtxCreate → rc=801 (CUDA_ERROR_NOT_SUPPORTED)` 로 실패. A100은 MIG-capable 데이터센터 GPU이므로 GreenContext를 지원함:

```bash
# A100에서 재실행 (이전에 INCONCLUSIVE였던 Stage 4)
python mamba-cuda-graph/stages/stage4_libsmctrl.py
# 예상: cuGreenCtxCreate rc=0, cuGreenCtxStreamCreate rc=0
```

---

## 6. 수정 우선순위 및 마이그레이션 체크리스트

```text
필수 — 실험 전 완료
──────────────────────────────────────────────────────
[ ] C-1  plot_srm.py — --hw-config 인자 추가, 하드코딩 상수 3개 제거
[ ] C-2  --device a100_40gb 명시 (auto 금지) → BW 1000 GB/s fallback 방지
[ ] C-3  results/ 디렉토리 분리 또는 compute_decision_matrix.py에 --device-tag 필터 추가
[ ] C-4  A100에서 Stage 2 재실행 (smctrl + layer_latency 측정 → decision_matrix 재생성)
[ ] C-5  configs/hardware.yaml rtx_5060ti 수정 (BW: 448→288, TFLOPS: 55→200)
[ ] W-1  libsmctrl 소스 클론 → A100 환경에서 make → LIBSMCTRL_PATH 설정
[ ] W-1  verify_sm_control() 통과 확인 (slowdown ratio ≥ 1.3)

권장 — 실험 중 또는 후
──────────────────────────────────────────────────────
[ ] W-2  Stage 1 A100 결과 기반으로 policy_layer_wise.py SM 비율 재조정
[ ] W-3  클러스터 환경 perf_event_paranoid 설정 확인 (ncu 프로파일링 권한)
[ ] W-4  Stage 1 재실행 시 --device a100_40gb 명시 (sm_sweep_steps 일관성)
[ ] W-5  _run_decode_step() recurrent 모드 구현 고려 (large-batch 실험 시)
[ ] 선택  stage4_libsmctrl.py A100에서 재실행 → GreenContext claim 검증
```

---

## 7. 이슈 요약 테이블

| ID | 파일 | 위치 | 증상 | 해결 |
| --- | --- | --- | --- | --- |
| C-1 | `plot_srm.py` | L57–60 | SRM BW ceiling 5배 낮게 그려짐 | `--hw-config` 인자 추가 |
| C-2 | `metrics.py` | L159–177 | `theoretical_bw_GBs=1000.0` 고정 | `--device a100_40gb` 명시 |
| C-3 | `compute_decision_matrix.py` | L55,101,145 | RTX+A100 데이터 혼합 → saturation 오판 | results/ 분리 또는 device-tag 필터 |
| C-4 | Stage 2 전체 | — | RTX 기반 decision matrix로 A100 Policy 결정 | Stage 2 A100 재실행 |
| C-5 | `hardware.yaml` | rtx_5060ti | BW/TFLOPS 오기재 | 수치 수정 |
| W-1 | `libsmctrl_wrapper.py` | — | MPS fallback → SM 제한 무효 | 재빌드 + 경로 설정 |
| W-2 | `policy_layer_wise.py` | L28–30 | SM 비율 미검증 상수 | A100 Stage 1 후 재조정 |
| W-3 | `run_ncu_profile.py` | — | ncu 권한 오류 | 클러스터 환경 확인 |
| W-4 | `run_ssm_prefill_sweep.py` | L55–62 | sweep step 불일치 | `--device a100_40gb` 명시 |
| W-5 | `run_concurrent_eval.py` | L221–233 | decode TPOT 과대 추정 | recurrent 모드 구현 (선택) |
