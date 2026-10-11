# M1 + V0 multi-GPU 기판 프로브 결과 (2026-10-12)

> **지위**: 기판 **기술치(descriptor)**. 서빙 성능 수치 0 · 판정 0 · 기판 동등성 서술 0 · Claim 변경 0.
> Job `pdmux-multigpu-probe-d947939-20261011182445`(slug `job-t3yp83vl6792`), `resourcespec-a100x8`, commit `d947939`,
> 컨테이너 2026-10-11T18:25:29Z–18:28:14Z(UTC), exit 0, `fetch.sh` DONE + sha256 통과.
> 설계: `reports/multigpu_scaling_design_2026-10-12.md` §6 M1 + AMD 메모 §7 V0. 원자료: `pdmux-multigpu-probe-d947939-20261011182445/`.

## 1. 단계 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| F1 기판 | PASS | 단, 런 안의 `f1/topo_m.json`은 파서 결함으로 `n_gpu: 1`이다(§4). 원문 `f1/topo_m.txt`가 정본 |
| F2 all-reduce | PASS | nccl-tests 바이너리가 이미지에 없어 torch.distributed NCCL로 대체. 부분집합은 GPU 0..N-1 고정 |
| F3 P2P | PASS | 복사 대역폭 + 소형 메시지 지연(V0) |
| F4 green ctx | PASS | 8/8 GPU |
| F5 RDMA | PASS(기술치) | IB 장치 0, mooncake 미설치 |

## 2. 기판 사실

- **GPU**: NVIDIA A100-SXM4-80GB ×8, driver 580.105.08, cc 8.0, MIG Disabled, power limit 400 W, max SM clock 1410 MHz (`f1/gpu.csv`).
- **연결 위상**: 서로 다른 GPU 쌍(순서쌍 56개) 전부 **NV12**(NVLink 12개 묶음). PCIe 경유 쌍은 없다(`f1/topo_m.txt`).
- **NUMA**: GPU0–3 → NUMA 0(CPU 0-23,48-71), GPU4–7 → NUMA 1(CPU 24-47,72-95). **N=8은 NUMA 두 개에 걸치고**, N≤4를 GPU 0..N-1로 잡으면 NUMA 하나 안이다 → 설계 C-1(부분집합 위상·NUMA 공변량)의 실제 축.
- **CPU**: cgroup `cpu.max = 8800000 100000`(=88 CPU), `nproc` 96. 프로브 동안 `nr_throttled 3`, `throttled_usec 192401`(종료 시점 누적).
- **NCCL**: 2.27.7(torch). `NCCL_*` env 미설정.
- **all-reduce**(torch.distributed, 20 iters, GPU 0..N-1; 1 KiB → 512 MiB 스윕):

  | N | 1 KiB 시간 | 1 MiB busbw | 16 MiB busbw | 512 MiB busbw |
  |---|---|---|---|---|
  | 2 | 37.8 µs | 22.3 GB/s | 131.2 GB/s | 191.7 GB/s |
  | 4 | 38.4 µs | 46.1 GB/s | 138.5 GB/s | 216.2 GB/s |
  | 8 | 40.2 µs | 26.6 GB/s | 127.2 GB/s | 222.1 GB/s |

  호스트 쪽 Python 루프 타이밍이라 작은 메시지 시간에는 dispatch 오버헤드가 들어 있다. 서빙 중 all-reduce 비용의 추정치로 쓰지 않는다.
- **P2P**: 모든 쌍 `can_device_access_peer = True`. 256 MiB 복사 270.7–274.2 GB/s(56쌍, 중앙값 273.3).
- **소형 메시지 복사 지연(V0)**: 2 KiB·64 KiB·512 KiB·1 MiB 모두 쌍별 중앙값 27.3–30.1 µs다(중앙값의 중앙값 28.6–28.7 µs). **크기와 무관**하다(1 MiB를 273 GB/s로 보내면 이론상 ≈4 µs). 따라서 이 값은 링크 지연이 아니라 **매 반복의 이벤트·동기화·eager dispatch 오버헤드가 지배한 값**이다. **AMD V1 crossing L0 입력으로 쓰지 않는다**. V1은 사전등록한 cudagraph 캡처 마이크로벤치로 따로 잰다.
- **green context**: 8/8 GPU에서 요청한 (16,92)·(44,64)·(92,16) 쌍이 전부 생성됐다. driver read-out `smCount`가 요청값과 일치하고 `get_sm_available = 108`이다. **생성/read-out 일치 ≠ 실현 SM**(%smid census는 하지 않았다).
- **RDMA·mooncake**: `/sys/class/infiniband` 비어 있음, 네트워크 링크는 `eth0`·`lo`뿐, `ibv_devinfo`·`rdma` 도구 없음, `mooncake` 모듈 없음.

## 3. 설계에 주는 함의 (판정 아님 — 다음 단계 입력)

1. **x8 노드는 전 쌍 NVLink(NV12)다.** 설계 R-1의 "PCIe N≥4에서 custom AR arm이 무처치" 위험은 이 노드 형상에서는 발생하지 않는다.
   다만 기본 경로가 실제로 CA v1을 고르는지는 M2g에서 커널 이름으로 확인해야 한다(이 프로브는 서빙 경로를 실행하지 않았다).
2. **NUMA 경계는 N=4→8에서 처음 넘는다.** N 부분집합을 GPU 0..N-1로 고정하면 NUMA 효과가 N=8에만 실린다. PREREG(M2.5)에서 부분집합 배치(예: N=4를 NUMA 균형 {0,1,4,5}로 하는 대조)를 정해야 한다.
3. **CPU quota는 GPU당 11개로 x1과 같은 비율**이다. 단일 프론트엔드 병목(C-4)은 여전히 M1.5에서 확인해야 한다.
4. **AMD 트랙 V2(PD-disagg 대조군)**: 이 노드에는 IB NIC와 mooncake가 없다. 설계 메모가 가정한 mooncake 전송 경로는 이 이미지·노드 그대로는 쓸 수 없다. 노드 안 NVLink 전송 백엔드가 가능한지는 확인하지 않았다(사용자 결정 사항: 이미지에 전송 백엔드 설치 vs 대조군 (i) 재정의).
5. GPU별 1-GPU 서빙 셀 편차(C-2)는 이번 프로브 범위 밖이다(사전등록 필요).

## 4. 프로브 결함과 수리

- `parse_topo`가 `nvidia-smi topo -m` 헤더 행의 ANSI 밑줄 코드(`ESC[4m`)를 처리하지 못했다. 그래서 **첫 데이터 행을 헤더로 오인**해 `n_gpu: 1`과 `all_pairs_nvlink: true`(1×1에서 공허하게 참)를 냈고, F1 사유의 `classes: {}`도 이 결함의 산물이다.
- 수리(`scripts/vessl/multigpu_probe_helper.py`): ANSI 제거, 헤더 판정은 두 번째 토큰이 GPU/CPU 라벨일 때만, `all_pairs_nvlink`는 n≥2 ∧ 행 수 = n일 때만 참.
  회귀 테스트 2건 추가(`tests/test_multigpu_probe.py`: 실제 출력 형태의 ANSI 헤더, 1열 공허 참 방지).
- 수리한 파서로 원문을 다시 파싱한 결과: `n_gpu 8`, `{'NVLINK(12)': 56}`, `all_pairs_nvlink True`. 런 디렉터리 안의 원 산출물은 sha256 보존을 위해 고치지 않았다.
- nccl-tests 바이너리가 없어서 all-reduce의 GPU 부분집합 위상 변형(예: NUMA 교차 N=2)은 측정하지 않았다.
