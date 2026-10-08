# 컨테이너 경로 power cap 프로브 — VESSL Job `job-x3lzellrkm8z` (2026-10-07)

Job `pdmux-power-cap-probe-20261007083542`, A100 SXM ×1(`resourcespec-a100x1`, cluster-betelgeuse us-west-2), 공개 이미지 digest 고정, 볼륨 없음.
스크립트 `../probe_container_caps.sh`(base64 인라인). 상태 `succeeded`, 실행 ~5 s. 회수: `job_logs.txt`, `job_show.json`, `SHA256SUMS`.

## 판정: 컨테이너 경로(Workspace/Job)에서 power limit 설정 **불가 — 실측 확정**

| 항목 | 관측 |
|---|---|
| 사용자 | uid=0 (root) |
| CapEff | `0xa80425fb` = chown, dac_override, fowner, fsetid, kill, setgid, setuid, setpcap, net_bind_service, net_raw, sys_chroot, mknod, audit_write, setfcap — **CAP_SYS_ADMIN(bit 21) 없음** (Docker 기본 집합) |
| 디바이스 | `/dev/nvidiactl`, `/dev/nvidia0` = `crw-rw-rw- root root` |
| 전력 제약값 | power.limit 400 W, default 400 W, **min 100 W, max 400 W**(NVML 제약 [100000, 400000] mW 일치) |
| `nvidia-smi -pl 400`(현재값과 같은 no-op) | `Insufficient Permissions`, rc=4 |
| NVML `nvmlDeviceSetPowerManagementLimit`(같은 값) | `NVMLError Insufficient Permissions` |
| `nvidia-smi -lgc` / `-rgc` | "current user does not have permission to change clocks", rc=4 |
| `nvidia-smi -pm 1` | "already Enabled", rc=0(변경 없음 — 권한 검증 아님) |
| 종료 후 | 400 W, persistence Enabled, 상태 불변 |

원인: root여도 컨테이너 bounding set에 CAP_SYS_ADMIN이 없고(CapBnd 동일), `vesslctl`·job JSON에 capability를 추가할 필드가 없다.
따라서 VESSL 안의 `-pl` 경로는 **VM Cluster**(passthrough VM, full root)뿐이다(`reports/power_cap_viability_2026-10-07.md` §2-1).
부수 확인: 이 기판 A100-SXM4-80GB의 설정 가능 범위가 100–400 W임이 실측됐다(사전등록 초안 §2 B0-1의 "미검증" 해소, 단 다른 기판에서는 다시 읽어야 함).
