#!/bin/bash
# =============================================================================
# interact_scheduler.sh — salloc 기반 interact 작업 상시 가동 스케줄러
#
# salloc로 GPU 슬롯을 확보하여 작업을 실행하고,
# 작업이 끝나거나 시간 제한에 걸리면 즉시 다음 salloc을 요청한다.
#
# USAGE (로그인 노드에서 실행)
#   bash slurm/interact_scheduler.sh [OPTIONS]
#
# OPTIONS
#   --model MODEL      zamba2 | falcon_h1 | all   (default: zamba2)
#   --device DEVICE    hardware key or 'auto'     (default: auto)
#   --script SCRIPT    슬롯 안에서 실행할 스크립트  (default: slurm/run_all.sh)
#   --time HH:MM:SS    salloc 시간 제한            (default: 36:00:00)
#   --partition PART   Slurm 파티션               (default: amd_a100nv_8)
#   --max-runs N       최대 실행 횟수 (0=무제한)   (default: 0)
#   --retry-fail       실패 시에도 재시도           (default: 성공 시만 재시도)
#   --log-dir DIR      스케줄러 로그 디렉토리       (default: logs/scheduler)
#   --dry-run          실제 salloc 없이 동작 확인
#
# EXAMPLES
#   bash slurm/interact_scheduler.sh
#   bash slurm/interact_scheduler.sh --model all --max-runs 3
#   bash slurm/interact_scheduler.sh --time 10:00:00 --retry-fail
#   bash slurm/interact_scheduler.sh --script slurm/run_stage3.sh --model falcon_h1
#
# STOP
#   Ctrl+C  또는  touch logs/scheduler/STOP
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 기본값
# ---------------------------------------------------------------------------
MODEL="zamba2"
DEVICE="auto"
SCRIPT="slurm/run_all.sh"
TIME_LIMIT="36:00:00"
PARTITION="amd_a100nv_8"
MAX_RUNS=0
RETRY_FAIL=0
LOG_DIR="logs/scheduler"
DRY_RUN=0

# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --script)     SCRIPT="$2";     shift 2 ;;
        --time)       TIME_LIMIT="$2"; shift 2 ;;
        --partition)  PARTITION="$2";  shift 2 ;;
        --max-runs)   MAX_RUNS="$2";   shift 2 ;;
        --retry-fail) RETRY_FAIL=1;    shift   ;;
        --log-dir)    LOG_DIR="$2";    shift 2 ;;
        --dry-run)    DRY_RUN=1;       shift   ;;
        *) echo "[error] unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

STOP_FILE="$LOG_DIR/STOP"
SCHED_LOG="$LOG_DIR/scheduler_$(date '+%Y%m%d_%H%M%S').log"

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$SCHED_LOG"
}

cleanup() {
    echo ""
    log "중단 신호 수신 — 현재 salloc 슬롯은 계속 실행됩니다."
    log "  진행 중인 작업을 강제로 끝내려면: scancel --user=$USER"
    # STOP 파일을 남겨 루프가 종료되도록
    touch "$STOP_FILE"
    exit 0
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# 시작 배너
# ---------------------------------------------------------------------------
cat <<EOF | tee -a "$SCHED_LOG"
╔══════════════════════════════════════════════════════════╗
║     salloc 상시 가동 스케줄러                            ║
╚══════════════════════════════════════════════════════════╝
  model      = $MODEL
  device     = $DEVICE
  script     = $SCRIPT
  time-limit = $TIME_LIMIT
  partition  = $PARTITION
  max-runs   = $([ "$MAX_RUNS" -eq 0 ] && echo "무제한" || echo "$MAX_RUNS 회")
  retry-fail = $([ "$RETRY_FAIL" -eq 1 ] && echo "yes" || echo "no")
  log        = $SCHED_LOG
  stop-file  = $STOP_FILE
  started    = $(date '+%Y-%m-%d %H:%M:%S')

  [중지] Ctrl+C  또는  touch $STOP_FILE
EOF
echo ""

# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------
RUN_COUNT=0

while true; do
    # STOP 파일 확인
    if [[ -f "$STOP_FILE" ]]; then
        log "STOP 파일 감지 — 스케줄러를 종료합니다."
        rm -f "$STOP_FILE"
        break
    fi

    # 최대 실행 횟수 확인
    if [[ $MAX_RUNS -gt 0 && $RUN_COUNT -ge $MAX_RUNS ]]; then
        log "최대 실행 횟수 ($MAX_RUNS회) 도달 — 스케줄러를 종료합니다."
        break
    fi

    RUN_COUNT=$(( RUN_COUNT + 1 ))
    log "━━━ Run #${RUN_COUNT} 시작 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # salloc 명령어 구성
    SALLOC_CMD=(
        salloc
        --job-name   "interact-${MODEL}-${RUN_COUNT}"
        --partition  "$PARTITION"
        --nodes      1
        --ntasks-per-node 1
        --cpus-per-task   16
        --gres       gpu:1
        --time       "$TIME_LIMIT"
        --comment    pytorch
    )

    # salloc 안에서 실행할 명령 (환경 설정 + 스크립트)
    INNER_CMD="
        module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true
        module load cuda/13.0.2                2>/dev/null || true
        module load gcc/15.2.0                 2>/dev/null || true
        source '${REPO_ROOT}/bin/activate'
        cd '${REPO_ROOT}'
        bash '${SCRIPT}' '${MODEL}' '${DEVICE}'
    "

    log "salloc 슬롯 요청 중... (partition=$PARTITION, time=$TIME_LIMIT)"

    T_START=$(date +%s)
    EXIT_CODE=0

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[dry-run] ${SALLOC_CMD[*]} bash -c \"...\""
        sleep 2
    else
        "${SALLOC_CMD[@]}" bash -c "$INNER_CMD" || EXIT_CODE=$?
    fi

    ELAPSED=$(( $(date +%s) - T_START ))
    ELAPSED_FMT="$(printf '%dh %dm %ds' $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 )))"

    log "salloc 종료: exit_code=$EXIT_CODE  elapsed=$ELAPSED_FMT"

    # 재시도 여부 결정
    if [[ -f "$STOP_FILE" ]]; then
        log "STOP 파일 감지 — 스케줄러를 종료합니다."
        rm -f "$STOP_FILE"
        break
    fi

    if [[ $EXIT_CODE -ne 0 && $RETRY_FAIL -eq 0 ]]; then
        log "작업 실패 (exit=$EXIT_CODE). --retry-fail 없이는 중단합니다."
        log "  재시도하려면: bash $0 --retry-fail (나머지 옵션 동일)"
        break
    fi

    log "→ 즉시 다음 슬롯을 요청합니다."
    echo ""
done

# ---------------------------------------------------------------------------
# 종료 요약
# ---------------------------------------------------------------------------
cat <<EOF | tee -a "$SCHED_LOG"

╔══════════════════════════════════════════════════════════╗
║  스케줄러 종료                                           ║
╚══════════════════════════════════════════════════════════╝
  총 실행 횟수 : $RUN_COUNT
  종료 시각    : $(date '+%Y-%m-%d %H:%M:%S')
  로그 파일    : $SCHED_LOG
EOF
