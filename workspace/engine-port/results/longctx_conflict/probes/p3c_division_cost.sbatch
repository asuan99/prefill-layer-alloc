#!/bin/bash
#SBATCH --job-name=p3cdiv
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.err
#
# P3-C DIVISION_COST -- PREREG_PROBES_2026-09-08.md sec 3-C (added 2026-09-09,
# repair of a preregistration gap in sec 3: "division 3개의 비용을 기존
# 1-division config와 대조해 병기" assumed a comparable artifact already
# existed. It did not -- w1probe (max_total_num_tokens=2,723,789, R=48) has a
# DIFFERENT --mem-fraction-static; cp2048 (3,074,249) has R=8. Neither isolates
# division count. Audit block B12 is therefore NOT closed by P3 alone.
#
# ★ONE VARIABLE ONLY vs P3's `d44` boot (job 905701, same node class, same
# everything): config path swaps from `probes/pdmux_homog5.yml` (5 groups, 3
# divisions) to `stage0_xctrl/pdmux_d44.yml` (3 groups, 1 division). Model,
# backend, ctx (16384/16896 cap), all common flags, all env vars,
# PDMUX_R2_FIXED_DSM=44, and the 8-request smoke (in=512 out=32 seed=7) are
# byte-identical to P3's d44 arm. Do NOT "improve" anything else -- that
# breaks the contrast (coordinator instruction).
#
# Labels (prereg sec 3-C, applied programmatically below against P3's own
# banner_d44.txt as the reference -- read from $P3_D44_REF_DIR, default the
# already-completed P3 job 905701):
#   HOMOG_FREE    Delta max_total_num_tokens < 5% AND capture-list length equal
#   HOMOG_COSTED  either condition fails -> homogenization has a declared cost;
#                 any campaign using it must caveat its operating point differs
#                 from the canonical 1-division campaign (B12 "quantified", not "closed")
#   CONTROL_FAILED boot failure -> 1 retry
#
# The 5% threshold is deliberately looser than this repo's PRACTICAL_FLOOR
# (3%) -- an engineering gauge for "is the memory budget practically the
# same", NOT a performance threshold (gate #93). No performance claim is made
# about what this Delta (if any) does to goodput/ITL/TTFT.
#
# usage: sbatch p3c_division_cost.sbatch
#        sbatch --export=ALL,P3_D44_REF_DIR=<path to another p3_<jobid> dir> p3c_division_cost.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
STAGE0=$ROOT/workspace/engine-port/results/stage0_xctrl
OUT="$PROBES/p3c_${SLURM_JOB_ID}"; mkdir -p "$OUT"

P3_D44_REF_DIR="${P3_D44_REF_DIR:-$PROBES/p3_905701}"
REF_BANNER="$P3_D44_REF_DIR/banner_d44.txt"
if [ ! -f "$REF_BANNER" ]; then
  echo "P3C_ABORT: reference banner not found: $REF_BANNER" | tee "$OUT/BOOT_FAILURES.txt"
  exit 3
fi
echo "P3_D44_REF_DIR=$P3_D44_REF_DIR"
cp "$REF_BANNER" "$OUT/reference_banner_d44_homog5.txt"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$STAGE0/pdmux_d44.yml"          # ★ONLY changed variable vs P3's d44 boot
CTX=16384
CTXCAP=$((CTX + 512))
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

echo "P3C model=$MODEL backend=$BACKEND cfg=$CFG ctx=$CTX ctxcap=$CTXCAP job=$SLURM_JOB_ID"
echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

wait_health () {
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

D=44
BOOT_OK=0
for attempt in 1 2; do
  PORT=$((39600 + (SLURM_JOB_ID + attempt + RANDOM) % 300))
  SRVLOG="$OUT/srv_d${D}.log"
  RUNID="p3c_d${D}_${SLURM_JOB_ID}_a${attempt}"
  export PDMUX_TELEMETRY_PATH="$OUT/p3c_d${D}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"
  export PDMUX_WORKLOAD_ID="p3c_d${D}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$D

  echo ""
  echo "########## P3C D=$D attempt=$attempt port=$PORT cfg=$CFG ##########"
  T0=$(date +%s)
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    T1=$(date +%s)
    BOOT_S=$((T1 - T0))
    echo "boot_ok=1 D=$D attempt=$attempt boot_s=$BOOT_S"
    BOOT_OK=1
    BANNER="$OUT/banner_d${D}.txt"; : > "$BANNER"
    echo "--- PD-mux enable banner + sm_counts ---" | tee -a "$BANNER"
    grep -oE "PD-Multiplexing enabled with .*" "$SRVLOG" | sort -u | tee -a "$BANNER"
    echo "--- max_total_num_tokens ---" | tee -a "$BANNER"
    grep -oE "max_total_num_tokens=[0-9]+.*" "$SRVLOG" | head -1 | tee -a "$BANNER"
    echo "--- cudagraph capture list ---" | tee -a "$BANNER"
    grep -oE "Capture cuda graph bs \[[^]]*\]" "$SRVLOG" | sort -u | tee -a "$BANNER"
    grep -c "Capturing batches" "$SRVLOG" | sed 's/^/N_CAPTURING_BATCHES_LOG_LINES /' | tee -a "$BANNER"
    grep -c "Capture cuda graph failed" "$SRVLOG" | sed 's/^/N_CAPTURE_FAILURES /' | tee -a "$BANNER"
    echo "BOOT_SECONDS $BOOT_S" | tee -a "$BANNER"

    echo "--- smoke: 8 requests (in=512 out=32, identical to P3) ---" | tee -a "$BANNER"
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len 512 --random-output-len 32 --random-range-ratio 1.0 \
      --num-prompts 8 --seed 7 \
      --output-details --output-file "$OUT/smoke_d${D}.jsonl" \
      > "$OUT/bench_d${D}.log" 2>&1
    BRC=$?
    echo "smoke_rc=$BRC D=$D" | tee -a "$BANNER"
    grep -E "Successful requests|Median TTFT|Median ITL" "$OUT/bench_d${D}.log" | tee -a "$BANNER"
  else
    echo "BOOT_FAILED D=$D attempt=$attempt" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
  fi
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10

  [ "$BOOT_OK" = 1 ] && break
  echo "retrying (prereg sec 3-C CONTROL_FAILED -> 1 retry)"
done

echo "=== P3-C label (HOMOG_FREE / HOMOG_COSTED / CONTROL_FAILED) ==="
python3 - "$OUT" "$REF_BANNER" "$BOOT_OK" <<'PY'
import json, os, re, sys

out, ref_banner, boot_ok = sys.argv[1], sys.argv[2], sys.argv[3] == "1"

def parse_banner(path):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m_tok = re.search(r"max_total_num_tokens=(\d+)", text)
    m_cap = re.search(r"Capture cuda graph bs \[([^\]]*)\]", text)
    m_grp = re.search(r"PD-Multiplexing enabled with (\d+) stream groups", text)
    m_boot = re.search(r"BOOT_SECONDS (\d+)", text)
    tokens = int(m_tok.group(1)) if m_tok else None
    capture_list = [int(x) for x in m_cap.group(1).split(",")] if m_cap else None
    n_groups = int(m_grp.group(1)) if m_grp else None
    boot_s = int(m_boot.group(1)) if m_boot else None
    return dict(max_total_num_tokens=tokens, capture_list=capture_list,
                n_stream_groups=n_groups, boot_seconds=boot_s)

result = {"probe": "P3C_DIVISION_COST"}

if not boot_ok:
    result["verdict"] = "CONTROL_FAILED"
    result["reason"] = "boot failed on both attempts"
    json.dump(result, open(os.path.join(out, "P3C_LABEL.json"), "w"), indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0)

ref = parse_banner(ref_banner)
new = parse_banner(os.path.join(out, "banner_d44.txt"))
result["reference"] = ref
result["new"] = new

if ref is None or new is None or ref["max_total_num_tokens"] is None or new["max_total_num_tokens"] is None:
    result["verdict"] = "CONTROL_FAILED"
    result["reason"] = "banner missing/unparseable on one side"
else:
    delta_pct = abs(new["max_total_num_tokens"] - ref["max_total_num_tokens"]) / ref["max_total_num_tokens"] * 100.0
    len_ref = len(ref["capture_list"]) if ref["capture_list"] is not None else None
    len_new = len(new["capture_list"]) if new["capture_list"] is not None else None
    capture_len_equal = (len_ref is not None and len_ref == len_new)
    capture_exact_equal = (ref["capture_list"] == new["capture_list"])
    result["delta_max_total_num_tokens_pct"] = delta_pct
    result["capture_list_len_ref"] = len_ref
    result["capture_list_len_new"] = len_new
    result["capture_list_len_equal"] = capture_len_equal
    result["capture_list_exact_equal"] = capture_exact_equal
    if delta_pct < 5.0 and capture_len_equal:
        result["verdict"] = "HOMOG_FREE"
        result["reading"] = "homogenization is effectively free -- B12 closed"
    else:
        result["verdict"] = "HOMOG_COSTED"
        result["reading"] = ("homogenization has a declared cost; campaigns using it must "
                              "caveat their operating point differs from the canonical "
                              "1-division campaign (B12 quantified, not closed)")

json.dump(result, open(os.path.join(out, "P3C_LABEL.json"), "w"), indent=2, sort_keys=True)
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "P3C_DONE_${SLURM_JOB_ID}"
