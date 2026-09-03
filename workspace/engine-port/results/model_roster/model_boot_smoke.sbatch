#!/bin/bash
#SBATCH --job-name=mboot
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/model_roster/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/model_roster/%x_%j.err
#
# MODEL BOOT SMOKE -- which (model x backend x pdmux) cells EXIST on this tree?
#
# Why: MODEL_ROSTER_2026-08-28.md sec 4-3 already recorded that "Zamba2 전 계열 ctx 4096,
# 긴 프롬프트 설계에 쓸 수 없다", and the W1 smoke (job 900671) hit exactly that wall.
# The roster lists long-context candidates but says their ENGINE SUPPORT is unverified
# for everything except Nemotron-Nano (jobs 896760/764/767).  Gate #83 says the engine
# forces mutually exclusive support regions (Nemotron-H + triton = boot refusal,
# Zamba2 + flashinfer = scheduler death), so model and backend cannot be chosen
# independently.  This job MEASURES that support table instead of assuming it.
#
# ★ WHAT THIS BUYS: boot_ok / serves-a-long-prompt, per cell.  NOTHING ELSE.
#    No performance number here is a measurement of anything (n=1, 8 prompts, no
#    warmup control, no repetition).  PROJECT_STATUS "방법론 게이트" #26: a smoke's
#    PASS carries no information about any policy question.
#
# usage: sbatch model_boot_smoke.sbatch
#        sbatch --export=ALL,MB_MODELS="ibm-granite/granite-4.0-h-micro-base" model_boot_smoke.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/model_roster
SLO=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
OUT="$ROOT/boot_smoke_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"

MODELS="${MB_MODELS:-ibm-granite/granite-4.0-h-micro-base tiiuae/Falcon-H1-3B-Base nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base}"
BACKENDS="${MB_BACKENDS:-triton flashinfer}"
MODES="${MB_MODES:-cp2048 d44}"
CTX="${MB_CTX:-16384}"
INLEN="${MB_INLEN:-8000}"
OUTLEN="${MB_OUTLEN:-200}"
NP="${MB_NP:-8}"
# rev2: piecewise capture is broken for falcon_h1/nemotron_h in this env, so the cp
# arms carry the engine's own recommended workaround.  Set MB_PIECEWISE=on to test
# the other side of that axis.
[ "${MB_PIECEWISE:-off}" = "on" ] && PW="" || PW="--disable-piecewise-cuda-graph"
SGPT="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

unset PDMUX_CHUNK_PROBE_PATH PDMUX_SLO_SCHED PDMUX_SLO_MODE

echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2
echo "MBOOT models=[$MODELS] backends=[$BACKENDS] modes=[$MODES] ctx=$CTX in=$INLEN np=$NP piecewise=${MB_PIECEWISE:-off} job=$SLURM_JOB_ID"

SUMMARY="$OUT/boot_smoke.jsonl"; : > "$SUMMARY"

mode_flags () {
  case "$1" in
    cp2048) echo "--chunked-prefill-size 2048 $PW" ;;
    cp8192) echo "--chunked-prefill-size 8192 $PW" ;;
    d44)    echo "--enable-pdmux --pdmux-config-path $SLO/pdmux_d44.yml --chunked-prefill-size -1 --disable-overlap-schedule $PW" ;;
    *) echo "__BAD__" ;;
  esac
}

for MODEL in $MODELS; do
 for BACKEND in $BACKENDS; do
  for MODE in $MODES; do
    FLAGS=$(mode_flags "$MODE"); [ "$FLAGS" = "__BAD__" ] && continue
    TAG="$(echo "$MODEL" | tr '/' '_')__${BACKEND}__${MODE}"
    SRVLOG="$OUT/srv_${TAG}.log"
    PORT=$((39000 + RANDOM % 500))
    echo ""
    echo "########## $MODEL | backend=$BACKEND | mode=$MODE | port=$PORT ##########"
    T0=$(date +%s)
    python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
      --dtype bfloat16 --attention-backend "$BACKEND" $FLAGS \
      --disable-radix-cache --mem-fraction-static 0.82 \
      --max-running-requests 48 --context-length "$CTX" \
      --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
    PID=$!
    OK=0
    for i in $(seq 1 90); do            # bounded: 90 x 5s = 7.5 min per cell
      [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { OK=1; break; }
      kill -0 "$PID" 2>/dev/null || { echo "  SERVER DIED"; break; }
      sleep 5
    done
    T1=$(date +%s)
    echo "  boot_ok=$OK boot_s=$((T1-T0))"

    FAILSIG=""
    if [ "$OK" != 1 ]; then
      # the failure SIGNATURE is the product here -- gate #83 is about which cells
      # the engine refuses, so the refusal text is data, not noise.
      FAILSIG=$(grep -oiE "assert[^\"]*|ValueError:[^\"]*|NotImplementedError[^\"]*|not supported[^\"]*|is not compatible[^\"]*" "$SRVLOG" \
                | grep -viE "pynvml|Future" | tail -2 | tr '\n' ' ' | cut -c1-300)
      echo "  FAIL: $FAILSIG"
      tail -12 "$SRVLOG" | grep -viE "pynvml|Future|^\s*$" | tail -5
    else
      grep -oE "chunked_prefill_size=[-0-9]+|piecewise_cuda_graph_max_tokens=[0-9]+|enable_pdmux=[A-Za-z]+|attention_backend='[a-z]+'" "$SRVLOG" | sort -u | head -5
      grep -E "max_total_num_tokens" "$SRVLOG" | head -1
    fi

    THR=""; TTFT=""; SERVED=0
    if [ "$OK" = 1 ]; then
      BLOG="$OUT/bench_${TAG}.log"
      python -m sglang.bench_serving --backend sglang --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" --dataset-name random \
        --dataset-path "$SGPT" \
        --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0 \
        --num-prompts "$NP" --request-rate 2 --seed 11 > "$BLOG" 2>&1
      BRC=$?
      SERVED=$(grep -oE "Successful requests:\s+[0-9]+" "$BLOG" | grep -oE "[0-9]+$" | head -1)
      THR=$(grep -oE "Request throughput \(req/s\):\s+[0-9.]+" "$BLOG" | grep -oE "[0-9.]+$" | head -1)
      TTFT=$(grep -oE "Median TTFT \(ms\):\s+[0-9.]+" "$BLOG" | grep -oE "[0-9.]+$" | head -1)
      echo "  bench rc=$BRC served=${SERVED:-0}/$NP thr=${THR:-NA} medTTFT=${TTFT:-NA}ms"
      [ "${SERVED:-0}" = "0" ] && tail -6 "$BLOG" | grep -viE "^\s*$" | tail -3
    fi

    printf '{"model":"%s","backend":"%s","mode":"%s","ctx":%s,"boot_ok":%s,"boot_s":%s,"served":%s,"np":%s,"throughput":"%s","median_ttft_ms":"%s","fail":"%s"}\n' \
      "$MODEL" "$BACKEND" "$MODE" "$CTX" "$OK" "$((T1-T0))" "${SERVED:-0}" "$NP" "${THR:-}" "${TTFT:-}" \
      "$(echo "$FAILSIG" | sed 's/"/\\"/g')" >> "$SUMMARY"

    kill "$PID" 2>/dev/null; sleep 3
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 6
  done
 done
done

echo ""
echo "==================== SUPPORT TABLE ===================="
python3 - "$SUMMARY" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
models=sorted({r["model"] for r in rows}); modes=sorted({r["mode"] for r in rows})
bks=sorted({r["backend"] for r in rows})
print(f"{'model':<46}" + "".join(f"{b+'/'+m:>18}" for b in bks for m in modes))
for mo in models:
    line=f"{mo:<46}"
    for b in bks:
        for md in modes:
            r=next((x for x in rows if x["model"]==mo and x["backend"]==b and x["mode"]==md),None)
            if not r: line+=f"{'-':>18}"
            elif r["boot_ok"]!=1: line+=f"{'BOOT_FAIL':>18}"
            elif int(r["served"])<1: line+=f"{'boot,no-serve':>18}"
            else: line+=f"{'OK '+str(r['served'])+'/'+str(r['np']):>18}"
    print(line)
print()
for r in rows:
    if r["boot_ok"]!=1 and r["fail"]:
        print(f"  FAIL {r['model']} {r['backend']} {r['mode']}: {r['fail'][:180]}")
PY
echo "MBOOT_DONE_${SLURM_JOB_ID}"
