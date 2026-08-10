#!/bin/bash
# Shared Phase-A (G3 correctness) helper functions for g2_holb_observer.sbatch
# AND g2_holb_smoke.sbatch. Sourced, not executed. Both scripts run the exact
# same code path, per the coordinator's post-mortem instruction (2026-08-06):
# "smoke test the ACTUAL sbatch path before resubmitting -- not a paraphrase
# of it."
#
# -----------------------------------------------------------------------------
# ROOT CAUSE of job 874601's spurious "G3 FAIL" (diagnosed by the coordinator,
# confirmed here independently):
#
# The original prompt generator built the Phase-A greedy prompt out of literal
# decimal digit STRINGS (`random.randint(1000, 30000)` joined by spaces),
# unrelated to any tokenizer. Re-tokenized by the *target model's own*
# tokenizer, that text is NOT ~2000 tokens:
#
#     Zyphra/Zamba2-2.7B                 : 11361 tokens (5.7x the target)
#     ibm-granite/granite-4.0-h-micro-base:  6005 tokens (3.0x the target)
#
# For Zamba2 (ctx=4096) that blows past the context window, so /generate
# returns `{"error": {...}}` instead of a normal response -- the old
# extractor did `json.load(...)['text']`, which raised `KeyError: 'text'` on
# every single call (16/16 in that job), producing empty shas that trivially
# "matched" (empty == empty) and were mislabelled RESULT=FAIL (compared
# `off=none on=none` -- the coordinator's smoking gun). Granite (ctx=8192)
# happened to survive at 6005 tokens by luck of a bigger window and produced
# a real, matching, non-empty sha -- but was NOT walking the intended
# ~2000-token / 512-chunk-boundary path either, so even that "PASS" measured
# the wrong thing.
#
# FIX: build the prompt exactly the way sglang.bench_serving's own
# `--dataset-name random-ids` dataset does
# (benchmark/datasets/random.py:sample_random_requests, the
# `random_sample=False` branch): sample consecutive-offset token ids modulo
# the tokenizer's vocab size and decode them. Re-encoding that text lands
# close to the target length (measured: Zamba2 2117, Granite 2552 -- both
# comfortably under their context windows, and both within ~10-28% of the
# 2000 target instead of 3-5.7x over it -- bench_serving's own harness does
# not guarantee an exact round trip either, see its comment "Need to truncate
# input_len as server encode will add special token").
#
# SECOND FIX: prefer `output_ids` (token-level) over `text` (string-level) for
# the equivalence check when the server provides it (it does whenever
# `skip_tokenizer_init=False`, true for all four Gate-2 arms) -- stronger,
# immune to any incidental decode-string differences.
#
# THIRD FIX: distinguish "could not measure" from "measured and it differs".
# `greedy_call` never raises on a malformed/error response; it always prints
# exactly one `KEY=value` line, eval-safe (values are `shlex.quote`d), so the
# caller can tell STATUS=OK from STATUS=ERROR and label the result
# PASS / FAIL / UNDETERMINED instead of forcing everything into PASS/FAIL.
# -----------------------------------------------------------------------------

# build_g2holb_prompt: writes $PROMPT_TXT and $PROMPT_META.
# Requires: MODEL, INLEN, PROMPT_TXT, PROMPT_META (caller-set globals).
build_g2holb_prompt () {
  python3 - "$MODEL" "$INLEN" "$PROMPT_TXT" "$PROMPT_META" <<'PYEOF'
import json, sys
import numpy as np
from transformers import AutoTokenizer

model, inlen, out_path, meta_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
rng = np.random.RandomState(4242)
offset = int(rng.randint(0, tok.vocab_size))
ids = [int((offset + j) % tok.vocab_size) for j in range(inlen)]
text = tok.decode(ids)
reenc = tok.encode(text)
with open(out_path, "w") as f:
    f.write(text)
meta = {
    "model": model, "target_len": inlen, "vocab_size": tok.vocab_size,
    "offset": offset, "reencoded_len": len(reenc),
    "generator": "bench_serving random-ids replica (sample_random_requests, random_sample=False branch)",
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"PROMPT_BUILD model={model} target_len={inlen} reencoded_len={len(reenc)} vocab_size={tok.vocab_size}")
PYEOF
}

# greedy_call PORT DUMP_PATH -> prints exactly one eval-safe "KEY=value ..." line to stdout:
#   STATUS=OK METHOD=output_ids|text SHA=<hex> N_OUTPUT_IDS=<n> HTTP_CODE=200
#   STATUS=ERROR DETAIL=<shlex-quoted, single-line, truncated> HTTP_CODE=<code|000>
#   STATUS=TIMEOUT DETAIL=<...> HTTP_CODE=000        <- bounded-wall-clock abort
# Also writes the full raw HTTP response body (marker line stripped) to
# DUMP_PATH (audit trail), and -- only when STATUS is not OK -- an explicit
# machine-readable measurement-failure record to DUMP_PATH.status.json.
# Requires: PROMPT_TXT, OUTLEN (caller-set globals).
#
# 2026-08-06 post-mortem #2 (coordinator correction after the first fix): the
# ORIGINAL bug was not a schema mismatch -- it was a genuine 400 Bad Request
# ("input longer than context length") that the harness never looked at,
# because the byte-length prompt (11360 bytes) was mistaken for a token count
# (should have been ~2000 tokens; it re-tokenized to 11362). The body-only
# checks below (missing 'error' key etc.) happened to catch that case
# correctly, but relying on JSON-body heuristics alone is fragile -- a non-200
# response might not even have a JSON body. The HTTP status code is checked
# FIRST and explicitly, independent of what (if anything) is in the body.
#
# -----------------------------------------------------------------------------
# 2026-08-09 post-mortem #3 (job 876699): greedy_call was a RAW curl with NO
# timeout of any kind. In job 876699 the first real multi-chunk /generate under
# `--enable-deterministic-inference` + `--chunked-prefill-size 512` + Granite
# (2552-token prompt, 5 chunks) never returned: the server logged its last line
# at 23:34:43 and the call was still outstanding when SLURM killed the job at
# 00:26:19 -- ~52 minutes in ONE call, which ate the whole 1 h budget so the
# remaining arms never started. No CUDA error fired, no watchdog fired. The
# same prompt on the same path with OFF + chunk512 completed in ~1 s.
#
# FOURTH FIX -- bounded wall clock + a FOURTH status value:
#   * `--max-time` (total) and `--connect-timeout`, both env-overridable:
#         G2_GREEDY_MAX_TIME        (default 180, seconds)
#         G2_GREEDY_CONNECT_TIMEOUT (default 10,  seconds)
#   * curl's exit status is captured explicitly (it used to be swallowed by the
#     pipe) and rc=28 is reported as STATUS=TIMEOUT, distinct from STATUS=ERROR.
#
# Why 180 s is not a guess. Every greedy_call response archived in this
# directory carries the server's own `meta_info.e2e_latency`. Across all 64
# archived greedy_call dumps (jobs 874602 / 874628 / 874633 / 874635 / 875344 /
# 875346 / 875610 / 875611 / 876699; both models; all four arms; prompts
# 2085-2121 tokens, 96 output tokens):
#       min 0.742 s   median 0.978 s   p90 2.534 s   max 6.605 s
# (the 6.6 s tail is the *first* chunk512 call of a fresh server -- kernel
# warm-up -- and it is the slowest single greedy_call ever recorded here).
# 180 s is therefore ~27x the slowest observed call and ~184x the median. It is
# also exactly the per-call timeout the SIBLING concurrency path already ships
# (g2ctrl_gate.py:51, g2_concurrent_gate.py:54, g2ea_gate.py:63 all do
# `urlopen(..., timeout=180)`), and that path fires 16 of these simultaneously,
# so a single sequential call can only be cheaper. Raise G2_GREEDY_MAX_TIME
# explicitly (and say so in the run log) if a future arm is legitimately slower.
#
# ***  STATUS=TIMEOUT IS A MEASUREMENT FAILURE, NEVER A MISMATCH AND NEVER A  ***
# ***  GATE FAILURE (methodology gate #21).  ***
# It carries no SHA at all, so it can never be compared against a baseline and
# can never be scored as "the outputs differ". Every caller in this directory
# routes any non-OK status to UNDETERMINED / "no data", and the .status.json
# sidecar makes "we did not measure" distinguishable from "we measured and it
# was clean" in the artifacts, not just in stdout. Because the call now returns,
# one hung arm costs at most G2_GREEDY_MAX_TIME instead of the whole job: the
# caller records the arm as UNDETERMINED and proceeds to the next arm.
# -----------------------------------------------------------------------------
greedy_call () {
  local PORT="$1" DUMP="$2"
  local MARK="___HTTP_CODE___"
  local MAX_TIME="${G2_GREEDY_MAX_TIME:-180}"
  local CONNECT_TIMEOUT="${G2_GREEDY_CONNECT_TIMEOUT:-10}"
  # curl's stdout goes to a scratch file (deleted below) instead of straight
  # into the pipe, so that its exit status is observable. Under the callers'
  # `set -uo pipefail` a piped curl failure was previously invisible here.
  #
  # The scratch file must exist and be readable no matter what, because the
  # filter below reads it on stdin: if the redirection failed, python3 would
  # never start, greedy_call would print NOTHING, and the caller's `eval` +
  # `set -u` would kill the whole job on the next "$STATUS" expansion. That
  # invariant -- exactly one KEY=value line, always -- is load-bearing.
  local CURLRAW="${DUMP}.curlraw"
  if ! : > "$CURLRAW" 2>/dev/null; then
    CURLRAW="$(mktemp "${TMPDIR:-/tmp}/greedy_call.XXXXXX" 2>/dev/null)" || CURLRAW=/dev/null
  fi
  local RC=0
  local T0=$SECONDS
  curl -s --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
    -w "\n${MARK}:%{http_code}" "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d "$(python3 - "$PROMPT_TXT" "$OUTLEN" <<'PYEOF'
import json, sys
prompt_path, max_new = sys.argv[1], int(sys.argv[2])
print(json.dumps({"text": open(prompt_path).read(),
                  "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}))
PYEOF
)" > "$CURLRAW" || RC=$?
  local ELAPSED=$((SECONDS - T0))
  python3 -c "
import sys, json, hashlib, shlex

_emitted = {}

def out(**kw):
    _emitted.update(kw)
    print(' '.join(f'{k}={shlex.quote(str(v))}' for k, v in kw.items()))

CURL_RC = int(sys.argv[1])
ELAPSED = sys.argv[2]
MAX_TIME = sys.argv[3]
CONNECT_TIMEOUT = sys.argv[4]

MARK = '___HTTP_CODE___:'
raw = sys.stdin.read()
idx = raw.rfind('\n' + MARK)
if CURL_RC == 28:
    # curl exit 28 == operation timed out (--max-time or --connect-timeout).
    # MEASUREMENT FAILURE. No SHA is emitted, so this can never be scored as a
    # byte-mismatch; callers must record it as UNDETERMINED / no-data.
    body = raw if idx == -1 else raw[:idx]
    out(STATUS='TIMEOUT', HTTP_CODE='000',
        DETAIL='UNDETERMINED(curl_timeout rc=28 elapsed_s~%s max_time_s=%s connect_timeout_s=%s):'
               ' call did not return in time -- MEASUREMENT FAILURE, not a mismatch'
               ' (job 876699 hang class); partial_body=%r' % (ELAPSED, MAX_TIME, CONNECT_TIMEOUT, body[:200]))
elif CURL_RC != 0:
    # Transport-level failure with a non-timeout cause (connection refused,
    # reset, empty reply, ...). Also a measurement failure, also never a
    # mismatch -- kept distinct from TIMEOUT so triage is not guesswork.
    body = raw if idx == -1 else raw[:idx]
    out(STATUS='ERROR', HTTP_CODE='000',
        DETAIL='curl_transport_failed rc=%s elapsed_s~%s: %r' % (CURL_RC, ELAPSED, body[:200]))
elif idx == -1:
    # curl itself failed (connection refused etc.) before writing the -w suffix.
    out(STATUS='ERROR', HTTP_CODE='000', DETAIL=f'no_http_code_marker_curl_may_have_failed:{raw[:200]!r}')
    body = raw
else:
    body = raw[:idx]
    code = raw[idx + len('\n' + MARK):].strip()
    if code != '200':
        out(STATUS='ERROR', HTTP_CODE=code, DETAIL=f'UNDETERMINED(http {code}):{body[:300]!r}')
    else:
        try:
            d = json.loads(body)
        except Exception as e:
            out(STATUS='ERROR', HTTP_CODE=code, DETAIL=f'json_parse_failed:{e}:{body[:200]!r}')
            d = None
        if d is not None:
            if isinstance(d, list):
                out(STATUS='ERROR', HTTP_CODE=code, DETAIL=f'response_is_list_len={len(d)}:{body[:200]!r}')
            elif not isinstance(d, dict):
                out(STATUS='ERROR', HTTP_CODE=code, DETAIL=f'response_is_{type(d).__name__}:{body[:200]!r}')
            elif 'error' in d:
                out(STATUS='ERROR', HTTP_CODE=code, DETAIL=f'http_200_but_error_key:{str(d[\"error\"])[:300]!r}')
            elif d.get('output_ids'):
                payload = json.dumps(d['output_ids'])
                sha = hashlib.sha256(payload.encode()).hexdigest()
                out(STATUS='OK', HTTP_CODE=code, METHOD='output_ids', SHA=sha, N_OUTPUT_IDS=len(d['output_ids']))
            elif 'text' in d:
                sha = hashlib.sha256(d['text'].encode()).hexdigest()
                out(STATUS='OK', HTTP_CODE=code, METHOD='text', SHA=sha, N_OUTPUT_IDS=0)
            else:
                out(STATUS='ERROR', HTTP_CODE=code, DETAIL=f'no_output_ids_or_text_keys={sorted(d.keys())}')
with open('$DUMP', 'w') as f:
    f.write(body)
if _emitted.get('STATUS') != 'OK':
    # Explicit, machine-readable 'we could not measure this' record. Written
    # ONLY on the failure paths, so a healthy run's artifact set is unchanged
    # byte-for-byte. Consumers that only see the dump file (which on a timeout
    # is empty or partial) must not mistake it for a measured-and-different
    # result -- that is methodology gate #21.
    import time
    _rec = dict(_emitted)
    _rec.update({'curl_rc': CURL_RC, 'elapsed_s_approx': ELAPSED,
                 'max_time_s': MAX_TIME, 'connect_timeout_s': CONNECT_TIMEOUT,
                 'dump_path': '$DUMP', 'wall_ts': time.time(),
                 'note': 'greedy_call measurement failure: NOT a byte-mismatch and NOT a gate failure'})
    with open('$DUMP' + '.status.json', 'w') as f:
        json.dump(_rec, f, indent=2)
" "$RC" "$ELAPSED" "$MAX_TIME" "$CONNECT_TIMEOUT" < "$CURLRAW"
  rm -f "$CURLRAW"
}
