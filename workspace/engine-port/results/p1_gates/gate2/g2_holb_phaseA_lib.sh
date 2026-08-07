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
# Also writes the full raw HTTP response body (marker line stripped) to
# DUMP_PATH (audit trail).
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
greedy_call () {
  local PORT="$1" DUMP="$2"
  local MARK="___HTTP_CODE___"
  curl -s -w "\n${MARK}:%{http_code}" "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d "$(python3 - "$PROMPT_TXT" "$OUTLEN" <<'PYEOF'
import json, sys
prompt_path, max_new = sys.argv[1], int(sys.argv[2])
print(json.dumps({"text": open(prompt_path).read(),
                  "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}))
PYEOF
)" | python3 -c "
import sys, json, hashlib, shlex

def out(**kw):
    print(' '.join(f'{k}={shlex.quote(str(v))}' for k, v in kw.items()))

MARK = '___HTTP_CODE___:'
raw = sys.stdin.read()
idx = raw.rfind('\n' + MARK)
if idx == -1:
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
"
}
