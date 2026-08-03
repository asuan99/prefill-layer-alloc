# D54 anchor attempt (2026-08-03) — CANCELLED, two reusable findings preserved

Status: jobs **872920** (T8) and **872921** (Hs8) were submitted to measure a
[prefill 16, decode 54] block-replicated anchor for the sticky `G_LEVER`
threshold, then **cancelled by explicit coordinator instruction** after ~17
minutes because the run's own gate showed the data was invalid, for reasons
independent of (but convergent with) claims-auditor's C2 estimand-identification
finding. **Nothing was resubmitted.** No performance/policy claim is made
anywhere in this note — this is an instrumentation/reproducibility record only,
not reviewed by claims-auditor, and it does not touch any canonical document.

```
$ sacct -j 872920,872921 --format=JobID,JobName,State,Elapsed,ExitCode
872920   s8d54   CANCELLED+  00:17:21  0:0
872920.batch     CANCELLED  00:17:22  0:15
872921   s8d54   CANCELLED+  00:17:21  0:0
872921.batch     CANCELLED  00:17:22  0:15
```

Root `.out`/`.err`: none produced by these jobs (`--output`/`--error` were
pointed at this directory in `s8_sweep_d54.sbatch`; verified via `find` on the
project root after cancellation). An earlier pair of jobs, 872918/872919,
*did* write `s8d54_872918.{out,err}` / `s8d54_872919.{out,err}` to the project
root before this fix — they were cancelled within seconds (before booting a
server) and their stray logs were moved into
`_stray_root_logs/` in this directory rather than left at the root.

Design as submitted (for context, not as a result to cite): `PDMUX_R2_POLICY=fixed`,
sticky **OFF**, arms T8 (Qwen2.5-7B, triton) + Hs8 (Nemotron-H-8B, flashinfer),
cells {d16, d54} only, 4 blocks/arm alternating cell order by block parity,
workload/server-args copied field-for-field from job 865493. Config artifacts:
`s8_sweep_d54.sbatch`, `pdmux_p16_d54.yml`, `d54_block_ratio.py` (unused —
no valid data reached it). `c2_anchor.py` in `../s8_frontier/` was extended
additively (d54 cell + `<job>_blk<N>` job-id pattern) to stay parseable; no
existing definition (SPLIT_HI, unit, statistic, gate) was changed, and a
regression check against `--job 865493` reproduced byte-identical output
after the edit.

---

## Finding 1 — keepalive prompt overflows the context cap: s8_scaleup is not reproducible as-is

`s8_keepalive_prompt_224.txt` (the keepalive prompt file, **byte-identical**
to the one 865493 used — it already existed on disk and was reused, not
regenerated) tokenizes to **1793 tokens** (Qwen2.5-7B tokenizer). `CTXCAP =
CTX + OUTTOK + 256 = 1024 + 512 + 256 = 1792`. Every keepalive request this
run made was rejected:

```
[2026-08-03 20:22:22] [http_server] Error: The input (1793 tokens) is longer
  than the model's context length (1792 tokens).
[2026-08-03 20:22:22] INFO: 127.0.0.1:45676 - "POST /generate HTTP/1.1" 400 Bad Request
```
(from `s8_deconf_T8_C1024_d16_872920_blk1_srv.log`; the same message repeats
for every one of the ~11,700 keepalive attempts per rep.)

Client-side this reads as `"keepalive_done": 0, "keepalive_errors": ~11700`
per rep across **both arms, both cells** (d16 and d54 alike). Job 865493, with
the exact same file and exact same printed `context_length=1792`, reported
`"keepalive_done": 528, "keepalive_errors": 0` (T8 d16 r1) — i.e. the request
that overflows today did not overflow on 2026-07-27.

Both srv.logs from today print `max_total_num_tokens=...` / `context_len=1792`
identically to 865493's, so the *configured* cap did not change; what changed
is that the **rejection now fires where it previously did not**, for the
identical 1793-token input against the identical 1792-token cap. This is most
consistent with engine-tree churn between 2026-07-27 and today (a
context-length check that got stricter, or an off-by-one that moved) — **root
cause not investigated, out of scope for this note.**

**Reproducibility consequence**: any future attempt to reproduce/extend
`results/s8_scaleup` with the current engine tree and `KEEPA_REPS=224` will
hit the same wall. **Trivial fix, not applied here**: `KEEPA_REPS=223` or
lower brings the keepalive prompt under the 1792-token cap (or raise
`CTXCAP`). Left for whoever picks this back up.

---

## Finding 2 — independent convergence with claims-auditor's C2 estimand finding (the more important one)

Two independent routes reached the same conclusion in this session, from
different evidence:

**Route A (code read, before any GPU run):** `_init_sticky_partition`'s
docstring (`multiplexing_mixin.py:206-231`) states that with
`PDMUX_STICKY_PARTITION` OFF (the default, and what all of `s8_scaleup`
including 865493 used), `adjust_stream_groups` falls back to the
unpartitioned `(0, total_sm)` stream group **whenever decode is busy but no
prefill batch is in flight**. Continuous prefill presence is therefore not a
side effect of the workload — it is *required* for the target decode-SM
division to be realized at all. `s0dc_client.py`'s `keepalive_worker`
docstring says exactly this: "keeps the pdmux split partition active...
without this the arm silently reverts to the full-108 stream."

**Route B (measured collapse, this run):** when Finding 1's bug killed 100%
of keepalive requests, in-window co-residency collapsed from 865493's
~90–100% to **31–34%** (see Finding 3 below) — i.e. removing the
keepalive-driven prefill-saturation device was sufficient, on its own, to
break the partition realization that 865493 relied on.

**Route C (telemetry cross-tab, this run, computed after the coordinator's
flag):** time-weighted joint distribution of `decode_sms` and
`prefill_active_batch_size>0` over decode-busy `runtime_snapshot` rows,
block-1 only (the only reps that ran to completion):

| file (arm/cell) | @ decode_sms==D, prefill_active | @ decode_sms==108, prefill_active |
|---|---|---|
| T8 d16  | 379/387 = **0.979** | 0/837 = **0.000** |
| T8 d54  | 1035/1046 = **0.989** | 0/619 = **0.000** |
| Hs8 d16 | 312/320 = **0.975** | 0/615 = **0.000** |
| Hs8 d54 | 744/747 = **0.996** | 0/547 = **0.000** |

This is the same shape claims-auditor reported independently (`prefill_active>0`
@ `decode_sms==D` = 0.943–0.996, @108 = 0.0000, 8/8 files, zero bidirectional
violations) — obtained here from a differently-broken run (keepalive dead)
against a working run (865493, keepalive alive) via a completely different
comparison (before/after a workload-mechanism failure), not from re-deriving
the auditor's own telemetry table.

**Put together**: C2's high realized-partition residency was never a property
of decode-SM control by itself — it was produced by a specific workload
device (keepalive saturation) that manufactures near-continuous prefill
presence, and `decode_sms==D` is realized if and only if that device is
working. Pull the device out (as Finding 1's bug did, by accident) and the
partition — and with it the entire measured contrast — does not reproduce.
This is the same fact the audit already identified from the SPLIT-population
selection angle; here it shows up as an operational failure mode instead of a
labeling artifact, which is why it is being recorded as convergent rather than
duplicate evidence.

---

## Finding 3 — preserved measurements (context only, not a result)

`REALIZED_PIN` (all-busy definition, `gate=0.80`), block 1 only — the only
reps that completed before cancellation:

| arm | cell | frac | verdict |
|---|---|---|---|
| T8  | d16 | 0.316 | FAIL |
| T8  | d54 | 0.628 | FAIL |
| Hs8 | d16 | 0.342 | FAIL |
| Hs8 | d54 | 0.577 | FAIL |

Block 2 (`d54` for both arms) had booted a server and started emitting
telemetry when cancelled, but completed zero reps — that boot's files
(`*_872920_blk2_*`, `*_872921_blk2_*`) are present but contain no
measurement rows and should not be used for anything.

`CLOCKS` (`nvidia-smi clocks.sm`, block 1, T8 only shown; Hs8 pattern is the
same): all four d16 reps and all four d54 reps sit at **mean 1396–1410 MHz**
(min dips to 1155 MHz in 2/8 reps, max 1410 MHz throughout) — i.e. flat and
indistinguishable between the two partitioned cells, consistent with
claims-auditor's report that C2's *split* cells all clock near 1410 MHz.
**This campaign did not run an `np` (unsplit/108) cell**, so it cannot
independently corroborate the auditor's separate observation that the
*unsplit* np cell drops to ~1290 MHz on T8 — that check would need a cell this
design didn't include.

All raw artifacts (srv.log, telemetry.jsonl, rep raw.jsonl, rep clocks.csv,
result.txt) for the above are preserved under this directory with the
`_872920_blk1_*` / `_872921_blk1_*` / `_872920_blk2_*` / `_872921_blk2_*`
filename stems; nothing was deleted or overwritten.

---

## Not done / explicitly out of scope for this note

- Root cause of the context-length check change (Finding 1) — not investigated.
- Any fix, re-run, or resubmission — **none performed**, per instruction.
- S1 (sticky-ON, dual-workload, claims-auditor's proposed design) — not
  submitted; awaiting user decision.
- Any performance or policy interpretation of the numbers above.
