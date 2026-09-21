"""`results/r2_correctness/r2_correctness.sbatch`: the lambda0-audit read-outs
cost ZERO additional scored boots and cannot reach the verdict.

WHAT THIS PINS
--------------
The rule-layer audit of the campaign's stage-0 registration
(`results/r2_eval/lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` sec 3-8)
asked the correctness gate to carry three read-outs:

  I1  the boot banner at this ctx / mem-fraction (`max_mamba_cache_size`,
      `max_total_num_tokens`, `available_gpu_mem`)          RECORDED, not judged
  I2  8 x (256 in, 512 out) at concurrency 1                floor probe
  I3  (256,512) and (8192,64) at `--request-rate inf --max-concurrency 64`
                                                            upper-bound probe

"Zero additional boots" is only true, and "cannot reach the verdict" is only
true, because all three run against the WARM-UP server:

  * `r2_correctness_check.py` opens `boots.txt` and then, per label in it,
    `gen_<label>.json` / `srv_<label>.log` / `tel_<label>.jsonl` /
    `green_<label>.json`.  The warm-up boot is never written to `boots.txt`.
  * the warm-up boot starts without `PDMUX_TELEMETRY_PATH`, so it writes no
    telemetry file for anything to be read out of.
  * three per-boot checks are WHOLE-FILE predicates over a scored boot's own
    artifacts -- B3 ("cuda graph: True" on *every* Decode line), B5 (*every*
    controller_decision fixed/unlimited), B8 (*every* overlap snapshot on the D
    division) -- so the same load attached to a scored boot WOULD change the
    verdict's inputs.  That is why it is not attached there.

PINNED HERE
-----------
1.  `R2C_INSTRUMENT` defaults to 0, and at 0 the block starts no process at all
    (jobs 907100 / 907456 reproduce command-for-command).
2.  At 1 it runs exactly three `sglang.bench_serving` invocations with the
    registered flags, all of them against the WARM-UP port.
3.  With the warm-up boot down it runs nothing and says so (a measurement
    failure is reported as a measurement failure -- lesson 21 / gate #21).
4.  The scored boots' server command and client command are unchanged, and the
    verdict rule file is untouched.
5.  ★NEGATIVE CONTROLS (lesson #53): mutants that remove the guard, that point
    the probes at a scored port, or that drop `--request-rate inf` must each
    make one of the tests above fail.

The block is executed for real, extracted verbatim from the sbatch between its
two marker comments, with a stub `python` on PATH.  Nothing here boots a server
or touches a GPU.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_ROOT.parents[1]
SBATCH = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness.sbatch"
CHECKER = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness_check.py"
ENGINE_BENCH = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(os.environ.get("PDMUX_ROOT", str(TRACK_ROOT.parents[2])),
                     "sglang_engine_dev", "python")
) / "sglang" / "bench_serving.py"

BEGIN = "# --- BEGIN r2c instrumentation"
END = "# --- END r2c instrumentation ---"
WBEGIN = "# --- BEGIN r2c warmup launch"
WEND = "# --- END r2c warmup launch ---"
SYNC = TRACK_ROOT / "scripts" / "bootstrap" / "sync_engine_tree.sh"

# Same constant as tests/test_r2_correctness_ctx.py: a harness change may not
# move the judging rule.
CHECKER_SHA256 = "ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30"

# The scored boots' server command, byte for byte.  This is the thing that must
# not drift when instrumentation is bolted on; `R2C_INSTRUMENT` is not allowed
# to add, remove or reorder a single server argument.
SERVER_CMD_PINNED = '''server_cmd () {   # $1 = port
  echo python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \\
    --dtype bfloat16 --attention-backend "$ABE" --disable-radix-cache \\
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \\
    --random-seed "$SEED" --host 127.0.0.1 --port "$1" \\
    ${NKVS:+--triton-attention-num-kv-splits $NKVS} \\
    --enable-pdmux --pdmux-config-path "$CONFIG" --chunked-prefill-size -1 \\
    --disable-overlap-schedule --decode-log-interval 1
}'''

BANNER = [
    "[2026-09-09 11:57:18] Mamba Cache is allocated. max_mamba_cache_size: 48, "
    "conv_state size: 0.09GB, ssm_state size: 6.46GB",
    "[2026-09-09 11:57:18] KV Cache is allocated. #tokens: 2618868, "
    "K size: 19.98 GB, V size: 19.98 GB",
    "[2026-09-09 11:57:27] max_total_num_tokens=2618868, chunked_prefill_size=-1, "
    "max_prefill_tokens=16384, max_running_requests=48, context_len=16384, "
    "available_gpu_mem=13.38 GB",
]


def sbatch_text() -> str:
    return SBATCH.read_text(encoding="utf-8")


def instrument_block(text: str | None = None) -> str:
    """The instrumentation block, verbatim, without its marker lines."""
    text = sbatch_text() if text is None else text
    start = text.index(BEGIN)
    start = text.index("\n", start) + 1
    stop = text.index(END)
    return text[start:stop]


def warmup_block(text: str | None = None) -> str:
    """The warm-up launch block, verbatim, without its marker lines."""
    text = sbatch_text() if text is None else text
    start = text.index(WBEGIN)
    start = text.index("\n", start) + 1
    stop = text.index(WEND)
    return text[start:stop]


def non_comment_lines(text: str):
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            yield line


class _BlockRunner(unittest.TestCase):
    """Run the extracted block in bash with a stub `python` on PATH."""

    BASEPORT = 31234

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.out = self.tmp / "job_local"
        self.out.mkdir()
        (self.out / "srv_warmup.log").write_text(
            "boot noise\n" + "\n".join(BANNER) + "\nmore noise\n", encoding="utf-8"
        )
        self.stub_dir = self.tmp / "stub"
        self.stub_dir.mkdir()
        self.calls_path = self.tmp / "calls.txt"
        stub = self.stub_dir / "python"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "{ printf 'CALL'; for a in \"$@\"; do printf ' %s' \"$a\"; done; "
            "printf '\\n'; } >> " + shlex.quote(str(self.calls_path)) + "\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

    def run_block(self, instr, warm_ok=1, block=None):
        if block is None:
            block = instrument_block()
        script = self.tmp / "block.sh"
        script.write_text(
            "set -uo pipefail\n"
            f"INSTR={shlex.quote(str(instr))}\n"
            f"WARM_OK={shlex.quote(str(warm_ok))}\n"
            f"OUT={shlex.quote(str(self.out))}\n"
            f"BASEPORT={self.BASEPORT}\n"
            'MODEL="a/model"\n'
            f"SGPT_RAW={shlex.quote(str(self.tmp / 'sharegpt.json'))}\n"
            + block,
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PATH"] = str(self.stub_dir) + os.pathsep + env["PATH"]
        return subprocess.run(
            ["/usr/bin/bash", str(script)],
            capture_output=True, text=True, timeout=300, env=env,
        )

    def calls(self):
        if not self.calls_path.exists():
            return []
        return [
            shlex.split(line[len("CALL "):])
            for line in self.calls_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("CALL ")
        ]


class TestDefaultIsOff(_BlockRunner):
    def test_knob_default_is_zero(self):
        self.assertIn('INSTR="${R2C_INSTRUMENT:-0}"', sbatch_text())

    def test_off_starts_no_process_and_writes_nothing(self):
        result = self.run_block(instr=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls(), [], "the default path must run nothing")
        self.assertFalse((self.out / "instrument").exists())

    def test_mutant_without_the_guard_is_caught(self):
        """★NEGATIVE CONTROL for test_off_starts_no_process_and_writes_nothing."""
        block = instrument_block()
        guard = 'if [ "$INSTR" = "1" ] && [ "$WARM_OK" = "1" ]; then'
        self.assertIn(guard, block)
        mutant = block.replace(guard, "if true; then")
        self.run_block(instr=0, block=mutant)
        self.assertNotEqual(self.calls(), [], "the guard must be what stops it")


class TestReadOutsWhenEnabled(_BlockRunner):
    def setUp(self):
        super().setUp()
        # No return-code assertion: the block's last statement is a summary
        # `grep` over a bench log, which legitimately exits 1 when the stub
        # produced no summary lines.  The sbatch does not branch on it either
        # (`set -uo pipefail`, no `-e`); what matters is what was invoked.
        self.result = self.run_block(instr=1)
        self.assertNotIn("syntax error", self.result.stderr)
        self.recorded = self.calls()

    def test_exactly_three_bench_serving_invocations(self):
        self.assertEqual(len(self.recorded), 3, self.recorded)
        for call in self.recorded:
            self.assertEqual(call[:2], ["-m", "sglang.bench_serving"], call)

    def test_every_probe_targets_the_warmup_port(self):
        for call in self.recorded:
            self.assertIn("--port", call)
            self.assertEqual(call[call.index("--port") + 1], str(self.BASEPORT), call)

    def test_mutant_pointing_at_a_scored_port_is_caught(self):
        """★NEGATIVE CONTROL for test_every_probe_targets_the_warmup_port.

        A scored boot runs on BASEPORT+i; if the read-outs drifted onto one of
        those, they would be loading a boot whose whole-file predicates (B3, B5,
        B8) the scorer evaluates.
        """
        self.calls_path.unlink(missing_ok=True)
        mutant = instrument_block().replace('--port "$BASEPORT"',
                                            '--port "$((BASEPORT + 1))"')
        self.run_block(instr=1, block=mutant)
        ports = {c[c.index("--port") + 1] for c in self.calls()}
        self.assertNotIn(str(self.BASEPORT), ports)

    def test_i2_is_shape_a_at_concurrency_one(self):
        call = self.recorded[0]
        self._assert_pairs(call, {
            "--random-input-len": "256", "--random-output-len": "512",
            "--num-prompts": "8", "--max-concurrency": "1",
            "--request-rate": "inf", "--seed": "7",
        })

    def test_i3_is_the_two_campaign_shapes_at_rate_inf(self):
        shapes = [(c[c.index("--random-input-len") + 1],
                   c[c.index("--random-output-len") + 1]) for c in self.recorded[1:]]
        self.assertEqual(shapes, [("256", "512"), ("8192", "64")])
        for call in self.recorded[1:]:
            self._assert_pairs(call, {
                "--num-prompts": "300", "--request-rate": "inf",
                "--max-concurrency": "64", "--seed": "41",
            })

    def test_input_length_is_exact_on_every_probe(self):
        """`--tokenize-prompt` + a real dataset is what made probe C's input
        token count land on 130 x 8192 with zero error."""
        for call in self.recorded:
            self.assertIn("--tokenize-prompt", call)
            self.assertIn("--dataset-path", call)
            self._assert_pairs(call, {"--random-range-ratio": "1.0",
                                      "--dataset-name": "random"})

    def test_every_probe_is_wall_clock_bounded(self):
        """The gate itself must survive a hung read-out.

        Worst case here is 600 + 900 + 1500 s = 50 min of instrumentation; the
        scored boots need ~20 min after it, which is why the script's own
        `#SBATCH --time` was raised to 02:30:00 (D11).  The structural worst,
        where every timeout in the script fires, is ~161 min and exceeds even
        that -- registered in newpair_prereg/ sec 9(f).
        """
        block = instrument_block()
        self.assertIn("timeout 600 python -m sglang.bench_serving", block)
        self.assertIn('"I3a_shapeA 256 512 900" "I3b_shapeB 8192 64 1500"', block)
        self.assertIn('timeout "$I_TMO" python -m sglang.bench_serving', block)

    def test_per_request_details_are_kept(self):
        for call in self.recorded:
            self.assertIn("--output-details", call)
            self.assertIn("--output-file", call)

    def test_mutant_dropping_rate_inf_is_caught(self):
        """★NEGATIVE CONTROL: `--request-rate inf` is the whole point of I3.

        With a finite rate the arrival process is a Poisson realisation seeded
        by `--seed`, which is exactly the free surface that killed the lambda0
        registration (audit deaths N2/S1).
        """
        self.calls_path.unlink(missing_ok=True)
        mutant = instrument_block().replace("--num-prompts 300 --request-rate inf",
                                            "--num-prompts 300 --request-rate 0.7")
        self.run_block(instr=1, block=mutant)
        rates = {c[c.index("--request-rate") + 1] for c in self.calls()}
        self.assertNotEqual(rates, {"inf"})

    def test_client_bottleneck_check_reads_the_engine_not_the_client(self):
        """confound #7, corrected.

        `--max-concurrency 64` makes bench_serving's own `Concurrency:` metric
        (sum of e2e latencies / duration, queueing included) sit near 64 BY
        CONSTRUCTION, so it cannot distinguish "engine saturated" from "client
        bottlenecked".  The engine's own admitted batch can.
        """
        block = instrument_block()
        self.assertIn('grep -oE "#running-req: [0-9]+" "$OUT/srv_warmup.log"', block)
        produced = (self.out / "instrument" / "I3_max_running_req.txt")
        self.assertTrue(produced.exists())

    def test_log_offsets_delimit_every_read_out(self):
        marks = dict(
            line.split()
            for line in (self.out / "instrument" / "I_log_offsets.txt")
            .read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(
            sorted(marks),
            ["I2_end", "I2_start", "I3a_shapeA_end", "I3a_shapeA_start",
             "I3b_shapeB_end", "I3b_shapeB_start"],
        )

    def test_i1_banner_is_captured_from_the_warmup_log(self):
        captured = (self.out / "instrument" / "I1_banner_warmup.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_mamba_cache_size: 48", captured)
        self.assertIn("max_total_num_tokens=2618868", captured)
        self.assertIn("available_gpu_mem=13.38 GB", captured)
        self.assertNotIn("boot noise", captured)

    def test_artifacts_stay_inside_the_instrument_subdirectory(self):
        """Nothing the scorer opens may appear or change."""
        top = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(top, ["instrument", "srv_warmup.log"])

    def _assert_pairs(self, call, expected):
        for flag, value in expected.items():
            self.assertIn(flag, call, call)
            self.assertEqual(call[call.index(flag) + 1], value, (flag, call))


class TestWarmupDown(_BlockRunner):
    def test_nothing_runs_and_the_skip_is_announced(self):
        result = self.run_block(instr=1, warm_ok=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertIn("INSTRUMENT_SKIPPED_WARMUP_DOWN", result.stdout)


class TestScoredPathUntouched(unittest.TestCase):
    def test_server_command_is_byte_identical(self):
        text = sbatch_text()
        start = text.index("server_cmd () {")
        stop = text.index("\n}", start) + 2
        self.assertEqual(text[start:stop], SERVER_CMD_PINNED)

    def test_scored_client_invocation_is_unchanged(self):
        self.assertIn(
            'timeout 900 python3 "$HERE/r2_correctness_client.py" --port "$PORT" \\\n'
            '    --boot "$LABEL" --out "$OUT/gen_${LABEL}.json" --n-probes 8 \\',
            sbatch_text(),
        )

    def test_verdict_rule_file_is_untouched(self):
        digest = hashlib.sha256(CHECKER.read_bytes()).hexdigest()
        self.assertEqual(digest, CHECKER_SHA256)

    def test_bench_serving_appears_only_inside_the_block(self):
        text = sbatch_text()
        block = instrument_block(text)
        outside = text.replace(block, "")
        self.assertEqual(outside.count("sglang.bench_serving"), 0)
        self.assertEqual(block.count("sglang.bench_serving"), 2)  # I2 + the I3 loop

    def test_block_runs_after_warmup_health_and_before_teardown(self):
        text = sbatch_text()
        health = text.index('if wait_health "$BASEPORT" "$WPID" 180; then')
        begin = text.index(BEGIN)
        end = text.index(END)
        teardown = text.index('stop_server "$BASEPORT" "$WPID"')
        self.assertLess(health, begin)
        self.assertLess(end, teardown)

    def test_warmup_boot_is_never_named_in_bootstxt(self):
        text = sbatch_text()
        appends = [ln for ln in non_comment_lines(text) if ">> \"$OUT/boots.txt\"" in ln]
        self.assertEqual(len(appends), 1, appends)
        self.assertIn('echo -n "$LABEL "', appends[0])


class TestWarmupGreenReadOut(unittest.TestCase):
    """D8: the warm-up boot's split is MEASURED when the read-outs run on it,
    and the default path's argv is unchanged.

    The read-outs' numbers are quoted as "at D44", so `target != realized` --
    the confound that made a "D108 anchor" turn out to be 16 SM -- has to be
    closed for the boot they actually run on.  The cost of closing it must be
    zero on the default path, which is what the byte-identity test below is.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.out = self.tmp / "job_local"
        self.out.mkdir()
        self.argv = self.tmp / "argv.txt"

    def run_warmup(self, instr, block=None):
        if block is None:
            block = warmup_block()
        script = self.tmp / "warm.sh"
        script.write_text(
            "set -uo pipefail\n"
            f"INSTR={shlex.quote(str(instr))}\n"
            f"OUT={shlex.quote(str(self.out))}\n"
            "DSM=44\nBASEPORT=31234\n"
            f'env () {{ printf "%s\\n" "$@" >> {shlex.quote(str(self.argv))}; }}\n'
            'server_cmd () { echo python -m sglang.launch_server --port "$1"; }\n'
            + block + "\nwait\n",
            encoding="utf-8",
        )
        result = subprocess.run(["/usr/bin/bash", str(script)],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.argv.read_text(encoding="utf-8").splitlines()

    DEFAULT_ARGV = [
        "PDMUX_R2_POLICY=fixed", "PDMUX_R2_FIXED_DSM=44",
        "python", "-m", "sglang.launch_server", "--port", "31234",
    ]

    def test_default_path_argv_is_byte_identical(self):
        """★The whole safety argument for D8: at INSTR=0 nothing is added.

        `$WARM_GREEN` is deliberately UNQUOTED so an empty value word-splits to
        nothing rather than to an empty argument.
        """
        self.assertEqual(self.run_warmup(instr=0), self.DEFAULT_ARGV)

    def test_mutant_quoting_the_variable_is_caught(self):
        """★NEGATIVE CONTROL for the test above: quoting it injects an empty
        argv entry, which is exactly the drift the test must catch."""
        block = warmup_block()
        self.assertIn("$WARM_GREEN \\", block)
        mutant = block.replace("$WARM_GREEN \\", '"$WARM_GREEN" \\')
        self.assertNotEqual(self.run_warmup(instr=0, block=mutant),
                            self.DEFAULT_ARGV)

    def test_instrumented_path_adds_exactly_the_read_out_pair(self):
        argv = self.run_warmup(instr=1)
        extra = [a for a in argv if a not in self.DEFAULT_ARGV]
        self.assertEqual(extra, [
            "PDMUX_GREEN_READOUT=1",
            f"PDMUX_GREEN_READOUT_PATH={self.out}/instrument/green_warmup.json",
        ])
        self.assertEqual(len(argv), len(self.DEFAULT_ARGV) + 2)
        self.assertTrue((self.out / "instrument").is_dir())

    def test_the_read_out_file_is_not_a_name_the_scorer_can_open(self):
        """`green_warmup.json` is not `green_<label>.json` for any label the
        scored loop can emit, and it lives under instrument/ besides."""
        text = sbatch_text()
        self.assertIn('LABEL="L$n_l"', text)
        self.assertIn('LABEL="TD$n_td"', text)
        self.assertIn('green_${LABEL}.json', text)
        self.assertIn("instrument/green_warmup.json", text)
        self.assertNotIn("green_warmup", CHECKER.read_text(encoding="utf-8"))

    def test_realization_is_reported_next_to_the_read_outs(self):
        block = instrument_block()
        self.assertIn("sm_counts (prefill_sm, decode_sm)", block)
        self.assertIn("I1_green_warmup.txt", block)
        self.assertIn("WARMUP_GREEN_READOUT_MISSING", block)


class TestWallLimitAndProvenance(unittest.TestCase):
    def test_wall_limit_is_in_the_directive_not_the_command_line(self):
        """D11.  A wall cap changes no measured quantity, so raising it in the
        directive is free and removes the `sbatch --time=...` failure point."""
        self.assertIn("#SBATCH --time=02:30:00", sbatch_text())

    def test_flashinfer_backend_is_in_the_provenance_manifest(self):
        """D10.  Switching NemotronH off triton made this file load bearing for
        a CONCLUSION (the O-tier equality argument), not just for speed."""
        sync = SYNC.read_text(encoding="utf-8")
        entry = '"${runtime_python}/sglang/srt/layers/attention/flashinfer_backend.py" \\'
        self.assertIn(entry, sync)
        # Appended, never inserted: every earlier manifest stays a prefix.
        self.assertLess(sync.index('configs/mamba_utils.py'), sync.index(entry))
        self.assertLess(sync.index(entry), sync.index('> "${manifest_tmp}"'))

    def test_flashinfer_wheel_version_is_recorded(self):
        """The manifest hashes sglang's wrapper; the wheel is a separate
        artifact that no manifest line covers."""
        self.assertIn("flashinfer_version", sbatch_text())


class TestScorerCannotSeeTheReadOuts(unittest.TestCase):
    def test_checker_never_mentions_the_warmup_boot_or_the_instrument_dir(self):
        src = CHECKER.read_text(encoding="utf-8")
        self.assertNotIn("warmup", src)
        self.assertNotIn("instrument", src)

    def test_checker_opens_only_per_label_artifacts(self):
        """The five path templates, and no directory walk."""
        src = CHECKER.read_text(encoding="utf-8")
        templates = set(re.findall(r'f"([a-z_]+)_\{lb\}\.[a-z]+"', src))
        self.assertEqual(templates, {"gen", "srv", "tel", "green"})
        for walker in ("glob", "listdir", "scandir", "os.walk", "iterdir"):
            self.assertNotIn(walker, src)


class TestEngineCitationsInThePreRegistration(unittest.TestCase):
    """The registration cites engine lines; if the tree moves, say so here."""

    def test_rate_inf_skips_every_sleep(self):
        if not ENGINE_BENCH.exists():
            self.skipTest(f"{ENGINE_BENCH} not present")
        lines = ENGINE_BENCH.read_text(encoding="utf-8", errors="replace").splitlines()
        cited = "\n".join(lines[942:945])  # bench_serving.py:943-945, 1-based
        self.assertIn('if request_rate == float("inf")', cited)
        self.assertIn("continue", cited)

    def test_client_seed_is_process_global(self):
        if not ENGINE_BENCH.exists():
            self.skipTest(f"{ENGINE_BENCH} not present")
        lines = ENGINE_BENCH.read_text(encoding="utf-8", errors="replace").splitlines()
        cited = "\n".join(lines[1704:1706])  # :1705-1706
        self.assertIn("random.seed(args.seed)", cited)
        self.assertIn("np.random.seed(args.seed)", cited)


if __name__ == "__main__":
    sys.path.insert(0, str(TRACK_ROOT / "benchmarks"))
    unittest.main()
