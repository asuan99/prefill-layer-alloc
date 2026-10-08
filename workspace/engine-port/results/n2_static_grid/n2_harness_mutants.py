"""N2(3-a) U-2 probe HARNESS mutants (CPU only) -- the harness half of the lesson-53 discipline (PREREG_N2_3A rev12
H-1, H-4…H-7; format = n2_mutants.MUTANTS: (id, description, expect, [(file, old, new), ...]); expect "kill" or
"equivalent"). HARNESS_MUTANTS can be appended to n2_mutants.MUTANTS once n2_mutants copies the harness files and runs
test_n2_harness (its FILES / TESTS tuples would grow by HARNESS_FILES / HARNESS_TESTS below).

Run (harness mutants only, sequential, alone):  PYTHONDONTWRITEBYTECODE=1 python3 n2_harness_mutants.py
Each mutant is applied to a fresh copy of the harness files + the registered reducer/calculator + the fake engine +
test_n2_harness.py in its own temp directory; a no-mutation control runs first; anchors must match exactly once
(else BROKEN-MUTANT); bytecode disabled; 900 s per mutant (TIMEOUT row, never a kill).
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MK, LG, RN = "n2_marks.py", "n2_loadgen.py", "n2_probe_run.py"
HARNESS_FILES = (MK, LG, RN)
SUPPORT = ("n2_probe_reduce.py", "n2_feasibility.py", "fake_n2_engine.py")
HARNESS_TESTS = ("test_n2_harness.py",)

HARNESS_MUTANTS = [
    ("H01", "H-4: connection mapping (ClientConnectionError) before timeout", "kill", [(LG, '    if isinstance(root, TimeoutError):\n        return "cell_timeout"\n    nbytes = getattr(exc, "bytes_received", 0)\n', '    nbytes = getattr(exc, "bytes_received", 0)\n    import aiohttp\n    if isinstance(root, aiohttp.ClientConnectionError) and not nbytes:\n        return "http_connection_error"\n    if isinstance(root, TimeoutError):\n        return "cell_timeout"\n')]),
    ("H02", "H-4: connector error after response bytes still VI", "kill", [(LG, "_connector_errors()) and not nbytes:", "_connector_errors()):")]),
    ("H03", "H-4: server_down without an observed exit code", "kill", [(LG, "    if server_exit_code() is not None:\n", "    if True:\n")]),
    ("H04", "F3: co-load payload assertion removed", "kill", [(LG, "            if bad:\n                raise CoLoadPayloadInvalid", "            if False:\n                raise CoLoadPayloadInvalid")]),
    ("H05", "H-1: cell_end only on success (not in finally)", "kill", [(MK, "            self.t_end = self.mw.now()\n            self.mw.cell_end(self.cell_id, t=self.t_end)\n", "            self.t_end = self.mw.now()\n            if cause is None:\n                self.mw.cell_end(self.cell_id, t=self.t_end)\n")]),
    ("H06", "H-1: coload_sent time read after the co requests are sent", "kill", [(LG, "            mw.coload_sent(cell_id, t_co)\n            for bd in co_bodies:", "            for bd in co_bodies:"), (LG, "            done_ev = asyncio.Event()\n", "            await asyncio.sleep(0.01)\n            t_co = clock()\n            st[\"t_coload\"] = t_co\n            mw.coload_sent(cell_id, t_co)\n            done_ev = asyncio.Event()\n")]),
    ("H07", "H-1: co-load paced (10 ms between requests)", "kill", [(LG, "                co_tasks.append(asyncio.ensure_future(stream(session, url, r, bd, on_co_token, clock)))\n", "                co_tasks.append(asyncio.ensure_future(stream(session, url, r, bd, on_co_token, clock)))\n                await asyncio.sleep(0.01)\n")]),
    ("H08", "R11-1: co-load K - 1 requests", "kill", [(LG, "for j in range(spec.k)]", "for j in range(spec.k - 1)]")]),
    ("H09", "§4-2: trigger at c + 0 (A1 window incomplete)", "kill", [(LG, "TRIGGER_EXTRA = 71 ", "TRIGGER_EXTRA = 0 ")]),
    ("H10", "§4-2: E-alpha cancel at +100 (A2 short)", "kill", [(LG, "EA_CANCEL_TOKENS = 16 + 128 ", "EA_CANCEL_TOKENS = 100 ")]),
    ("H11", "H-5: no 2 x flush_interval wait before stop", "kill", [(RN, 'await asyncio.sleep(2 * opts.get("flush_interval_s", FLUSH_INTERVAL_S))', "await asyncio.sleep(0)")]),
    ("H12", "H-5: RunnerStop not mapped to runner_stop", "kill", [(LG, '    if isinstance(exc, RunnerStop):\n        return "runner_stop"\n', "")]),
    ("H13", "H-1: id uniqueness check removed", "kill", [(MK, "        if ident in pool:\n            raise MarkError", "        if False:\n            raise MarkError")]),
    ("H14", "H-6: marks file opened for append", "kill", [(MK, 'open(path, "x", encoding="utf-8")', 'open(path, "a", encoding="utf-8")')]),
    ("H15", "H-6: attempt directory reused (n = 1, exist_ok)", "kill", [(RN, "    n = (prev[-1][0] + 1) if prev else 1\n", "    n = 1\n"), (RN, "    os.mkdir(path)  ", "    os.makedirs(path, exist_ok=True)  ")]),
    ("H16", "H-7: stop rule compares ordered lists", "kill", [(RN, 'set(results[-2].get("reason_codes", [])) == set(last.get("reason_codes", []))', 'results[-2].get("reason_codes", []) == last.get("reason_codes", [])')]),
    ("H17", "H-7: two-same-codes stop removed", "kill", [(RN, "    if len(results) >= 2 and results[-2]", "    if len(results) >= 99 and results[-2]")]),
    ("H18", "R11-2: boot_plan not written", "kill", [(MK, "        for d, n in sorted(self.plan.items()):", "        for d, n in []:")]),
    ("H19", "H-4 (3): boot exception classified after the runner's own stop", "kill", [(RN, "                raise n2_marks.Classified(classify(e), e) from e", "                raise")]),
    ("H20", "§4-2 step 6: engine-idle wait removed from the drain", "kill", [(LG, "            await wait_engine_idle(", "            0 and await wait_engine_idle(")]),
    ("H21", "§3-1: sticky knob not set", "kill", [(RN, 'PDMUX_STICKY_PARTITION="1",', 'PDMUX_STICKY_PARTITION="0",')]),
    ("H22", "§11-1-like sticky wiring check removed", "kill", [(RN, 'if "sticky partition ENABLED" not in txt or want not in txt:', "if False:")]),
    ("H23", "H-6: absolute raw paths in probe.json provenance", "kill", [(RN, 'telemetry=f"tel_{b}.jsonl"', 'telemetry=os.path.abspath(f"tel_{b}.jsonl")')]),
    ("H24", "Δℓ sign flipped (ℓ_co − ℓ_d)", "kill", [(LG, "    return statistics.median(lags) - l_co", "    return l_co - statistics.median(lags)")]),
    ("H25", "fix_s = whole cell duration", "kill", [(LG, 'fix = (cell.t_end - cell.t_start) - (st["t_active_end"] - st["t_first_send"])', "fix = (cell.t_end - cell.t_start)")]),
    ("H26", "ABORT_TRACE co-load max_new_tokens check removed", "kill", [(LG, "    if coload and (max_new_tokens is None or int(max_new_tokens) != 1):", "    if False:")]),
    ("H27", "server exit code never consulted by the boot classifier", "kill", [(RN, "return lg.classify_exception(exc, lambda: server.exit_code(grace))", "return lg.classify_exception(exc, lambda: None)")]),
    ("H28", "§4-2: upper median instead of lower median", "kill", [(LG, "    return v[int((len(v) - 1) * 0.5)]", "    return v[int(len(v) * 0.5)]")]),
    ("H29", "§4-2 step 5: cancel margin ignores delta", "kill", [(LG, '>= st["base"] + EA_CANCEL_TOKENS + spec.delta:', '>= st["base"] + EA_CANCEL_TOKENS:')]),
    ("H30", "co-load fired without waiting for the trigger", "kill", [(LG, '            if not await _wait_or_fail(st["trigger"], tasks, "trigger"):', '            if not await _wait_or_fail(asyncio.Event(), [], "trigger") and False:')]),
]


def run(selected=None):
    names = HARNESS_FILES + SUPPORT
    src = {fn: open(os.path.join(HERE, fn)).read() for fn in names}
    tests = {fn: open(os.path.join(HERE, fn)).read() for fn in HARNESS_TESTS}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "N2_N3_SH": os.path.join(HERE, "..", "n3_capacity", "n3_capacity.sh")}
    rows = []
    muts = [m for m in HARNESS_MUTANTS if not selected or m[0] in selected]
    for mid, desc, expect, reps in [("CTRL", "no mutation", "pass", [])] + muts:
        files = dict(src)
        broken = False
        for fn, old, new in reps:
            if files[fn].count(old) != 1:
                rows.append((mid, desc, "BROKEN-MUTANT", f"anchor count {files[fn].count(old)} in {fn}"))
                broken = True
                break
            files[fn] = files[fn].replace(old, new)
        if not broken:
            d = tempfile.mkdtemp(prefix=f"n2hmut_{mid}_")
            try:
                for fn, c in {**files, **tests}.items():
                    with open(os.path.join(d, fn), "w") as fh:
                        fh.write(c)
                try:
                    r = subprocess.run([sys.executable, "-m", "unittest", *[t[:-3] for t in HARNESS_TESTS]], cwd=d,
                                       env=env, capture_output=True, text=True, timeout=900)
                except subprocess.TimeoutExpired:
                    rows.append((mid, desc, "TIMEOUT", "900 s"))
                    print(f"{mid:5s} {desc:62s} {'TIMEOUT':20s} 900 s", flush=True)
                    continue
                pyc = any(x.endswith(".pyc") for _, _, fs in os.walk(d) for x in fs)
                tail = [ln for ln in r.stderr.strip().split("\n") if ln.startswith(("OK", "FAILED"))]
                tail = tail[-1] if tail else r.stderr[-200:]
                failed = r.returncode != 0
                if expect == "pass":
                    status = "CONTROL-OK" if not failed else "CONTROL-FAILED"
                elif expect == "kill":
                    status = "KILLED" if failed else "SURVIVED"
                else:
                    status = "EQUIVALENT-SURVIVED" if not failed else "EQUIVALENT-KILLED"
                rows.append((mid, desc, status, f"{tail}{' pyc!' if pyc else ''}"))
            finally:
                shutil.rmtree(d, ignore_errors=True)
        print(f"{rows[-1][0]:5s} {rows[-1][1]:62s} {rows[-1][2]:20s} {rows[-1][3]}", flush=True)
    return rows


if __name__ == "__main__":
    run(set(sys.argv[1:]) or None)
