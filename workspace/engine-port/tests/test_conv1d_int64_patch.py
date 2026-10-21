"""CPU checks for src/patches/triton_causal_conv1d_int64.patch (decision 15, 2026-10-20).

What is (and is not) tested here:
  * the patch applies forward to the pristine upstream file and reverses to the same bytes;
  * it is line-neutral (only kernel lines 69 and 77 change) and touches no file another tracked
    patch touches;
  * the static check (scripts/bootstrap/check_conv1d_int64.py) passes on the patched file and FAILS
    on the pristine file and on every reverted/weakened mutant (lesson 53: a check that cannot fail
    on the reverted change proves nothing);
  * sync_engine_tree.sh wires it (grep guard after the admission hook) and hashes the file as the
    LAST manifest entry;
  * optional: TTIR of the kernel compiled for sm_80 on the CPU has no i32 stride multiply (skipped
    when triton is not importable).  No kernel is LAUNCHED: numerical/bit equality and the >94.8k
    failure/repair are GPU gate items, not claimed here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
PATCH = ENG / "src" / "patches" / "triton_causal_conv1d_int64.patch"
SYNC = ENG / "scripts" / "bootstrap" / "sync_engine_tree.sh"
CHECK = ENG / "scripts" / "bootstrap" / "check_conv1d_int64.py"
REL = "sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"
MARK = "int64: token offsets"
PRISTINE_SHA256 = "1700fae290fb1c28d6968f1f2732067aeba26e92898653f3ec6fcbed3c24e5cb"  # SGLang v0.5.10(.post1)
RUNTIME = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(os.environ.get("PDMUX_ROOT", str(Path(__file__).resolve().parents[4])),
                    "sglang_engine_dev", "python")
)


def _load_check():
    spec = importlib.util.spec_from_file_location("check_conv1d_int64_under_test", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CK = _load_check()


def _patch(tmp, direction, pf=PATCH):
    with open(pf, "rb") as fh:
        return subprocess.run(["patch", direction, "--batch", "-p1", "-d", str(tmp)], stdin=fh,
                              capture_output=True)


class _TreeCopy(unittest.TestCase):
    """A temp copy of the runtime file, brought to the PRISTINE state (reverse-applied if the
    runtime already carries the patch)."""

    def setUp(self):
        src = RUNTIME / REL
        if not src.is_file():
            self.skipTest(f"runtime tree absent: {src}")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.target = self.tmp / REL
        self.target.parent.mkdir(parents=True)
        shutil.copy(src, self.target)
        if MARK in self.target.read_text():
            r = _patch(self.tmp, "--reverse")
            self.assertEqual(r.returncode, 0, r.stderr.decode() + r.stdout.decode())
        import hashlib
        if hashlib.sha256(self.target.read_bytes()).hexdigest() != PRISTINE_SHA256:
            self.skipTest("runtime causal_conv1d_triton.py is neither pristine v0.5.10 nor pristine+patch")
        self.pristine = self.target.read_text()

    def patched(self):
        r = _patch(self.tmp, "--forward")
        self.assertEqual(r.returncode, 0, r.stderr.decode() + r.stdout.decode())
        return self.target.read_text()


class TestApply(_TreeCopy):
    def test_forward_then_reverse_is_identity(self):
        new = self.patched()
        self.assertNotEqual(new, self.pristine)
        self.assertIn(MARK, new)
        r = _patch(self.tmp, "--reverse")
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        self.assertEqual(self.target.read_text(), self.pristine)

    def test_second_forward_is_refused(self):
        self.patched()
        r = _patch(self.tmp, "--forward")
        self.assertNotEqual(r.returncode, 0)  # sync relies on the grep guard, not on patch idempotence

    def test_line_neutral_two_lines_only(self):
        old, new = self.pristine.splitlines(), self.patched().splitlines()
        self.assertEqual(len(old), len(new))
        changed = [i + 1 for i, (a, b) in enumerate(zip(old, new)) if a != b]
        self.assertEqual(changed, [69, 77])
        for ln in (83, 267, 304):  # lines cited by the audit stay put
            self.assertEqual(old[ln - 1], new[ln - 1])

    def test_guard_marker_absent_from_pristine(self):
        self.assertNotIn(MARK, self.pristine)


class TestStaticCheck(_TreeCopy):
    """Lesson 53: every mutant that undoes or weakens the change must FAIL the check."""

    def test_patched_passes(self):
        self.assertEqual(CK.source_violations(self.patched()), [])

    def test_pristine_fails_both(self):
        v = CK.source_violations(self.pristine)
        self.assertEqual(len(v), 2, v)

    def _mutants(self):
        new = self.patched()
        co = "chunk_offset = tl.program_id(1).to(tl.int64)"
        ss = "sequence_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)"
        self.assertEqual(new.count(co), 1)
        self.assertEqual(new.count(ss), 1)
        return {
            "revert_chunk_offset": new.replace(co, "chunk_offset = tl.program_id(1)"),
            "revert_start": new.replace(ss, "sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)"),
            "int32_cast": new.replace(co, "chunk_offset = tl.program_id(1).to(tl.int32)"),
            "cast_in_comment_only": new.replace(co, "chunk_offset = tl.program_id(1)  # .to(tl.int64)"),
            "late_reassign": new.replace(co, co + "\n    chunk_offset = tl.program_id(1)"),
        }

    def test_every_mutant_fails(self):
        for name, text in self._mutants().items():
            with self.subTest(mutant=name):
                self.assertNotEqual(CK.source_violations(text), [], name)

    def test_cli_exit_codes(self):
        self.patched()
        self.assertEqual(CK.main([str(self.target)]), 0)
        self.target.write_text(self.pristine)
        self.assertEqual(CK.main([str(self.target)]), 1)
        self.assertEqual(CK.main([str(self.tmp / "absent.py")]), 2)


def _triton_ok():
    try:
        import triton  # noqa: F401
        from triton.backends.compiler import GPUTarget  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_triton_ok(), "triton not importable -- TTIR check NOT RUN")
class TestTTIR(_TreeCopy):
    """Compile (no launch) for sm_80; count stride multiplies by integer width."""

    def _muls(self, text):
        self.target.write_text(text)
        try:
            return CK.ttir_stride_muls(self.target)
        except CK.InputError as exc:
            self.skipTest(f"TTIR compile unavailable here: {exc}")

    def test_patched_all_i64_pristine_and_halves_not(self):
        new = self.patched()
        self.assertEqual(CK.ttir_violations(self._muls(new)), [])
        self.assertNotEqual(CK.ttir_violations(self._muls(self.pristine)), [])
        only_chunk = self.pristine.replace("chunk_offset = tl.program_id(1)",
                                           "chunk_offset = tl.program_id(1).to(tl.int64)")
        # the audit's one-line proposal leaves the conv_state tail read (seqlen-derived) in int32
        self.assertNotEqual(CK.ttir_violations(self._muls(only_chunk)), [])


UNIT = ENG / "scripts" / "vessl" / "conv1d_int64_unit.py"


@unittest.skipUnless(_triton_ok(), "triton not importable -- interpreter self-test NOT RUN")
class TestUnitHarnessInterpreter(_TreeCopy):
    """The GPU gate (c)-2 harness, self-tested on the CPU with TRITON_INTERPRET=1 at a tiny shape
    (T=37 -> 5 chunks of BLOCK_M=8, non-contiguous view stride 96).  Says nothing about >94.8k: the
    point is that the harness can tell MATCH from MISMATCH (both checks) before it is trusted on GPU."""

    def _run(self, text, *extra):
        f = self.tmp / "k" / "causal_conv1d_triton.py"
        f.parent.mkdir(exist_ok=True)
        f.write_text(text)
        env = dict(os.environ, TRITON_INTERPRET="1")
        r = subprocess.run([sys.executable, "-I", str(UNIT), "--kernel-file", str(f), "--tokens", "37",
                            "--width-inproj", "96", "--dim", "40", "--offset", "50", "--dtype", "fp32",
                            "--device", "cpu", *extra], capture_output=True, text=True, env=env, timeout=600)
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if not lines:
            self.skipTest(f"interpreter run unavailable: {r.stderr[-300:]}")
        return r.returncode, json.loads(lines[-1])

    def test_pristine_and_patched_match_with_identical_bytes(self):
        rc0, d0 = self._run(self.pristine)
        rc1, d1 = self._run(self.patched())
        rc2, d2 = self._run(self.target.read_text(), "--layout", "contig")
        self.assertEqual((rc0, rc1, rc2), (0, 0, 0))
        self.assertEqual(d1["stride_x_token"], 96)
        self.assertEqual(d2["stride_x_token"], 40)
        self.assertEqual(len({d0["out_sha256"], d1["out_sha256"], d2["out_sha256"]}), 1)

    def test_harness_kills_output_and_state_mutants(self):
        new = self.patched()
        out_mut = "prior_tokens = x_base + (token_offset - 1) * stride_x_token"
        st_mut = "idx_tokens_last = (seqlen - state_len) + tl.arange("
        self.assertEqual(new.count(out_mut), 1)
        self.assertEqual(new.count(st_mut), 1)
        # decision 20: the unit exit code judges only the conv STATE; an output mutant is killed by the ORCHESTRATOR's
        # sha comparison with the original kernel's run (+ the descriptive out_ok)
        _rc0, base = self._run(new)
        rc, d = self._run(new.replace(out_mut, out_mut.replace("- 1", "- 2")))
        self.assertEqual((rc, d["out_ok"]), (0, False))
        self.assertNotEqual(d["out_sha256"], base["out_sha256"])
        rc, d = self._run(new.replace(st_mut, "idx_tokens_last = (seqlen - state_len - 1) + tl.arange("))
        self.assertEqual((rc, d["state_ok"]), (1, False))


class TestSyncWiring(unittest.TestCase):
    def setUp(self):
        self.sync = SYNC.read_text(encoding="utf-8")

    def test_guarded_application_after_admission(self):
        self.assertIn("triton_causal_conv1d_int64.patch", self.sync)
        self.assertIn(f'grep -q "{MARK}"', self.sync)
        adm = self.sync.index("admission_probe_scheduler_hook.patch")
        chunk = self.sync.index("chunk_probe_scheduler_hook.patch")
        conv = self.sync.index("triton_causal_conv1d_int64.patch")
        self.assertLess(chunk, adm)
        self.assertLess(adm, conv)
        # the guard marker only appears in ADDED lines of the patch
        body = PATCH.read_text(encoding="utf-8").split("\n--- a/", 1)[1]
        self.assertTrue(all(l.startswith("+") for l in body.splitlines() if MARK in l))

    def test_manifest_entries_appended(self):
        guard = self.sync.index('"${runtime_python}/sglang/srt/multiplex/multigpu_guard.py" \\')
        adm = self.sync.index('"${runtime_python}/sglang/srt/multiplex/admission_probe.py" \\')
        conv = self.sync.index(f'"${{runtime_python}}/{REL}" \\')
        dec = self.sync.index('"${runtime_python}/sglang/srt/multiplex/decode_probe.py" \\')
        end = self.sync.index('> "${manifest_tmp}"')
        self.assertLess(guard, adm)
        self.assertLess(adm, conv)
        self.assertLess(conv, dec)  # 2026-10-20 decision 16 O-2: entry 30 appended after it
        self.assertLess(dec, end)

    def test_no_other_tracked_patch_touches_the_file(self):
        others = [p for p in (ENG / "src" / "patches").glob("*.patch") if p != PATCH]
        others.append(ENG / "env" / "devtree_manual_edits.patch")
        for p in others:
            self.assertNotIn("causal_conv1d_triton.py", p.read_text(encoding="utf-8", errors="replace"), p.name)


if __name__ == "__main__":
    unittest.main()
