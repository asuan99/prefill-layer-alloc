"""Gate for PDMUX_GREEN_READOUT (A1 rev2 절차 2 — the `green` axis' responder).

WHY THIS EXISTS.  `DESIGN_A1_REV2_STICKY_2026-08-25.md` §3.1 registers a
decision channel for the question "did decode actually run on a green context of
D SM?".  Two things are NOT that channel and the design says so explicitly:
`smid_l0_census.driver_readout` (returns the CURRENT context's `primary_sm`, so
it reports `mismatched` unconditionally, and registers itself as descriptive
only) and `realizedprobe` (separate process, after the server is dead -- the
Stage 0 D108 failure mode).  The channel is the stream's OWN green context:
`cuStreamGetGreenCtx` -> `cuGreenCtxGetDevResource(..., SM)`.

These tests pin the properties that channel has to have:

  1. DEFAULT OFF -- without the flag the helper returns None before touching
     ctypes, and the mixin emits nothing;
  2. the read-out is PER STREAM, not per process.  ★This is the 2026-08-21
     audit item R2 lesson made structural: a producer that answers the same way
     for every stream -- either way -- was invisible to that battery's 50
     checks.  `test_a_constant_answer_producer_is_detected` is the mutation
     test: the discriminating assertion must FAIL on a constant-answer double;
  3. the failure contract is PER FIELD -- a driver that raises leaves `error`
     and no `green_sm`, and never kills the boot;
  4. `target_*_sm` comes from the engine's `sm_counts` table and `green_sm`
     from the driver, so the two numbers a consumer compares are NOT derived
     from each other (an identity is not evidence).
"""

import ctypes
import os
import sys
import unittest
from types import SimpleNamespace

# See test_trace_force_prefill.py / test_sticky_partition.py: another test
# installs src/multiplex/profile.py as sys.modules["profile"], shadowing the
# stdlib module and breaking the sglang import chain below.
sys.modules.pop("profile", None)

from sglang.srt.multiplex.green_readout import (
    CUdevResource,
    ENV_FLAG,
    green_sm_readout,
    maybe_read_green_contexts,
    read_stream_groups,
)

# pdmux_e1_d16.yml shape: [(108,0), (92,16), (0,108)].
SM_COUNTS = [(108, 0), (92, 16), (0, 108)]

PREFILL_PTRS = (0x1000, 0x1010, 0x1020)
DECODE_PTRS = (0x2000, 0x2010, 0x2020)


class FakeStream:
    def __init__(self, ptr):
        self.cuda_stream = ptr


def fake_groups():
    return [
        (FakeStream(PREFILL_PTRS[i]), FakeStream(DECODE_PTRS[i]))
        for i in range(len(SM_COUNTS))
    ]


class FakeCuDriver:
    """Known-answer stand-in for `libcuda.so.1`.

    `ctypes.cast(ref, POINTER(...))[0] = value` is the public-API way to write
    through a `byref` out-parameter; the private `._obj` back door is
    deliberately avoided (same convention as `smid_l0_census._FakeCuDriver`).

    `sm_by_ptr` maps a stream pointer to the SM count its green context holds.
    A pointer that is absent has NO green context (handle 0).
    """

    def __init__(self, sm_by_ptr):
        self.sm_by_ptr = dict(sm_by_ptr)
        self._handle_to_sm = {}

    def cuStreamGetGreenCtx(self, ptr, ref):
        value = ptr.value if hasattr(ptr, "value") else ptr
        if value in self.sm_by_ptr:
            handle = 0x9000 + (value & 0xFFF)
            self._handle_to_sm[handle] = self.sm_by_ptr[value]
        else:
            handle = 0
        ctypes.cast(ref, ctypes.POINTER(ctypes.c_void_p))[0] = handle
        return 0

    def cuGreenCtxGetDevResource(self, handle, ref, kind):
        value = handle.value if hasattr(handle, "value") else handle
        sm = self._handle_to_sm.get(value, 0)
        res = ctypes.cast(ref, ctypes.POINTER(CUdevResource))
        payload = (ctypes.c_uint * 3)(sm, 8, 8)
        ctypes.memmove(
            ctypes.addressof(res.contents) + CUdevResource._u.offset,
            ctypes.byref(payload),
            ctypes.sizeof(payload),
        )
        return 0


class RaisingCuDriver:
    def cuStreamGetGreenCtx(self, ptr, ref):
        raise OSError("driver call refused")


def _sm_counts_seen(readout):
    """Every smCount the driver reported, per (index, role)."""
    seen = {}
    for row in readout["groups"]:
        for role in ("prefill", "decode"):
            entry = row[role]
            if isinstance(entry, dict) and entry.get("green_sm"):
                seen[(row["stream_index"], role)] = entry["green_sm"]["smCount"]
    return seen


def _distinguishes_streams(readout):
    """The discriminating property: the producer gave at least two different
    answers.  A producer that answers identically for every stream fails this."""
    values = set(_sm_counts_seen(readout).values())
    return len(values) >= 2


class GreenReadoutDefaultOffTest(unittest.TestCase):
    def test_absent_flag_returns_none(self):
        self.assertIsNone(
            maybe_read_green_contexts(fake_groups(), SM_COUNTS, env={})
        )

    def test_explicit_zero_returns_none(self):
        self.assertIsNone(
            maybe_read_green_contexts(fake_groups(), SM_COUNTS, env={ENV_FLAG: "0"})
        )

    def test_flag_on_produces_one_row_per_stream_group(self):
        out = maybe_read_green_contexts(
            fake_groups(),
            SM_COUNTS,
            env={ENV_FLAG: "1"},
            lib=FakeCuDriver({DECODE_PTRS[1]: 16, PREFILL_PTRS[1]: 92}),
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(out["groups"]), len(SM_COUNTS))
        self.assertEqual(
            [row["stream_index"] for row in out["groups"]], [0, 1, 2]
        )


class GreenReadoutPerStreamTest(unittest.TestCase):
    """Property 2 -- the read-out is per stream, and the check that says so
    must be able to fail."""

    def _readout(self, driver):
        return read_stream_groups(
            fake_groups(), SM_COUNTS, sticky_idx=1, lib=driver
        )

    def test_each_half_of_the_division_reports_its_own_sm_count(self):
        out = self._readout(FakeCuDriver({PREFILL_PTRS[1]: 92, DECODE_PTRS[1]: 16}))
        seen = _sm_counts_seen(out)
        self.assertEqual(seen[(1, "prefill")], 92)
        self.assertEqual(seen[(1, "decode")], 16)
        self.assertTrue(_distinguishes_streams(out))

    def test_a_constant_answer_producer_is_detected(self):
        """★MUTATION TEST.  Every stream green with the SAME size is exactly the
        degenerate instrument audit item R2 caught; the discriminating
        assertion above must be FALSE here, or it is not discriminating."""
        constant = FakeCuDriver(
            {p: 108 for p in PREFILL_PTRS + DECODE_PTRS}
        )
        out = self._readout(constant)
        self.assertFalse(_distinguishes_streams(out))
        self.assertEqual(set(_sm_counts_seen(out).values()), {108})

    def test_plain_group_streams_report_a_null_green_context(self):
        out = self._readout(FakeCuDriver({PREFILL_PTRS[1]: 92, DECODE_PTRS[1]: 16}))
        plain = out["groups"][2]["decode"]
        self.assertTrue(plain["green_ctx_is_null"])
        self.assertNotIn("green_sm", plain)


class GreenReadoutContractTest(unittest.TestCase):
    def test_target_sm_comes_from_the_engine_table_not_the_driver(self):
        """Property 4 -- the driver is allowed to disagree, and the record must
        keep BOTH numbers so the disagreement is visible."""
        out = read_stream_groups(
            fake_groups(),
            SM_COUNTS,
            sticky_idx=1,
            lib=FakeCuDriver({DECODE_PTRS[1]: 44}),  # engine asked for 16
        )
        row = out["groups"][1]
        self.assertEqual(row["target_decode_sm"], 16)
        self.assertEqual(row["decode"]["green_sm"]["smCount"], 44)

    def test_sticky_target_index_is_marked(self):
        out = read_stream_groups(fake_groups(), SM_COUNTS, sticky_idx=1,
                                 lib=FakeCuDriver({}))
        self.assertEqual(
            [row["is_sticky_target"] for row in out["groups"]],
            [False, True, False],
        )
        self.assertEqual(out["sticky_target_index"], 1)

    def test_no_sticky_target_marks_nothing(self):
        out = read_stream_groups(fake_groups(), SM_COUNTS, sticky_idx=None,
                                 lib=FakeCuDriver({}))
        self.assertEqual(
            [row["is_sticky_target"] for row in out["groups"]],
            [False, False, False],
        )

    def test_a_raising_driver_leaves_an_error_and_no_answer(self):
        """Property 3 -- per-field failure, and nothing escapes."""
        out = read_stream_groups(fake_groups(), SM_COUNTS, lib=RaisingCuDriver())
        entry = out["groups"][1]["decode"]
        self.assertIn("error", entry)
        self.assertNotIn("green_sm", entry)
        self.assertNotIn("green_ctx_is_null", entry)

    def test_a_stream_without_a_handle_is_an_error_entry(self):
        groups = [(SimpleNamespace(), SimpleNamespace())]
        out = read_stream_groups(groups, [(108, 0)], lib=FakeCuDriver({}))
        self.assertIn("error", out["groups"][0]["decode"])

    def test_nsys_is_not_consulted(self):
        """The axis is registered as non-circular with Q2ae; a reference to nsys
        anywhere in the producer would break that."""
        import sglang.srt.multiplex.green_readout as module
        with open(module.__file__) as handle:
            source = handle.read()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        self.assertNotIn("nsys", code.split('"""')[-1])


class SmTripletTest(unittest.TestCase):
    def test_union_payload_is_parsed_as_three_uints(self):
        res = CUdevResource()
        payload = (ctypes.c_uint * 3)(16, 8, 8)
        ctypes.memmove(
            ctypes.addressof(res) + CUdevResource._u.offset,
            ctypes.byref(payload),
            ctypes.sizeof(payload),
        )
        self.assertEqual(
            res.sm_triplet(),
            {"smCount": 16, "minSmPartitionSize": 8, "smCoscheduledAlignment": 8},
        )

    def test_layout_matches_the_census_struct(self):
        """The layout is copied from smid_l0_census; if either drifts, the
        payload is read at the wrong offset and every smCount is garbage."""
        self.assertEqual(ctypes.sizeof(CUdevResource), 4 + 92 + 48)
        self.assertEqual(CUdevResource._u.offset, 96)


class GreenReadoutDriverLoadTest(unittest.TestCase):
    def test_missing_driver_is_reported_not_raised(self):
        saved = os.environ.get("LD_LIBRARY_PATH")
        try:
            out = green_sm_readout(0x1234, lib=RaisingCuDriver())
            self.assertIn("error", out)
            self.assertEqual(
                out["channel"], "cuStreamGetGreenCtx + cuGreenCtxGetDevResource"
            )
        finally:
            if saved is not None:
                os.environ["LD_LIBRARY_PATH"] = saved


if __name__ == "__main__":
    unittest.main()
