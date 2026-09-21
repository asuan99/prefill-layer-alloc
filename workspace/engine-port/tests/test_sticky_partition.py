"""Gate for PDMUX_STICKY_PARTITION (hold the decode division while decode is busy).

WHY THIS EXISTS.  `initialize_stream_groups` hard-codes
`SM_COUNTS = [(total,0)] + divisions + [(0,total)]`, and
`adjust_stream_groups` falls back to that trailing PLAIN group whenever the
decode batch is busy but no prefill batch is in flight.  In job 872077 the
cell's decode division was therefore realized over only 4-19% of
decode-active time (T8 d16 0.038 -> d54 0.093, Ha8 0.104 -> 0.187); the rest
ran unpartitioned at 108 SM.  The cell label was a TARGET, not an
allocation, and the dilution factor differed per cell.

These tests pin the three properties the flag has to have:

  1. default OFF, and OFF is behaviourally identical to the pre-patch
     selector -- asserted against an independent re-implementation of the
     pre-patch branch over the full (decode_bs x prefill-in-flight x config)
     grid, so past campaigns stay reproducible;
  2. ON holds the target division on EVERY decode-busy state, including the
     prefill-idle state that is the whole point, and releases to the
     full-SM-prefill group only when the decode batch is empty;
  3. combinations whose partition index is owned by another mechanism are
     REJECTED at init rather than silently given some behaviour.

The mixin is loaded from the installed runtime tree when SGLANG_ENGINE_DEV is
set (same convention as test_dual_worker.py / test_trace_force_prefill.py), so
this also checks that sync_engine_tree.sh actually installed the patch.
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# See test_trace_force_prefill.py: test_profile_controller.py installs
# src/multiplex/profile.py as sys.modules["profile"], which shadows the stdlib
# module and breaks the sglang import chain below.
sys.modules.pop("profile", None)

from sglang.srt.multiplex import pdmux_context
from sglang.srt.multiplex.controller import FixedPolicy, GenericDynamicPolicy
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin

# [(108,0), (92,16), (0,108)] -- one division, the pdmux_e1_d16.yml shape.
SM_COUNTS_1DIV = [(108, 0), (92, 16), (0, 108)]
DIVISIONS_1DIV = [[92, 16, 0]]

# [(108,0), (54,54), (64,44), (0,108)] -- the pdmux_e1_d54.yml shape, whose
# second row is a guard-satisfier that the decode-batch-size selector WILL
# drift onto once decode_bs >= 48.
SM_COUNTS_2DIV = [(108, 0), (54, 54), (64, 44), (0, 108)]
DIVISIONS_2DIV = [[54, 54, 0], [64, 44, 48]]


class FakeBatch:
    def __init__(self, size):
        self.reqs = [SimpleNamespace(rid=f"r{i}") for i in range(size)]

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs


class FakeScheduler(SchedulerMultiplexMixin):
    """Minimal stand-in exposing only what `adjust_stream_groups` touches."""

    def __init__(self, sm_counts, manual_divisions, sticky=False, fixed_idx=None):
        self.sm_counts = list(sm_counts)
        self.stream_groups = [(f"p{i}", f"d{i}") for i in range(len(sm_counts))]
        self.real_sm_group_num = len(sm_counts)
        self.pdmux_config = SimpleNamespace(
            manual_divisions=[list(row) for row in manual_divisions],
            decode_bs_divisor=36,
        )
        self.running_batch = FakeBatch(0)
        self.split_prefill_batch = None
        self.sticky_partition_enabled = sticky
        self._sticky_fixed_idx = fixed_idx
        self.attn_backend_updates = []
        self.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                update_decode_attn_backend=self.attn_backend_updates.append
            )
        )


def legacy_expected_idx(sched):
    """Independent re-implementation of the PRE-PATCH selector.

    Deliberately written from the pre-patch source rather than by calling the
    patched method, so `test_off_matches_pre_patch_selector` is a real
    comparison and not a tautology.
    """
    if not sched.running_batch.is_empty() and sched.split_prefill_batch:
        decode_bs = sched.running_batch.batch_size()
        manual_divisions = sched.pdmux_config.manual_divisions
        if manual_divisions:
            stream_idx = None
            for i in range(len(manual_divisions)):
                _, _, threshold = manual_divisions[i]
                if decode_bs >= threshold:
                    stream_idx = i + 1
        else:
            stream_idx = max(
                1,
                min(
                    sched.real_sm_group_num - 2,
                    decode_bs
                    * (sched.real_sm_group_num - 2)
                    // sched.pdmux_config.decode_bs_divisor,
                ),
            )
        return stream_idx
    if not sched.running_batch.is_empty():
        return sched.real_sm_group_num - 1
    return 0


class StickyPartitionSelectorTest(unittest.TestCase):
    """(1) and (2): the selector itself, driven over the full state grid."""

    def setUp(self):
        self._saved = (pdmux_context.STREAM_GROUPS, pdmux_context.CURRENT_STREAM_IDX,
                       pdmux_context.CURRENT_STREAM_GROUP)

        def _restore():
            (pdmux_context.STREAM_GROUPS, pdmux_context.CURRENT_STREAM_IDX,
             pdmux_context.CURRENT_STREAM_GROUP) = self._saved

        self.addCleanup(_restore)

    def _install(self, sched):
        pdmux_context.STREAM_GROUPS = sched.stream_groups
        pdmux_context.CURRENT_STREAM_IDX = 0
        pdmux_context.CURRENT_STREAM_GROUP = sched.stream_groups[0]

    def _run(self, sched, decode_bs, prefill):
        sched.running_batch = FakeBatch(decode_bs)
        sched.split_prefill_batch = FakeBatch(2) if prefill else None
        self._install(sched)
        idx, group = sched.adjust_stream_groups()
        # the returned index must be what the module global now says, and the
        # attention backend must have been pointed at the same index -- the
        # telemetry writer reads the same global, so a mismatch here would be
        # exactly the "label != what ran" failure this flag exists to remove.
        self.assertEqual(idx, pdmux_context.get_current_stream_idx())
        self.assertIs(group, sched.stream_groups[idx])
        self.assertEqual(sched.attn_backend_updates[-1], idx)
        return idx

    def test_off_matches_pre_patch_selector(self):
        for sm_counts, divisions in (
            (SM_COUNTS_1DIV, DIVISIONS_1DIV),
            (SM_COUNTS_2DIV, DIVISIONS_2DIV),
            (SM_COUNTS_2DIV, []),  # no manual divisions -> decode_bs_divisor path
        ):
            for decode_bs in (0, 1, 4, 47, 48, 96):
                for prefill in (False, True):
                    with self.subTest(
                        n=len(sm_counts), div=len(divisions), bs=decode_bs, pf=prefill
                    ):
                        sched = FakeScheduler(sm_counts, divisions, sticky=False)
                        got = self._run(sched, decode_bs, prefill)
                        self.assertEqual(got, legacy_expected_idx(sched))

    def test_off_still_reverts_to_the_unpartitioned_group(self):
        """The behaviour the flag exists to change must still be there when OFF."""
        sched = FakeScheduler(SM_COUNTS_1DIV, DIVISIONS_1DIV, sticky=False)
        self.assertEqual(self._run(sched, decode_bs=4, prefill=False), 2)
        self.assertEqual(sched.sm_counts[2], (0, 108))

    def test_on_holds_the_division_while_prefill_is_idle(self):
        sched = FakeScheduler(SM_COUNTS_1DIV, DIVISIONS_1DIV, sticky=True)
        for decode_bs in (1, 4, 47, 48, 96):
            with self.subTest(bs=decode_bs):
                self.assertEqual(self._run(sched, decode_bs, prefill=False), 1)
                self.assertEqual(sched.sm_counts[1], (92, 16))

    def test_on_leaves_the_prefill_in_flight_state_unchanged(self):
        for sm_counts, divisions in (
            (SM_COUNTS_1DIV, DIVISIONS_1DIV),
            (SM_COUNTS_2DIV, DIVISIONS_2DIV),
            (SM_COUNTS_2DIV, []),
        ):
            for decode_bs in (1, 4, 47, 48, 96):
                with self.subTest(div=len(divisions), bs=decode_bs):
                    off = FakeScheduler(sm_counts, divisions, sticky=False)
                    on = FakeScheduler(sm_counts, divisions, sticky=True)
                    self.assertEqual(
                        self._run(on, decode_bs, prefill=True),
                        self._run(off, decode_bs, prefill=True),
                    )

    def test_on_releases_to_full_sm_prefill_when_decode_is_empty(self):
        """Documented decode-empty choice: release, do not hold."""
        sched = FakeScheduler(SM_COUNTS_1DIV, DIVISIONS_1DIV, sticky=True)
        for prefill in (False, True):
            with self.subTest(pf=prefill):
                self.assertEqual(self._run(sched, decode_bs=0, prefill=prefill), 0)
                self.assertEqual(sched.sm_counts[0], (108, 0))

    def test_on_with_fixed_target_ignores_the_guard_satisfier_row(self):
        """A config with a guard row must not drift onto it under sticky.

        pdmux_e1_d54.yml carries [64,44,48] purely to satisfy the
        `_build_r2_policy` D16/24/34/44 guard.  The decode-batch-size selector
        would take it at decode_bs >= 48, which is a DIFFERENT partition from
        the cell label -- the same class of error the flag exists to remove.
        """
        with_fixed = FakeScheduler(
            SM_COUNTS_2DIV, DIVISIONS_2DIV, sticky=True, fixed_idx=1
        )
        without_fixed = FakeScheduler(SM_COUNTS_2DIV, DIVISIONS_2DIV, sticky=True)
        for prefill in (False, True):
            self.assertEqual(self._run(with_fixed, 48, prefill), 1)
            self.assertEqual(self._run(without_fixed, 48, prefill), 2)
        # below the guard threshold both agree
        self.assertEqual(self._run(with_fixed, 4, False), 1)
        self.assertEqual(self._run(without_fixed, 4, False), 1)


class StickyPartitionInitTest(unittest.TestCase):
    """(1) default-OFF and (3) explicit rejection of undefined combinations."""

    ENV_KEYS = (
        "PDMUX_STICKY_PARTITION",
        "PDMUX_LA_COORD",
        "PDMUX_SLO_SCHED",
        "PDMUX_FIXED_DECODE_SM_FILE",
    )

    def setUp(self):
        saved = {k: os.environ.get(k) for k in self.ENV_KEYS}

        def _restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(_restore)
        for k in self.ENV_KEYS:
            os.environ.pop(k, None)

    @staticmethod
    def _fake(sm_counts=SM_COUNTS_1DIV, policy_name="", policy=None):
        return SimpleNamespace(
            sm_counts=list(sm_counts),
            real_sm_group_num=len(sm_counts),
            r2_policy_name=policy_name,
            r2_policy=policy,
        )

    def _init(self, obj):
        SchedulerMultiplexMixin._init_sticky_partition(obj)
        return obj

    def test_runtime_default_is_off(self):
        self.assertNotIn("PDMUX_STICKY_PARTITION", os.environ)
        obj = self._init(self._fake())
        self.assertFalse(obj.sticky_partition_enabled)
        self.assertIsNone(obj._sticky_fixed_idx)
        # and the installed runtime must literally default the env var to "0"
        src = Path(
            os.environ.get("SGLANG_ENGINE_DEV")
            or os.path.join(
                os.environ.get("PDMUX_ROOT", str(Path(__file__).resolve().parents[4])),
                "sglang_engine_dev", "python",
            )
        ) / "sglang/srt/multiplex/multiplexing_mixin.py"
        self.assertIn('"PDMUX_STICKY_PARTITION", "0"', src.read_text())

    def test_enabled_without_policy_uses_the_engine_selector(self):
        os.environ["PDMUX_STICKY_PARTITION"] = "1"
        obj = self._init(self._fake())
        self.assertTrue(obj.sticky_partition_enabled)
        self.assertIsNone(obj._sticky_fixed_idx)

    def test_enabled_with_fixed_policy_resolves_the_division_index(self):
        os.environ["PDMUX_STICKY_PARTITION"] = "1"
        obj = self._init(
            self._fake(SM_COUNTS_2DIV, "fixed", FixedPolicy(54))
        )
        self.assertEqual(obj._sticky_fixed_idx, 1)
        obj = self._init(
            self._fake(SM_COUNTS_2DIV, "fixed", FixedPolicy(44))
        )
        self.assertEqual(obj._sticky_fixed_idx, 2)

    def test_fixed_target_matching_only_a_plain_group_is_rejected(self):
        """decode_sms=108 matches the trailing PLAIN group, which is not a division."""
        os.environ["PDMUX_STICKY_PARTITION"] = "1"
        with self.assertRaises(RuntimeError):
            self._init(self._fake(SM_COUNTS_1DIV, "fixed", FixedPolicy(108)))
        with self.assertRaises(RuntimeError):
            self._init(self._fake(SM_COUNTS_1DIV, "fixed", FixedPolicy(34)))

    def test_undefined_combinations_are_rejected(self):
        os.environ["PDMUX_STICKY_PARTITION"] = "1"
        for key in ("PDMUX_LA_COORD", "PDMUX_SLO_SCHED", "PDMUX_FIXED_DECODE_SM_FILE"):
            with self.subTest(env=key):
                os.environ[key] = "1"
                try:
                    with self.assertRaises(RuntimeError):
                        self._init(self._fake())
                finally:
                    os.environ.pop(key)
        with self.subTest(env="PDMUX_R2_POLICY=generic"):
            with self.assertRaises(RuntimeError):
                self._init(
                    self._fake(
                        SM_COUNTS_1DIV,
                        "generic",
                        GenericDynamicPolicy((16, 24, 34, 44)),
                    )
                )
        with self.subTest(env="no division"):
            with self.assertRaises(RuntimeError):
                self._init(self._fake([(108, 0), (0, 108)]))

    def test_rejections_do_not_fire_when_the_flag_is_off(self):
        """OFF must stay OFF-compatible with every other mode."""
        for key in ("PDMUX_LA_COORD", "PDMUX_SLO_SCHED", "PDMUX_FIXED_DECODE_SM_FILE"):
            os.environ[key] = "1"
        obj = self._init(
            self._fake(SM_COUNTS_1DIV, "generic", GenericDynamicPolicy((16, 24, 34, 44)))
        )
        self.assertFalse(obj.sticky_partition_enabled)


if __name__ == "__main__":
    unittest.main()
