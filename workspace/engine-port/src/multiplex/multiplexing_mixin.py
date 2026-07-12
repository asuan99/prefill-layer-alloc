"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

import os

from sglang.srt.distributed.parallel_state import set_pdmux_status
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
    set_current_stream_idx,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:

    def init_pdmux(self: Scheduler):
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(self.server_args.pdmux_config_path)
        initialize_stream_groups(self.gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self: Scheduler,
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if (
            os.environ.get("PDMUX_SLO_SCHED")
            and not self.running_batch.is_empty()
            and self.split_prefill_batch
        ):
            # diagnostic pin: fix the split idx (SLO code path, no dynamic adjustment) to isolate
            # code-path overhead from the controller's dynamic switching.
            _pin = os.environ.get("PDMUX_SLO_PIN")
            if _pin:
                _pidx = int(_pin)
                set_current_stream_idx(_pidx)
                self.tp_worker.model_runner.update_decode_attn_backend(_pidx)
                return _pidx, self.stream_groups[_pidx]
            # SLO-aware step-level split (step A): shift the prefill<->decode boundary toward
            # whichever SLO is more at risk. TPOT signal = measured decode-iteration EMA (one
            # token/iter, set in event_loop_pdmux); TTFT signal = prefill backlog depth. Pure SLO
            # feedback, no layer-type -> (D)-safe (adjusts only the step split, at prefill bounds).
            # Assumes config divisions ordered by INCREASING decode SM (idx+1 = more decode).
            # v2: TPOT-band targeting with deadband/hysteresis (avoid v1 bang-bang overshoot &
            # oscillation). Keep decode TPOT under SLO with margin; hand SM to prefill only when
            # decode has real slack AND prefill is backlogged. Start decode-safe, relax toward prefill.
            _tpot_slo = float(os.environ.get("PDMUX_TPOT_SLO_MS", "60"))
            _qtarget = float(os.environ.get("PDMUX_QDEPTH_TARGET", "4"))
            _hi_frac = float(os.environ.get("PDMUX_TPOT_HI", "0.85"))
            _lo_frac = float(os.environ.get("PDMUX_TPOT_LO", "0.65"))
            _lo, _hi = 1, self.real_sm_group_num - 2
            _idx = getattr(self, "_slo_idx", (_lo + _hi) // 2)  # v6: start NEUTRAL (1 step from either
            # regime optimum) not decode-heavy — a slow descent from _hi starved prefill (transient backlog).
            _tpot = getattr(self, "_slo_tpot_ema", 0.0)
            _qd = len(self.waiting_queue)
            _dwell = getattr(self, "_slo_dwell", 0)
            if _dwell > 0:
                self._slo_dwell = _dwell - 1     # v3: min dwell after a switch (anti-oscillation)
                _new = _idx
            elif _tpot > _hi_frac * _tpot_slo:
                _new = min(_hi, _idx + 1)        # TPOT tight -> more decode SM
            elif _tpot < _lo_frac * _tpot_slo and _qd > _qtarget:  # v5: TPOT has real margin & prefill backlog
                _new = max(_lo, _idx - 1)        # -> more prefill SM (stops the descent before decode gets tight)
            else:
                _new = _idx                      # no pressure -> hold
            if _new != _idx:
                self._slo_dwell = int(os.environ.get("PDMUX_SLO_DWELL", "3"))
                self._slo_skip = 2               # skip switch-drain iters in the TPOT EMA
                logger.info(
                    "SLO-SCHED %d->%d decode_sm=%d tpot=%.1fms qd=%d",
                    _idx, _new, self.sm_counts[_new][1], _tpot, _qd,
                )
            _idx = _new
            self._slo_idx = _idx
            set_current_stream_idx(_idx)
            self.tp_worker.model_runner.update_decode_attn_backend(_idx)
            return _idx, self.stream_groups[_idx]
        if not self.running_batch.is_empty() and self.split_prefill_batch:
            decode_bs = self.running_batch.batch_size()
            manual_divisions = self.pdmux_config.manual_divisions
            if manual_divisions:
                for i in range(len(manual_divisions)):
                    _, _, threshold = manual_divisions[i]
                    if decode_bs >= threshold:
                        stream_idx = i + 1
            else:
                stream_idx = max(
                    1,
                    min(
                        self.real_sm_group_num - 2,
                        decode_bs
                        * (self.real_sm_group_num - 2)
                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)
        elif not self.running_batch.is_empty():
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(self: Scheduler, sm_count: int) -> bool:
        if self.split_prefill_batch:
            return False

        # add new request
        batch = self.get_new_batch_prefill()
        if batch and not batch.is_empty():
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            self.split_prefill_batch = batch
            return True
        return False

    @torch.inference_mode()
    def event_loop_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")

        import time as _time
        _slo_on = bool(os.environ.get("PDMUX_SLO_SCHED"))
        self._slo_last_t = _time.perf_counter()
        while True:
            if _slo_on:
                # measure per-iteration wall time = TPOT (one token/iter when decode active);
                # v3: skip the iterations right after a partition switch — the switch drains the
                # GPU (2x synchronize) and that drain would otherwise be misread as a huge TPOT
                # spike, which drove the v1/v2 oscillation.
                _now = _time.perf_counter()
                _dt = (_now - self._slo_last_t) * 1000.0
                self._slo_last_t = _now
                _ema = getattr(self, "_slo_tpot_ema", 0.0)
                # v4: reject outlier spikes. The adjust block does 2x synchronize (drain) at
                # EVERY prefill boundary (frequent in prefill-bound), which shows up as a ~200ms+
                # spike vs the true ~40ms TPOT. Accept _dt only within a band around the running
                # EMA, else the signal is poisoned and the controller oscillates.
                _cap = (3.0 * _ema) if _ema else 1000.0
                if (not self.running_batch.is_empty()) and 0.0 < _dt < max(_cap, 90.0):
                    self._slo_tpot_ema = (0.7 * _ema + 0.3 * _dt) if _ema else _dt
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if self.running_batch and not self.running_batch.is_empty():
                    decode_result = self.run_batch(self.running_batch)
                    decode_done = True
                else:
                    decode_done = False
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(self.running_batch, decode_result)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True

    @torch.inference_mode()
    def event_loop_pdmux_coord(self: Scheduler):
        """Coordinated per-type layer-aware event loop (R0c/A, PDMUX_LA_COORD=1).

        Same as event_loop_pdmux, except when BOTH a decode step and a concurrent
        prefill are active, the decode step is run in layer-TYPE windows: attn windows
        run on a decode-HEAVY coordinated pair (protect the SM-sensitive no-GQA attn),
        mamba windows on a decode-LIGHT pair (release SM), and prefill advances a share
        on the *complementary* prefill-half of the SAME pair per window. So releasing an
        insensitive layer's SM hands prefill the exact complement, coordinated (no overlap).
        """
        from sglang.srt.managers.utils import GenerationBatchResult

        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        # persist across iterations (like event_loop_pdmux): when wait_prefill_kernel_done
        # is set but the allreduce flag wasn't ready, the next iteration's completion step
        # reuses the prefill_result/prefill_exe_done from when the prefill actually ran.
        decode_result = None
        prefill_result = None
        prefill_exe_done = None
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # LIGHT = min decode-SM middle group (mamba windows), HEAVY = max (attn windows)
        _mids = list(range(1, self.real_sm_group_num - 1))
        LIGHT = min(_mids, key=lambda i: self.sm_counts[i][1])
        HEAVY = max(_mids, key=lambda i: self.sm_counts[i][1])
        logger.info(
            f"PD-Multiplexing COORD per-type la: LIGHT idx{LIGHT} {self.sm_counts[LIGHT]} "
            f"(mamba), HEAVY idx{HEAVY} {self.sm_counts[HEAVY]} (attn)"
        )
        MR = self.tp_worker.model_runner
        # (a) substrate-isolation variant: PDMUX_LA_COORD_OPT removes the two known
        # implementation overheads — per-window CPU synchronize (-> GPU-side wait_stream/
        # events) and decode SM-pinning (decode-only windows after prefill -> full-SM
        # normal stream). Leaves the *structural* cost (prefill overlaps window 0 only).
        coord_opt = bool(os.environ.get("PDMUX_LA_COORD_OPT"))
        # version-4: faithful sim mechanism — prefill SLICED across mamba (release) windows
        # via lightweight MR.forward (no run_batch), overlapping decode in EVERY release
        # window (not just window 0). Directly attacks the single-window-overlap killer.
        coord_v4 = bool(os.environ.get("PDMUX_LA_COORD_V4"))
        # prefill-type-aware boundary: shift the prefill<->decode SM split by the PREFILL's
        # current layer type (not decode's, as v4/R0d did). attn-prefill (more SM-sensitive)
        # -> prefill-heavy partition (protect prefill); mamba-prefill -> decode-heavy (release
        # prefill SM to decode). Prefill chunk is windowed by its type; decode is sliced across.
        coord_pf = bool(os.environ.get("PDMUX_LA_COORD_PF"))
        FULL = self.real_sm_group_num - 1  # normal full-SM decode stream (0,108)

        while True:
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False

            decode_active = bool(self.running_batch) and not self.running_batch.is_empty()
            prefill_active = (
                self.split_prefill_batch is not None
                and not self.split_prefill_batch.is_empty()
                and not wait_prefill_kernel_done
            )

            if decode_active and prefill_active and coord_pf:
                # ===== prefill-type-aware boundary shift =====
                # Window the PREFILL chunk by prefill-layer-type; the (P,D) split follows
                # PREFILL's type: attn-prefill -> LIGHT pair (prefill 92 / decode 16, protect the
                # more-sensitive prefill), mamba-prefill -> HEAVY pair (prefill 34 / decode 74,
                # release prefill SM to decode). Decode (whole 54-layer step) is sliced across the
                # prefill windows on the complementary decode-half. GPU-ordered.
                num_layers = self.model_config.num_hidden_layers
                decode_mwb = self.running_batch.get_model_worker_batch()
                decode_fb = ForwardBatch.init_new(decode_mwb, MR)
                MR.init_decode_metadata_coord(decode_fb, HEAVY)
                if getattr(self.split_prefill_batch, "split_forward_batch", None) is None:
                    _mwb = self.split_prefill_batch.get_model_worker_batch()
                    self.split_prefill_batch.split_forward_batch = ForwardBatch.init_new(_mwb, MR)
                    self.split_prefill_batch.seq_lens_cpu_cache = _mwb.seq_lens_cpu
                pfb = self.split_prefill_batch.split_forward_batch
                ext = self.split_prefill_batch.extend_num_tokens
                fwd_total = (
                    max(1, self.pdmux_config.split_forward_token_budget // ext)
                    if ext > 0 else num_layers
                )
                _pf_chunk = os.environ.get("PDMUX_PF_CHUNK")  # fix #2: cap prefill layers/step
                if _pf_chunk:                                 # (match prefill workload to its low SM)
                    fwd_total = max(1, min(fwd_total, int(_pf_chunk)))
                p_start = self.split_prefill_batch.split_index
                k_total = min(fwd_total, num_layers - p_start)
                # prefill windows (by prefill-layer-type) over the chunk [p_start, p_start+k_total)
                pf_windows = [
                    (max(s, p_start), min(e, p_start + k_total), a)
                    for (s, e, a) in MR.model.la_coord_windows()
                    if e > p_start and s < p_start + k_total
                ]
                n_pw = max(1, len(pf_windows))
                # fix #1: distribute decode ∝ each window's decode-SM (more decode work in the
                # high-decode-SM mamba windows, less in the low-SM attn windows) — match decode
                # workload to its SM instead of the flawed even split.
                win_dsm = [
                    (self.sm_counts[LIGHT][1] if a else self.sm_counts[HEAVY][1])
                    for (_s, _e, a) in pf_windows
                ]
                total_dsm = sum(win_dsm) or 1
                cum_dsm = 0
                logits_output = None
                plog = None
                prefill_done = True
                prev_d = prev_p = None
                d_done = 0
                for wi, (ps, pe, is_attn) in enumerate(pf_windows):
                    idx = LIGHT if is_attn else HEAVY  # prefill-attn->prefill-heavy(92); mamba->decode-heavy(74)
                    p_s, d_s = self.stream_groups[idx][0], self.stream_groups[idx][1]
                    with torch.cuda.stream(p_s):
                        set_pdmux_status(True)
                        if prev_d is not None:
                            p_s.wait_stream(prev_d)
                        if prev_p is not None:
                            p_s.wait_stream(prev_p)
                        _pout = MR.forward(pfb, split_forward_count=(pe - ps))
                        if _pout.logits_output is not None:
                            plog = _pout.logits_output
                    cum_dsm += win_dsm[wi]
                    d_target = num_layers if wi == n_pw - 1 else min(
                        num_layers, round(num_layers * cum_dsm / total_dsm)
                    )
                    if d_target > d_done:
                        with torch.cuda.stream(d_s):
                            set_pdmux_status(False)
                            if prev_d is not None:
                                d_s.wait_stream(prev_d)
                            if prev_p is not None:
                                d_s.wait_stream(prev_p)
                            r = MR.forward_split_decode(decode_fb, (d_done, d_target))
                            if r is not None:
                                logits_output = r
                        d_done = d_target
                    prev_d, prev_p = d_s, p_s
                if d_done < num_layers:  # safety: finish decode
                    d_s = self.stream_groups[HEAVY][1]
                    with torch.cuda.stream(d_s):
                        set_pdmux_status(False)
                        if prev_d is not None:
                            d_s.wait_stream(prev_d)
                        if prev_p is not None:
                            d_s.wait_stream(prev_p)
                        r = MR.forward_split_decode(decode_fb, (d_done, num_layers))
                        if r is not None:
                            logits_output = r
                    prev_d = d_s
                if prev_d is not None:
                    prev_d.synchronize()
                if prev_p is not None:
                    prev_p.synchronize()
                next_ids = MR.sample(logits_output, decode_fb)
                self.running_batch.output_ids = next_ids
                decode_result = GenerationBatchResult(logits_output=logits_output)
                decode_result.next_token_ids = next_ids
                decode_done = True
                self.split_prefill_batch.split_index = pfb.split_index
                if pfb.split_index >= num_layers:
                    self.split_prefill_batch.split_prefill_finished = True
                    prefill_exe_done = (prev_p or prev_d).record_event()
                    pf_next = MR.sample(plog, pfb) if plog is not None else None
                    prefill_result = GenerationBatchResult(logits_output=plog)
                    prefill_result.next_token_ids = pf_next
                    self.split_prefill_batch.output_ids = pf_next
                stream_idx = HEAVY
                set_current_stream_idx(HEAVY)
                stream_group = self.stream_groups[HEAVY]
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
            elif decode_active and prefill_active and coord_v4:
                # ===== version-4: sim's multi-window overlap, done cheaply =====
                # Prefill is SLICED across the mamba (release) windows via a lightweight
                # MR.forward on a PERSISTENT split_forward_batch (NO run_batch, NO per-window
                # get_model_worker_batch -> avoids inefficient_v1's 698ms). In each mamba window
                # decode runs the mamba layers on the decode-half (16 SM) while prefill advances a
                # slice on the disjoint prefill-half (92 SM); attn windows protect decode (74 SM)
                # and pause prefill. Windows serialize on GPU (consecutive green-ctx partitions
                # share physical SMs); within a window decode-slice ∥ prefill-slice = sim's vision.
                num_layers = self.model_config.num_hidden_layers
                windows = MR.model.la_coord_windows()
                decode_mwb = self.running_batch.get_model_worker_batch()
                decode_fb = ForwardBatch.init_new(decode_mwb, MR)
                MR.init_decode_metadata_coord(decode_fb, HEAVY)
                # persistent prefill forward batch (threads hidden state across windows & steps)
                if getattr(self.split_prefill_batch, "split_forward_batch", None) is None:
                    _mwb = self.split_prefill_batch.get_model_worker_batch()
                    self.split_prefill_batch.split_forward_batch = ForwardBatch.init_new(_mwb, MR)
                    self.split_prefill_batch.seq_lens_cpu_cache = _mwb.seq_lens_cpu
                pfb = self.split_prefill_batch.split_forward_batch
                ext = self.split_prefill_batch.extend_num_tokens
                fwd_total = (
                    max(1, self.pdmux_config.split_forward_token_budget // ext)
                    if ext > 0 else num_layers
                )
                remaining_pf = min(fwd_total, num_layers - self.split_prefill_batch.split_index)
                mamba_wins = sum(1 for (_s, _e, _a) in windows if not _a) or 1
                per_win = -(-remaining_pf // mamba_wins)   # ceil divide over release windows
                logits_output = None
                plog = None
                prefill_done = True
                prev_d = prev_p = None
                for wi, (s, e, is_attn) in enumerate(windows):
                    idx = HEAVY if is_attn else LIGHT
                    set_current_stream_idx(idx)
                    p_s, d_s = self.stream_groups[idx][0], self.stream_groups[idx][1]
                    with torch.cuda.stream(d_s):
                        set_pdmux_status(False)
                        if prev_d is not None:
                            d_s.wait_stream(prev_d)     # serialize windows (shared physical SMs)
                        if prev_p is not None:
                            d_s.wait_stream(prev_p)
                        r = MR.forward_split_decode(decode_fb, (s, e))
                        if r is not None:
                            logits_output = r
                    k = min(per_win, remaining_pf) if (not is_attn and remaining_pf > 0) else 0
                    if k > 0:
                        with torch.cuda.stream(p_s):
                            set_pdmux_status(True)
                            if prev_d is not None:
                                p_s.wait_stream(prev_d)
                            if prev_p is not None:
                                p_s.wait_stream(prev_p)
                            _pout = MR.forward(pfb, split_forward_count=k)
                            if _pout.logits_output is not None:
                                plog = _pout.logits_output
                        remaining_pf -= k
                        prev_p = p_s
                    prev_d = d_s
                if prev_d is not None:
                    prev_d.synchronize()
                if prev_p is not None:
                    prev_p.synchronize()
                # sample decode (after final window)
                next_ids = MR.sample(logits_output, decode_fb)
                self.running_batch.output_ids = next_ids
                decode_result = GenerationBatchResult(logits_output=logits_output)
                decode_result.next_token_ids = next_ids
                decode_done = True
                # prefill completion bookkeeping (mirror baseline; steps 7-8 consume these)
                self.split_prefill_batch.split_index = pfb.split_index
                if pfb.split_index >= num_layers:
                    self.split_prefill_batch.split_prefill_finished = True
                    prefill_exe_done = (prev_p or prev_d).record_event()
                    pf_next = MR.sample(plog, pfb) if plog is not None else None
                    prefill_result = GenerationBatchResult(logits_output=plog)
                    prefill_result.next_token_ids = pf_next
                    # run_batch normally sets batch.output_ids (merge_batch cats it into the
                    # running_batch); we bypass run_batch for prefill, so set it here — exactly
                    # like the decode side sets self.running_batch.output_ids above.
                    self.split_prefill_batch.output_ids = pf_next
                # settle stream refs on HEAVY pair for steps 7-8
                stream_idx = HEAVY
                set_current_stream_idx(HEAVY)
                stream_group = self.stream_groups[HEAVY]
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
            elif decode_active and prefill_active:
                # ===== coordinated per-type windowed interleave =====
                num_layers = self.model_config.num_hidden_layers
                windows = MR.model.la_coord_windows()
                n_win = len(windows)
                decode_mwb = self.running_batch.get_model_worker_batch()
                decode_fb = ForwardBatch.init_new(decode_mwb, MR)
                MR.init_decode_metadata_coord(decode_fb, HEAVY)  # attn windows share heavy backend

                ext = self.split_prefill_batch.extend_num_tokens
                fwd_total = (
                    max(1, self.pdmux_config.split_forward_token_budget // ext)
                    if ext > 0
                    else num_layers
                )
                logits_output = None
                prefill_done = True
                full_d = self.stream_groups[FULL][1]   # normal full-SM decode stream
                prev_d = None                          # GPU-side window ordering (opt)
                prefill_evt = None
                for wi, (s, e, is_attn) in enumerate(windows):
                    if coord_opt and wi > 0:
                        # pin removal: prefill already ran in window 0, so decode-only
                        # windows run on the full-SM (unmasked) stream instead of the
                        # LIGHT/HEAVY sub-partition. attn backend (HEAVY) is stream-agnostic.
                        p_s = self.stream_groups[LIGHT][0]
                        d_s = full_d
                    else:
                        idx = HEAVY if is_attn else LIGHT
                        set_current_stream_idx(idx)
                        p_s, d_s = self.stream_groups[idx][0], self.stream_groups[idx][1]
                    with torch.cuda.stream(d_s):
                        set_pdmux_status(False)
                        if coord_opt:
                            if prev_d is not None:
                                d_s.wait_stream(prev_d)      # order windows on GPU (no CPU block)
                            if wi == 1 and prefill_evt is not None:
                                d_s.wait_event(prefill_evt)  # full-SM decode waits prefill-tail done
                        r = MR.forward_split_decode(decode_fb, (s, e))
                        if r is not None:
                            logits_output = r
                    prev_d = d_s
                    # advance the WHOLE prefill chunk once, in the first (mamba/LIGHT) window
                    # where prefill gets the large complementary partition (avoids 19x run_batch).
                    if wi == 0 and self.split_prefill_batch.split_index < num_layers:
                        k = min(fwd_total, num_layers - self.split_prefill_batch.split_index)
                        self.split_prefill_batch.split_forward_count = k
                        with torch.cuda.stream(p_s):
                            set_pdmux_status(True)
                            prefill_result = self.run_batch(self.split_prefill_batch)
                            if coord_opt:
                                prefill_evt = p_s.record_event()
                        nxt = self.split_prefill_batch.split_index + k
                        if nxt >= num_layers:
                            self.split_prefill_batch.split_prefill_finished = True
                            prefill_exe_done = p_s.record_event()
                        self.split_prefill_batch.split_index = nxt
                    if not coord_opt:
                        d_s.synchronize()
                        if wi == 0:
                            p_s.synchronize()
                if coord_opt and prev_d is not None:
                    prev_d.synchronize()                     # single final decode sync before sample

                # sample decode (after final window)
                next_ids = MR.sample(logits_output, decode_fb)
                # run_batch normally sets batch.output_ids; next prepare_for_decode reads it
                # as input_ids. We bypass run_batch for decode, so set it here.
                self.running_batch.output_ids = next_ids
                decode_result = GenerationBatchResult(logits_output=logits_output)
                decode_result.next_token_ids = next_ids
                decode_done = True
                # settle stream refs on HEAVY pair for steps 7-8
                stream_idx = HEAVY
                set_current_stream_idx(HEAVY)
                stream_group = self.stream_groups[HEAVY]
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
            else:
                # ---- fallback: decode-only or prefill-only (original behavior) ----
                with torch.cuda.stream(decode_stream):
                    set_pdmux_status(False)
                    if decode_active:
                        decode_result = self.run_batch(self.running_batch)
                        decode_done = True
                    else:
                        decode_done = False
                with torch.cuda.stream(prefill_stream):
                    set_pdmux_status(True)
                    if prefill_active:
                        prefill_done = True
                        ext = self.split_prefill_batch.extend_num_tokens
                        forward_count = (
                            max(1, self.pdmux_config.split_forward_token_budget // ext)
                            if ext > 0
                            else self.model_config.num_hidden_layers
                        )
                        next_split_index = min(
                            self.split_prefill_batch.split_index + forward_count,
                            self.model_config.num_hidden_layers,
                        )
                        forward_count = next_split_index - self.split_prefill_batch.split_index
                        self.split_prefill_batch.split_forward_count = forward_count
                        prefill_result = self.run_batch(self.split_prefill_batch)
                        if next_split_index == self.model_config.num_hidden_layers:
                            self.split_prefill_batch.split_prefill_finished = True
                            prefill_exe_done = prefill_stream.record_event()
                        self.split_prefill_batch.split_index = next_split_index
                    elif wait_prefill_kernel_done:
                        prefill_done = True
                    else:
                        prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(self.running_batch, decode_result)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )
                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch
                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
