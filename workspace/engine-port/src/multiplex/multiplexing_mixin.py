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

            if decode_active and prefill_active:
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
                for wi, (s, e, is_attn) in enumerate(windows):
                    idx = HEAVY if is_attn else LIGHT
                    set_current_stream_idx(idx)
                    p_s, d_s = self.stream_groups[idx][0], self.stream_groups[idx][1]
                    with torch.cuda.stream(d_s):
                        set_pdmux_status(False)
                        r = MR.forward_split_decode(decode_fb, (s, e))
                        if r is not None:
                            logits_output = r
                    # advance the WHOLE prefill chunk once, in the first (mamba/LIGHT) window
                    # where prefill gets the large complementary partition (avoids 19x run_batch).
                    if wi == 0 and self.split_prefill_batch.split_index < num_layers:
                        k = min(fwd_total, num_layers - self.split_prefill_batch.split_index)
                        self.split_prefill_batch.split_forward_count = k
                        with torch.cuda.stream(p_s):
                            set_pdmux_status(True)
                            prefill_result = self.run_batch(self.split_prefill_batch)
                        nxt = self.split_prefill_batch.split_index + k
                        if nxt >= num_layers:
                            self.split_prefill_batch.split_prefill_finished = True
                            prefill_exe_done = p_s.record_event()
                        self.split_prefill_batch.split_index = nxt
                    d_s.synchronize()
                    if wi == 0:
                        p_s.synchronize()

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
