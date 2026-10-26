#!/usr/bin/env python3
"""FAKE SGLang + PD-mux engine for the CPU tests of the cost-term harness.  NOT engine code, no GPU, no model.

One deterministic engine core (`EngineCore`) with a KNOWN ground truth, driven two ways:
  * virtual time  -- `VirtualTransport` advances an event heap; no sleeps, bit-for-bit reproducible (the e2e test);
  * real time     -- `python3 ct_fake_engine.py <sglang launch_server argv>` serves /health, /generate (JSON and SSE),
                     /abort_request over HTTP and sleeps the modelled durations (the HTTP / process-lifecycle test).

What the core imitates (each item mirrors a code fact of SGLang v0.5.10 + this project's src/multiplex):
  * boot log lines the harness parses: ServerArgs repr, 'PD-Multiplexing enabled ... sm_counts', 'Load weight end ...
    type=... mem usage=', 'Mamba Cache is allocated' (cap + 1 slots), 'KV Cache is allocated', 'Capture cuda graph bs',
    'max_total_num_tokens=... max_running_requests=...', 'PD-mux sticky partition ENABLED ... (fixed target index=i)',
    'decode probe ENABLED', 'admission probe ENABLED', 'PD-mux green read-out: {...}' (+ JSON file);
  * decode_forward records (schema pdmux.decode_probe/v1, RuntimeEvent top-level layout) one per decode forward:
    seq, mode, bs_real, bs_padded (capture-list bisect_left), graph, t_host_start (= forward start),
    decode_fwd_host_ms, decode_fwd_gpu_ms (None when 'unresolved').  Order: t_host_start(k) < t_host_start(k) + dur
    = t_host_start(k + 1); TEL decode_iteration (stage-2 interface) has t_host_start <= t_launch < t_sync < next start;
  * admission: ignore_eos reserves input + max_new_tokens (schedule_policy add_one_req_ignore_eos), cap = slots,
    FIFO without skipping, a prefill batch closes after the request that makes it reach max_prefill_tokens;
  * vanilla loop: prefill batches run before decode (event_loop_normal); PD-mux: one split-prefill span at a time,
    decode continues concurrently and is slower by `kappa` while a span is in flight; sticky ON keeps decode on the
    division, with an EMPTY decode batch prefill uses group 0 = 108 SM (multiplexing_mixin.adjust_stream_groups);
    108u (sticky OFF) decodes on the plain group unless a span is in flight;
  * I-5 admission_window records (terminal_by_category, decode_bs_hist) per 1 s window.

Ground truth (ms): T_dec = (a + b * bs_padded + c * bs * ctx_mean) with a, b, c scaled (108 / D_real)^g and by the
model composition; T_prefill(L, P) = (u + v L + w L^2) (108 / P)^theta, the first request of a new L in a boot x
`warm_mult`.  Stage 2: a PD-mux span that starts while bs_dec decode requests run is slower by (1 + pf_coex bs_dec)
(so sigma' = (1 + pf_coex bs) / (1 + pf_coex) for the T3 reference with ONE decode request); a decode step launched
while a span is in flight is slower by kappa (rho truth), and every `stall_every`-th such step lasts stall_mult x the
undisturbed step.  TP (--tp-size 2): per-rank banners ('TPr]' prefix), weights / mamba / KV bytes per rank = 1/tp,
rank-suffixed files (decode_probe.tp0.jsonl, tel.tp1.jsonl, green.tp0.json; multigpu_guard.rank_suffixed_path) with
tp_rank / dp_rank fields, decode x tp_dec_mult, prefill x tp_pf_mult.
TEL phase events follow multiplexing_mixin._phase_on_*: prefill_span_start at issue, prefill_span_end when the loop
observes completion (a decode-step boundary while decoding), decode_iteration emitted at t_sync (timestamp = t_sync,
fields iter_seq, t_launch, t_sync, bs, stream_idx, prefill_sms, decode_sms, prefill_in_flight, ...).
Faults (truth keys) exist to make each harness mutant observable: realize, client_drift(_bs), i4_unresolved_every,
retract_at_step, capture_max, weight_dependent, no_vanilla_probe, stall_every.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import ct_spec as S  # noqa: E402

sys.path.insert(0, str(HERE.parents[2] / "scripts" / "costmodel"))
import analytic_terms as AT  # noqa: E402

GIB = 1024 ** 3
DECODE_SCHEMA = "pdmux.decode_probe/v1"
ADMISSION_SCHEMA = "pdmux.admission/v1"
TERMINAL = ("kv", "cap", "prefill_seat", "max_prefill_tokens", "r2_latch", "other", "no_waiting", "admitted_all",
            "unclassified")
LLAMA_PARAMS = 8_030_261_248
NANO_BUDGET = 50_816_000_000        # pool + weights (bytes) at mem 0.6 in the fake; pool(a4) is then ~ the PREREG value

DEFAULT_TRUTH = {
    "a108": 8.5, "b108": 0.15, "c108": 8.0e-6, "ga": 0.5, "gb": 0.8, "gc": 0.9,
    "vanilla_mult": 1.03, "kappa": 1.25,
    "pf_u": 15.0, "pf_v": 0.1, "pf_w4": 3.0e-7, "theta": 0.8, "warm_mult": 1.5,
    "host_frac": 0.05, "gpu_frac": 0.9, "client_lag_s": 0.0003,
    "client_drift": 0.0, "client_drift_bs": None,
    "i4_unresolved_every": 0, "realize": {}, "retract_at_step": None, "capture_max": None,
    "weight_dependent": 0.0, "no_vanilla_probe": False, "boot_delay_s": 0.05,
    "pf_coex": 0.0, "stall_every": 0, "stall_mult": 6.0, "tp_dec_mult": 0.62, "tp_pf_mult": 0.6,
    # decision 30 long points: the server dies (illegal-access traceback) on an input of at least this many tokens;
    # the logged / enforced context_len is clamped to this value (a server that does not honour --context-length)
    "die_on_input_ge": None, "context_clamp": None,
    # audit r1 m2: the server dies when the N-th request (counting warm-up) arrives -- a death in the middle of a grid boot
    "die_after_requests": None,
}


# ============================================================================ model / memory facts
def _arg(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
    return default


def model_facts(model_path):
    """-> dict(arch, n_A, n_M, params, s, k, s_ssm) from the model path (config.json if present)."""
    p = Path(str(model_path))
    if "llama" in str(model_path).lower():
        k = 32 * 2 * 8 * 128 * 2
        return {"arch": "LlamaForCausalLM", "n_A": 32, "n_M": 0, "params": LLAMA_PARAMS, "s": 0, "k": k, "s_ssm": 0,
                "family": "llama"}
    cfgp = p / "config.json"
    if cfgp.exists():
        a = AT.arch_from_config(json.loads(cfgp.read_text()))
    else:
        a = AT.nano9b_v2()
    return {"arch": "NemotronHForCausalLM", "n_A": a.n_A, "n_M": a.n_M, "params": AT.params_total(a),
            "s": AT.state_bytes_per_request(a), "k": AT.kv_bytes_per_token(a),
            "s_ssm": a.n_M * AT.ssm_state_bytes_per_layer(a), "family": "nano",
            "alpha": AT.prefill_flops(a, 1)["linear_per_token"]}


def read_pdmux_config(path):
    text = Path(path).read_text()
    rows = [tuple(int(x) for x in m) for m in re.findall(r"^\s*-\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*[0-9]+\s*\]", text, re.M)]
    groups = [(S.TOTAL_SM, 0)] + [(p, d) for p, d in rows] + [(0, S.TOTAL_SM)]
    return groups


# ============================================================================ the core
class Req:
    __slots__ = ("rid", "n_in", "max_new", "gen", "t_arr", "sink", "done", "aborted", "ids")

    def __init__(self, rid, n_in, max_new, sink, ids=None):
        self.rid, self.n_in, self.max_new, self.sink = rid, n_in, max_new, sink
        self.gen, self.t_arr, self.done, self.aborted, self.ids = 0, None, False, False, ids


class EngineCore:
    def __init__(self, argv, env, truth=None, log_write=None, wall0=1_790_000_000.0):
        self.argv, self.env = list(argv), dict(env)
        self.truth = dict(DEFAULT_TRUTH, **(truth or {}))
        self.log_write = log_write or (lambda s: None)
        self.wall0 = wall0
        a = self.argv
        self.model_path = _arg(a, "--model-path")
        self.load_format = _arg(a, "--load-format", "auto")
        self.cap = int(_arg(a, "--max-running-requests", 48))
        self.mamba_cap = _arg(a, "--max-mamba-cache-size")
        self.ctx_len = int(_arg(a, "--context-length", 131072))
        if self.truth.get("context_clamp"):
            self.ctx_len = min(self.ctx_len, int(self.truth["context_clamp"]))
        self.dead = False
        self.max_prefill = int(_arg(a, "--max-prefill-tokens", 16384))
        self.pdmux = bool(_arg(a, "--enable-pdmux", False))
        self.eager = bool(_arg(a, "--disable-cuda-graph", False))
        self.mf = float(_arg(a, "--mem-fraction-static", 0.88))
        self.tp = int(_arg(a, "--tp-size", 1))
        self.m = model_facts(self.model_path)
        self.sticky = self.env.get("PDMUX_STICKY_PARTITION") == "1"
        self.fixed_d = int(self.env["PDMUX_R2_FIXED_DSM"]) if self.env.get("PDMUX_R2_FIXED_DSM") else None
        self.groups = read_pdmux_config(_arg(a, "--pdmux-config-path")) if self.pdmux else None
        self.realize = {int(k): v for k, v in (self.truth.get("realize") or {}).items()}
        # memory
        wbytes = self.m["params"] * 2 / self.tp            # per rank
        pool = NANO_BUDGET - wbytes
        slots = (self.cap + 1) if self.m["family"] == "nano" else 0
        self.mamba_bytes = slots * self.m["s"] / self.tp
        self.kv_bytes = pool - self.mamba_bytes
        self.kv_tokens = int(self.kv_bytes // (self.m["k"] / self.tp))
        self.weight_gib = wbytes / GIB
        mx = min(256, self.cap) if self.truth.get("capture_max") is None else int(self.truth["capture_max"])
        self.capture = [] if self.eager else S.default_capture_list(mx)
        # state
        self.waiting, self.running, self.span = [], [], None
        self.kv_used, self.seq, self.step_n, self.span_seq, self.iter_seq = 0, 0, 0, 0, 0
        self.seen_L = set()
        self.files = {}
        self.dec_p = self.env.get("PDMUX_DECODE_PROBE_PATH")
        if self.dec_p and not self.pdmux and self.truth.get("no_vanilla_probe"):
            self.dec_p = None
        self.adm_p = self.env.get("PDMUX_ADMISSION_PATH")
        self.tel_p = self.env.get("PDMUX_TELEMETRY_PATH") if self.pdmux and self.env.get("PDMUX_PHASE_EVENTS") == "1" else None
        self.co_steps = 0
        self.win = None
        self.sticky_idx = None
        if self.pdmux and self.sticky:
            divs = [i for i, (p, d) in enumerate(self.groups) if 1 <= i <= len(self.groups) - 2 and d == self.fixed_d]
            if not divs:
                raise RuntimeError(f"PDMUX_STICKY_PARTITION: no green-context division with decode_sms={self.fixed_d}")
            self.sticky_idx = divs[0]

    # ---------------------------------------------------------------- output helpers
    def rank_path(self, path, r):
        """multigpu_guard.rank_suffixed_path: unchanged at tp = 1, '<stem>.tp<r><suffix>' otherwise."""
        if self.tp <= 1:
            return path
        p = Path(path)
        return str(p.with_name(f"{p.stem}.tp{r}{p.suffix}"))

    def _emit(self, path, rec):
        if not path:
            return
        for r in range(self.tp):
            rp = self.rank_path(path, r)
            f = self.files.get(rp)
            if f is None:
                f = self.files[rp] = open(rp, "a", buffering=1)
            f.write(json.dumps(dict(rec, tp_rank=r, dp_rank=0) if self.tp > 1 else rec) + "\n")

    def _log(self, now, text):
        self.log_write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.wall0 + now))}] {text}\n")

    def close(self):
        for f in self.files.values():
            f.close()
        self.files = {}

    # ---------------------------------------------------------------- SM model
    def _real(self, d):
        return int(self.realize.get(d, d))

    def decode_sm(self, in_flight):
        if not self.pdmux:
            return S.TOTAL_SM
        if self.sticky:
            return self._real(self.fixed_d)
        return self._real(self.fixed_d) if in_flight else S.TOTAL_SM

    def prefill_sm(self, decode_busy):
        if not self.pdmux or not decode_busy:
            return S.TOTAL_SM
        return S.TOTAL_SM - self._real(self.fixed_d)

    # ---------------------------------------------------------------- ground truth
    def _scales(self):
        m = self.m
        if m["family"] == "llama":
            return {"a": 0.85, "b": 0.30, "c": m["k"] / 16384.0, "w": 8.0, "v": 1.0}
        nano = AT.nano9b_v2()
        return {"a": (m["params"] / AT.params_total(nano)) * (1.0 + self.truth["weight_dependent"] * (self.load_format != "dummy")),
                "b": 0.2 + 0.8 * m["n_M"] / 27.0, "c": m["k"] / 16384.0, "w": m["n_A"] / 4.0,
                "v": m["alpha"] / AT.prefill_flops(nano, 1)["linear_per_token"]}

    def t_dec_ms(self, d_sm, bs_pad, bs, ctx_mean):
        t, sc = self.truth, self._scales()
        r = S.TOTAL_SM / d_sm
        ms = (t["a108"] * sc["a"] * r ** t["ga"] + t["b108"] * sc["b"] * r ** t["gb"] * bs_pad
              + t["c108"] * sc["c"] * r ** t["gc"] * bs * ctx_mean)
        if not self.pdmux:
            ms *= t["vanilla_mult"]
        if self.tp > 1:
            ms *= t["tp_dec_mult"]
        return ms

    def t_pf_ms(self, L, p_sm):
        t, sc = self.truth, self._scales()
        ms = (t["pf_u"] + t["pf_v"] * sc["v"] * L + t["pf_w4"] * sc["w"] * L * L) * (S.TOTAL_SM / p_sm) ** t["theta"]
        if self.tp > 1:
            ms *= t["tp_pf_mult"]
        if L not in self.seen_L:
            self.seen_L.add(L)
            ms *= t["warm_mult"]
        return ms

    def pad(self, bs):
        if self.eager or not self.capture:
            return bs, "eager"
        for c in self.capture:
            if c >= bs:
                return c, "hit"
        return bs, "eager"

    # ---------------------------------------------------------------- boot
    def boot_lines(self):
        sa = (f"server_args=ServerArgs(model_path='{self.model_path}', tokenizer_path='{self.model_path}', "
              f"load_format='{self.load_format}', revision='{_arg(self.argv, '--revision', '')}', "
              f"trust_remote_code=True, context_length={_arg(self.argv, '--context-length', 131072)}, mem_fraction_static={self.mf}, "
              f"max_running_requests={self.cap}, chunked_prefill_size={_arg(self.argv, '--chunked-prefill-size', 8192)}, "
              f"max_prefill_tokens={self.max_prefill}, schedule_conservativeness=1.0, random_seed={_arg(self.argv, '--random-seed', 1)}, "
              f"attention_backend='{_arg(self.argv, '--attention-backend', 'flashinfer')}', "
              f"max_mamba_cache_size={self.mamba_cap}, cuda_graph_max_bs=256, "
              f"disable_radix_cache={bool(_arg(self.argv, '--disable-radix-cache', False))}, "
              f"disable_cuda_graph={self.eager}, enable_pdmux={self.pdmux}, "
              f"pdmux_config_path={repr(_arg(self.argv, '--pdmux-config-path')) if self.pdmux else None}, tp_size={self.tp})")
        L = [sa]
        if self.pdmux:
            L.append(f"PD-Multiplexing enabled with {len(self.groups)} stream groups, sm_counts (prefill_sm, decode_sm): "
                     f"{[tuple(g) for g in self.groups]}")
        for r in range(self.tp):
            pre = f"TP{r}] " if self.tp > 1 else ""
            L.append(f"{pre}Load weight end. elapsed=0.12 s, type={self.m['arch']}, avail mem=62.13 GB, mem usage={self.weight_gib:.2f} GB.")
            if self.m["family"] == "nano":
                slots = self.cap + 1
                ssm = slots * self.m["s_ssm"] / self.tp / GIB
                conv = slots * (self.m["s"] - self.m["s_ssm"]) / self.tp / GIB
                L.append(f"{pre}Mamba Cache is allocated. max_mamba_cache_size: {self.cap}, conv_state size: {conv:.2f}GB, ssm_state size: {ssm:.2f}GB ")
            kv_gib = self.kv_tokens * self.m["k"] / self.tp / 2 / GIB
            L.append(f"{pre}KV Cache is allocated. #tokens: {self.kv_tokens}, K size: {kv_gib:.2f} GB, V size: {kv_gib:.2f} GB")
        if not self.eager:
            L.append(f"{'TP0] ' if self.tp > 1 else ''}Capture cuda graph bs {self.capture}")
        for r in range(self.tp):
            pre = f"TP{r}] " if self.tp > 1 else ""
            if self.dec_p:
                L.append(f"{pre}decode probe ENABLED (I-4 events=True, I-6; all arms; pdmux={self.pdmux} capture_bs={self.capture}) -> {self.rank_path(self.dec_p, r)}")
            if self.adm_p:
                L.append(f"{pre}admission probe ENABLED (all arms; cap_slot_relation=identical cap_source=user) -> {self.rank_path(self.adm_p, r)}")
            if self.pdmux and self.sticky:
                L.append(f"{pre}PD-mux sticky partition ENABLED: decode holds its division while decode is busy (fixed target "
                         f"index={self.sticky_idx}); the unpartitioned (0, total_sm) group is used only when the decode batch is empty")
            if self.pdmux and self.env.get("PDMUX_GREEN_READOUT") == "1":
                g = self.green()
                if self.tp > 1:
                    g.update(tp_rank=r, dp_rank=0)
                L.append(f"{pre}PD-mux green read-out: {g}")
                if self.env.get("PDMUX_GREEN_READOUT_PATH"):
                    Path(self.rank_path(self.env["PDMUX_GREEN_READOUT_PATH"], r)).write_text(json.dumps(g, indent=2, sort_keys=True))
        L.append(f"max_total_num_tokens={self.kv_tokens}, chunked_prefill_size={_arg(self.argv, '--chunked-prefill-size', 8192)}, "
                 f"max_prefill_tokens={self.max_prefill}, max_running_requests={self.cap}, context_len={self.ctx_len}, "
                 f"available_gpu_mem=28.01 GB")
        if self.dec_p:
            self._emit(self.dec_p, {"event": "decode_probe_boot", "record_schema": DECODE_SCHEMA, "capture_bs": self.capture,
                                    "arm_enable_pdmux": self.pdmux, "schema": "pdmux.runtime-event/v1"})
        return L

    def green(self):
        groups = []
        last = len(self.groups) - 1
        for i, (p, d) in enumerate(self.groups):
            div = 1 <= i <= last - 1
            row = {"stream_index": i, "target_prefill_sm": p, "target_decode_sm": d,
                   "is_sticky_target": self.sticky_idx == i}
            for side, sm in (("prefill", p), ("decode", d)):
                if not div:
                    row[side] = {"green_ctx_is_null": True, "green_sm": None}
                    continue
                if side == "decode":
                    real = self._real(d)
                else:                       # a mis-realized decode half leaves its complement to prefill
                    real = S.TOTAL_SM - self._real(d) if d in self.realize else sm
                row[side] = {"green_ctx_is_null": False, "green_sm": {"smCount": real}}
            groups.append(row)
        return {"probe": "pdmux_green_readout", "sticky_target_index": self.sticky_idx, "groups": groups}

    # ---------------------------------------------------------------- requests
    def submit(self, req, now):
        req.t_arr = now
        self.waiting.append(req)

    def abort(self, rid, now):
        for q in (self.waiting, self.running):
            for r in list(q):
                if r.rid == rid and not r.done:
                    r.aborted = r.done = True
                    q.remove(r)
                    if q is self.running:
                        self.kv_used -= r.n_in + r.max_new
                    r.sink(now, r.gen, True, "abort")
        if self.span:
            for r in self.span[0]:
                if r.rid == rid:
                    r.aborted = True

    def _n_active(self):
        return len(self.running) + (len(self.span[0]) if self.span else 0)

    def _admit(self, now):
        """FIFO, ignore_eos full reservation, cap = slots, <= max_prefill tokens per batch (first always)."""
        batch, toks, cat = [], 0, None
        if not self.waiting:
            self._decision(now, "no_waiting")
            return batch
        while self.waiting:
            r = self.waiting[0]
            if self._n_active() + len(batch) + 1 > self.cap:
                cat = "cap"
                break
            if self.kv_used + r.n_in + r.max_new > self.kv_tokens:
                cat = "kv"
                break
            if batch and toks >= self.max_prefill:
                # add_one_req_ignore_eos (ignore_eos + radix off, schedule_policy.py:641-724) has no per-request
                # input check; budget_state() returns OTHER once rem_input_tokens <= 0, i.e. AFTER the request that
                # crosses max_prefill_tokens was added
                cat = "max_prefill_tokens"
                break
            batch.append(self.waiting.pop(0))
            toks += r.n_in
            self.kv_used += r.n_in + r.max_new
        self._decision(now, cat or "admitted_all")
        return batch

    # ---------------------------------------------------------------- I-5 windows
    def _decision(self, now, cat):
        if not self.adm_p:
            return
        self._roll(now)
        self.win["cats"][cat] = self.win["cats"].get(cat, 0) + 1

    def _roll(self, now):
        if self.win is None:
            self.win = {"t0": now, "cats": {}, "hist": {}, "n_dec": 0}
        if now - self.win["t0"] >= S.PROBE_WINDOW_S:
            self._emit(self.adm_p, {"event": "admission_window", "record_schema": ADMISSION_SCHEMA,
                                    "t_mono_start": self.win["t0"], "t_mono_end": now,
                                    "n_decisions": sum(self.win["cats"].values()),
                                    "terminal_by_category": {c: self.win["cats"].get(c, 0) for c in TERMINAL},
                                    "n_decode_iter": self.win["n_dec"],
                                    "decode_bs_hist": {str(k): v for k, v in sorted(self.win["hist"].items())}})
            self.win = {"t0": now, "cats": {}, "hist": {}, "n_dec": 0}

    def flush(self, now):
        if self.adm_p and self.win is not None:
            self.win["t0"] -= S.PROBE_WINDOW_S      # force a roll
            self._roll(now)

    # ---------------------------------------------------------------- one engine action
    def step(self, now):
        """-> (t_next or None, deliveries [(t, req, cum, finished)])."""
        out = []
        if self.span and self.span[1] <= now:
            reqs, sidx_ = self.span[0], self.span[3]
            self.span = None
            self._tel(now, "prefill_span_end", span_seq=self.span_seq, n_done=len(reqs), running_bs=len(self.running),
                      queue_len=len(self.waiting), stream_idx=sidx_)
            for r in reqs:
                if r.aborted:
                    self.kv_used -= r.n_in + r.max_new
                    continue
                r.gen = 1
                fin = r.gen >= r.max_new
                out.append((now, r, 1, fin))
                if fin:
                    r.done = True
                    self.kv_used -= r.n_in + r.max_new
                else:
                    self.running.append(r)
        if not self.pdmux:
            batch = self._admit(now) if self.waiting else []
            if batch:
                dur = sum(self.t_pf_ms(r.n_in, S.TOTAL_SM) for r in batch) / 1000.0
                self._log(now, f"Prefill batch, #new-seq: {len(batch)}, #new-token: {sum(r.n_in for r in batch)}, "
                               f"#running-req: {len(self.running)}, #queue-req: {len(self.waiting)}, cuda graph: False")
                t_end = now + dur
                for r in batch:
                    r.gen = 1
                    fin = r.gen >= r.max_new
                    out.append((t_end, r, 1, fin))
                    if fin:
                        r.done = True
                        self.kv_used -= r.n_in + r.max_new
                    else:
                        self.running.append(r)
                return t_end, out
        else:
            if self.span is None and self.waiting:
                batch = self._admit(now)
                if batch:
                    busy = bool(self.running)
                    p = self.prefill_sm(busy)
                    coex = (1.0 + self.truth["pf_coex"] * len(self.running)) if busy else 1.0
                    dur = sum(self.t_pf_ms(r.n_in, p) for r in batch) * coex / 1000.0
                    self.span_seq += 1
                    sidx = self._stream_idx(busy)
                    self.span = (batch, now + dur, p, sidx)
                    gp, gd = self.groups[sidx]
                    self._tel(now, "prefill_span_start", span_seq=self.span_seq, n_new=len(batch),
                              extend_tokens=sum(r.n_in for r in batch), running_bs=len(self.running),
                              queue_len=len(self.waiting), cap=self.cap, stream_idx=sidx, prefill_sms=gp,
                              decode_sms=gd)
                    self._log(now, f"Prefill batch, #new-seq: {len(batch)}, #new-token: {sum(r.n_in for r in batch)}, "
                                   f"#running-req: {len(self.running)}, #queue-req: {len(self.waiting)}, cuda graph: False")
            elif self.span is not None and self.waiting:
                self._decision(now, "prefill_seat")
        if self.running:
            return self._decode(now, out)
        if self.span:
            return self.span[1], out
        return None, out

    def _decode(self, now, out):
        in_flight = self.span is not None
        d_sm = self.decode_sm(in_flight)
        bs = len(self.running)
        self.last_decode_bs = bs
        bs_pad, graph = self.pad(bs)
        ctx = sum(r.n_in + r.gen for r in self.running) / bs
        ms = self.t_dec_ms(d_sm, bs_pad, bs, ctx) * (self.truth["kappa"] if in_flight else 1.0)
        if in_flight:
            self.co_steps += 1
            se = int(self.truth.get("stall_every") or 0)
            if se and self.co_steps % se == 0:
                ms = ms / self.truth["kappa"] * self.truth["stall_mult"]
        dur = ms / 1000.0
        self.seq += 1
        self.step_n += 1
        ev = int(self.truth.get("i4_unresolved_every") or 0)
        gpu = None if (ev and self.seq % ev == 0) else ms * self.truth["gpu_frac"]
        sidx = None
        if self.pdmux:
            sidx = self.sticky_idx if self.sticky else (len(self.groups) - 1 if not in_flight else self._fixed_idx())
        if self.dec_p:
            self._emit(self.dec_p, {"run_id": self.env.get("PDMUX_RUN_ID", "unlabeled"), "workload_id": "fake",
                                    "phase": "decode_probe", "event": "decode_forward", "timestamp_monotonic_s": now + dur,
                                    "request_id": "", "schema": "pdmux.runtime-event/v1", "record_schema": DECODE_SCHEMA,
                                    "seq": self.seq, "forward_pass_id": self.seq, "mode": "decode", "bs_real": bs,
                                    "graph": graph, "bs_padded": bs_pad, "bs_pad_expected": bs_pad,
                                    "pad_check": "ok" if graph == "hit" else "na",
                                    "eager_reason": None if graph == "hit" else "bs_over_max",
                                    "pdmux_stream_idx": sidx, "t_host_start": now,
                                    "decode_fwd_host_ms": ms * self.truth["host_frac"], "decode_fwd_gpu_ms": gpu,
                                    "thread": "MainThread", "dropped_events": 0})
        self.iter_seq += 1
        if self.pdmux:
            gp, gd = self.groups[sidx]
            # emitted on decode sync (multiplexing_mixin._phase_on_decode_sync): timestamp = t_sync
            self._tel(now + 0.95 * dur, "decode_iteration", iter_seq=self.iter_seq, t_launch=now + 0.02 * dur,
                      t_sync=now + 0.95 * dur, bs=bs, stream_idx=sidx, prefill_sms=gp, decode_sms=gd,
                      prefill_in_flight=in_flight, prefill_kernel_done_wait=False, queue_len=len(self.waiting))
        if self.adm_p:
            self._roll(now)
            self.win["n_dec"] += 1
            self.win["hist"][bs] = self.win["hist"].get(bs, 0) + 1
        if self.step_n % 40 == 0:
            self._log(now, f"Decode batch, #running-req: {bs}, #full token: {int(ctx * bs)}, cuda graph: {graph == 'hit'}, "
                           f"gen throughput (token/s): 0.00, #queue-req: {len(self.waiting)}")
        ra = self.truth.get("retract_at_step")
        if ra is not None and self.step_n == int(ra):
            self._log(now, "KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 10")
        t_end = now + dur
        keep = []
        for r in self.running:
            r.gen += 1
            fin = r.gen >= r.max_new
            out.append((t_end, r, r.gen, fin))
            if fin:
                r.done = True
                self.kv_used -= r.n_in + r.max_new
            else:
                keep.append(r)
        self.running = keep
        return t_end, out

    def _stream_idx(self, decode_busy):
        """multiplexing_mixin.adjust_stream_groups: empty decode -> group 0 (108, 0); busy -> sticky / fixed division."""
        if not decode_busy:
            return 0
        return self.sticky_idx if self.sticky else self._fixed_idx()

    def _fixed_idx(self):
        for i, (p, d) in enumerate(self.groups):
            if 1 <= i <= len(self.groups) - 2 and d == self.fixed_d:
                return i
        return None

    def _tel(self, now, event, **f):
        if self.tel_p:
            self._emit(self.tel_p, dict(run_id=self.env.get("PDMUX_RUN_ID", "unlabeled"), workload_id="fake",
                                        phase="benchmark", event=event, timestamp_monotonic_s=now, request_id="",
                                        schema="pdmux.runtime-event/v1", dropped_events=0, **f))

    # ---------------------------------------------------------------- client-side lag (fault: drift)
    def client_time(self, t, req, cum):
        lag = self.truth["client_lag_s"]
        dr = self.truth.get("client_drift") or 0.0
        dbs = self.truth.get("client_drift_bs")
        if dr and (dbs is None or getattr(self, "last_decode_bs", None) == dbs):
            lag += dr * max(0.0, t - (req.t_arr or 0.0))
        return t + lag


# ============================================================================ virtual-time transport
class Handle:
    def __init__(self, rid, n_in, max_new, t_send):
        self.rid, self.n_in, self.max_new, self.t_send = rid, n_in, max_new, t_send
        self.chunks, self.done, self.finish, self.error, self.aborted = [], False, None, None, False

    def record(self):
        return {"rid": self.rid, "t_send": self.t_send, "chunks": [list(c) for c in self.chunks],
                "t_first": self.chunks[0][0] if self.chunks else None,
                "t_done": self.chunks[-1][0] if self.done and self.chunks else None,
                "prompt_tokens": self.n_in, "completion_tokens": self.chunks[-1][1] if self.chunks else 0,
                "finish": self.finish, "error": self.error, "aborted": self.aborted}


class VirtualTransport:
    """Deterministic event heap: engine ticks and client deliveries in time order."""

    def __init__(self, core):
        self.core = core
        self.t = 0.0
        self.heap, self.n = [], 0
        self.tick_pending = False
        self.handles = {}

    def now(self):
        return self.t

    def clock_pair(self):
        return [self.core.wall0 + self.t, self.t]

    def _push(self, t, kind, payload):
        self.n += 1
        heapq.heappush(self.heap, (t, self.n, kind, payload))

    def _kick(self):
        if not self.tick_pending:
            self._push(self.t, "tick", None)
            self.tick_pending = True

    def start(self, rid, ids, max_new):
        h = Handle(rid, len(ids), max_new, self.t)
        self.handles[rid] = h
        die = self.core.truth.get("die_on_input_ge")
        if self.core.dead:
            h.error, h.done = "connection refused (server dead)", True
            return h
        if len(ids) + max_new >= self.core.ctx_len:       # SGLang: input + max_new >= context_len -> HTTP 400
            h.error, h.done = (f"400 The input ({len(ids)} tokens) plus max_new_tokens is longer than the model's "
                               f"context length ({self.core.ctx_len} tokens)"), True
            return h
        self.core.n_started = getattr(self.core, "n_started", 0) + 1
        da = self.core.truth.get("die_after_requests")
        if (die is not None and len(ids) >= int(die)) or (da is not None and self.core.n_started >= int(da)):
            self.core.dead = True
            self.core.log_write("Traceback (most recent call last):\nRuntimeError: CUDA error: an illegal memory access "
                                "was encountered\n")
            h.error, h.done = "connection reset (server died)", True
            return h
        # deliveries come back from core.step(); the sink is only called by core.abort (handled in self.abort)
        self.core.submit(Req(rid, len(ids), max_new, lambda *a, **k: None, ids), self.t)
        self._kick()
        return h

    def abort(self, h):
        self.core.abort(h.rid, self.t)
        h.aborted = True
        h.done = True
        h.finish = "abort"

    def _pop(self):
        t, _, kind, payload = heapq.heappop(self.heap)
        self.t = max(self.t, t)
        if kind == "tick":
            self.tick_pending = False
            t_next, outs = self.core.step(self.t)
            for (te, r, cum, fin) in outs:
                h = self.handles.get(r.rid)
                if h is None or h.aborted:
                    continue
                self._push(self.core.client_time(te, r, cum), "deliver", (h, cum, fin, "length"))
            if t_next is not None:
                self._push(t_next, "tick", None)
                self.tick_pending = True
        elif kind == "deliver":
            h, cum, fin, finish = payload
            if not h.aborted:
                h.chunks.append((self.t, cum))
                if fin:
                    h.done, h.finish = True, finish
        elif kind == "sleep":
            pass

    def wait(self, pred, timeout=1e9):
        t_lim = self.t + timeout
        while not pred():
            if not self.heap:
                raise RuntimeError("virtual engine idle while the client is waiting (deadlock)")
            if self.heap[0][0] > t_lim:
                return False
            self._pop()
        return True

    def sleep(self, dt):
        t_end = self.t + dt
        self._push(t_end, "sleep", None)
        while self.heap and self.heap[0][0] <= t_end:
            self._pop()
        self.t = max(self.t, t_end)

    def health(self):
        return True

    def greedy(self, ids, max_new):
        h = self.start(f"greedy-{self.n}", ids, max_new)
        self.wait(lambda: h.done)
        key = hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()
        return {"ok": h.finish == "length", "output_ids": [int(key[i:i + 4], 16) for i in range(0, 4 * max_new, 4)][:max_new]}


# ============================================================================ real-time HTTP server
def serve(argv, env):
    import http.server
    import queue
    import socketserver
    truth = json.loads(env.get("FAKE_CT_TRUTH", "{}") or "{}")
    lock = threading.Condition()
    out = sys.stdout

    def logw(s):
        out.write(s)
        out.flush()
    t0 = time.perf_counter()
    try:
        core = EngineCore(argv, env, truth, logw, wall0=time.time() - t0)
    except RuntimeError as exc:
        print(f"Traceback (most recent call last):\nRuntimeError: {exc}", flush=True)
        sys.exit(1)
    time.sleep(core.truth.get("boot_delay_s", 0.05))
    for ln in core.boot_lines():
        logw(f"[boot] {ln}\n")
    port = int(_arg(argv, "--port"))

    def engine_loop():
        while True:
            with lock:
                now = time.perf_counter()
                t_next, outs = core.step(now)
                imm = [(r, cum, fin) for (te, r, cum, fin) in outs if te <= now]
                later = [(r, cum, fin) for (te, r, cum, fin) in outs if te > now]
                for r, cum, fin in imm:
                    r.sink(now, cum, fin, "length")
                if t_next is None:
                    lock.wait(timeout=0.01)
                    continue
            while True:
                rem = t_next - time.perf_counter()
                if rem <= 0:
                    break
                time.sleep(min(rem, 0.002) if rem > 0.0005 else 0)
            with lock:
                for r, cum, fin in later:
                    r.sink(time.perf_counter(), cum, fin, "length")
    threading.Thread(target=engine_loop, daemon=True).start()

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/abort_request":
                with lock:
                    core.abort(str(body.get("rid")), time.perf_counter())
                    lock.notify_all()
                return self._send(200, {})
            sp = body.get("sampling_params") or {}
            ids = body.get("input_ids") or [1, 2, 3, 4]
            n = int(sp.get("max_new_tokens", 16))
            rid = str(body.get("rid") or hashlib.sha256(os.urandom(8)).hexdigest()[:12])
            q = queue.Queue()
            req = Req(rid, len(ids), n, lambda t, cum, fin, finish="length": q.put((cum, fin, finish)), ids)
            with lock:
                core.submit(req, time.perf_counter())
                lock.notify_all()
            if not body.get("stream"):
                while True:
                    cum, fin, finish = q.get()
                    if fin:
                        break
                key = hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()
                oids = [int(key[i % 60:i % 60 + 4], 16) for i in range(n)]
                return self._send(200, {"text": "", "output_ids": oids,
                                        "meta_info": {"prompt_tokens": len(ids), "completion_tokens": cum,
                                                      "finish_reason": {"type": finish}}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(b):
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
            try:
                while True:
                    cum, fin, finish = q.get()
                    mi = {"prompt_tokens": len(ids), "completion_tokens": cum,
                          "finish_reason": {"type": finish} if fin else None}
                    chunk(("data: " + json.dumps({"text": "", "meta_info": mi}) + "\n\n").encode())
                    if fin:
                        break
                chunk(b"data: [DONE]\n\n")
                chunk(b"")
            except Exception:  # noqa: BLE001 - client went away -> abort like the engine does
                with lock:
                    core.abort(rid, time.perf_counter())

    class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        request_queue_size = 512

    Srv(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:2] == ["-m", "sglang.launch_server"]:
        a = a[2:]
    serve(a, dict(os.environ))
