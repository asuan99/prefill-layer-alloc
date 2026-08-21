"""
P0-B SHIM -- inject Breakable-CUDA-Graph break points at decode layer boundaries
WITHOUT modifying the dev tree.

★ NOT production code. Probe only. Never import this from a campaign harness.

------------------------------------------------------------------------------
WHY A SHIM AND NOT `--forward-hooks`
------------------------------------------------------------------------------
SGLang has a first-class hook extension point (`--forward-hooks` ->
model_executor/hook_manager.py:register_forward_hooks, glob `target_modules` +
`hook_factory`), and it would be the obvious way to attach a break at each
layer boundary. It does not work for this probe, for one verified ordering
reason:

    model_runner.py:656   self.init_device_graphs()          <-- CAPTURE happens here
    model_runner.py:666   register_forward_hooks(self.model, ...)   <-- hooks attach AFTER

Hooks registered through `--forward-hooks` are therefore absent from the
captured decode graph and would only fire on eager paths -- i.e. exactly not
the thing under test. This shim instead wraps the layer objects right after
`ModelRunner.load_model()` (defined model_runner.py:1072, called :508), which runs BEFORE
init_device_graphs(), so the wrappers are present during capture.

(An upstream-side fix would be to move the hook registration above
init_device_graphs. That is a dev-tree change and is deliberately NOT done
here: this probe promises zero writes to the engine tree.)

------------------------------------------------------------------------------
WHAT IT DOES
------------------------------------------------------------------------------
Wraps `ModelRunner.load_model` so that, after the model is built, every
`PDMUX_BCG_BREAK_EVERY`-th decoder layer's `forward` is wrapped to call
`break_graph()` first. With PDMUX_BCG_BREAK_EVERY=0 (the default) NOTHING is
wrapped -- that is the null-break arm, which must be measured so that the
standing cost of the BCG code path is separated from the per-break cost.

    PDMUX_BCG_BREAK_EVERY=0   BCG path on, zero breaks     (null-break control)
    PDMUX_BCG_BREAK_EVERY=k   break before every k-th layer, DECODE STEPS ONLY

The decode-only gate is not cosmetic: the same layer objects are invoked by
`forward_split_prefill` (models/zamba2.py:766-772), the fully-eager PD-mux
layer-axis prefill split. An ungated wrapper would add a Python call per prefill
layer as well, and the measured delta would confound two changes at once.

Env:
    PDMUX_BCG_BREAK_EVERY  int, default 0
    PDMUX_BCG_SHIM_VERBOSE 1 to log every wrapped layer index

Usage (never edits anything on disk):
    PYTHONPATH=<this dir>:$PYTHONPATH \
    python -c "import p0b_break_shim; p0b_break_shim.install(); \
               import runpy; runpy.run_module('sglang.launch_server', run_name='__main__')" \
           --model-path ... <normal server flags>
"""

import logging
import os
import sys

logger = logging.getLogger("p0b_break_shim")

_STATE = {"installed": False, "wrapped": 0, "total_layers": 0, "break_every": 0,
          "break_graph": None, "import_error": None}


def status():
    """Machine-readable state, printed by the harness after boot."""
    return dict(_STATE, break_graph=(_STATE["break_graph"] is not None))


def _resolve_break_graph():
    """Import the backported BCG entry point. Returns (fn, error_string)."""
    try:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (  # noqa: E501
            break_graph,
        )
        return break_graph, None
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def install():
    """Idempotent. Must be called BEFORE sglang.launch_server runs."""
    if _STATE["installed"]:
        return
    _STATE["installed"] = True

    break_every = int(os.environ.get("PDMUX_BCG_BREAK_EVERY", "0"))
    _STATE["break_every"] = break_every

    break_graph, err = _resolve_break_graph()
    _STATE["break_graph"] = break_graph
    _STATE["import_error"] = err

    if break_every > 0 and break_graph is None:
        # Refuse to run a break arm without the break primitive. Silently
        # degrading to zero breaks would make the k-break arm identical to the
        # null arm and the per-break cost would score as ~0 -- an identity, not
        # a measurement (methodology lesson #9).
        raise SystemExit(
            f"PDMUX_BCG_BREAK_EVERY={break_every} but break_graph() is not "
            f"importable ({err}). The BCG backport is not installed; this arm "
            f"is UNAVAILABLE, not a zero-cost result.")

    from sglang.srt.model_executor.model_runner import ModelRunner

    _orig_load = ModelRunner.load_model

    def _patched_load(self, *a, **kw):
        out = _orig_load(self, *a, **kw)
        if break_every > 0:
            _wrap_layers(self.model, break_every, break_graph)
        return out

    ModelRunner.load_model = _patched_load
    print(f"[p0b_break_shim] installed; break_every={break_every} "
          f"break_graph={'yes' if break_graph else 'no'}", file=sys.stderr)


def _layer_list(model):
    """Locate the decoder layer container. Kept deliberately narrow: this probe
    targets the hybrid models this project serves, all of which expose
    `model.model.layers` (Zamba2Model.layers = nn.ModuleList(...),
    models/zamba2.py:478)."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise SystemExit(
            "[p0b_break_shim] could not find model.model.layers; this shim "
            "does not know this architecture. UNAVAILABLE, not a result.")
    return layers


def _forward_batch_of(args, kwargs):
    """Find the ForwardBatch in a layer call. Zamba2 layer signatures are
    (hidden_states, forward_batch) and (hidden_states, original_hidden_states,
    forward_batch); both pass it positionally, but accept a kwarg too."""
    fb = kwargs.get("forward_batch")
    if fb is not None:
        return fb
    for v in args:
        if hasattr(v, "forward_mode"):
            return v
    return None


def _wrap_layers(model, break_every, break_graph):
    layers = _layer_list(model)
    _STATE["total_layers"] = len(layers)
    verbose = os.environ.get("PDMUX_BCG_SHIM_VERBOSE") == "1"
    n = 0
    for i, layer in enumerate(layers):
        if i == 0 or i % break_every != 0:
            continue
        orig = layer.forward

        def _broken(*a, _orig=orig, **kw):
            # ★ DECODE ONLY. Zamba2 calls these same layer objects from TWO
            # sites: the graphed decode forward (models/zamba2.py:600-608) and
            # forward_split_prefill (:766-772), which is the PD-mux layer-axis
            # prefill split and is FULLY EAGER. Without this gate the wrapper
            # fires on prefill too, and delta_k would measure
            # "graph break + eager per-layer Python call on prefill" -- two
            # variables at once, and not the quantity the gate is written about.
            fb = _forward_batch_of(a, kw)
            if fb is not None and fb.forward_mode.is_decode():
                break_graph()
            return _orig(*a, **kw)

        layer.forward = _broken
        n += 1
        if verbose:
            print(f"[p0b_break_shim] break before layer {i}", file=sys.stderr)
    _STATE["wrapped"] = n
    print(f"[p0b_break_shim] wrapped {n} of {len(layers)} layers "
          f"(break_every={break_every})", file=sys.stderr)
