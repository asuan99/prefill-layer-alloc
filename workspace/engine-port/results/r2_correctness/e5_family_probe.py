#!/usr/bin/env python3
"""E5 (F-e) trigger-family probe -- rebuilds the whole table in
`E5_FAMILY_RA4_2_2026-09-14.md` from scratch.  GPU 0; CPU only.

WHY THIS FILE EXISTS
--------------------
The E5 disposition in `rerun_prereg/PREREG_RERUN_2026-09-13.md` §6-5 registers a
TRIGGER REGEX but **no executable anywhere in the repository evaluates it** (see
§1 of the .md).  A human applies it by hand.  RA4-2 then reported that the
registered family misses at least one hard error, and proposed a correction --
by enumerating the family from the same place the registration came from.
`G-RA4-2` forbids re-using that enumeration, so this probe derives the corpus
independently:

  POSITIVES, part 1 (EXECUTED): reproduced here, on this venv, and the exact
      final traceback line is the datum.  Nothing is transcribed by hand.
  POSITIVES, part 2 (BINARY-ATTESTED): string literals extracted from the
      installed libtorch with the command in `STRINGS_CMD` below, for the
      messages whose call sites are not reachable from Python.  Marked as NOT
      EXECUTED, because "the binary contains it" is weaker than "it fired".
  NEGATIVES: three reproduced non-inference RuntimeErrors, the REAL
      `torch.OutOfMemoryError` line from job 907959 (the H1 channel that must
      NOT be intercepted), and two forms of the one dispatcher message that
      mentions `InferenceMode` without being an inference-tensor error.

Then every candidate regex is run under THREE engines, because the registered
form uses a PCRE inline-flag group that POSIX ERE does not have.

    python3 e5_family_probe.py            # table
    python3 e5_family_probe.py --json out.json
"""
import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import traceback

import torch
import torch.nn as nn

# How the binary-attested literals below were obtained (torch 2.9.1+cu130):
STRINGS_CMD = (
    "TL=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib; "
    "for f in libtorch_cpu.so libtorch_python.so libc10.so libc10_cuda.so; do "
    "strings -n 4 \"$TL/$f\"; done | "
    "grep -iE 'inference tensor|InferenceMode|inference mode|is_inference|inference_mode' | "
    "grep -iE 'not allowed|cannot|Expected |INTERNAL ASSERT|forbidden|do not track|"
    "Was it created|A view was created|only be reached'"
)

# ---------------------------------------------------------------- executed set
def _exc_line(fn):
    """Run fn, return (type_name, the `Xxx: message` line of the traceback)."""
    try:
        fn()
    except BaseException as exc:                                    # noqa: BLE001
        buf = io.StringIO()
        traceback.print_exc(file=buf)
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        name = type(exc).__module__ + "." + type(exc).__qualname__
        short = type(exc).__qualname__
        for line in lines:
            if line.startswith(short) or line.startswith(name):
                return name, line
        return name, lines[-1]
    return None, None


def p_inplace_outside():
    with torch.inference_mode():
        t = torch.ones(4)
    t.add_(1)


def p_requires_grad_outside():
    with torch.inference_mode():
        t = torch.ones(4)
    t.requires_grad_(True)


def p_saved_for_backward():
    x = torch.ones(4, requires_grad=True)
    with torch.inference_mode():
        t = torch.ones(4) * 3
    (t * x).sum().backward()


def p_version_counter():
    with torch.inference_mode():
        t = torch.ones(4)
    print(t._version, file=io.StringIO())


def p_view_rebase_direct():
    """RA4-2's case: a view of an nn.Parameter made inside, mutated outside."""
    p = nn.Parameter(torch.ones(4, 4))
    with torch.inference_mode():
        v = p[0]
    v.add_(1)


def p_view_rebase_indirect():
    """Same family, INDIRECT branch -- the base is what gets mutated."""
    p = nn.Parameter(torch.ones(4, 4))
    base = p * 2
    with torch.inference_mode():
        view = base[0]
    base.add_(1)
    view.grad_fn                                                    # noqa: B018


def p_unsafe_set_version():
    with torch.inference_mode():
        t = torch.ones(4)
    torch._C._autograd._unsafe_set_version_counter((t,), (3,))


def n_no_grad_view_rebase():
    """Specificity control: the SAME code path, no_grad instead of inference."""
    p = nn.Parameter(torch.ones(4, 4))
    with torch.no_grad():
        v = p[0]
    v.add_(1)


def n_shape():
    torch.ones(2, 3) @ torch.ones(4, 5)


def n_dtype():
    torch.ones(3, dtype=torch.int32).add_(torch.ones(3, dtype=torch.complex64))


EXECUTED_POS = [
    ("X1", p_inplace_outside),
    ("X2", p_requires_grad_outside),
    ("X3", p_saved_for_backward),
    ("X4", p_version_counter),
    ("X5", p_view_rebase_direct),
    ("X6", p_view_rebase_indirect),
    ("X7", p_unsafe_set_version),
]
EXECUTED_NEG = [("Y1", n_no_grad_view_rebase), ("Y2", n_shape), ("Y3", n_dtype)]

# -------------------------------------------------- binary-attested (not run)
BINARY_POS = {
    "B1": "Cannot set version_counter for inference tensor",
    "B2": ("Expected this function to only be reached in inference mode and when "
           "all the inputs are inference tensors. You should NOT call this "
           "function directly as native::_make_dual. Please use the dispatcher, "
           "i.e., at::_make_dual. Please file an issue if you come across this "
           "error otherwise."),
    "B3": ("Expected this method to only be reached in inference mode and when "
           "all the inputs are inference tensors. You should NOT call this "
           "method directly as native::_fw_primal. Please use the dispatcher, "
           "i.e., at::_fw_primal. Please file an issue if you come across this "
           "error otherwise."),
    "B4": ('InferenceMode::is_enabled() && self.is_inference() INTERNAL ASSERT '
           'FAILED at "/pytorch/aten/src/ATen/native/VariableMethodStubs.cpp":66,'
           ' please report a bug to PyTorch. '),
    "B5": ('InferenceMode::is_enabled() && primal.is_inference() && '
           'tangent.is_inference() INTERNAL ASSERT FAILED at '
           '"/pytorch/aten/src/ATen/native/AutogradComposite.cpp":23, please '
           'report a bug to PyTorch. '),
}

# The real H1 channel, transcribed from job_907959/srv_TD1.log:2592 (that log is
# outside version control -- size -- so the line is carried here).
OOM_907959 = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 198.00 MiB. "
    "GPU 0 has a total capacity of 79.25 GiB of which 55.19 MiB is free. "
    "Including non-PyTorch memory, this process has 79.19 GiB memory in use. Of "
    "the allocated memory 77.25 GiB is allocated by PyTorch, with 156.00 MiB "
    "allocated in private pools (e.g., CUDA Graphs), and 1014.25 MiB is reserved "
    "by PyTorch but unallocated."
)
# `ambiguous_autogradother_kernel` (KernelFunction.cpp:23) appends an
# `InferenceMode` HINT to an error that has nothing to do with inference
# tensors.  PyTorch puts a newline before the hint, so a line-oriented scan does
# not see both tokens on one line (Y5); a log writer that joins lines does (Y6).
AMBIG_HEAD = ("aten::foo has kernels registered to both CompositeImplicitAutograd "
              "and a backend mapped to AutogradOther.")
AMBIG_HINT = ("If you only want to run inference instead of training, in C++, add "
              "`c10::InferenceMode mode;` before model.forward(); in Python, use "
              "`torch.inference_mode()` as a context manager.")
ONNX_DECOY = "The exported ONNX inference model failed shape inference."

REGEXES = {
    # as registered in PREREG_RERUN_2026-09-13.md rev4 §6-5
    "REGISTERED_rev4": r"RuntimeError.*(?i:inference tensor)|RuntimeError.*InferenceMode",
    # as proposed by RA4-2 in VERDICT_rerun_rev4_2026-09-14.md:94
    "RA4_2_proposed": r"RuntimeError.*(?i:inference[ _](tensor|mode))|RuntimeError.*InferenceMode",
    # this probe's candidate: one alternative, POSIX-ERE-safe, \b-guarded
    "VERIFIED_B3": r"RuntimeError.*[Ii]nference[ _]?([Tt]ensors?|[Mm]ode)\b",
}


def build_corpus():
    rows = []
    for tag, fn in EXECUTED_POS:
        kind, line = _exc_line(fn)
        rows.append({"id": tag, "class": "POS", "provenance": "executed",
                     "exc": kind, "line": line})
    for tag, msg in BINARY_POS.items():
        rows.append({"id": tag, "class": "POS", "provenance": "binary-attested",
                     "exc": "builtins.RuntimeError (expected)",
                     "line": "RuntimeError: " + msg})
    for tag, fn in EXECUTED_NEG:
        kind, line = _exc_line(fn)
        rows.append({"id": tag, "class": "NEG", "provenance": "executed",
                     "exc": kind, "line": line})
    rows.append({"id": "Y4", "class": "NEG", "provenance": "job_907959 log",
                 "exc": "torch.OutOfMemoryError", "line": OOM_907959})
    rows.append({"id": "Y5", "class": "NEG", "provenance": "binary-attested",
                 "exc": "builtins.RuntimeError",
                 "line": "RuntimeError: " + AMBIG_HEAD})
    rows.append({"id": "Y6", "class": "NEG", "provenance": "binary-attested/joined",
                 "exc": "builtins.RuntimeError",
                 "line": "RuntimeError: " + AMBIG_HEAD + " " + AMBIG_HINT})
    rows.append({"id": "Y7", "class": "NEG", "provenance": "binary-attested",
                 "exc": "builtins.RuntimeError",
                 "line": "RuntimeError: " + ONNX_DECOY})
    return rows


def grep_count(flag, pattern, path):
    proc = subprocess.run(["/usr/bin/grep", "-c", flag, "-e", pattern, path],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return int(proc.stdout.strip() or 0)


def score(rows):
    pos = [r for r in rows if r["class"] == "POS"]
    neg = [r for r in rows if r["class"] == "NEG"]
    files = {}
    for name, group in (("pos", pos), ("neg", neg)):
        fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        fh.write("\n".join(r["line"] for r in group) + "\n")
        fh.close()
        files[name] = fh.name
    out = {}
    for name, pat in REGEXES.items():
        rx = re.compile(pat)
        out[name] = {
            "python_re": {
                "pos": sum(bool(rx.search(r["line"])) for r in pos),
                "neg": sum(bool(rx.search(r["line"])) for r in neg),
                "missed_pos": [r["id"] for r in pos if not rx.search(r["line"])],
                "fired_neg": [r["id"] for r in neg if rx.search(r["line"])],
            },
            "gnu_grep_E": {"pos": grep_count("-E", pat, files["pos"]),
                           "neg": grep_count("-E", pat, files["neg"])},
            "gnu_grep_P": {"pos": grep_count("-P", pat, files["pos"]),
                           "neg": grep_count("-P", pat, files["neg"])},
        }
    return out, len(pos), len(neg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rows = build_corpus()
    results, npos, nneg = score(rows)

    print(f"torch {torch.__version__}   positives={npos}  negatives={nneg}")
    print(f"grep: {subprocess.run(['/usr/bin/grep','--version'],capture_output=True,text=True).stdout.splitlines()[0]}")
    print("\nCORPUS")
    for r in rows:
        print(f"  {r['id']:3s} {r['class']} {r['provenance']:22s} {r['line'][:96]}")
    print(f"\n{'regex':17s} {'-E':>9s} {'-P':>9s} {'python':>9s}   (pos/neg)")
    for name, res in results.items():
        e, p, y = res["gnu_grep_E"], res["gnu_grep_P"], res["python_re"]
        print(f"{name:17s} {e['pos']}/{e['neg']:<7d} {p['pos']}/{p['neg']:<7d} "
              f"{y['pos']}/{y['neg']:<7d}  missed={y['missed_pos']} fired={y['fired_neg']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"torch": torch.__version__, "corpus": rows,
                       "regexes": REGEXES, "results": results}, fh,
                      indent=1, ensure_ascii=False)
    # fail-closed: the recommended form must be 12/12 under all three engines
    ok = all(results["VERIFIED_B3"][eng]["pos"] == npos
             for eng in ("gnu_grep_E", "gnu_grep_P", "python_re"))
    print("\nVERIFIED_B3 covers every positive under every engine:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
