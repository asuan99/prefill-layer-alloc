"""
T5 smoke: verify the EXACT upstream sgl_kernel.spatial green-ctx path works on
A100-SXM4-80GB / CUDA 13. Two checks, matching pdmux_context.py usage:
  1) get_sm_available (pure-python torch path) -> expect 108 on A100.
  2) create_greenctx_stream_by_value (compiled greenctx_stream.cu, JIT-built here)
     -> split 108 SM into a prefill/decode partition pair, like divide_sm().

This does NOT boot a full server; it validates the SM-control mechanism (the
riskiest layer). No venv/site-packages pollution: JIT build goes to a scratch dir.
"""
import os, sys, json, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
result = {"phase": "start", "checks": {}}

def emit(**kw):
    result.update(kw)

try:
    import torch
    from torch.utils.cpp_extension import load
    emit(torch_version=torch.__version__, cuda_version=torch.version.cuda)

    assert torch.cuda.is_available(), "CUDA not available in this job"
    dev = 0
    props = torch.cuda.get_device_properties(dev)
    sm = props.multi_processor_count
    cc = torch.cuda.get_device_capability(dev)
    result["checks"]["device"] = props.name
    result["checks"]["compute_capability"] = f"{cc[0]}.{cc[1]}"

    # ---- Check 1: get_sm_available (pure-python path, identical to spatial.get_sm_available) ----
    result["checks"]["get_sm_available"] = int(sm)
    result["checks"]["get_sm_available_ok"] = (sm == 108)
    print(f"[check1] device={props.name} cc={cc} SM_available={sm} (expect 108)")

    # ---- Check 2: JIT-build the REAL greenctx_stream.cu and create a partition pair ----
    print("[check2] JIT-compiling real upstream greenctx_stream.cu ...")
    t0 = time.time()
    ext = load(
        name="spatial_greenctx_t5",
        sources=[os.path.join(HERE, "greenctx_stream.cu"),
                 os.path.join(HERE, "spatial_reg.cc")],
        extra_include_paths=[HERE],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )
    result["checks"]["jit_build_sec"] = round(time.time() - t0, 1)
    print(f"[check2] build ok in {result['checks']['jit_build_sec']}s")

    # Emulate divide_sm() arch-constraint for cc8 (A100): multiple=2 -> pick 64/44.
    smA, smB = 64, 44  # prefill_sm, decode_sm ; 64+44=108
    res = ext.create_greenctx_stream_by_value(smA, smB, dev)
    # returns [streamA_ptr, streamB_ptr, smCountA, smCountB]
    result["checks"]["requested_split"] = [smA, smB]
    result["checks"]["greenctx_return"] = [int(x) for x in res]
    result["checks"]["actual_smA"] = int(res[2])
    result["checks"]["actual_smB"] = int(res[3])
    result["checks"]["streamA_nonzero"] = bool(res[0] != 0)
    result["checks"]["streamB_nonzero"] = bool(res[1] != 0)
    result["checks"]["greenctx_ok"] = bool(res[0] != 0 and res[1] != 0 and res[2] > 0 and res[3] > 0)
    print(f"[check2] requested={smA}/{smB} -> actual smA={res[2]} smB={res[3]} "
          f"streamA={hex(res[0])} streamB={hex(res[1])}")

    # ---- Check 3: run a trivial kernel on each partition stream to prove usability ----
    sA = torch.cuda.ExternalStream(stream_ptr=res[0], device=torch.device(f"cuda:{dev}"))
    sB = torch.cuda.ExternalStream(stream_ptr=res[1], device=torch.device(f"cuda:{dev}"))
    x = torch.randn(4096, 4096, device=f"cuda:{dev}")
    with torch.cuda.stream(sA):
        yA = x @ x
    with torch.cuda.stream(sB):
        yB = x @ x
    torch.cuda.synchronize()
    result["checks"]["matmul_on_partitions_ok"] = bool(torch.isfinite(yA).all() and torch.isfinite(yB).all())
    print("[check3] matmul on both partition streams: ok")

    result["phase"] = "success"
except Exception as e:
    result["phase"] = "error"
    result["error"] = repr(e)
    result["traceback"] = traceback.format_exc()
    print("ERROR:", e)
    traceback.print_exc()

with open(os.path.join(HERE, "t5_result.json"), "w") as f:
    json.dump(result, f, indent=2)
print("\n=== T5 RESULT ===")
print(json.dumps(result, indent=2))
sys.exit(0 if result.get("phase") == "success" else 1)
