# py3.14 compatibility shims for the sglang v0.5.10 + triton 3.5.1 stack.
# Both are FAITHFUL to py<=3.13 behavior (no semantic/perf change), so this env
# is valid for perf runs, not just a boot smoke.
import sys
if sys.version_info >= (3, 14):
    # (1) Restore ast.Num/Str/etc. aliases removed in py3.14. Triton 3.5.1
    #     (code_generator.py:1172/1174) constructs ast.Num(0); on py<=3.13 these
    #     were deprecated aliases returning ast.Constant. Faithful restore:
    #     ast.Num(x) -> ast.Constant(x). No change to generated kernels.
    import ast as _ast
    if not hasattr(_ast, "Num"):
        _ast.Num = _ast.Constant
        _ast.Str = _ast.Constant
        _ast.Bytes = _ast.Constant
        _ast.NameConstant = _ast.Constant
        _ast.Ellipsis = _ast.Constant
    # (2) torch.compile raises on py3.14 (torch 2.9.1). sglang defaults
    #     enable_torch_compile=False, so forcing eager is perf-neutral for our runs.
    try:
        import torch
        def _noop_compile(model=None, *args, **kwargs):
            if model is None:
                return lambda f: f
            return model
        torch.compile = _noop_compile
    except Exception:
        pass
