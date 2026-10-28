# LOCAL CPU TEST STUB ONLY -- never installed into an engine tree, never on a GPU path.
# Fakes the GPU-only `sgl_kernel` package (absent / ABI-broken on the local sm_120 machine) so that the real
# `sglang.srt.multiplex.multiplexing_mixin` and its event loop can be IMPORTED on CPU.  Any attribute of the fake
# package is an inert callable; a test that actually needed a kernel would get a dummy object, so the stub is only
# valid for tests that drive the loop with fakes (pdmux_loop_fakes).  Used by run_diag_tests_with_stub.sh.
import importlib.abc
import importlib.machinery
import sys
import types

_STUB_ROOTS = ("sgl_kernel",)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Dummy()

    def __getattr__(self, n):
        return _Dummy()

    def __mro_entries__(self, bases):
        return (object,)


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Finder())
