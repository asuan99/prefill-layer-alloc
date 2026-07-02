// Minimal pybind wrapper around the REAL, unmodified upstream greenctx_stream.cu
// (sha256 de20703f...). Avoids pulling sgl_kernel_ops.h; exposes the exact
// create_greenctx_stream_by_value() used by SGLang's srt/multiplex/pdmux_context.py.
#include <torch/extension.h>

#include "greenctx_stream.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_greenctx_stream_by_value", &create_greenctx_stream_by_value,
        "Real upstream sgl_kernel.spatial green-ctx stream creator (smA, smB, device)");
}
