"""Print the FULL traceback of decode_ssm (ssm-scan seq_len=1) for a model."""
import sys, os, traceback
sys.path.insert(0, os.path.abspath(".."))
from experiments.common import kernels as K
model = sys.argv[1] if len(sys.argv) > 1 else "zamba2_7b"
cfg = K.load_layer_cfg(model)
print(f"[diag] {model} cfg: n_heads={cfg['n_heads']} head_dim={cfg['head_dim']} "
      f"d_state={cfg['d_state']} n_groups={cfg['n_groups']} chunk_size={cfg['chunk_size']}")
for b in (1, 8, 64):
    try:
        fn, *_ = K.build_decode_ssm_fn(cfg, batch=b)
        fn()
        import torch; torch.cuda.synchronize()
        print(f"[diag] decode_ssm b={b}: OK")
    except Exception as e:
        print(f"[diag] decode_ssm b={b}: {type(e).__name__}: {e}")
        traceback.print_exc()
        break
