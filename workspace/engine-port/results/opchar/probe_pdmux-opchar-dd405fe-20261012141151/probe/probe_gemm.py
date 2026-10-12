
import torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
print("gemm ok")
