# multigpu_probe SUMMARY (facts only; no performance verdict, no substrate-equivalence verdict)

| stage | status | reason |
|---|---|---|
| F1 | PASS | GPUs visible: 8; classes: {} |
| F2 | PASS | all-reduce measured: 2(torch.distributed) 4(torch.distributed) 8(torch.distributed) (descriptor, see f2/) |
| F3 | PASS | peer-access matrix + copy GB/s + small-message latency recorded (f3/) |
| F4 | PASS | all pairs created on all 8/8 GPUs (created != realized; census TODO) |
| F5 | PASS | infiniband devices: 0; mooncake absent (descriptor; RDMA usability not tested) |

TODO (needs pre-registration): per-GPU 1-GPU serving-cell deviation (same cell on each GPU).
