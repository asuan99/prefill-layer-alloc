"""
run_chunked_ssm_sweep.py

기존 run_ssm_prefill_sweep.py를 대체하는 chunked prefill 방식 SSM sweep 스크립트.

Chunked prefill 방식으로 mamba_chunk_scan_combined를 실행해 cooperative barrier
deadlock 없이 Green Context SM 제한 하에서 직접 latency를 측정한다.

사용법:
    # Step 1: cooperative 안전성 확인 (소규모 테스트)
    python stage1_sm_scaling/run_chunked_ssm_sweep.py \\
        --model zamba2 \\
        --device a100-sxm4-80gb \\
        --prefill-chunk-tokens 256 512 1024 \\
        --sm-counts 14 27 54 108 \\
        --seq-lens 1024 2048 \\
        --batch-sizes 1 4 \\
        --n-warmup 2 --n-measure 5 \\
        --output-dir results/stage1/chunked/

    # Step 2: 전체 sweep
    python stage1_sm_scaling/run_chunked_ssm_sweep.py \\
        --model zamba2 \\
        --device a100-sxm4-80gb \\
        --prefill-chunk-tokens 512 1024 2048 4096 \\
        --sm-counts 14 27 40 54 68 81 94 108 \\
        --seq-lens 256 512 1024 2048 4096 8192 \\
        --batch-sizes 1 4 16 32 \\
        --n-warmup 3 --n-measure 10 \\
        --output-dir results/stage1/chunked/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import math
from pathlib import Path

import torch

from src.smctrl.green_ctx_controller import SMController
from src.hardware_config import get_hardware_config
from stage1_sm_scaling.chunked_ssm_runner import run_chunked_ssm_sweep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Chunked prefill SSM SM sweep — cooperative barrier 우회 방식으로 "
            "Green Context 하에서 mamba_chunk_scan_combined를 직접 측정."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",   default="zamba2",
                        choices=["zamba2", "falcon_h1"],
                        help="모델 이름 (configs/models.yaml 키)")
    parser.add_argument("--device",  default="auto",
                        help="hardware config 태그 (hardware.yaml). 예: a100-sxm4-80gb")
    parser.add_argument(
        "--prefill-chunk-tokens", nargs="+", type=int,
        default=[512, 1024, 2048, 4096],
        metavar="PCT",
        help=(
            "kernel 호출당 토큰 수 sweep (핵심 독립 변수). "
            "cooperative 안전 조건: batch × (pct//256) × n_heads ≤ sm_count. "
            "Zamba2 bs=1: pct ≤ sm_count × 256 / 112."
        ),
    )
    parser.add_argument(
        "--sm-counts", nargs="+", type=int,
        default=[14, 27, 40, 54, 68, 81, 94, 108],
        help="측정할 SM 수 목록",
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[512, 1024, 2048, 4096],
        help="시퀀스 길이 목록",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 4, 16],
        help="배치 크기 목록",
    )
    parser.add_argument("--n-warmup",  type=int, default=3)
    parser.add_argument("--n-measure", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/stage1/chunked/"),
        help="CSV 출력 디렉토리",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="SM control 검증 skip (빠른 테스트용)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    hw_cfg  = get_hardware_config(args.device)
    total_sm = hw_cfg["sm_count"]

    print(f"\n=== Chunked SSM Prefill SM Sweep ===")
    print(f"  model              : {args.model}")
    print(f"  device             : {hw_cfg['name']}  ({total_sm} SM)")
    print(f"  prefill_chunk_tokens: {args.prefill_chunk_tokens}")
    print(f"  sm_counts          : {args.sm_counts}")
    print(f"  seq_lens           : {args.seq_lens}")
    print(f"  batch_sizes        : {args.batch_sizes}")
    print()

    smctrl = SMController(
        device_id=0,
        total_sm_count=total_sm,
        preset_sm_counts=args.sm_counts,
    )

    if not args.skip_verify:
        print("SM control 검증 중…")
        if not smctrl.verify_sm_control(verbose=True):
            raise RuntimeError(
                "SMController 검증 실패. Green Context가 실제로 SM을 제한하는지 확인:\n"
                "  latency(25% SM) / latency(100% SM) >= 2.0이어야 함\n"
                "  현재 flat하다면 driver가 Green Context를 지원하지 않는 환경임"
            )
        print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "device", "seq_len", "batch_size",
        "sm_count", "sm_ratio_pct",
        "prefill_chunk_tokens", "n_kernel_calls",
        "n_blocks_per_call", "cooperative_safe",
        "latency_ms", "latency_std_ms",
    ]

    # device 태그: 공백/특수문자 제거
    dev_tag = hw_cfg["name"].lower().replace("nvidia ", "").replace(" ", "-")
    output_file = args.output_dir / f"ssm_chunked_{args.model}_{dev_tag}.csv"

    # sweep 조합 수 계산
    total = sum(
        1
        for pct in args.prefill_chunk_tokens
        for sm  in args.sm_counts
        for seq in args.seq_lens
        if seq >= pct
        for _  in args.batch_sizes
    )

    done = 0
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pct in args.prefill_chunk_tokens:
            for sm in args.sm_counts:
                for seq in args.seq_lens:
                    if seq < pct:
                        # 청크 크기가 전체 시퀀스보다 크면 의미 없음 — skip
                        continue
                    for bs in args.batch_sizes:
                        done += 1
                        print(
                            f"[{done}/{total}] pct={pct:5d} sm={sm:3d} "
                            f"seq={seq:6d} bs={bs:2d} …",
                            end=" ",
                            flush=True,
                        )

                        try:
                            result = run_chunked_ssm_sweep(
                                model_name=args.model,
                                seq_len=seq,
                                batch_size=bs,
                                prefill_chunk_tokens=pct,
                                sm_count=sm,
                                smctrl=smctrl,
                                n_warmup=args.n_warmup,
                                n_measure=args.n_measure,
                            )
                            print(
                                f"lat={result['latency_ms']:.2f}ms  "
                                f"safe={result['cooperative_safe']}"
                            )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print("OOM — skipped")
                            result = {
                                "latency_ms":           float("nan"),
                                "latency_std_ms":       float("nan"),
                                "n_kernel_calls":       math.ceil(seq / pct),
                                "n_blocks_per_call":    -1,
                                "cooperative_safe":     False,
                            }
                        except RuntimeError as e:
                            print(f"ERROR: {e}")
                            result = {
                                "latency_ms":           float("nan"),
                                "latency_std_ms":       float("nan"),
                                "n_kernel_calls":       math.ceil(seq / pct),
                                "n_blocks_per_call":    -1,
                                "cooperative_safe":     False,
                            }

                        row = {
                            "model":                args.model,
                            "device":               dev_tag,
                            "seq_len":              seq,
                            "batch_size":           bs,
                            "sm_count":             sm,
                            "sm_ratio_pct":         round(sm / total_sm * 100, 1),
                            "prefill_chunk_tokens": pct,
                            "n_kernel_calls":       result["n_kernel_calls"],
                            "n_blocks_per_call":    result["n_blocks_per_call"],
                            "cooperative_safe":     result["cooperative_safe"],
                            "latency_ms":           result["latency_ms"],
                            "latency_std_ms":       result["latency_std_ms"],
                        }
                        writer.writerow(row)
                        f.flush()

    print(f"\nDone. 결과 저장: {output_file}")


if __name__ == "__main__":
    main()
