"""
analyze_chunk_size.py

run_chunked_ssm_sweep.py 결과 CSV를 읽어서:
  1. cooperative_safe=False인 케이스 확인
  2. 같은 (seq_len, batch_size, sm_count)에서 prefill_chunk_tokens별 latency 비교
  3. SM scaling curve 확인 (chunked prefill의 실측 sm→latency 관계)
  4. 최적 prefill_chunk_tokens 권장

사용법:
    python stage1_sm_scaling/analyze_chunk_size.py \\
        --csv results/stage1/chunked/ssm_chunked_zamba2_a100-sxm4-80gb.csv

    # 기존 wave-model SSM CSV와 비교
    python stage1_sm_scaling/analyze_chunk_size.py \\
        --csv results/stage1/chunked/ssm_chunked_zamba2_a100-sxm4-80gb.csv \\
        --wave-csv results/stage1/ssm_scaling_zamba2_a100-sxm4-80gb.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _fmt_df(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_string(index=False)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def report_unsafe(df: pd.DataFrame) -> None:
    _print_section("cooperative_safe=False 케이스")
    unsafe = df[~df["cooperative_safe"].astype(bool)]
    if len(unsafe) == 0:
        print("  없음 — 모든 케이스가 cooperative 조건 충족")
        return
    print(f"  총 {len(unsafe)}개 케이스:")
    cols = ["seq_len", "batch_size", "sm_count", "prefill_chunk_tokens",
            "n_blocks_per_call", "latency_ms"]
    print(_fmt_df(unsafe, [c for c in cols if c in unsafe.columns]))


def report_chunk_overhead(
    df: pd.DataFrame,
    ref_seq: int = 2048,
    ref_bs:  int = 4,
    ref_sm:  int = 108,
) -> None:
    """prefill_chunk_tokens별 overhead 분석 (full SM 기준)."""
    _print_section(
        f"prefill_chunk_tokens별 kernel launch overhead "
        f"(seq={ref_seq}, bs={ref_bs}, sm={ref_sm} 기준)"
    )

    sub = df[
        (df["seq_len"]    == ref_seq) &
        (df["batch_size"] == ref_bs)  &
        (df["sm_count"]   == ref_sm)
    ].sort_values("prefill_chunk_tokens")

    if sub.empty:
        print(f"  조건에 맞는 데이터 없음 (seq={ref_seq}, bs={ref_bs}, sm={ref_sm})")
        return

    # baseline: 가장 큰 pct (단일 kernel에 가장 가까움)
    baseline_row = sub[sub["prefill_chunk_tokens"] == sub["prefill_chunk_tokens"].max()]
    baseline_lat = baseline_row["latency_ms"].values[0] if len(baseline_row) > 0 else None

    print(f"  {'pct':>6} {'n_calls':>8} {'latency_ms':>12} {'overhead':>10} {'safe':>6}")
    for _, row in sub.iterrows():
        pct     = int(row["prefill_chunk_tokens"])
        n_calls = int(row["n_kernel_calls"]) if "n_kernel_calls" in row.index else ref_seq // pct
        lat     = row["latency_ms"]
        safe    = bool(row["cooperative_safe"])
        if baseline_lat and not np.isnan(baseline_lat) and not np.isnan(lat):
            overhead = (lat - baseline_lat) / baseline_lat * 100
            overhead_str = f"{overhead:+.1f}%"
        else:
            overhead_str = "N/A"
        print(f"  {pct:>6} {n_calls:>8} {lat:>12.3f} {overhead_str:>10} {str(safe):>6}")


def report_sm_scaling(
    df: pd.DataFrame,
    ref_pct: int = 512,
    ref_seq: int = 2048,
    ref_bs:  int = 4,
) -> None:
    """SM scaling curve 실측값 (chunked prefill)."""
    _print_section(
        f"SM scaling curve — 실측 (pct={ref_pct}, seq={ref_seq}, bs={ref_bs})"
    )

    sub = df[
        (df["prefill_chunk_tokens"] == ref_pct) &
        (df["seq_len"]    == ref_seq) &
        (df["batch_size"] == ref_bs)
    ].sort_values("sm_count")

    if sub.empty:
        print(f"  조건에 맞는 데이터 없음 (pct={ref_pct}, seq={ref_seq}, bs={ref_bs})")
        return

    full_sm_rows = sub[sub["sm_count"] == sub["sm_count"].max()]["latency_ms"].values
    full_sm_lat  = full_sm_rows[0] if len(full_sm_rows) > 0 else None

    print(f"  {'sm':>5} {'sm_ratio':>10} {'latency_ms':>12} {'throughput':>12} {'safe':>6}")
    for _, row in sub.iterrows():
        sm_cnt = int(row["sm_count"])
        lat    = row["latency_ms"]
        safe   = bool(row["cooperative_safe"])
        sm_rat = row.get("sm_ratio_pct", sm_cnt)
        if full_sm_lat and not np.isnan(full_sm_lat) and not np.isnan(lat) and lat > 0:
            tp = full_sm_lat / lat
            tp_str = f"{tp:.3f}x"
        else:
            tp_str = "N/A"
        print(f"  {sm_cnt:>5} {sm_rat:>9.1f}% {lat:>12.3f} {tp_str:>12} {str(safe):>6}")


def report_recommendation(df: pd.DataFrame) -> None:
    """최적 prefill_chunk_tokens 권장."""
    _print_section("권장 prefill_chunk_tokens")

    safe_df = df[df["cooperative_safe"].astype(bool) & df["latency_ms"].notna()]
    if safe_df.empty:
        print("  cooperative_safe=True인 케이스가 없음 — pct를 줄이거나 sm_count를 높여 재시도")
        return

    min_pct = int(safe_df["prefill_chunk_tokens"].min())
    print(f"  최소 안전 prefill_chunk_tokens: {min_pct}")
    print()
    print(
        "  cooperative 안전 조건 공식:\n"
        "    n_blocks_per_call = batch × (pct // ssd_chunk_size) × n_heads ≤ sm_count\n"
        "    (A100 max_blocks_per_sm = 1 가정, conservative)\n"
    )

    # 모델별 권장값 계산 (Zamba2 기준)
    n_heads   = 112   # Zamba2 n_mamba_heads
    ssd_chunk = 256

    print("  Zamba2 (n_heads=112, ssd_chunk=256) 기준 권장 pct:")
    print(f"    {'sm_count':>10} {'batch':>6} {'max_safe_pct':>14}")
    for sm in sorted(df["sm_count"].unique()):
        for bs in sorted(df["batch_size"].unique()):
            max_pct = int(sm * ssd_chunk / bs)  # max_blocks_per_sm=1 가정
            # pct는 ssd_chunk의 배수여야 함
            max_pct = (max_pct // ssd_chunk) * ssd_chunk
            if max_pct >= ssd_chunk:
                print(f"    {int(sm):>10} {int(bs):>6} {max_pct:>14}")


def compare_with_wave_model(
    chunked_df: pd.DataFrame,
    wave_csv: Path,
    ref_pct:  int = 512,
) -> None:
    """Chunked prefill 실측값 vs wave-model 합성값 비교."""
    _print_section("Chunked prefill 실측 vs Wave-model 합성값 비교")

    try:
        wave_df = pd.read_csv(wave_csv)
    except Exception as e:
        print(f"  wave CSV 로드 실패: {e}")
        return

    # chunked_df: prefill_chunk_tokens=ref_pct, cooperative_safe=True
    ch = chunked_df[
        (chunked_df["prefill_chunk_tokens"] == ref_pct) &
        chunked_df["cooperative_safe"].astype(bool)
    ][["seq_len", "batch_size", "sm_count", "latency_ms"]].rename(
        columns={"latency_ms": "lat_chunked"}
    )

    wv = wave_df[["seq_len", "batch_size", "sm_count", "latency_ms"]].rename(
        columns={"latency_ms": "lat_wave"}
    )

    merged = ch.merge(wv, on=["seq_len", "batch_size", "sm_count"], how="inner")
    if merged.empty:
        print("  매칭되는 (seq_len, batch_size, sm_count) 쌍 없음")
        return

    merged["error_pct"] = (
        (merged["lat_chunked"] - merged["lat_wave"]).abs()
        / merged["lat_wave"].clip(lower=1e-9) * 100
    )
    rmse = np.sqrt(((merged["lat_chunked"] - merged["lat_wave"]) ** 2).mean())
    mape = merged["error_pct"].mean()

    print(f"  매칭 포인트: {len(merged)}개")
    print(f"  RMSE: {rmse:.3f} ms")
    print(f"  MAPE: {mape:.2f}%")
    print()

    # 가장 큰 오차 상위 5개
    worst = merged.nlargest(5, "error_pct")
    print("  최대 오차 상위 5개:")
    print(_fmt_df(
        worst,
        [c for c in ["seq_len", "batch_size", "sm_count",
                     "lat_chunked", "lat_wave", "error_pct"]
         if c in worst.columns]
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="run_chunked_ssm_sweep.py 결과 CSV 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", required=True, type=Path,
                        help="chunked sweep 결과 CSV")
    parser.add_argument("--wave-csv", type=Path, default=None,
                        help="기존 wave-model SSM CSV (비교용, 선택 사항)")
    parser.add_argument("--ref-pct",  type=int, default=512,
                        help="SM scaling curve 표시에 사용할 prefill_chunk_tokens (기본: 512)")
    parser.add_argument("--ref-seq",  type=int, default=2048,
                        help="overhead 분석 기준 seq_len (기본: 2048)")
    parser.add_argument("--ref-bs",   type=int, default=4,
                        help="overhead 분석 기준 batch_size (기본: 4)")
    parser.add_argument("--ref-sm",   type=int, default=108,
                        help="overhead 분석 기준 sm_count (기본: 108)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n분석 대상: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  총 {len(df)}행  |  pct={sorted(df['prefill_chunk_tokens'].unique())}  "
          f"|  sm={sorted(df['sm_count'].unique())}")

    report_unsafe(df)
    report_chunk_overhead(df, ref_seq=args.ref_seq, ref_bs=args.ref_bs, ref_sm=args.ref_sm)
    report_sm_scaling(df, ref_pct=args.ref_pct, ref_seq=args.ref_seq, ref_bs=args.ref_bs)
    report_recommendation(df)

    if args.wave_csv:
        compare_with_wave_model(df, args.wave_csv, ref_pct=args.ref_pct)

    print("\n완료.")


if __name__ == "__main__":
    main()
