"""
DEPRECATED — 이 파일은 deprecated/run_concurrent_eval.py 로 이동됨.

폐기 사유:
  1. decode 가 seq_len=1 SSM 1개 / KV read 없어 Problem 4 (decode SM under-use)를 자기 부정.
  2. TTFT 가 wall-clock 합성값이라 측정값이 아님.
  3. 단일 스트림 순차 interleaving — 진짜 kernel-level 동시 실행 아님.

상세 사유는 deprecated/run_concurrent_eval.py docstring 참조.

대안:
  - 진짜 concurrent execution: stage3_hm_eval/coexec_microbench.py
  - 이득 projection:           stage3_hm_eval/projection_analysis.py
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_deprecated = os.path.join(_here, "deprecated")

print("=" * 70)
print("  ERROR: run_concurrent_eval.py 는 DEPRECATED 입니다.")
print("  이 파일을 직접 실행하지 마십시오.")
print()
print("  폐기 사유:")
print("    1. decode seq_len=1 SSM — KV read 없어 Problem 4 전제 자기 부정")
print("    2. TTFT = wall-clock 합성값, CUDA event 측정치 아님")
print("    3. 단일 스트림 순차 실행 — spatial sharing 없음")
print()
print("  대안:")
print("    python stage3_hm_eval/coexec_microbench.py   (실측 concurrent overlap)")
print("    python stage3_hm_eval/projection_analysis.py (이득 projection)")
print()
print("  원본 코드 보관: stage3_hm_eval/deprecated/run_concurrent_eval.py")
print("=" * 70)

sys.exit(1)
