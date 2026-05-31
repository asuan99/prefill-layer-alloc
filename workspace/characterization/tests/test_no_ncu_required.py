"""
test_no_ncu_required.py — 헤드라인 산출물이 ncu CSV 없이 완결됨을 검증.

검증 전략:
  1. grep: 헤드라인 스크립트가 ncu_runner / NCURunner 를 import 하지 않음.
     (직접 import 하면 ncu 바이너리 탐색이 일어나 runtime 의존성 발생)
  2. grep: 헤드라인 플롯 스크립트가 기본 입력으로 ncu_*.csv 를 glob 하지 않음.
  3. 임포트 테스트: 헤드라인 스크립트들이 ncu/CUPTI 없이 import 가능.
  4. demo 모드: plot_motivation.py --demo 가 figure 를 생성하고 종료.

헤드라인 스크립트 목록 (ncu 없이 완결돼야 함):
  - stage1_sm_scaling/plot_saturation.py       (free-SM zone CSV/figure)
  - stage2_overhead/compute_decision_matrix.py  (decision matrix)
  - serving-eval/plot_motivation.py             (Fig A/B/C, counter-free)
  - stage3_hm_eval/coexec_microbench.py         (concurrent execution sweep)
  - stage3_hm_eval/projection_analysis.py       (PROJECTION figure)

Non-headline (optional, may import ncu_runner):
  - stage1_sm_scaling/run_ncu_profile.py        (OPTIONAL ENRICHMENT)
  - src/profiling/ncu_runner.py
  - src/profiling/_ncu_target.py
  - src/profiling/cupti_monitor.py              (ncu 대안, OPTIONAL)
"""

from __future__ import annotations

import ast
import subprocess
import sys
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Workspace root
# ---------------------------------------------------------------------------

_WORKSPACE = Path(__file__).parent.parent.parent  # /workspace/
_CHAR      = _WORKSPACE / "characterization"
_SERVING   = _WORKSPACE / "serving-eval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports_ncu(path: Path) -> list[str]:
    """Return list of ncu-related import statements found in the file."""
    hits = []
    try:
        tree = ast.parse(_source(path))
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            src = ast.unparse(node)
            if "ncu_runner" in src or "NCURunner" in src or "cupti_monitor" in src:
                hits.append(src)

    return hits


def _globs_ncu_csv(path: Path) -> list[str]:
    """Return lines that unconditionally glob for ncu_*.csv as a primary required input.

    Lines with the comment marker '# ncu_optional' are excluded — those globs are
    already guarded by 'if ncu_files:' and produce empty output when ncu is absent.
    """
    hits = []
    source = _source(path)
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if 'glob("ncu_' in stripped or "glob('ncu_" in stripped:
            # Skip lines explicitly marked as optional enrichment
            if "ncu_optional" in line or "OPTIONAL" in line:
                continue
            hits.append(f"line {i}: {stripped}")
    return hits


# ---------------------------------------------------------------------------
# Headline scripts
# ---------------------------------------------------------------------------

_HEADLINE_SCRIPTS = [
    _CHAR / "stage1_sm_scaling" / "plot_saturation.py",
    _CHAR / "stage2_overhead" / "compute_decision_matrix.py",
    _SERVING / "plot_motivation.py",
    _CHAR / "stage3_hm_eval" / "coexec_microbench.py",
    _CHAR / "stage3_hm_eval" / "projection_analysis.py",
]

# Scripts that ARE allowed to import ncu_runner (optional enrichment)
_OPTIONAL_NCU_SCRIPTS = [
    _CHAR / "stage1_sm_scaling" / "run_ncu_profile.py",
    _CHAR / "src" / "profiling" / "ncu_runner.py",
    _CHAR / "src" / "profiling" / "_ncu_target.py",
    _CHAR / "src" / "profiling" / "cupti_monitor.py",
    _SERVING / "profile_layer_heterogeneity.py",
    _SERVING / "plot_layer_heterogeneity.py",
]

_OPTIONAL_NCU_NAMES = {p.name for p in _OPTIONAL_NCU_SCRIPTS}


# ---------------------------------------------------------------------------
# Test 1: headline scripts do NOT import ncu_runner or NCURunner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", _HEADLINE_SCRIPTS, ids=[p.name for p in _HEADLINE_SCRIPTS])
def test_headline_script_no_ncu_import(script: Path) -> None:
    """Headline script must not import ncu_runner / NCURunner / cupti_monitor."""
    assert script.exists(), f"Script not found: {script}"
    hits = _imports_ncu(script)
    assert not hits, (
        f"{script.name} imports ncu-related modules — headline scripts must be "
        f"counter-free.\nFound:\n  " + "\n  ".join(hits)
    )


# ---------------------------------------------------------------------------
# Test 2: headline plot scripts do not glob ncu_*.csv as default required input
# ---------------------------------------------------------------------------

_HEADLINE_PLOT_SCRIPTS = [
    _CHAR / "stage1_sm_scaling" / "plot_saturation.py",
    _SERVING / "plot_motivation.py",
    _CHAR / "stage3_hm_eval" / "projection_analysis.py",
]


@pytest.mark.parametrize(
    "script", _HEADLINE_PLOT_SCRIPTS,
    ids=[p.name for p in _HEADLINE_PLOT_SCRIPTS]
)
def test_headline_plot_no_ncu_csv_glob(script: Path) -> None:
    """Headline plot scripts must not glob for ncu_*.csv as a primary input."""
    assert script.exists(), f"Script not found: {script}"
    hits = _globs_ncu_csv(script)
    assert not hits, (
        f"{script.name} globs for ncu_*.csv — this creates an implicit ncu dependency.\n"
        "Found:\n  " + "\n  ".join(hits)
    )


# ---------------------------------------------------------------------------
# Test 3: optional scripts ARE marked as optional in their docstring
# ---------------------------------------------------------------------------

_OPTIONAL_KEYWORDS = ["OPTIONAL", "optional enrichment", "critical path ではない",
                      "critical path 아님", "OPTIONAL ENRICHMENT"]


@pytest.mark.parametrize(
    "script", _OPTIONAL_NCU_SCRIPTS,
    ids=[p.name for p in _OPTIONAL_NCU_SCRIPTS]
)
def test_optional_ncu_script_marked(script: Path) -> None:
    """Optional ncu scripts must state 'OPTIONAL' or equivalent in their docstring."""
    if not script.exists():
        pytest.skip(f"{script.name} not found — skip")

    source = _source(script)
    # Check first 50 lines (docstring area)
    header = "\n".join(source.splitlines()[:50])
    found = any(kw.lower() in header.lower() for kw in _OPTIONAL_KEYWORDS)
    assert found, (
        f"{script.name} is an optional/ncu script but its header does not contain\n"
        f"any of {_OPTIONAL_KEYWORDS}.\n"
        "Add 'OPTIONAL ENRICHMENT' or 'OPTIONAL' at the top of the docstring."
    )


# ---------------------------------------------------------------------------
# Test 4: projection_analysis.py --demo produces output without real data
# ---------------------------------------------------------------------------

def test_projection_demo_runs() -> None:
    """projection_analysis.py --demo must complete without error."""
    script = _CHAR / "stage3_hm_eval" / "projection_analysis.py"
    if not script.exists():
        pytest.skip("projection_analysis.py not found")

    result = subprocess.run(
        [sys.executable, str(script), "--demo",
         "--output-dir", "/tmp/test_projection_demo"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"projection_analysis.py --demo failed (exit {result.returncode}):\n"
        f"STDOUT: {result.stdout[-1000:]}\n"
        f"STDERR: {result.stderr[-500:]}"
    )
    # Check that at least one PROJECTION file was created
    out_dir = Path("/tmp/test_projection_demo")
    proj_files = list(out_dir.glob("projection_*.png")) + list(out_dir.glob("projection_*.txt"))
    assert proj_files, (
        "projection_analysis.py --demo ran but produced no projection_*.{png,txt} files"
    )


# ---------------------------------------------------------------------------
# Test 5: plot_motivation.py --demo produces fig B and fig C without ncu
# ---------------------------------------------------------------------------

def test_motivation_demo_no_ncu() -> None:
    """plot_motivation.py --demo must generate figures without ncu/GPU counters."""
    script = _SERVING / "plot_motivation.py"
    if not script.exists():
        pytest.skip("plot_motivation.py not found")

    result = subprocess.run(
        [sys.executable, str(script), "--demo",
         "--output-dir", "/tmp/test_motivation_demo"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"plot_motivation.py --demo failed (exit {result.returncode}):\n"
        f"STDOUT: {result.stdout[-1000:]}\n"
        f"STDERR: {result.stderr[-500:]}"
    )
    out_dir = Path("/tmp/test_motivation_demo")
    figs = list(out_dir.glob("fig_*.png"))
    assert len(figs) >= 2, (
        f"Expected >= 2 fig_*.png from --demo, got {len(figs)}: {[f.name for f in figs]}"
    )
