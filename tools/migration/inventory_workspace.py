#!/usr/bin/env python3
"""Inventory every file under the workspace root that is NOT tracked by git.

Walks the whole workspace (default /scratch/ehmoon/whlee), removes files tracked
by any git repository living inside it, and assigns every remaining file to one
rule in RULES (first match wins).  Each rule carries a decision (what to do with
the file when leaving this machine) and where the migration bundle stores it.

Outputs (into --out):
  untracked_files.tsv.gz  path, bytes, rule, decision, bundle   (one row per file)
  summary.tsv             per-rule file count and bytes
  hf_models.tsv           per Hub repo: bytes, revision
A rule named UNCLASSIFIED must end up empty; anything there needs a human look.

Written for the 2026-09-17 migration (handoff-report/session_handoff_2026-09-17.md).
"""
import argparse
import collections
import gzip
import os
import re
import subprocess
import sys

MUST, RECOMMENDED, OPTIONAL, SKIP = "필수", "권장", "선택", "불필요"
PLA = "prefill-layer-alloc"
EXT = f"{PLA}/workspace/engine-port/external"
CACHE_DIR = r"(^|/)([^/]*triton[^/]*cache[^/]*|[^/]*cache[^/]*triton[^/]*|__pycache__|\.pytest_cache)/"

# (name, regex on workspace-relative path, decision, bundle location, reason)
RULES = [
    ("migration_bundle_itself", r"^_migration_[0-9-]+/", SKIP, "-", "이 번들 자체"),
    ("pla_git_dir", rf"^{PLA}/\.git/", MUST, "A_git (bundle)", "git 이력 - bundle로 저장, config(토큰)은 제외"),
    ("build_caches", CACHE_DIR, SKIP, "-", "triton/python 캐시, 재생성"),
    ("pla_results_untracked", rf"^{PLA}/workspace/engine-port/results/", MUST, "C_results_untracked", "원시 telemetry·로그, GPU 재실행 외 복구 불가"),
    ("hf_hub_models", rf"^{PLA}/hf_cache/hub/models--", SKIP, "D_misc/hf_hub_revisions.tsv (리비전만)", "제외 - 사용자 결정(2026-09-17): 다른 장치에서 재다운로드"),
    ("hf_hub_datasets", rf"^{PLA}/hf_cache/hub/datasets--", SKIP, "D_misc/hf_hub_revisions.tsv (리비전만)", "제외 - 사용자 결정(2026-09-17): 다른 장치에서 재다운로드"),
    ("hf_hub_other", rf"^{PLA}/hf_cache/hub/", SKIP, "-", "HF 캐시 잠금·태그"),
    ("hf_raw_traces", rf"^{PLA}/hf_cache/raw/", MUST, "D_misc/hf_cache_raw_traces", "벤치 입력 trace(필터 사본), upstream 변동 위험"),
    ("hf_converted_configs", rf"^{PLA}/hf_cache/mamba[^/]*-sglang/", RECOMMENDED, "D_misc/hf_converted_model_dirs", "수작업 변환 config·tokenizer(가중치는 hub 심볼릭 링크)"),
    ("hf_datasets_cache", rf"^{PLA}/hf_cache/(json|anon8231489123___share_gpt_vicuna_unfiltered|modules|xet)/", SKIP, "-", "HF datasets/modules/xet 캐시, 재다운로드"),
    ("hf_cache_meta", rf"^{PLA}/hf_cache/[^/]+$", SKIP, "-", "HF 캐시 메타파일"),
    ("external_git_clones", rf"^{EXT}/(sglang-latest|muxwise|bullet)/", SKIP, "E_env/external_clone_pins.tsv", "로컬 수정 0(HEAD 대비 M 0), 커밋 고정 재클론"),
    ("external_zenodo_and_logs", rf"^{EXT}/", RECOMMENDED, "D_misc/repo_untracked_misc", "muxwise-zenodo(git 아님)·clone 로그"),
    ("pla_stray_venvs", rf"^{PLA}/(bin/|lib/|lib64$|include/|pyvenv\.cfg$)|^{PLA}/workspace/serving-eval/\.venv/", SKIP, "-", "venv, 재설치"),
    ("pla_other_untracked", rf"^{PLA}/", MUST, "D_misc/repo_untracked_misc", "로그·그림·루트 SLURM 출력 등 git 밖 파일"),
    ("ssm_scope_git_dir", r"^SSM-Scope/\.git/", SKIP, "-", "별도 저장소, 원격과 차이 0"),
    ("ssm_scope_untracked", r"^SSM-Scope/", RECOMMENDED, "-", "별도 저장소의 미추적 파일(있으면 확인)"),
    ("engine_dev_tree", r"^sglang_engine_dev/", RECOMMENDED, "E_env/sglang_engine_dev_src", "v0.5.10+sync+patch로 재구성 가능, 바이트 동일 사본"),
    ("engine_venv_config", r"^sglang_engine_venv/(pyvenv\.cfg|lib/python[^/]+/site-packages/sitecustomize\.py)$", RECOMMENDED, "E_env (+ git env/sitecustomize.py)", "venv 설정·py3.14 shim"),
    ("engine_venv", r"^sglang_engine_venv/", SKIP, "E_env/../venv_packages (git)", "재설치, purge로 이미 손상"),
    ("vllm_venv_and_pip_cache", r"^(vllm_venv|\.pip_cache)/", SKIP, "-", "재설치·다운로드 캐시"),
    ("vllm_bench_scripts", r"^vllm_bench/", RECOMMENDED, "D_misc/vllm_bench", "git 밖 스크립트 3개"),
    ("claude_workspace_tools", r"^(CLAUDE\.md$|\.claude/)", MUST, "B_claude/claude_memory_tools", "에이전트·스킬·CLAUDE.md (git tools/claude에도 사본)"),
    ("vscode_settings", r"^\.vscode/", OPTIONAL, "D_misc/workspace_dotfiles", "편집기 설정"),
    ("vsix_installer", r"^Microsoft\.VisualStudio\.Services\.VSIXPackage$", SKIP, "-", "VS Code 확장 설치본"),
    ("outer_triton_cache", r"^\.triton_cache_smid/", SKIP, "-", "triton 캐시"),
    ("UNCLASSIFIED", r"", MUST, "?", "규칙 없음 - 사람이 확인"),
]
RULES = [(n, re.compile(p), d, b, r) for n, p, d, b, r in RULES]


def git_tracked(repo_abs, root):
    """Workspace-relative paths of files tracked by the repo at repo_abs."""
    out = subprocess.run(["git", "-C", repo_abs, "ls-files", "-z"],
                         capture_output=True, check=True).stdout
    rel = os.path.relpath(repo_abs, root)
    return {os.path.join(rel, p) for p in out.decode("utf-8", "surrogateescape").split("\0") if p}


def walk(root):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError as e:
            print(f"WARN cannot list {d}: {e}", file=sys.stderr)
            continue
        with it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                else:
                    try:
                        yield e.path, e.stat(follow_symlinks=False).st_size
                    except OSError:
                        yield e.path, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/ehmoon/whlee")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)

    tracked = set()
    repos = [f"{PLA}", "SSM-Scope"]  # external clones are classified wholesale
    for r in repos:
        if os.path.isdir(os.path.join(root, r, ".git")):
            tracked |= git_tracked(os.path.join(root, r), root)

    n_tracked = b_tracked = 0
    stats = collections.defaultdict(lambda: [0, 0])
    hf = collections.defaultdict(int)
    hf_purged = collections.defaultdict(int)
    with gzip.open(os.path.join(args.out, "untracked_files.tsv.gz"), "wt", encoding="utf-8",
                   errors="surrogateescape") as fh:
        fh.write("path\tbytes\trule\tdecision\tbundle\n")
        for abs_path, size in walk(root):
            rel = os.path.relpath(abs_path, root)
            if rel in tracked:
                n_tracked += 1
                b_tracked += size
                continue
            for name, rx, decision, bundle, _ in RULES:
                if rx.search(rel):
                    break
            stats[name][0] += 1
            stats[name][1] += size
            m = re.match(rf"^{PLA}/hf_cache/hub/((models|datasets)--[^/]+)/", rel)
            if m:
                hf[m.group(1)] += size
                if os.path.basename(rel).startswith("ToBeDelete_"):
                    hf_purged[m.group(1)] += 1
            fh.write(f"{rel}\t{size}\t{name}\t{decision}\t{bundle}\n")

    with open(os.path.join(args.out, "summary.tsv"), "w", encoding="utf-8") as fh:
        fh.write("rule\tdecision\tfiles\tbytes\tbundle\treason\n")
        fh.write(f"git_tracked\t(git)\t{n_tracked}\t{b_tracked}\tA_git (bundle)\tgit 추적 파일\n")
        for name, _, decision, bundle, reason in RULES:
            if name in stats:
                c, b = stats[name]
                fh.write(f"{name}\t{decision}\t{c}\t{b}\t{bundle}\t{reason}\n")
    with open(os.path.join(args.out, "hf_models.tsv"), "w", encoding="utf-8") as fh:
        # purged_files > 0: the scratch purge renamed blobs/refs to ToBeDelete_*,
        # so this local copy is already incomplete and must be re-downloaded.
        fh.write("repo\tbytes\trevision\tpurged_files\n")
        for repo, b in sorted(hf.items()):
            rev = "?"
            for name in ("main", "ToBeDelete_main"):
                ref = os.path.join(root, PLA, "hf_cache/hub", repo, "refs", name)
                if os.path.exists(ref):
                    rev = open(ref).read().strip()
                    break
            fh.write(f"{repo}\t{b}\t{rev}\t{hf_purged[repo]}\n")
    unclassified = stats.get("UNCLASSIFIED", [0, 0])[0]
    print(f"tracked {n_tracked} files; untracked rules: "
          + ", ".join(f"{k}={v[0]}" for k, v in sorted(stats.items())))
    return 1 if unclassified else 0


if __name__ == "__main__":
    sys.exit(main())
