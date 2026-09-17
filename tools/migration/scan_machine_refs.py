#!/usr/bin/env python3
"""List machine-specific references that must be revisited after moving machines.

Scans the Claude Code tool definitions (workspace CLAUDE.md, .claude/agents,
.claude/skills) line by line, plus the tool-sync scripts, and prints a Markdown
checklist grouped by file.  Also prints per-directory counts for the rest of the
repository so the size of the script/doc retarget is visible.

Re-run on the new machine (with --live pointing at the new workspace root) to
confirm nothing machine-specific is left:
  python3 tools/migration/scan_machine_refs.py --live <new root> > report.md

Written for the 2026-09-17 migration (handoff-report/session_handoff_2026-09-17.md).
"""
import argparse
import collections
import glob
import os
import re
import subprocess

# category -> (regex, what to change once the target is known)
CATEGORIES = {
    "경로": (r"/scratch/ehmoon|/home01/ehmoon|\$USER/whlee",
             "새 작업 루트·홈 경로로 치환(가능하면 환경변수/상대경로화)"),
    "메모리 경로": (r"-scratch-ehmoon-whlee|projects/[^ ]*/memory",
                "새 작업 루트의 slug(`/`→`-`)로 치환, 메모리 디렉터리도 그 위치로 이동"),
    "스케줄러": (r"\bsbatch\b|\bsqueue\b|\bsacct\b|\bsalloc\b|\bscancel\b|\bsinfo\b|amd_a100nv|cas_v100|--comment|showappl|--gres|SLURM|Slurm|slurm",
             "새 클러스터 스케줄러·파티션·제출 정책(`--comment` 등)에 맞게 수정, 스케줄러가 없으면 실행 규약 자체 재작성"),
    "모듈·환경": (r"module load|conda/pytorch|cuda/13|gcc/15|sglang_engine_venv|sglang_engine_dev|python3\.14|sync_engine_tree",
              "새 환경의 Python/CUDA/torch 스택과 venv·dev tree 경로로 수정"),
    "하드웨어·기관": (r"A100|108 ?SM|\bSM\b 분할|SXM4|PCIe|glogin|\bgpu[34][0-9]\b|KISTI|Neuron|뉴론|ksc\.re\.kr",
                "새 GPU(모델·SM 수·인터커넥트)와 기관 정보로 수정 - 실험 기판 동일성 판단과 연결"),
    "git 원격·자격증명": (r"\bpush\b|토큰|\btoken\b(?!-)|\borigin\b|\bremote\b",
                    "새 머신 자격증명 방식(토큰 평문 URL 여부)에 맞게 push/원격 규칙 재검토"),
}
COMPILED = {k: re.compile(v[0]) for k, v in CATEGORIES.items()}


def scan_file(path):
    hits = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            cats = [k for k, rx in COMPILED.items() if rx.search(line)]
            if cats:
                text = line.strip().replace("|", "\\|")
                hits.append((i, cats, text[:160] + ("…" if len(text) > 160 else "")))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="/scratch/ehmoon/whlee", help="workspace root with CLAUDE.md/.claude")
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    args = ap.parse_args()

    files = [os.path.join(args.live, "CLAUDE.md")]
    files += sorted(glob.glob(os.path.join(args.live, ".claude/agents/*.md")))
    files += sorted(glob.glob(os.path.join(args.live, ".claude/skills/*/SKILL.md")))
    files += [os.path.join(args.repo, p) for p in (
        "tools/claude/sync_claude_tools.sh", "tools/claude/test_sync_claude_tools.sh", "tools/claude/README.md")]

    print("# Claude 도구 머신 의존 참조 체크리스트\n")
    print("카테고리별 조치:\n")
    print("| 카테고리 | 조치 |\n|---|---|")
    for k, (_, action) in CATEGORIES.items():
        print(f"| {k} | {action} |")
    total = collections.Counter()
    body = []
    for f in files:
        if not os.path.exists(f):
            continue
        hits = scan_file(f)
        rel = os.path.relpath(f, args.live) if f.startswith(args.live) else f
        for _, cats, _ in hits:
            total.update(cats)
        if not hits:
            body.append(f"\n## `{rel}` — 참조 0\n")
            continue
        body.append(f"\n## `{rel}` — {len(hits)}줄\n")
        body.append("| 줄 | 카테고리 | 내용 |\n|---:|---|---|")
        for i, cats, text in hits:
            body.append(f"| {i} | {', '.join(cats)} | {text} |")
    print("\n카테고리별 줄 수(한 줄이 여러 카테고리에 걸칠 수 있음): "
          + ", ".join(f"{k} {total[k]}" for k in CATEGORIES))
    print("\n".join(body))

    # Repository-wide counts (scripts and docs), per top-level directory.
    out = subprocess.run(["git", "-C", args.repo, "grep", "-c", "-E", CATEGORIES["경로"][0]],
                         capture_output=True, text=True).stdout
    per_dir = collections.Counter()
    per_kind = collections.Counter()
    for line in out.splitlines():
        path, n = line.rsplit(":", 1)
        top = "/".join(path.split("/")[:3]) if path.startswith("workspace/") else path.split("/")[0]
        per_dir[top] += int(n)
        per_kind[os.path.splitext(path)[1] or "(없음)"] += 1
    print("\n## 참고: 저장소 전체의 절대경로 참조 (`git grep`, 줄 수)\n")
    print("| 디렉터리 | 줄 수 |\n|---|---:|")
    for d, n in per_dir.most_common(15):
        print(f"| `{d}` | {n} |")
    print("\n파일 확장자별 파일 수: " + ", ".join(f"`{k}` {v}" for k, v in per_kind.most_common(8)))


if __name__ == "__main__":
    main()
