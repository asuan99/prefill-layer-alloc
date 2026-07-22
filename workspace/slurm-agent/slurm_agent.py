#!/usr/bin/env python3
"""Track sbatch jobs and create experiment reports.

The module intentionally uses only the Python standard library. It is meant
to run on a login node, where it can poll squeue/sacct while the actual
experiment runs on a compute node.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE_DIR = Path(os.environ.get("SLURM_AGENT_STATE_DIR", REPO / ".slurm-agent"))
REPORT_DIR = Path(os.environ.get("SLURM_AGENT_REPORT_DIR", HERE / "reports"))
LOCK_PATH = STATE_DIR / "watcher.lock"
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str | None) -> float:
    if not value:
        return time.time()
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except ValueError:
        return time.time()


def ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout, result.stderr


def job_id_from_sbatch(output: str) -> str:
    match = re.search(r"(?m)^\s*(\d+(?:_[\d-]+)?)(?:;[^\s]+)?\s*$", output)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {output.strip()!r}")
    return match.group(1)


def option_value(args: list[str], names: set[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg in names and index + 1 < len(args):
            return args[index + 1]
        for name in names:
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def find_script(args: list[str], cwd: Path) -> Path | None:
    value_options = {
        "-A", "-D", "-J", "-N", "-o", "-e", "-p", "-t", "-w", "-x",
        "--account", "--chdir", "--comment", "--cpus-per-task", "--error",
        "--export", "--gres", "--job-name", "--mem", "--nodes", "--output",
        "--partition", "--time", "--wrap", "--dependency", "--array",
    }
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in value_options:
            index += 2
            continue
        if any(arg.startswith(name + "=") for name in value_options):
            index += 1
            continue
        if arg.startswith("-"):
            index += 1
            continue
        candidate = Path(arg).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.is_file() or arg.endswith((".sh", ".sbatch")):
            return candidate.resolve()
        index += 1
    return None


def parse_sbatch_directives(script: str) -> dict[str, str]:
    directives: dict[str, str] = {}
    short_names = {
        "-J": "job-name", "-o": "output", "-e": "error", "-p": "partition",
        "-t": "time", "-A": "account", "-D": "chdir",
    }
    for line in script.splitlines():
        if not re.match(r"^\s*#SBATCH\b", line):
            continue
        body = re.sub(r"^\s*#SBATCH\s+", "", line).strip()
        try:
            tokens = shlex.split(body, comments=True)
        except ValueError:
            tokens = body.split()
        if not tokens:
            continue
        key_token = tokens[0]
        if "=" in key_token:
            key, value = key_token.split("=", 1)
        else:
            key = key_token
            value = tokens[1] if len(tokens) > 1 else ""
        key = short_names.get(key, key.lstrip("-").replace("-", "_"))
        directives[key] = value
    return directives


def infer_experiment(script: str, script_path: Path | None, directives: dict[str, str], args: list[str]) -> dict[str, Any]:
    job_name = option_value(args, {"--job-name", "-J"}) or directives.get("job_name") or "unknown"
    comment = option_value(args, {"--comment"}) or directives.get("comment", "")
    labels = sorted({label.upper() for label in re.findall(r"\b(E\d(?:_[A-Za-z0-9]+)*)\b", script, re.I)})
    experiment_paths = sorted(set(re.findall(r"(?:experiments|stage\d+_[A-Za-z0-9_]+)/[A-Za-z0-9_./-]+", script)))
    commands = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("echo"):
            continue
        if re.search(r"\b(?:python|python3|bash|srun|torchrun)\b", stripped):
            commands.append(stripped[:240])
    wrapped = option_value(args, {"--wrap"})
    if wrapped:
        commands.append(wrapped[:240])
    clues = []
    if comment:
        clues.append(f"comment={comment}")
    if labels:
        clues.append("labels=" + ", ".join(labels))
    if experiment_paths:
        clues.append("paths=" + ", ".join(experiment_paths[:5]))
    if wrapped:
        clues.append("wrap=" + wrapped[:120])
    if not clues and script_path:
        clues.append(f"script={script_path.name}")
    purpose = f"{job_name}: " + "; ".join(clues) if clues else job_name
    return {
        "job_name": job_name,
        "comment": comment,
        "purpose": purpose,
        "purpose_basis": ["SBATCH metadata", "script path/commands", "script comments"],
        "experiment_labels": labels,
        "experiment_paths": experiment_paths,
        "commands": commands[:20],
    }


def replace_tokens(value: str, job_id: str, job_name: str = "job") -> str:
    replacements = {
        "%j": job_id, "%A": job_id.split("_", 1)[0], "%x": job_name,
        "$USER": os.environ.get("USER", "unknown"),
        "${USER}": os.environ.get("USER", "unknown"),
        "${SLURM_JOB_ID}": job_id, "$SLURM_JOB_ID": job_id,
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return os.path.expandvars(value)


def literal_paths(script: str, directives: dict[str, str], job_id: str, cwd: Path, job_name: str = "job") -> set[Path]:
    values = list(directives.values())
    values.extend(re.findall(r"(?:(?:\"|')([^\"']*(?:results|reports|logs|output)[^\"']*)(?:\"|'))", script))
    values.extend(re.findall(r"(?<![A-Za-z0-9_])((?:results(?:_v\d+)?|reports|logs)/[^\s;\"']+)", script))
    paths: set[Path] = set()
    for raw in values:
        raw = replace_tokens(raw, job_id, job_name).rstrip(",)")
        if not any(marker in raw for marker in ("results", "reports", "logs", "output")):
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if any(ch in str(candidate) for ch in "*?["):
            paths.update(Path(p).resolve() for p in candidate.parent.glob(candidate.name))
        else:
            paths.add(candidate.resolve())
    return paths


def candidate_roots() -> list[Path]:
    roots = [REPO / name for name in ("logs", "results", "reports", "slurm/logs")]
    workspace = REPO / "workspace"
    if workspace.exists():
        for child in workspace.iterdir():
            if child.is_dir():
                roots.extend(child / name for name in ("logs", "results", "reports"))
    return [root for root in roots if root.exists()]


def csv_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            numeric: dict[str, list[float]] = {}
            row_count = 0
            for row in reader:
                row_count += 1
                for index, value in enumerate(row[: len(header)]):
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    numeric.setdefault(header[index], []).append(number)
        metrics = {
            name: {"min": min(values), "max": max(values)}
            for name, values in list(numeric.items())[:8] if values
        }
        return {"kind": "csv", "rows": row_count, "columns": header[:20], "numeric": metrics}
    except (OSError, csv.Error):
        return {"kind": "csv", "read_error": True}


def collect_artifacts(meta: dict[str, Any], script: str, directives: dict[str, str]) -> list[dict[str, Any]]:
    job_id = str(meta["job_id"])
    submitted = parse_time(meta.get("submitted_at")) - 90
    cutoff = time.time() + 30
    job_name = meta.get("inference", {}).get("job_name", "job")
    paths = literal_paths(script, directives, job_id, Path(meta["cwd"]), job_name)
    for root in candidate_roots():
        try:
            for path in root.rglob("*"):
                if not path.is_file() or any(part in {".triton_cache", ".sglang_cache", "__pycache__"} for part in path.parts):
                    continue
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    continue
                if submitted <= modified <= cutoff and (job_id in path.name or path.suffix in {".csv", ".json", ".md", ".txt", ".log"}):
                    paths.add(path.resolve())
        except OSError:
            continue
    artifacts = []
    for path in sorted(paths):
        if not path.exists():
            continue
        try:
            stat = path.stat()
            item = {"path": str(path), "size": stat.st_size, "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")}
            if path.suffix.lower() == ".csv":
                item.update(csv_summary(path))
            artifacts.append(item)
        except OSError:
            continue
    return artifacts[:250]


def sacct_record(job_id: str) -> dict[str, str]:
    fields = "JobIDRaw,State,ExitCode,Elapsed,Start,End,MaxRSS,AllocTRES,NodeList"
    code, stdout, stderr = run_command(["sacct", "-X", "-n", "-P", "-j", job_id, "--format", fields])
    if code != 0:
        return {"state": "UNKNOWN", "sacct_error": (stderr or stdout).strip()[:500]}
    for line in stdout.splitlines():
        parts = line.split("|")
        if not parts or parts[0] not in {job_id, job_id.split("_", 1)[0]}:
            continue
        keys = fields.split(",")
        record = dict(zip(keys, parts))
        state = record.get("State", "UNKNOWN").split("+", 1)[0].split(None, 1)[0]
        normalized = {key.lower(): value for key, value in record.items()}
        normalized["state"] = state
        return normalized
    return {"state": "UNKNOWN", "sacct_error": "job not present in accounting output"}


def squeue_state(job_id: str) -> str | None:
    code, stdout, _ = run_command(["squeue", "-h", "-j", job_id, "-o", "%T"])
    if code == 0 and stdout.strip():
        return stdout.strip().splitlines()[0].strip()
    return None


def slurm_user() -> str:
    """Return the scheduler user selector; numeric UID works on this cluster."""
    return os.environ.get("SLURM_AGENT_SLURM_USER") or os.environ.get("SLURM_USER_ID") or str(os.getuid())


def scontrol_job(job_id: str) -> dict[str, str]:
    code, stdout, _ = run_command(["scontrol", "show", "job", "-o", job_id])
    if code != 0 or not stdout.strip():
        return {}
    result: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)=(\"[^\"]*\"|\S+)", stdout):
        result[match.group(1)] = match.group(2).strip('"')
    return result


def adopt_job(job_id: str, start_watcher: bool = True) -> int:
    """Register a job submitted before the wrapper was installed."""
    ensure_dirs()
    existing = STATE_DIR / f"job_{job_id}.json"
    if existing.exists():
        if start_watcher:
            ensure_watcher()
        print(f"[slurm-agent] already registered job {job_id}", file=sys.stderr)
        return 0
    details = scontrol_job(job_id)
    if not details:
        print(f"[slurm-agent] cannot inspect job {job_id} with scontrol", file=sys.stderr)
        return 2
    script_path = Path(details["Command"]) if details.get("Command") else None
    script = script_path.read_text(encoding="utf-8", errors="replace") if script_path and script_path.exists() else ""
    directives = parse_sbatch_directives(script)
    inferred = infer_experiment(script, script_path, directives, [])
    if details.get("JobName"):
        inferred["job_name"] = details["JobName"]
        if not inferred["purpose"].startswith(details["JobName"] + ":"):
            inferred["purpose"] = f"{details['JobName']}: {inferred['purpose']}"
    meta: dict[str, Any] = {
        "job_id": job_id,
        "submitted_at": details.get("SubmitTime", now()),
        "cwd": details.get("WorkDir", str(REPO)),
        "script": str(script_path) if script_path else None,
        "sbatch_args": [],
        "sbatch_command": f"scontrol show job -o {job_id}",
        "directives": directives,
        "inference": inferred,
        "status": details.get("JobState", "UNKNOWN").split("+", 1)[0],
        "terminal_at": None,
        "accounting": {},
        "artifacts": [],
        "report": None,
        "adopted": True,
        "scontrol": details,
    }
    if meta["status"] in TERMINAL_STATES:
        meta["terminal_at"] = now()
        meta["accounting"] = sacct_record(job_id)
    write_job(meta)
    print(f"[slurm-agent] adopted existing job {job_id}: {inferred['purpose']}", file=sys.stderr)
    if start_watcher:
        ensure_watcher()
    return 0


def discover_jobs(start_watcher: bool = True) -> int:
    """Adopt all currently queued jobs belonging to the scheduler user."""
    code, stdout, stderr = run_command(["squeue", "-u", slurm_user(), "-h", "-o", "%i|%T"])
    if code != 0:
        print((stderr or stdout).strip(), file=sys.stderr)
        return code or 1
    job_ids = [line.split("|", 1)[0].strip() for line in stdout.splitlines() if line.strip()]
    adopted = 0
    for job_id in job_ids:
        if not (STATE_DIR / f"job_{job_id}.json").exists():
            adopted += adopt_job(job_id, start_watcher=False) == 0
    if job_ids and start_watcher:
        ensure_watcher()
    print(f"[slurm-agent] discovered {len(job_ids)} queued job(s), adopted {adopted}")
    return 0


def read_job(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_job(meta: dict[str, Any]) -> Path:
    ensure_dirs()
    path = STATE_DIR / f"job_{meta['job_id']}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def submit(args: list[str], auto_watch: bool = True) -> int:
    ensure_dirs()
    command_args = args if "--parsable" in args else ["--parsable", *args]
    cwd = Path.cwd().resolve()
    code, stdout, stderr = run_command(["sbatch", *command_args])
    if stdout:
        print(stdout, end="")
    if code != 0:
        if stderr:
            print(stderr, file=sys.stderr, end="")
        return code
    job_id = job_id_from_sbatch(stdout)
    script_path = find_script(args, cwd)
    script = script_path.read_text(encoding="utf-8", errors="replace") if script_path and script_path.exists() else ""
    directives = parse_sbatch_directives(script)
    inferred = infer_experiment(script, script_path, directives, args)
    meta: dict[str, Any] = {
        "job_id": job_id, "submitted_at": now(), "cwd": str(cwd),
        "script": str(script_path) if script_path else None, "sbatch_args": args,
        "sbatch_command": shlex.join(["sbatch", *args]), "directives": directives,
        "inference": inferred, "status": "SUBMITTED", "terminal_at": None,
        "accounting": {}, "artifacts": [], "report": None,
    }
    write_job(meta)
    print(f"[slurm-agent] registered job {job_id}; purpose: {inferred['purpose']}", file=sys.stderr)
    if auto_watch:
        ensure_watcher()
    return 0


def ensure_watcher() -> None:
    pid_path = STATE_DIR / "watcher.pid"
    try:
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            return
    except (ValueError, OSError):
        pass
    log_path = STATE_DIR / "watcher.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "observe"], cwd=str(REPO),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid) + "\n", encoding="utf-8")


def pending_jobs() -> list[Path]:
    ensure_dirs()
    result = []
    for path in sorted(STATE_DIR.glob("job_*.json")):
        try:
            meta = read_job(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not meta.get("terminal_at"):
            result.append(path)
    return result


def watch_once() -> int:
    changed = 0
    for path in pending_jobs():
        meta = read_job(path)
        job_id = str(meta["job_id"])
        live_state = squeue_state(job_id)
        if live_state:
            meta["status"] = live_state
            write_job(meta)
            continue
        accounting = sacct_record(job_id)
        state = accounting.get("state", "UNKNOWN")
        if state == "UNKNOWN":
            continue
        meta["status"] = state
        meta["accounting"] = accounting
        if state in TERMINAL_STATES:
            meta["terminal_at"] = now()
            report_path = create_report(meta)
            meta["report"] = str(report_path)
            write_job(meta)
            changed += 1
            print(f"[slurm-agent] job {job_id} -> {state}; report: {report_path}")
    return changed


def tail_lines(path: Path, limit: int = 80) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def interesting_log_lines(path: Path, limit: int = 40) -> list[str]:
    patterns = re.compile(r"(throughput|goodput|ttft|tpot|it[l]?|slo|accuracy|failed|failure|error|passed|result|healthy|done)", re.I)
    lines = [line.strip() for line in tail_lines(path, 240) if patterns.search(line)]
    return lines[-limit:]


def create_report(meta: dict[str, Any]) -> Path:
    script_path = Path(meta["script"]) if meta.get("script") else None
    script = script_path.read_text(encoding="utf-8", errors="replace") if script_path and script_path.exists() else ""
    directives = meta.get("directives", {})
    meta["artifacts"] = collect_artifacts(meta, script, directives)
    accounting = meta.get("accounting", {})
    state = meta.get("status", "UNKNOWN")
    title = meta.get("inference", {}).get("purpose", meta.get("job_id"))
    lines = [
        f"# SLURM 실험 보고서 — `{meta['job_id']}`", "",
        f"- 실험 식별(자동 추론): **{title}**", f"- 상태: **{state}**",
        f"- 제출 시각(UTC): `{meta.get('submitted_at', '')}`",
        f"- 종료 감지 시각(UTC): `{meta.get('terminal_at', now())}`",
        f"- 실행 시간: `{accounting.get('elapsed', 'unknown')}`",
        f"- 종료 코드: `{accounting.get('exitcode', 'unknown')}`",
        f"- 실행 노드: `{accounting.get('nodelist', 'unknown')}`", "",
        "## 실험 목적과 근거", "",
        "이 절의 목적/실험명은 SBATCH 메타데이터, 제출 스크립트의 경로·명령·주석을 이용해 자동 추론했습니다. 추론 결과는 실제 측정값과 구분해야 합니다.", "",
        f"- job name: `{meta.get('inference', {}).get('job_name', '')}`",
        f"- comment: `{meta.get('inference', {}).get('comment', '')}`",
        f"- experiment labels: `{', '.join(meta.get('inference', {}).get('experiment_labels', [])) or 'none'}`",
        f"- experiment paths: `{', '.join(meta.get('inference', {}).get('experiment_paths', [])) or 'none'}`",
        f"- 제출 명령: `{meta.get('sbatch_command', '')}`",
        f"- 스크립트: `{meta.get('script') or 'inline --wrap command'}`", "",
        "## 실행 결과", "",
        "자동 수집된 결과 파일과 로그의 핵심 문장을 아래에 기록했습니다. 수치의 의미와 통계적 유의성은 실험 설계에 따라 별도 검토가 필요합니다.", "",
    ]
    if not meta.get("artifacts"):
        lines.append("- 결과 파일: 발견되지 않음 (로그와 SLURM 상태를 확인하세요.)")
    else:
        lines.extend(["| 파일 | 크기 | 요약 |", "|---|---:|---|"])
        for item in meta["artifacts"]:
            if item.get("kind") == "csv":
                summary = f"CSV {item.get('rows', '?')} rows"
                numeric = item.get("numeric", {})
                if numeric:
                    summary += "; " + ", ".join(f"{key}[{values['min']:.4g},{values['max']:.4g}]" for key, values in numeric.items())
            else:
                summary = "artifact"
            lines.append(f"| `{item['path']}` | {item['size']} B | {summary} |")
    lines.extend(["", "### 로그 핵심 문장", ""])
    logs = [Path(item["path"]) for item in meta.get("artifacts", []) if Path(item["path"]).suffix in {".log", ".out", ".err", ".txt"}]
    seen: set[str] = set()
    for log in logs[:12]:
        interesting = interesting_log_lines(log)
        if not interesting:
            continue
        lines.append(f"#### `{log}`")
        lines.extend(f"- {line}" for line in interesting)
        seen.add(str(log))
    if not seen:
        lines.append("- 핵심 문장을 자동 추출하지 못했습니다.")
    lines.extend(["", "## 자동 판정", "", f"- SLURM 종료 상태: **{state}**."])
    if state != "COMPLETED":
        lines.append("- 잡이 정상 완료되지 않았으므로 결과 해석보다 오류 원인과 재현 가능성 확인이 우선입니다.")
    else:
        lines.append("- 잡은 정상 종료했지만, 정상 종료만으로 실험 가설이 지지되었다고 판정하지 않습니다.")
    lines.extend(["", "## 다음 검토 항목", "", "- 대표 CSV/JSON의 핵심 지표를 기준선 또는 이전 반복과 비교", "- 로그의 경고·예외·부분 결과 여부 확인", "- 실험 목적과 결과 파일의 모델/조건 일치 여부 확인", ""])
    report_path = REPORT_DIR / f"job_{meta['job_id']}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    run_optional_agent(meta, report_path)
    return report_path


def run_optional_agent(meta: dict[str, Any], report_path: Path) -> None:
    template = os.environ.get("SLURM_AGENT_CMD", "").strip()
    if not template:
        return
    output_path = report_path.with_suffix(".agent.md")
    prompt = f"""You are the experiment-results reviewer for SLURM job {meta['job_id']}.
Read the generated report at {report_path} and the listed artifacts. Do not modify code,
submit/cancel jobs, or invent measurements. Write a concise review in Korean with exactly
these sections: 요약, 관찰된 주요 수치, 이상/실패, 해석, 다음 실험. Distinguish observed
facts from inference and mention missing evidence explicitly.
"""
    rendered = template.format(report=str(report_path), metadata=str(STATE_DIR / f"job_{meta['job_id']}.json"), output=str(output_path))
    try:
        command = shlex.split(rendered)
        result = subprocess.run(command, input=prompt, text=True, capture_output=True, check=False, cwd=str(REPO))
    except (OSError, ValueError) as exc:
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Agent 검토 실패\n\n`{exc}`\n")
        return
    if result.returncode == 0:
        agent_text = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else result.stdout.strip()
        if agent_text:
            with report_path.open("a", encoding="utf-8") as handle:
                handle.write("\n## Agent 검토\n\n" + agent_text.rstrip() + "\n")
    else:
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Agent 검토 실패\n\nexit={result.returncode}\n{(result.stderr or result.stdout).strip()[-1000:]}\n")


def watch(poll: int) -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    ensure_dirs()
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        (STATE_DIR / "watcher.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
        try:
            while pending_jobs():
                watch_once()
                time.sleep(max(5, poll))
        finally:
            try:
                (STATE_DIR / "watcher.pid").unlink()
            except FileNotFoundError:
                pass
    return 0


def observe(poll: int) -> int:
    """Continuously discover raw sbatch jobs and process terminal jobs."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    ensure_dirs()
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        (STATE_DIR / "watcher.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
        try:
            while True:
                discover_jobs(start_watcher=False)
                watch_once()
                time.sleep(max(5, poll))
        finally:
            try:
                (STATE_DIR / "watcher.pid").unlink()
            except FileNotFoundError:
                pass


def list_jobs() -> int:
    paths = sorted(STATE_DIR.glob("job_*.json"))
    for path in paths:
        try:
            meta = read_job(path)
        except (OSError, json.JSONDecodeError):
            continue
        print(f"{meta.get('job_id')}\t{meta.get('status')}\t{meta.get('inference', {}).get('purpose', '')}\t{meta.get('report') or '-'}")
    return 0


def report(job_id: str) -> int:
    path = STATE_DIR / f"job_{job_id}.json"
    if not path.exists():
        print(f"No registered job: {job_id}", file=sys.stderr)
        return 2
    meta = read_job(path)
    if not meta.get("terminal_at"):
        meta["accounting"] = sacct_record(job_id)
        state = meta["accounting"].get("state", "UNKNOWN")
        meta["status"] = state
        if state in TERMINAL_STATES:
            meta["terminal_at"] = now()
    report_path = create_report(meta)
    meta["report"] = str(report_path)
    write_job(meta)
    print(report_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    submit_parser = sub.add_parser("submit", help="submit through sbatch and register the job")
    submit_parser.add_argument("--no-watch", action="store_true")
    submit_parser.add_argument("sbatch_args", nargs=argparse.REMAINDER)
    watch_parser = sub.add_parser("watch", help="poll registered jobs")
    watch_parser.add_argument("--poll", type=int, default=int(os.environ.get("SLURM_AGENT_POLL", "30")))
    observe_parser = sub.add_parser("observe", help="continuously discover and process sbatch jobs")
    observe_parser.add_argument("--poll", type=int, default=int(os.environ.get("SLURM_AGENT_POLL", "30")))
    sub.add_parser("watch-once", help="poll once; useful for testing/cron")
    sub.add_parser("discover", help="adopt all currently queued jobs")
    adopt_parser = sub.add_parser("adopt", help="adopt one already-submitted job")
    adopt_parser.add_argument("job_id")
    sub.add_parser("list", help="list registered jobs")
    report_parser = sub.add_parser("report", help="regenerate one report")
    report_parser.add_argument("job_id")
    ns = parser.parse_args()
    if ns.command == "submit":
        args = list(ns.sbatch_args)
        if args and args[0] == "--":
            args = args[1:]
        if not args:
            parser.error("submit requires sbatch arguments, e.g. submit -- script.sbatch")
        return submit(args, auto_watch=not ns.no_watch)
    if ns.command == "watch":
        return watch(ns.poll)
    if ns.command == "observe":
        return observe(ns.poll)
    if ns.command == "watch-once":
        return watch_once()
    if ns.command == "discover":
        return discover_jobs()
    if ns.command == "adopt":
        return adopt_job(ns.job_id)
    if ns.command == "list":
        return list_jobs()
    return report(ns.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
