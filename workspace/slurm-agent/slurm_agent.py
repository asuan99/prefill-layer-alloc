#!/usr/bin/env python3
"""Persistent SLURM observer and experiment-result reporter."""

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
STATE = Path(os.environ.get("SLURM_AGENT_STATE_DIR", REPO / ".slurm-agent"))
REPORTS = Path(os.environ.get("SLURM_AGENT_REPORT_DIR", HERE / "reports"))
LOCK = STATE / "observer.lock"
PID = STATE / "observer.pid"
LOG = STATE / "observer.log"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def ensure_dirs() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def run(command: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(command, text=True, capture_output=True, check=False)
        return p.returncode, p.stdout, p.stderr
    except OSError as exc:
        return 127, "", str(exc)


def log(message: str) -> None:
    ensure_dirs()
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}\n")


def job_file(job_id: str) -> Path:
    return STATE / f"job_{job_id}.json"


def read_job(job_id: str) -> dict[str, Any] | None:
    path = job_file(job_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_job(meta: dict[str, Any]) -> None:
    ensure_dirs()
    path = job_file(str(meta["job_id"]))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def uid_selector() -> str:
    return os.environ.get("SLURM_AGENT_SLURM_USER") or os.environ.get("SLURM_USER_ID") or str(os.getuid())


def normalize_state(value: str) -> str:
    return value.split("+", 1)[0].split(None, 1)[0].strip() or "UNKNOWN"


def sacct(job_id: str) -> dict[str, str]:
    fields = "JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,WorkDir,NodeList,MaxRSS"
    code, stdout, stderr = run(["sacct", "-X", "-n", "-P", "-j", job_id, "--format", fields])
    if code != 0:
        return {"state": "UNKNOWN", "error": (stderr or stdout).strip()[:500]}
    for line in stdout.splitlines():
        parts = line.split("|")
        if not parts or parts[0] not in {job_id, job_id.split("_", 1)[0]}:
            continue
        keys = fields.split(",")
        item = dict(zip(keys, parts))
        result = {key.lower(): value for key, value in item.items()}
        result["state"] = normalize_state(item.get("State", "UNKNOWN"))
        return result
    return {"state": "UNKNOWN", "error": "not found in accounting"}


def squeue(job_id: str | None = None) -> list[tuple[str, str]]:
    command = ["squeue", "-h"]
    if job_id:
        command += ["-j", job_id]
    else:
        command += ["-u", uid_selector()]
    command += ["-o", "%i|%T"]
    code, stdout, _ = run(command)
    if code != 0:
        return []
    return [tuple(line.split("|", 1)) for line in stdout.splitlines() if "|" in line]


def scontrol(job_id: str) -> dict[str, str]:
    code, stdout, _ = run(["scontrol", "show", "job", "-o", job_id])
    if code != 0:
        return {}
    return {m.group(1): m.group(2).strip('"') for m in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)=(\"[^\"]*\"|\S+)", stdout)}


def directives(script: str) -> dict[str, str]:
    short = {"-J": "job_name", "-o": "output", "-e": "error", "-p": "partition", "-t": "time"}
    result: dict[str, str] = {}
    for line in script.splitlines():
        if not re.match(r"^\s*#SBATCH\b", line):
            continue
        try:
            tokens = shlex.split(re.sub(r"^\s*#SBATCH\s+", "", line), comments=True)
        except ValueError:
            tokens = line.split()
        if not tokens:
            continue
        key, value = tokens[0], tokens[1] if len(tokens) > 1 else ""
        if "=" in key:
            key, value = key.split("=", 1)
        result[short.get(key, key.lstrip("-").replace("-", "_"))] = value
    return result


def find_script(command: str | None, job_name: str | None) -> Path | None:
    if command:
        candidate = Path(command)
        if candidate.exists():
            return candidate.resolve()
    candidates: list[Path] = []
    for root in (REPO / "slurm", REPO / "workspace"):
        if root.exists():
            candidates.extend(root.rglob("*.sbatch"))
            candidates.extend(root.rglob("*.sh"))
    for path in candidates:
        if job_name and path.stem == job_name:
            return path.resolve()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        d = directives(text)
        if job_name and d.get("job_name") == job_name:
            return path.resolve()
    return None


def infer(script: str, path: Path | None, job_name: str | None, comment: str | None) -> dict[str, Any]:
    labels = sorted({x.upper() for x in re.findall(r"\b(E\d(?:_[A-Za-z0-9]+)*)\b", script, re.I)})
    paths = sorted(set(re.findall(r"(?:experiments|stage\d+_[A-Za-z0-9_]+)/[A-Za-z0-9_./-]+", script)))
    commands = [line.strip()[:240] for line in script.splitlines() if re.search(r"\b(python|python3|srun|torchrun)\b", line) and not line.lstrip().startswith("#")]
    name = job_name or "unknown"
    clues = [x for x in (f"comment={comment}" if comment else "", f"labels={','.join(labels)}" if labels else "", f"paths={','.join(paths[:4])}" if paths else "") if x]
    return {"job_name": name, "comment": comment or "", "purpose": f"{name}: {'; '.join(clues) if clues else 'script=' + (path.name if path else 'unknown')}", "labels": labels, "paths": paths, "commands": commands[:20]}


def register(job_id: str, details: dict[str, str] | None = None, accounting: dict[str, str] | None = None) -> dict[str, Any]:
    details = details or {}
    accounting = accounting or {}
    command = details.get("Command")
    name = details.get("JobName") or accounting.get("jobname") or job_id
    path = find_script(command, name)
    script = path.read_text(encoding="utf-8", errors="replace") if path and path.exists() else ""
    d = directives(script)
    meta: dict[str, Any] = {
        "job_id": job_id,
        "submitted_at": details.get("SubmitTime") or accounting.get("start") or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "cwd": details.get("WorkDir") or accounting.get("workdir") or str(REPO),
        "script": str(path) if path else None,
        "job_name": name,
        "sbatch_command": f"scontrol show job -o {job_id}" if details else "unknown sbatch command",
        "directives": d,
        "inference": infer(script, path, name, details.get("Comment")),
        "status": normalize_state(details.get("JobState") or accounting.get("state", "UNKNOWN")),
        "terminal_at": None,
        "accounting": accounting,
        "artifacts": [],
    }
    write_job(meta)
    return meta


def discover_queue() -> int:
    count = 0
    for job_id, _ in squeue():
        if not read_job(job_id):
            register(job_id, scontrol(job_id))
            count += 1
    return count


def recent_accounting(hours: int = 72) -> list[dict[str, str]]:
    start = (dt.datetime.now() - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    fields = "JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,WorkDir,NodeList,MaxRSS"
    code, stdout, _ = run(["sacct", "-u", uid_selector(), "-S", start, "-X", "-n", "-P", "--format", fields])
    if code != 0:
        return []
    result = []
    for line in stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2 or not re.match(r"^\d+(?:_\d+)?$", parts[0]):
            continue
        item = dict(zip(fields.split(","), parts))
        item = {key.lower(): value for key, value in item.items()}
        item["state"] = normalize_state(item.get("state", "UNKNOWN"))
        result.append(item)
    return result


def backfill() -> int:
    count = 0
    for item in recent_accounting():
        job_id = item["jobidraw"]
        if not read_job(job_id):
            register(job_id, accounting=item)
            count += 1
    return count


def file_summary(path: Path) -> str:
    if path.suffix.lower() != ".csv":
        return "artifact"
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = list(reader)
        return f"CSV {len(rows)} rows; columns={','.join(header[:8])}"
    except (OSError, csv.Error):
        return "CSV unreadable"


def artifacts(meta: dict[str, Any]) -> list[dict[str, Any]]:
    job_id = str(meta["job_id"])
    paths: set[Path] = set()
    if meta.get("script"):
        script = Path(meta["script"])
        if script.exists():
            text = script.read_text(encoding="utf-8", errors="replace")
            for raw in re.findall(r"(?:results(?:_v\d+)?|reports|logs)/[^\s;\"']+", text):
                candidate = Path(raw.replace("%j", job_id).replace("%A", job_id))
                if not candidate.is_absolute():
                    candidate = Path(meta.get("cwd", REPO)) / candidate
                paths.add(candidate.resolve())
    roots = [REPO / x for x in ("logs", "results", "reports", "slurm/logs")]
    workspace = REPO / "workspace"
    if workspace.exists():
        for child in workspace.iterdir():
            if child.is_dir():
                roots.extend(child / x for x in ("logs", "results", "reports"))
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and job_id in path.name:
                    paths.add(path.resolve())
        except OSError:
            pass
    result = []
    for path in sorted(paths):
        if path.is_file():
            try:
                result.append({"path": str(path), "size": path.stat().st_size, "summary": file_summary(path)})
            except OSError:
                pass
    return result[:300]


def report(meta: dict[str, Any]) -> Path:
    meta["artifacts"] = artifacts(meta)
    state = meta.get("status", "UNKNOWN")
    acc = meta.get("accounting", {})
    lines = [
        f"# SLURM 실험 보고서 — `{meta['job_id']}`", "",
        f"- 실험 식별: **{meta.get('inference', {}).get('purpose', meta.get('job_name', 'unknown'))}**",
        f"- 상태: **{state}**", f"- 종료 코드: `{acc.get('exitcode', 'unknown')}`",
        f"- 실행 시간: `{acc.get('elapsed', 'unknown')}`", f"- 실행 노드: `{acc.get('nodelist', 'unknown')}`", "",
        "## 실험 목적과 근거", "",
        "목적은 SLURM 메타데이터와 제출 스크립트에서 자동 추론했습니다. 추론과 측정값은 구분해야 합니다.", "",
        f"- job name: `{meta.get('job_name', '')}`", f"- script: `{meta.get('script') or 'unknown'}`",
        f"- command: `{meta.get('sbatch_command', '')}`", "", "## 결과물", "",
    ]
    if meta["artifacts"]:
        lines += ["| 파일 | 크기 | 요약 |", "|---|---:|---|"]
        lines += [f"| `{x['path']}` | {x['size']} B | {x['summary']} |" for x in meta["artifacts"]]
    else:
        lines.append("결과 파일을 자동 발견하지 못했습니다.")
    lines += ["", "## 자동 판정", "", f"- SLURM 최종 상태는 `{state}`입니다."]
    lines += ["- 정상 종료만으로 실험 가설의 성공을 의미하지 않습니다." if state == "COMPLETED" else "- 정상 종료가 아니므로 오류 및 부분 결과를 우선 확인해야 합니다."]
    path = REPORTS / f"job_{meta['job_id']}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def process_pending() -> int:
    processed = 0
    for path in sorted(STATE.glob("job_*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("terminal_at"):
            continue
        job_id = str(meta["job_id"])
        live = dict(squeue(job_id))
        if live:
            meta["status"] = normalize_state(next(iter(live.values())))
            write_job(meta)
            continue
        acc = sacct(job_id)
        state = acc.get("state", "UNKNOWN")
        if state == "UNKNOWN":
            continue
        meta["status"] = state
        meta["accounting"] = acc
        if state in TERMINAL:
            meta["terminal_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            existing_report = REPORTS / f"job_{job_id}.md"
            path_report = existing_report if existing_report.exists() else report(meta)
            meta["report"] = str(path_report)
            write_job(meta)
            log(f"job={job_id} state={state} report={path_report}")
            processed += 1
    return processed


def poll_once() -> tuple[int, int]:
    adopted = discover_queue() + backfill()
    processed = process_pending()
    return adopted, processed


def observe(poll: int, once: bool = False) -> int:
    ensure_dirs()
    with LOCK.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        PID.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            while True:
                try:
                    poll_once()
                except Exception as exc:  # observer must survive one bad job/API call
                    log(f"poll-error={type(exc).__name__}: {exc}")
                if once:
                    return 0
                time.sleep(max(5, poll))
        finally:
            try:
                PID.unlink()
            except FileNotFoundError:
                pass


def daemon_start() -> int:
    ensure_dirs()
    if PID.exists():
        try:
            os.kill(int(PID.read_text().strip()), 0)
            return 0
        except (ValueError, OSError):
            PID.unlink(missing_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "observe"], cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(p.pid) + "\n", encoding="utf-8")
    return 0


def daemon_stop() -> int:
    if not PID.exists():
        return 0
    try:
        os.kill(int(PID.read_text().strip()), signal.SIGTERM)
    except (ValueError, OSError):
        pass
    PID.unlink(missing_ok=True)
    return 0


def submit(args: list[str]) -> int:
    command = args if "--parsable" in args else ["--parsable", *args]
    code, stdout, stderr = run(["sbatch", *command])
    if stdout:
        print(stdout, end="")
    if code == 0:
        daemon_start()
    elif stderr:
        print(stderr, file=sys.stderr, end="")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("observe"); p.add_argument("--poll", type=int, default=30); p.add_argument("--once", action="store_true")
    sub.add_parser("start"); sub.add_parser("stop"); sub.add_parser("status"); sub.add_parser("poll")
    p = sub.add_parser("submit"); p.add_argument("sbatch_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("report"); p.add_argument("job_id")
    ns = parser.parse_args()
    if ns.command == "observe": return observe(ns.poll, ns.once)
    if ns.command == "start": return daemon_start()
    if ns.command == "stop": return daemon_stop()
    if ns.command == "poll": poll_once(); return 0
    if ns.command == "status":
        if PID.exists():
            try: os.kill(int(PID.read_text().strip()), 0); print(f"running pid={PID.read_text().strip()}"); return 0
            except (ValueError, OSError): pass
        print("stopped"); return 1
    if ns.command == "submit":
        args = list(ns.sbatch_args); args = args[1:] if args[:1] == ["--"] else args
        return submit(args)
    if ns.command == "report":
        meta = read_job(ns.job_id)
        if not meta: print(f"unknown job {ns.job_id}", file=sys.stderr); return 2
        acc = sacct(ns.job_id); meta["status"] = acc.get("state", meta.get("status")); meta["accounting"] = acc
        path = report(meta); meta["report"] = str(path); meta["terminal_at"] = meta.get("terminal_at") or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"); write_job(meta); print(path); return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
