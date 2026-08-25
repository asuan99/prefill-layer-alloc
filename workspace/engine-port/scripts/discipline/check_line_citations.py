#!/usr/bin/env python3
"""문서의 `file:line` 인용이 **현재 트리와 어긋났는지** 검사한다.

왜 이 도구가 있는가 (설계 이력을 남긴다)
----------------------------------------
교훈 #80(*"출처 허위는 규칙 정본 파일 안이 가장 위험하다 — 다음 판본이 재검증 없이
승계한다"*)이 **2026-08-25 한 세션에서만 세 번** 발생했다:

  1. NSL ③ 사전등록 — 감사가 물려준 `:155`/`:249-254`가 트리와 불일치.
  2. NSL ② 설계 — `scheduler.py:2369`를 "도달"로 적었으나 엔진 assert가 닫는다.
  3. A1 rev2 — `multiplexing_mixin.py` 인용 **9건 전부 −41/−42 노후**.
     ★원인은 **그 세션 자신이 같은 파일에 넣은 배선**이었다: 인용을 검증한 시점과
     커밋 시점 사이에 파일이 밀렸다.

3번이 이 도구가 겨냥하는 형태다. **사람이 인용을 검증하는 시점과 그 인용이 읽히는
시점 사이에 파일이 변한다.** 그것은 재검토로 막을 수 없고 **지문(fingerprint)** 으로만 막힌다.

무엇을 하는가 / 못 하는가
-------------------------
`--snapshot`  문서의 모든 인용에 대해 **인용된 줄 범위의 내용 지문**을 매니페스트에 기록한다.
`--check`     지문을 다시 계산해 어긋난 것을 보고하고, ★**같은 내용이 파일 안 다른 곳에
              있으면 새 줄번호를 제안한다**(이것이 실질 가치다 — 드리프트는 거의 항상 이동이다).

★**정직한 한계**: 최초 `--snapshot`은 인용이 **옳은지 검증하지 못한다.** 그 시점의 내용을
기록할 뿐이다. 따라서 **스냅샷은 인용을 방금 검증한 직후에만 찍어야 한다.**
이 한계 때문에 `--check`와 **별개로** 항상 도는 검사 두 개를 둔다:

  * `RANGE`  인용한 줄이 파일 범위 밖이다(= 확실한 오류).
  * `BLANK`  인용한 범위가 **전부 공백/빈 줄**이다(= 거의 확실한 오류).

이 둘은 매니페스트 없이도 즉시 참/거짓이 갈린다.

사용:
    python3 check_line_citations.py --snapshot <doc.md> [...]   # 지문 기록
    python3 check_line_citations.py --check <doc.md> [...]      # 드리프트 검사
    python3 check_line_citations.py --check --all               # 매니페스트에 있는 전부
종료: 0 통과, 1 위반, 2 사용법/해석 오류.
"""
import argparse, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))      # .../engine-port
PROJECT_ROOT = os.path.abspath(os.path.join(TRACK_ROOT, "..", ".."))  # .../prefill-layer-alloc
# The editable engine tree is a SIBLING of the project, not inside it
# (CLAUDE.md: /scratch/.../sglang_engine_dev).  Getting this wrong makes every
# engine citation "UNRESOLVED", which is how the first run of this tool failed.
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
MANIFEST = os.path.join(HERE, "line_citations.json")

# `path/or/name.ext:12` or `...:12-34`, inside markdown backticks.
CITE = re.compile(
    r"`([A-Za-z0-9_./\-]+\.(?:py|sh|sbatch|yml|yaml|md)):(\d+)(?:-(\d+))?`"
)
# Files whose line numbers are not worth tracking (prose that moves freely).
SKIP_TARGET_EXT = (".md",)
# Where to look for a bare basename.  Deliberately NOT the whole filesystem.
SEARCH_ROOTS = tuple(r for r in (
    os.environ.get("SGLANG_ENGINE_DEV",
                   os.path.join(WORKSPACE_ROOT, "sglang_engine_dev")) + "/python/sglang/srt",
    os.path.join(TRACK_ROOT, "src"),
    os.path.join(TRACK_ROOT, "results"),
    os.path.join(TRACK_ROOT, "scripts"),
    os.path.join(TRACK_ROOT, "tests"),
) if os.path.isdir(r))
if not SEARCH_ROOTS:
    print("no search root exists; check SGLANG_ENGINE_DEV", file=sys.stderr)
    sys.exit(2)

_index = None


def _build_index():
    """basename -> [absolute paths].  Ambiguity is REPORTED, never guessed."""
    global _index
    if _index is not None:
        return _index
    _index = {}
    for root in SEARCH_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "deprecated")]
            for fn in filenames:
                _index.setdefault(fn, []).append(os.path.join(dirpath, fn))
    return _index


def resolve(cited):
    """Resolve a cited path to exactly one file, or return (None, reason)."""
    base = os.path.basename(cited)
    # An explicit repo-relative path wins if it exists.
    for root in (TRACK_ROOT, PROJECT_ROOT):
        cand = os.path.join(root, cited)
        if os.path.isfile(cand):
            return cand, None
    hits = _build_index().get(base, [])
    # If the citation carries directory components, use them to disambiguate.
    if len(hits) > 1 and "/" in cited:
        tail = cited.replace("/", os.sep)
        narrowed = [h for h in hits if h.endswith(tail)]
        if narrowed:
            hits = narrowed
    if not hits:
        return None, "no such file under the search roots"
    if len(hits) == 1:
        return hits[0], None

    # ★Two kinds of ambiguity, and they mean OPPOSITE things.
    #
    #  (a) the SAME file installed twice -- `src/multiplex/*.py` is the tracked
    #      overlay and `sglang_engine_dev/.../multiplex/*.py` is what
    #      `sync_engine_tree.sh` installed from it.  Byte-identical means the
    #      sync is in effect; resolve to the tree that ACTUALLY RUNS.
    #  (b) genuinely different files sharing a basename (e.g.
    #      `managers/scheduler.py` vs `dllm/mixin/scheduler.py`).  Guessing
    #      here is how a citation ends up pointing at code that never ran --
    #      the "wrong tree grep" of lesson #31.  Refuse, and say so.
    #
    # ★A DIVERGENT overlay/dev pair also lands in (b), which is the point: the
    # NVTX incident was exactly an overlay that no longer matched the engine.
    shas = {}
    for h in hits:
        with open(h, "rb") as fh:
            shas.setdefault(hashlib.sha256(fh.read()).hexdigest(), []).append(h)
    if len(shas) == 1:
        runs = [h for h in hits if "sglang_engine_dev" in h]
        return (runs[0] if runs else sorted(hits)[0]), None
    listed = " | ".join(sorted(os.path.relpath(h, WORKSPACE_ROOT) for h in hits))
    return None, (f"ambiguous across {len(shas)} DIFFERENT contents -- "
                  f"cite with a path component: {listed}")


def _norm(lines):
    return "\n".join(l.rstrip() for l in lines)


def read_range(path, a, b):
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().split("\n")
    if a < 1 or b > len(lines):
        return None, len(lines)
    return lines[a - 1:b], len(lines)


def fingerprint(chunk):
    """sha over the whole cited range + the most DISTINCTIVE line as anchor.

    ★The anchor is what turns "this drifted" into "it is now at line N", so it
    must be the line least likely to repeat.  The first non-empty line is the
    obvious choice and a bad one: a range starting at `else:` anchored on
    `else:` and the relocation search returned fifteen candidates (observed).
    Longest stripped line is a cheap proxy for distinctiveness.
    """
    text = _norm(chunk)
    idx = [i for i, l in enumerate(chunk) if l.strip()]
    if idx:
        best = max(idx, key=lambda i: len(chunk[i].strip()))
        anchor, offset = chunk[best].strip(), best
    else:
        anchor, offset = "", 0
    return hashlib.sha256(text.encode()).hexdigest()[:16], anchor, offset


def relocate(path, anchor, chunk_len):
    """If the cited content moved, say where to.  Exact-anchor match only."""
    if not anchor:
        return []
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().split("\n")
    return [i + 1 for i, l in enumerate(lines) if l.strip() == anchor]


BARE = re.compile(r"`:(\d+)(?:-(\d+))?`")
# ★A citation followed by [HIST] deliberately quotes an OLDER tree -- "the audit
# wrote :155, the current tree says :228".  Checking it against HEAD would
# report a drift that is the whole point of the sentence.  The marker is
# explicit so that skipping is a declared act, not a silent exemption.
HIST = "[HIST]"


def citations_in(doc):
    """Both `path.py:12` and the bare `:12` shorthand these documents use.

    ★The bare form is 12 of the 21 citations in the A1 design and it is where
    the one surviving stale number lived, so a scanner that only sees the full
    form reports OK on a document whose worst citation it never looked at.
    A bare citation inherits the last full-form file named before it -- which
    is exactly how a reader resolves it.
    """
    with open(doc, "r", errors="replace") as fh:
        text = fh.read()
    out = []
    events = [(m.start(), "full", m) for m in CITE.finditer(text)]
    events += [(m.start(), "bare", m) for m in BARE.finditer(text)]
    current = None
    for _pos, kind, m in sorted(events, key=lambda e: e[0]):
        if kind == "full":
            cited, a, b = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
            current = cited
        else:
            if current is None:
                continue
            cited, a, b = current, int(m.group(1)), int(m.group(2) or m.group(1))
        if cited.endswith(SKIP_TARGET_EXT) or b < a:
            continue
        if text[m.end():m.end() + 40].lstrip().startswith(HIST):
            continue
        out.append((cited, a, b))
    # de-duplicate, keep order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def load_manifest():
    if not os.path.isfile(MANIFEST):
        return {}
    with open(MANIFEST) as fh:
        return json.load(fh)


def save_manifest(man):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST)


def process(docs, snapshot, man, force=False):
    violations, recorded, checked = [], 0, 0
    for doc in docs:
        rel = os.path.relpath(os.path.abspath(doc), TRACK_ROOT)
        entries = man.setdefault(rel, {}) if snapshot else man.get(rel, {})
        for cited, a, b in citations_in(doc):
            key = f"{cited}:{a}-{b}"
            path, why = resolve(cited)
            if path is None:
                violations.append(f"{rel}  {key}  UNRESOLVED  ({why})")
                continue
            chunk, nlines = read_range(path, a, b)
            if chunk is None:
                violations.append(
                    f"{rel}  {key}  RANGE  file has {nlines} lines")
                continue
            if not _norm(chunk).strip():
                violations.append(f"{rel}  {key}  BLANK  cited range is empty")
                continue
            sha, anchor, offset = fingerprint(chunk)
            if snapshot:
                prev = entries.get(key)
                if prev is not None and prev["sha"] != sha and not force:
                    # ★THE LIMIT THIS TOOL DECLARES, FIRING.  A re-snapshot of
                    # an edited file silently rebased every citation into it --
                    # which is exactly how `a1_q3k1_rule.py:83` kept saying
                    # `N_MIN_DECODE_STEPS` after the constant moved to :98,
                    # while --check reported OK.  The tool certified a false
                    # citation because I re-baselined instead of re-checking.
                    # Overwriting a CHANGED baseline is now an explicit act.
                    moved = relocate(path, prev["anchor"], b - a + 1)
                    off0 = prev.get("anchor_offset", 0)
                    hint = (f"  -> cite `{cited}:{moved[0] - off0}`"
                            if len(moved) == 1 else "")
                    violations.append(
                        f"{rel}  {key}  REBASE-REFUSED  the baseline changed "
                        f"({prev['sha']} -> {sha}); fix the citation, or pass "
                        f"--force to re-baseline deliberately{hint}\n"
                        f"      anchor was: {prev['anchor'][:80]}")
                    continue
                entries[key] = {"sha": sha, "anchor": anchor,
                                "anchor_offset": offset,
                                "target": os.path.relpath(path, WORKSPACE_ROOT)}
                recorded += 1
                continue
            prev = entries.get(key)
            if prev is None:
                continue                      # not snapshotted: nothing to compare
            checked += 1
            if prev["sha"] != sha:
                moved = relocate(path, prev["anchor"], b - a + 1)
                off = prev.get("anchor_offset", 0)
                if len(moved) == 1:
                    # the anchor sits `off` lines into the cited range, so the
                    # corrected citation is the anchor's new line minus that.
                    ns = moved[0] - off
                    ne = ns + (b - a)
                    hint = (f"  -> cite `{cited}:{ns}`" if ns == ne
                            else f"  -> cite `{cited}:{ns}-{ne}`")
                elif moved:
                    hint = f"  -> anchor found at {moved} (not unique)"
                else:
                    hint = "  -> anchor not found (content changed, not moved)"
                violations.append(
                    f"{rel}  {key}  DRIFT  expected {prev['sha']} got {sha}"
                    f"{hint}\n      anchor: {prev['anchor'][:88]}")
    return violations, recorded, checked


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="with --snapshot: re-baseline citations whose content "
                         "changed.  Without it, a changed baseline is REFUSED.")
    ap.add_argument("--all", action="store_true",
                    help="with --check: every document already in the manifest")
    ap.add_argument("docs", nargs="*")
    args = ap.parse_args(argv)
    if args.snapshot == args.check:
        print("give exactly one of --snapshot / --check", file=sys.stderr)
        return 2

    man = load_manifest()
    docs = list(args.docs)
    if args.all:
        docs += [os.path.join(TRACK_ROOT, d) for d in man]
    docs = sorted({os.path.abspath(d) for d in docs if os.path.isfile(d)})
    if not docs:
        print("no documents given", file=sys.stderr)
        return 2

    violations, recorded, checked = process(docs, args.snapshot, man, args.force)
    if args.snapshot:
        save_manifest(man)
        print(f"--- snapshotted {recorded} citation(s) in {len(docs)} doc(s)")
        print("    ★ a snapshot records, it does NOT validate: only take one "
              "right after verifying the citations by hand.")
    for v in violations:
        print(f"  [FAIL] {v}")
    tag = "snapshot" if args.snapshot else "check"
    print(f"--- {tag}: {checked} compared, {len(violations)} violation(s)"
          f"  {'OK' if not violations else 'FAIL'}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
