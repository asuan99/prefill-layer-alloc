#!/usr/bin/env python3
"""Mutation ledger for the r2 judge (lesson 53: a judge that cannot fail proves nothing).  Local, CPU.

usage: preflight_judge_mutants.py [FIXTURE_DIR]
  Copies the judge tools into a temp dir, applies ONE exact-string substitution per mutant (the anchor must occur exactly once,
  else the row is BAD_ANCHOR), and runs preflight_judge_selftest.py (synthetic) and preflight_judge_fixture_selftest.py (real logs
  of pdmux-preflight-cl3-68f386b, read-only) on the copy.
  KILLED  = a self-test returned rc 1 and printed at least one 'FAIL ' case line (a clean detection, not a crash).
  CRASHED = a self-test died otherwise (traceback / other rc) -- reported, NOT counted as killed.
  SURVIVED = both self-tests passed on the mutant.
  The unmutated copy (CONTROL) must pass both, else the ledger is void.  rc 0 iff CONTROL ok and every mutant KILLED.
"""
import os, shutil, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__))
FILES = ['preflight_judge.py', 'sigcmp.py', 'pf_unittest_runner.py', 'preflight_judge_selftest.py',
         'preflight_judge_fixture_selftest.py', 'preflight_cl3_l2_ids.txt', 'preflight_cl3_targets.txt',
         'preflight_cl3_cpu_timing_declared.txt']
J, SG, RU = 'preflight_judge.py', 'sigcmp.py', 'pf_unittest_runner.py'
MUTANTS = [
    ('J01 F6 L2 hole removed', J, "        if pst.get(tid) != 'ok':\n            l2_holes.append(", "        if False:\n            l2_holes.append("),
    ('J02 F5 arm mask removed', SG, "    s = ARM_RE.sub('<arm>', s)\n", "    pass\n"),
    ('J03 F5 over-mask (whole path)', SG, "    s = ARM_RE.sub('<arm>', s)\n", "    s = re.sub(r'/data/runs/\\S+', '<arm>', s)\n"),
    ('J04 F3 strict -> first token wins', SG, "if cur is not None and len(toks) == 1: status[cur] = toks[0]",
     "if cur is not None and toks: status[cur] = toks[0]"),
    ('J05 F3 token on its own line ignored', SG, "m = TOK_END.search(line) or TOK_ALONE.match(line)", "m = TOK_END.search(line)"),
    ('J06 declaration applied on any spec', J, "excl = set(declared['ids']) if (declared and cpu_spec) else set()",
     "excl = set(declared['ids']) if declared else set()"),
    ('J07 declaration never applied', J, "excl = set(declared['ids']) if (declared and cpu_spec) else set()", "excl = set()"),
    ('J08 declared target allowed', J, "        if bad: inv.append(f'declared CPU-timing ids overlap", "        if False: inv.append(f'declared CPU-timing ids overlap"),
    ('J09 JSON/text disagreement ignored', J, "        if kind_of(jst[tid]) != kind_of(s):", "        if False:"),
    ('J10 failure blocks vs JSON ignored', J, "    if set(sg) != failed:", "    if False:"),
    ('J11 --require-structured ignored', J, "            if require_structured and r['source'] != 'json':", "            if False:"),
    ('J12 JSON ran check removed', J, "    if jran != ran:", "    if False:"),
    ('J13 record: declaration sha not compared', J, "        elif jdec.get('sha256') != dsha:", "        elif False:"),
    ('J14 record: applied without the file accepted', J, "    elif isinstance(jdec, dict) and jdec.get('applied'):", "    elif False:"),
    ('J15 scope always CPU', J, "'conclusion_scope': SCOPE_CPU if cpu_spec else SCOPE_GPU", "'conclusion_scope': SCOPE_CPU"),
    ('J16 spec: GPU contradiction ignored', J, "    if flag and gpu_seen:\n", "    if False:\n"),
    # (a mutant "return bool(flag), None" on the last line is EQUIVALENT: the contradiction branch has already returned -- not listed)
    ('J17 spec: contradiction treated as CPU', J, "        return False, 'CPU spec declared", "        return True, 'CPU spec declared"),
    ('J18 runner: subTest failure dropped', RU, "        if err is not None:\n            self._set(", "        if False:\n            self._set("),
    ('J19 runner: fixture holder id not normalised', RU, "    if m:\n        return m.group(1)", "    if False:\n        return m.group(1)"),
    ('J20 runner: ran count off by one', RU, "'ran': result.testsRun,", "'ran': result.testsRun + 1,"),
    ('J21 F6 reverted: pre-failing L2 compared by signature', J,
     "            continue                                   # pre not ok: already a hole above",
     "            if pst.get(tid) != 'ok' and tid in ps and ps[tid] != sig: changed.append(tid)\n            continue"),
]

def run(d, script, extra):
    r = subprocess.run([sys.executable, '-I', os.path.join(d, script)] + extra, capture_output=True, text=True, cwd=d)
    fails = [l for l in r.stdout.splitlines() if l.startswith('FAIL ')]
    if r.returncode == 0: return 'pass', ''
    if r.returncode == 1 and fails: return 'killed', fails[0][5:80]
    return 'crashed', (r.stdout + r.stderr).strip().splitlines()[-1][:120] if (r.stdout + r.stderr).strip() else f'rc {r.returncode}'

def trial(fixture, name=None, path=None, old=None, new=None):
    with tempfile.TemporaryDirectory() as d:
        for f in FILES: shutil.copy2(os.path.join(here, f), os.path.join(d, f))
        if path:
            p = os.path.join(d, path); t = open(p, encoding='utf-8').read()
            if t.count(old) != 1: return 'BAD_ANCHOR', f'anchor occurs {t.count(old)}x'
            open(p, 'w', encoding='utf-8').write(t.replace(old, new))
        res = [run(d, 'preflight_judge_selftest.py', []), run(d, 'preflight_judge_fixture_selftest.py', [fixture])]
    if all(s == 'pass' for s, _ in res): return 'SURVIVED' if path else 'CONTROL_OK', ''
    k = [(i, w) for i, (s, w) in enumerate(res) if s == 'killed']
    by = 'syn=%s fix=%s' % (res[0][0], res[1][0])
    if k: return 'KILLED', f"[{by}] {('synthetic', 'fixture')[k[0][0]]}: {k[0][1]}"
    return 'CRASHED', '; '.join(w for s, w in res if s == 'crashed')

def main(argv):
    fixture = argv[1] if len(argv) > 1 else os.path.normpath(os.path.join(here, '..', '..', 'preflight_cl3',
                                                                          'pdmux-preflight-cl3-68f386b-20261016114954'))
    if not os.path.isfile(os.path.join(fixture, 'logs', 'pre.rep1.log')):
        print(f'FIXTURE ABSENT: {fixture}'); return 2
    st, why = trial(fixture)
    print(f'CONTROL{"":40s} {st} {why}')
    if st != 'CONTROL_OK': print('LEDGER VOID'); return 1
    bad = []
    for name, path, old, new in MUTANTS:
        st, why = trial(fixture, name, path, old, new)
        print(f'{name:46s} {st:9s} {why}')
        if st != 'KILLED': bad.append(name)
    print('MUTANTS_ALL_KILLED' if not bad else f'NOT_KILLED: {bad}')
    return 0 if not bad else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
