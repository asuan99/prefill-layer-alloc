#!/usr/bin/env python3
"""Structured unittest runner for the merge preflight (judge r2, F3; 2026-10-17).

usage (cwd = the directory `python -m unittest` would have been run from):
  pf_unittest_runner.py --status-json OUT discover -s START_DIR      # == python -m unittest discover -s START_DIR -v
  pf_unittest_runner.py --status-json OUT names MOD [MOD ...]         # == python -m unittest -v MOD ...

Why: the judge used to read each test's status from the `-v` text with a regex.  In the container (PYTHONUNBUFFERED=1)
test stdout and ResourceWarnings land between "<name> (<id>) ... " and the status token, so statuses were lost
(PREFLIGHT_FAILNEW_ANALYSIS_2026-10-16 sec 2-(a), 4).  This runner prints the SAME -v text to stderr (failure blocks and
the "Ran N tests" line are unchanged, so sigcmp signatures are unchanged) and additionally writes, at the end, one JSON
record per test id taken from the TestResult callbacks, never from the text:
  {"schema": "pf_status/v1", "ran": testsRun, "status": {id: "ok"|"FAIL"|"ERROR"|"skipped <reason>"|"expected failure"|
   "unexpected success"}, "was_successful": bool}
Ids are normalised the way sigcmp names failure blocks: "setUpClass (mod.Cls)" -> "mod.Cls"; a failing subTest counts for
its parent id (ERROR outranks FAIL).  The JSON is written atomically after the run; a killed run leaves no JSON, and the
judge then treats the repetition as unreadable (fail-closed).  Exit code as unittest.main: 0 iff successful, else 1.
No -b (buffering): it would move captured output into the failure blocks and change signatures.
"""
import json, os, re, sys, unittest, warnings

_HOLDER = re.compile(r'^(?:setUpClass|tearDownClass|setUpModule|tearDownModule) \((.+)\)$')
_RANK = {'ok': 0, 'expected failure': 0, 'skipped': 1, 'unexpected success': 2, 'FAIL': 3, 'ERROR': 4}


def norm_id(test):
    tid = test.id()
    m = _HOLDER.match(tid)
    if m:
        return m.group(1)
    if isinstance(test, unittest.case._SubTest):          # noqa: SLF001 - the parent test id
        return test.test_case.id()
    return tid


class StatusResult(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pf_status = {}

    def _set(self, test, st):
        tid = norm_id(test)
        old = self.pf_status.get(tid)
        if old is None or _RANK[st.split(' ', 1)[0] if st.startswith('skipped') else st] >= \
                _RANK[old.split(' ', 1)[0] if old.startswith('skipped') else old]:
            self.pf_status[tid] = st

    def addSuccess(self, test):
        super().addSuccess(test); self._set(test, 'ok')

    def addFailure(self, test, err):
        super().addFailure(test, err); self._set(test, 'FAIL')

    def addError(self, test, err):
        super().addError(test, err); self._set(test, 'ERROR')

    def addSkip(self, test, reason):
        super().addSkip(test, reason); self._set(test, 'skipped ' + str(reason)[:200])

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err); self._set(test, 'expected failure')

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test); self._set(test, 'unexpected success')

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._set(test, 'FAIL' if issubclass(err[0], test.failureException) else 'ERROR')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 3 or argv[0] != '--status-json' or argv[2] not in ('discover', 'names'):
        sys.stderr.write(__doc__); return 2
    out, mode, rest = argv[1], argv[2], argv[3:]
    # `python -m unittest` puts the cwd (not this script's directory) first on sys.path
    sys.path[0] = os.getcwd()
    loader = unittest.TestLoader()
    if mode == 'discover':
        if len(rest) != 2 or rest[0] != '-s':
            sys.stderr.write('discover needs exactly: -s START_DIR\n'); return 2
        suite = loader.discover(rest[1])
    else:
        if not rest:
            sys.stderr.write('names needs at least one module\n'); return 2
        suite = loader.loadTestsFromNames(rest)
    # unittest.main(): warnings='default' unless -W was given
    runner = unittest.TextTestRunner(verbosity=2, resultclass=StatusResult, warnings=None if sys.warnoptions else 'default')
    result = runner.run(suite)
    rec = {'schema': 'pf_status/v1', 'ran': result.testsRun, 'status': dict(sorted(result.pf_status.items())),
           'was_successful': result.wasSuccessful()}
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=0, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    os.replace(tmp, out)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
