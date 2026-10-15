#!/usr/bin/env python3
"""Preflight judge (merge CL-3): pair PRE/POST unittest logs from ONE container and label the result.

usage: preflight_judge.py --pre LOG[+LOG..] [LOG2[+..]] --post LOG[+LOG..] [..] \
          [--l2 preflight_cl3_l2_ids.txt] [--targets preflight_cl3_targets.txt] [--setup-ok 0|1 ...] [--json OUT]
  --pre / --post take one or more repetitions (each repetition = LOG or LOG+LOG when a run was split).
Labels (exit code): PREFLIGHT_PASS (0) / PREFLIGHT_FAIL_NEW (1) / PREFLIGHT_UNVERIFIABLE (3).

Pairing: the unit is (test id, signature) from sigcmp.parse_multi (exception type, masked message, keys, DRIFT digest).
 NEW failure = a post failure whose id is not failing in pre with the identical signature and that is not in the L2 list
               (L2 = pass before, fail after, sealed-sha mirror checks; membership by id). A signature change on an id
               that fails in both is NEW too (same id, different cause). Fail-closed.
 Coverage    = every CL-3 target (targets file) must have run and passed in post; a skipped/import-failed/failed target
               is a HOLE -> UNVERIFIABLE (a hole is a gap in the evidence, not a pass).
 INVALID     = setup flags from the shell script (--invalid REASON), a run without a 'Ran N tests' line, repetitions of
               the same arm that disagree (flaky), or L2 never reproduced (post tree not actually different) -> UNVERIFIABLE.
Precedence: INVALID -> UNVERIFIABLE; else NEW -> FAIL_NEW; else holes -> UNVERIFIABLE; else PASS.
This file never reads pin_post.json; the local pin is not a reference for the VESSL environment.
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigcmp import parse_multi

SKIP_RE = re.compile(r'^(?:\S+) \(([\w.]+)\)[^\n]*? \.\.\. skipped (.*)$', re.M)

def read_list(p):
    return [l.strip() for l in open(p, encoding='utf-8') if l.strip() and not l.startswith('#')]

def read_targets(p):
    out = []
    for l in open(p, encoding='utf-8'):
        if not l.strip() or l.startswith('#'): continue
        s, k, v = l.rstrip('\n').split('\t'); out.append((s, k, v))
    return out

def load_arm(reps):
    runs = []
    for r in reps:
        st, sg, ran, summ = parse_multi(r)
        # sigcmp's status regex only knows skips quoted with single quotes; unittest also prints skipped "..." (setUpClass
        # skips such as the zamba2 import skip).  Scan the raw text so a skip can never read as "not seen -> fine".
        for path in r.split('+'):
            for m in SKIP_RE.finditer(open(path, encoding='utf-8', errors='replace').read()):
                st.setdefault(m.group(1), 'skipped ' + m.group(2).strip()[:80])
        runs.append({'status': st, 'sigs': sg, 'ran': ran, 'summary': summ})
    return runs

def mod_of(tid):
    return tid.rsplit('.', 1)[-1] if tid.startswith('unittest.loader._FailedTest.') else tid.split('.', 1)[0]

def target_state(status, sigs, kind, value):
    """-> (verified: bool, detail: dict)"""
    if kind == 'id':
        s = status.get(value)
        return s == 'ok', {'status': s or 'absent'}
    ok = fail = skip = 0; imp = 0
    for tid, s in status.items():
        if tid.startswith('unittest.loader._FailedTest.'):
            if mod_of(tid) == value: imp += 1
            continue
        if tid.split('.', 1)[0] != value: continue
        if s == 'ok': ok += 1
        elif s.startswith('skipped'): skip += 1
        else: fail += 1
    for tid in sigs:                                   # failure blocks the status regex may have missed
        if tid.startswith('unittest.loader._FailedTest.') and mod_of(tid) == value and not imp: imp += 1
    return (ok > 0 and fail == 0 and skip == 0 and imp == 0), {'ok': ok, 'fail': fail, 'skipped': skip, 'import_error': imp}

def judge(pre_reps, post_reps, l2, targets, invalid=()):
    pre, post = load_arm(pre_reps), load_arm(post_reps)
    inv = list(invalid)
    for name, arm in (('pre', pre), ('post', post)):
        for i, r in enumerate(arm):
            if not r['ran']: inv.append(f'{name} rep{i+1}: no "Ran N tests" line (run killed or log truncated)')
        for i, r in enumerate(arm[1:], 2):
            a, b = arm[0]['sigs'], r['sigs']
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if diff: inv.append(f'{name} rep1 vs rep{i} disagree on {len(diff)} failing id(s) (flaky), e.g. {diff[0]}')
    # use the union over repetitions only after the agreement check above: take rep1 as the arm's record
    ps, qs = pre[0]['sigs'], post[0]['sigs']
    pst, qst = pre[0]['status'], post[0]['status']
    new, changed, l2_hit, l2_sigdiff = [], [], [], []
    for tid, sig in sorted(qs.items()):
        if tid in ps:
            if ps[tid] != sig: changed.append(tid)
        elif tid in l2 and pst.get(tid) == 'ok':
            l2_hit.append(tid)
        else:
            new.append(tid)
    for tid in l2:
        if tid not in qs: pass
    l2_missing = sorted(t for t in l2 if t not in qs)
    if l2 and not l2_hit: inv.append('L2 never reproduced: post tree does not look different from pre (wrong tree?)')
    fixed = sorted(set(ps) - set(qs))
    holes, cov = [], {}
    for s, k, v in targets:
        ok, det = target_state(qst, qs, k, v)
        d = cov.setdefault(s, {'verified': 0, 'total': 0, 'items': {}}); d['total'] += 1; d['items'][v] = det
        if ok: d['verified'] += 1
        else:
            # a target that fails only with an L2 id is still verified-by-pair? no: L2 ids are not CL-3 targets
            holes.append(f'{s}:{v}')
    fail_new = sorted(new + changed)
    if inv: label = 'PREFLIGHT_UNVERIFIABLE'; rc = 3
    elif fail_new: label = 'PREFLIGHT_FAIL_NEW'; rc = 1
    elif holes: label = 'PREFLIGHT_UNVERIFIABLE'; rc = 3
    else: label = 'PREFLIGHT_PASS'; rc = 0
    return rc, {'label': label, 'invalid': inv, 'new_failures': new, 'signature_changed': changed,
                'l2_reproduced': len(l2_hit), 'l2_expected': len(l2), 'l2_not_failing_in_post': l2_missing,
                'fixed_in_post': fixed, 'holes': holes, 'coverage': {k: {'verified': v['verified'], 'total': v['total']} for k, v in cov.items()},
                'coverage_detail': cov,
                'pre': {'ran': pre[0]['ran'], 'summary': pre[0]['summary'], 'failing_ids': len(ps)},
                'post': {'ran': post[0]['ran'], 'summary': post[0]['summary'], 'failing_ids': len(qs)}}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', nargs='+', required=True); ap.add_argument('--post', nargs='+', required=True)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--l2', default=os.path.join(here, 'preflight_cl3_l2_ids.txt'))
    ap.add_argument('--targets', default=os.path.join(here, 'preflight_cl3_targets.txt'))
    ap.add_argument('--invalid', action='append', default=[]); ap.add_argument('--json')
    a = ap.parse_args(argv)
    rc, res = judge(a.pre, a.post, read_list(a.l2), read_targets(a.targets), a.invalid)
    print(res['label'])
    print(f"pre ran={res['pre']['ran']} failing={res['pre']['failing_ids']} | post ran={res['post']['ran']} failing={res['post']['failing_ids']}")
    print(f"L2 reproduced {res['l2_reproduced']}/{res['l2_expected']}; new={len(res['new_failures'])} sig_changed={len(res['signature_changed'])} fixed_in_post={len(res['fixed_in_post'])}")
    for k, v in res['coverage'].items(): print(f"  coverage {k}: {v['verified']}/{v['total']}")
    for x in res['invalid']: print('  INVALID', x)
    for x in res['new_failures']: print('  NEW    ', x)
    for x in res['signature_changed']: print('  CHANGED', x)
    for x in res['holes'][:40]: print('  HOLE   ', x)
    if a.json: json.dump(res, open(a.json, 'w'), indent=1, ensure_ascii=False)
    return rc

if __name__ == '__main__':
    sys.exit(main())
