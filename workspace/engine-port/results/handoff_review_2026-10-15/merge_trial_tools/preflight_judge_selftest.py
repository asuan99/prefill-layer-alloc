#!/usr/bin/env python3
"""Self-test of preflight_judge.py on synthetic unittest -v logs (no GPU, no repo checkout).
A control must give PASS and each mutant must move the label (lesson 53: a judge that cannot fail proves nothing).
usage: python3 preflight_judge_selftest.py   -> rc 0 iff every case behaves as expected."""
import os, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import preflight_judge as J

L2 = J.read_list(os.path.join(here, 'preflight_cl3_l2_ids.txt'))
TG = J.read_targets(os.path.join(here, 'preflight_cl3_targets.txt'))
SEP = '=' * 70

def split_id(tid):
    mod, rest = tid.split('.', 1); return rest.rsplit('.', 1)[-1], tid

def build(entries, ran=True):
    """entries: list of (id, kind, exc, msg); kind in ok|FAIL|ERROR|skip"""
    head, blocks = [], []
    for tid, kind, exc, msg in entries:
        if tid.startswith('unittest.loader._FailedTest.'):
            name = tid.rsplit('.', 1)[-1]; shown = tid
        else:
            name, shown = split_id(tid)
        if kind == 'ok': head.append(f'{name} ({shown}) ... ok')
        elif kind == 'skip': head.append(f'setUpClass ({shown.rsplit(".",1)[0]}) ... skipped "{msg}"')
        else:
            head.append(f'{name} ({shown}) ... {kind}')
            blocks.append(f'{SEP}\n{kind}: {name} ({shown})\n{"-"*70}\nTraceback (most recent call last):\n  File "x.py", line 1, in f\n{exc}: {msg}\n')
    tail = f'{"-"*70}\nRan {len(head)} tests in 1.0s\n\nFAILED (errors={len(blocks)})\n' if ran else ''
    return '\n'.join(head) + '\n\n' + '\n'.join(blocks) + '\n' + tail

def tids(s, k):
    out = []
    for sset, kind, v in TG:
        if sset != s: continue
        if kind == 'id': out.append(v)
        else: out += [f'{v}.Test.test_{i}' for i in range(k)]
    return out

TARGET_IDS = [t for s in sorted({x[0] for x in TG}) for t in tids(s, 3)]
ENV_FAIL = [('test_x_env.Test.test_needs_archive', 'ERROR', 'FileNotFoundError', 'no lambda* table at /p/q.json')]
BASE_OK = [(t, 'ok', '', '') for t in TARGET_IDS] + [('test_misc.Test.test_a', 'ok', '', '')]
def pre_entries(): return BASE_OK + [(t, 'ok', '', '') for t in L2] + ENV_FAIL
def post_entries():
    return BASE_OK + [(t, 'FAIL', 'AssertionError', 'DRIFT mirror != registered') for t in L2] + ENV_FAIL

def run(pre, post, **kw):
    with tempfile.TemporaryDirectory() as d:
        def w(name, txt):
            p = os.path.join(d, name); open(p, 'w').write(txt); return p
        pre_reps = [w(f'pre{i}', t) for i, t in enumerate(pre if isinstance(pre, list) else [pre])]
        post_reps = [w(f'post{i}', t) for i, t in enumerate(post if isinstance(post, list) else [post])]
        rc, res = J.judge(pre_reps, post_reps, L2, TG, kw.get('invalid', ()))
        return res['label'], res

def case(name, pre, post, want, **kw):
    lab, res = run(pre, post, **kw)
    ok = lab == want
    print(('ok   ' if ok else 'FAIL ') + f'{name}: {lab} (want {want})')
    if not ok:
        print('   ', {k: res[k] for k in ('invalid', 'new_failures', 'signature_changed', 'holes')})
    return ok

def main():
    B = build(pre_entries()), build(post_entries())
    allok = True
    allok &= case('control', B[0], B[1], 'PREFLIGHT_PASS')
    pe = post_entries() + [('test_new.Test.test_regress', 'FAIL', 'AssertionError', 'boom')]
    allok &= case('M1 new failing id in post', B[0], build(pe), 'PREFLIGHT_FAIL_NEW')
    pe = [(i, k, e, 'different cause') if i == ENV_FAIL[0][0] else (i, k, e, m) for i, k, e, m in post_entries()]
    allok &= case('M2 same failing id, other signature', B[0], build(pe), 'PREFLIGHT_FAIL_NEW')
    pe = [(i, 'skip', '', 'import failed') if i == TARGET_IDS[0] else (i, k, e, m) for i, k, e, m in post_entries()]
    pe = [(i, k, e, m) for i, k, e, m in pe if i != 'x']
    allok &= case('M3 target module skipped', B[0], build(pe), 'PREFLIGHT_UNVERIFIABLE')
    mod = [x[2] for x in TG if x[0] == 'A_import9'][0]
    pe = [(i, k, e, m) for i, k, e, m in post_entries() if not i.startswith(mod + '.')]
    pe.append((f'unittest.loader._FailedTest.{mod}', 'ERROR', 'ModuleNotFoundError', "No module named 'sgl_kernel'"))
    allok &= case('M4a import error only after the merge (pre ok)', B[0], build(pe), 'PREFLIGHT_FAIL_NEW')
    pre_imp = [(i, k, e, m) for i, k, e, m in pre_entries() if not i.startswith(mod + '.')]
    pre_imp.append((f'unittest.loader._FailedTest.{mod}', 'ERROR', 'ModuleNotFoundError', "No module named 'sgl_kernel'"))
    allok &= case('M4b import error in both arms (the local situation: a hole, not a pass)', build(pre_imp), build(pe), 'PREFLIGHT_UNVERIFIABLE')
    allok &= case('M5 L2 never fails (post == pre)', B[0], B[0], 'PREFLIGHT_UNVERIFIABLE')
    allok &= case('M6 post log truncated (no Ran line)', B[0], build(post_entries(), ran=False), 'PREFLIGHT_UNVERIFIABLE')
    allok &= case('M7 repetitions disagree (flaky)', B[0], [B[1], build(post_entries() + [('test_flaky.T.t', 'FAIL', 'AssertionError', 'x')])], 'PREFLIGHT_UNVERIFIABLE')
    v17 = [x[2] for x in TG if x[0] == 'C_venv17'][0]
    pre_v = [(i, 'ERROR', 'FileNotFoundError', 'sglang_engine_venv/bin/activate: No such file') if i == v17 else (i, k, e, m) for i, k, e, m in pre_entries()]
    post_v = [(i, 'ERROR', 'FileNotFoundError', 'sglang_engine_venv/bin/activate: No such file') if i == v17 else (i, k, e, m) for i, k, e, m in post_entries()]
    allok &= case('M8 venv test fails identically in both arms (shared, not a pass)', build(pre_v), build(post_v), 'PREFLIGHT_UNVERIFIABLE')
    pe = post_entries() + [('test_new.Test.test_regress', 'FAIL', 'AssertionError', 'boom')]
    pe = [(i, 'skip', '', 'x') if i == TARGET_IDS[0] else (i, k, e, m) for i, k, e, m in pe]
    allok &= case('M9 new failure outranks a hole', B[0], build(pe), 'PREFLIGHT_FAIL_NEW')
    allok &= case('M10 setup flag from the shell script', B[0], B[1], 'PREFLIGHT_UNVERIFIABLE', invalid=['pre tree not reconstructed'])
    pe = [(i, k, e, m) for i, k, e, m in post_entries() if i != L2[0]]
    allok &= case('M11 one L2 id absent from post (only 0 reproduced is invalid)', B[0], build(pe), 'PREFLIGHT_PASS')
    print('SELFTEST', 'PASS' if allok else 'FAIL')
    return 0 if allok else 1

if __name__ == '__main__':
    sys.exit(main())
