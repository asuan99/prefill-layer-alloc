#!/usr/bin/env python3
"""Lesson 53: pincheck must FAIL on a log whose signature was altered. usage: mutants.py PIN.json LOG(+LOG2)"""
import sys, re, tempfile, os, subprocess
HERE=os.path.dirname(os.path.abspath(__file__))
pre, post = sys.argv[1], sys.argv[2]
parts = post.split('+')
base = [open(p, encoding='utf-8', errors='replace').read() for p in parts]
def first_block(t, needle):
    i = t.find(needle); assert i >= 0, needle; return i
def mutate(name, fn):
    ts = list(base); ok = False
    for k, t in enumerate(ts):
        n = fn(t)
        if n != t: ts[k] = n; ok = True; break
    assert ok, f'mutant {name} did not apply'
    d = tempfile.mkdtemp(prefix='mut_', dir=HERE); fs = []
    for k, t in enumerate(ts):
        f = f'{d}/p{k}.log'; open(f, 'w').write(t); fs.append(f)
    r = subprocess.run([sys.executable, f'{HERE}/pincheck.py', pre, '+'.join(fs)], capture_output=True, text=True)
    summ = [l for l in r.stdout.split('\n') if l.startswith('pinned=')]
    print(f'{name:48s} rc={r.returncode}  {summ[0] if summ else r.stdout[-200:]}')
    return r.returncode
def sub1(a, b):  # replace first occurrence inside a FAIL/ERROR block region only
    def f(t):
        return t.replace(a, b, 1)
    return f
rcs = {}
# control: unchanged post -> signature_changed must be 0 (rc may be 1 only through new_in_post, reported separately)
d = tempfile.mkdtemp(prefix='mut_', dir=HERE); fs=[]
for k,t in enumerate(base):
    f=f'{d}/p{k}.log'; open(f,'w').write(t); fs.append(f)
r = subprocess.run([sys.executable, f'{HERE}/pincheck.py', pre, '+'.join(fs)], capture_output=True, text=True)
print('control (unmutated)'.ljust(48), 'rc=%d'%r.returncode, [l for l in r.stdout.split('\n') if l.startswith('pinned=')])
assert r.returncode == 0, 'control must pass'
rcs['M1 env-fail cause swapped (venv -> Permission denied)'] = mutate('M1 same id, other cause (venv->Permission denied)', sub1('sglang_engine_venv/bin/activate: No such file or directory','sglang_engine_venv/bin/activate: Permission denied'))
rcs['M2'] = mutate('M2 exception type ModuleNotFound->ImportError', sub1("ModuleNotFoundError: No module named 'sgl_kernel'","ImportError: No module named 'sgl_kernel'"))
rcs['M3'] = mutate('M3 DRIFT sha altered inside L2 block', lambda t: re.sub(r'(FAIL: test_unbroken_copy_runs.*?DRIFT sglang/srt/multiplex/holb_probe.py: mirror src/multiplex/holb_probe.py = )\w{16}', r'\g<1>deadbeefdeadbeef', t, count=1, flags=re.S))
rcs['M4'] = mutate('M4 ABORT_REG_DRIFT -> ABORT_CONFIG (one block)', sub1('ABORT_REG_DRIFT config','ABORT_CONFIG config'))
rcs['M5'] = mutate('M5 FAIL->ERROR kind swapped (one block)', lambda t: re.sub(r'^FAIL: (test_unbroken_copy_runs)', r'ERROR: \1', t, count=1, flags=re.M))
rcs['M6'] = mutate('M6 one failing block deleted (id vanishes)', lambda t: re.sub(r'={60,}\nFAIL: test_unbroken_copy_runs.*?(?=\n={60,}\n)', '', t, count=1, flags=re.S))
bad = [k for k,v in rcs.items() if v == 0]
print('MUTANTS_ALL_DETECTED' if not bad else f'MUTANT_NOT_DETECTED: {bad}')
sys.exit(1 if bad else 0)
