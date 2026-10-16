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
    """entries: list of (id, kind, exc, msg); kind in ok|FAIL|ERROR|skip, and (r2, shapes seen in the container log of
    pdmux-preflight-cl3-68f386b, PREFLIGHT_FAILNEW_ANALYSIS sec 2-(a), 4):
      ok_stdout  "<name> (<id>) ... <msg>" then "ok" on its own line (test stdout between header and token, PYTHONUNBUFFERED)
      ok_warn    header, docstring line with a ResourceWarning glued on, two warning lines, then "ok"
      ok_twice   " ... ok" and a second line "ok" (two tokens -> ambiguous -> must stay unread)"""
    head, blocks = [], []
    for tid, kind, exc, msg in entries:
        if tid.startswith('unittest.loader._FailedTest.'):
            name = tid.rsplit('.', 1)[-1]; shown = tid
        else:
            name, shown = split_id(tid)
        if kind == 'ok': head.append(f'{name} ({shown}) ... ok')
        elif kind == 'ok_stdout': head.append(f'{name} ({shown}) ... {msg or "SELFTEST OK (seeds = 1)"}\nok')
        elif kind == 'ok_warn':
            head.append(f'{name} ({shown})\nThe mirrored patch must reproduce ... /data/runs/j/pf_work/post/t.py:667: ResourceWarning: '
                        f'unclosed file <_io.BufferedReader>\n  r = subprocess.run(\nResourceWarning: Enable tracemalloc to get the '
                        f'object allocation traceback\nok')
        elif kind == 'ok_twice': head.append(f'{name} ({shown}) ... ok\nok')
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

def status_json(entries, ran_delta=0, override=None):
    """The pf_unittest_runner.py record for the same entries (r2 F3)."""
    import json as _j
    st = {}
    for tid, kind, exc, msg in entries:
        if kind == 'skip': st[tid.rsplit('.', 1)[0]] = 'skipped ' + msg
        elif kind.startswith('ok'): st[tid] = 'ok'
        else: st[tid] = kind
    st.update(override or {})
    return _j.dumps({'schema': 'pf_status/v1', 'ran': len(entries) + ran_delta, 'status': st, 'was_successful': False})

def run(pre, post, **kw):
    """kw: invalid, pre_json / post_json (list of JSON text or None per repetition, r2), declared (ids), cpu_spec, require_structured."""
    with tempfile.TemporaryDirectory() as d:
        def w(name, txt):
            p = os.path.join(d, name); open(p, 'w').write(txt); return p
        def reps(arm, logs, jsons):
            logs = logs if isinstance(logs, list) else [logs]
            out = []
            for i, t in enumerate(logs):
                p = w(f'{arm}{i}', t)
                if jsons and jsons[i] is not None: w(f'{arm}{i}.status.json', jsons[i])
                out.append(p)
            return out
        pre_reps, post_reps = reps('pre', pre, kw.get('pre_json')), reps('post', post, kw.get('post_json'))
        dec = None
        if kw.get('declared') is not None:
            dec = J.read_declared(w('declared.txt', '# test declaration\n' + ''.join(i + '\n' for i in kw['declared'])))
        rc, res = J.judge(pre_reps, post_reps, L2, TG, kw.get('invalid', ()), declared=dec, cpu_spec=kw.get('cpu_spec', False),
                          require_structured=kw.get('require_structured', False))
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
    allok &= r2_cases()
    allok &= runner_cases()
    allok &= record_cases()
    print('SELFTEST', 'PASS' if allok else 'FAIL')
    return 0 if allok else 1

def r2_cases():
    """judge r2 (2026-10-17): F3 strict/structured status, F5 arm-path mask, F6 L2 fail-closed, CPU-timing declaration."""
    ok = True
    pre_e, post_e = pre_entries(), post_entries()
    l2a, tgt = L2[0], 'test_holb_probe.TestSchedulerHooksInstalled.test_patch_reapplies_the_hooks_and_nothing_else'
    sub = lambda es, tid, kind, exc='', msg='': [(i, kind, exc, msg) if i == tid else (i, k, e, m) for i, k, e, m in es]
    # F3 text: the two container shapes are read as ok (control), an ambiguous segment is never ok
    ok &= case('S1 L2 id ok in pre with stdout between header and token -> L2 reproduced', build(sub(pre_e, l2a, 'ok_stdout')),
               build(post_e), 'PREFLIGHT_PASS')
    ok &= case('S2 target ok after interleaved ResourceWarnings -> verified', build(pre_e), build(sub(post_e, tgt, 'ok_warn')),
               'PREFLIGHT_PASS')
    ok &= case('S3 target segment with two status tokens -> unread -> HOLE', build(pre_e), build(sub(post_e, tgt, 'ok_twice')),
               'PREFLIGHT_UNVERIFIABLE')
    ok &= case('S3b L2 id with two tokens in pre -> unread -> L2_UNVERIFIED HOLE (not NEW)', build(sub(pre_e, l2a, 'ok_twice')),
               build(post_e), 'PREFLIGHT_UNVERIFIABLE')
    # F3 structured: JSON agreeing with the text passes; every disagreement is INVALID
    PJ, QJ = [status_json(pre_e)], [status_json(post_e)]
    ok &= case('S4 structured JSON agrees (control)', build(pre_e), build(post_e), 'PREFLIGHT_PASS', pre_json=PJ, post_json=QJ,
               require_structured=True)
    ok &= case('S4a structured required, JSON absent -> INVALID', build(pre_e), build(post_e), 'PREFLIGHT_UNVERIFIABLE',
               pre_json=PJ, require_structured=True)
    ok &= case('S4b JSON says FAIL where the text says ok -> INVALID', build(pre_e), build(post_e), 'PREFLIGHT_UNVERIFIABLE',
               pre_json=PJ, post_json=[status_json(post_e, override={tgt: 'FAIL'})])
    ok &= case('S4c JSON ran != text Ran -> INVALID', build(pre_e), build(post_e), 'PREFLIGHT_UNVERIFIABLE', pre_json=PJ,
               post_json=[status_json(post_e, ran_delta=1)])
    ok &= case('S4d JSON failure without a failure block (text status unread) -> INVALID', build(pre_e),
               build(sub(post_e, 'test_misc.Test.test_a', 'ok_twice')), 'PREFLIGHT_UNVERIFIABLE',
               pre_json=PJ, post_json=[status_json(post_e, override={'test_misc.Test.test_a': 'ERROR'})])
    ok &= case('S4e JSON ok where the text reads ERROR (failure block present) -> INVALID', build(pre_e), build(post_e), 'PREFLIGHT_UNVERIFIABLE',
               pre_json=PJ, post_json=[status_json(post_e, override={ENV_FAIL[0][0]: 'ok'})])
    ok &= case('S4g JSON skipped where the text reads ok (blocks agree) -> INVALID', build(pre_e), build(post_e), 'PREFLIGHT_UNVERIFIABLE',
               pre_json=PJ, post_json=[status_json(post_e, override={'test_misc.Test.test_a': 'skipped x'})])
    ok &= case('S4f structured: the ambiguous text segment is fine when JSON is ok', build(pre_e), build(sub(post_e, tgt, 'ok_twice')),
               'PREFLIGHT_PASS', pre_json=PJ, post_json=QJ)
    # F5: one cause in both arms (only the arm root differs) is one signature; another file is another signature
    env_id = ENV_FAIL[0][0]
    arm = lambda a, f='bench_d16_r0_o96.jsonl': f"[Errno 2] No such file: '/data/runs/pdmux-pf-x-20261016/pf_work/{a}/prefill-layer-alloc/results/{f}'"
    ok &= case('S5 same missing archive, arm root pre vs post -> same signature', build(sub(pre_e, env_id, 'ERROR', 'FileNotFoundError', arm('pre'))),
               build(sub(post_e, env_id, 'ERROR', 'FileNotFoundError', arm('post'))), 'PREFLIGHT_PASS')
    ok &= case('S5b arm root differs AND another file -> CHANGED', build(sub(pre_e, env_id, 'ERROR', 'FileNotFoundError', arm('pre'))),
               build(sub(post_e, env_id, 'ERROR', 'FileNotFoundError', arm('post', 'tel_A2_warm.jsonl'))), 'PREFLIGHT_FAIL_NEW')
    # F6: an L2 id that fails in pre (even with the identical signature) is a hole, not absorbed
    l2fail = [(i, 'FAIL', 'AssertionError', 'DRIFT mirror != registered') if i == l2a else (i, k, e, m) for i, k, e, m in pre_e]
    ok &= case('S6 L2 id fails in pre with the same signature -> L2_UNVERIFIED HOLE', build(l2fail), build(post_e), 'PREFLIGHT_UNVERIFIABLE')
    ok &= case('S6b L2 id skipped in pre -> HOLE', build(pre_e), build(post_e), 'PREFLIGHT_UNVERIFIABLE',
               pre_json=[status_json(pre_e, override={l2a: 'skipped absent'})], post_json=QJ)
    # declaration: applied only on the CPU spec; ids outside it are judged normally; a target can never be declared
    tim = ('test_e1_cap_vessl.TestBudgetEnforcer.test_overrunning_run_is_cut_at_the_deadline', 'FAIL', 'AssertionError', '60.7 not less than 50')
    tpre, tpost = pre_e + [(tim[0], 'ok', '', '')], post_e + [tim]
    ok &= case('S7 declared timing id fails only in post, CPU spec -> excluded -> PASS', build(tpre), build(tpost), 'PREFLIGHT_PASS',
               declared=[tim[0]], cpu_spec=True)
    ok &= case('S7b same on a GPU spec -> not applied -> FAIL_NEW', build(tpre), build(tpost), 'PREFLIGHT_FAIL_NEW',
               declared=[tim[0]], cpu_spec=False)
    other = ('test_e1_cap_vessl.TestBudgetEnforcer.test_control_both_kill_layers_removed_overruns', 'FAIL', 'AssertionError', '38.3 not greater than or equal to 60')
    ok &= case('S7c undeclared timing failure, CPU spec -> still FAIL_NEW', build(tpre + [(other[0], 'ok', '', '')]),
               build(tpost + [other]), 'PREFLIGHT_FAIL_NEW', declared=[tim[0]], cpu_spec=True)
    lab, res = run(build(pre_e), build(post_e), declared=[tgt], cpu_spec=True)
    good = lab == 'PREFLIGHT_UNVERIFIABLE' and any('overlap the CL-3 targets' in x for x in res['invalid'])
    print(('ok   ' if good else 'FAIL ') + f"S7d declaring a CL-3 target -> INVALID by the overlap rule (not merely a hole): {lab} {res['invalid'][:1]}")
    ok &= good
    l2pre_fail = [(i, 'FAIL', 'AssertionError', '38.3 not >= 60') if i == l2a else (i, k, e, m) for i, k, e, m in pre_e]
    ok &= case('S7e declared L2 id failing in pre, CPU spec -> leaves L2 -> PASS', build(l2pre_fail), build(post_e), 'PREFLIGHT_PASS',
               declared=[l2a], cpu_spec=True)
    ok &= case('S7f same on a GPU spec -> L2_UNVERIFIED HOLE', build(l2pre_fail), build(post_e), 'PREFLIGHT_UNVERIFIABLE',
               declared=[l2a], cpu_spec=False)
    lab, res = run(build(tpre), build(tpost), declared=[tim[0]], cpu_spec=True)
    d = res['declared_cpu_timing']
    good = (d['applied'] and d['ids'] == [tim[0]] and d['status'][tim[0]] == {'pre': ['ok'], 'post': ['FAIL']}
            and len(d['sha256']) == 64 and d['disclosure'].startswith('UNVERIFIED (CPU timing)') and res['conclusion_scope'] == J.SCOPE_CPU)
    print(('ok   ' if good else 'FAIL ') + 'S7g judge.json discloses the excluded ids, their statuses, the file sha and the CPU scope')
    ok &= good
    lab, res = run(build(tpre), build(tpost), declared=[tim[0]], cpu_spec=False)
    good = (not res['declared_cpu_timing']['applied']) and res['conclusion_scope'] == J.SCOPE_GPU
    print(('ok   ' if good else 'FAIL ') + 'S7h GPU spec: declaration recorded as not applied, GPU scope')
    ok &= good
    # the CPU-spec rule (preflight_cl3.sh calls `preflight_judge.py spec`): flag AND no GPU; flag with a GPU is a contradiction
    table = [({}, False, (False, False)), ({}, True, (False, False)),
             ({'PDMUX_CPU_ONLY': '1'}, False, (True, False)), ({'PDMUX_CPU_ONLY': '1'}, True, (False, True)),
             ({'PDMUX_SPEC': 'resourcespec-a100cpu'}, False, (True, False)), ({'PDMUX_SPEC': 'resourcespec-a100cpu'}, True, (False, True)),
             ({'PDMUX_SPEC': 'resourcespec-a100x1'}, False, (False, False)), ({'PDMUX_CPU_ONLY': '0', 'PDMUX_SPEC': 'resourcespec-a100x1'}, True, (False, False))]
    got = [(lambda r: (r[0], r[1] is not None))(J.spec_decision(env, g)) for env, g, _ in table]
    good = got == [w for _, _, w in table]
    print(('ok   ' if good else 'FAIL ') + f'S8 CPU-spec decision table (8 rows) {"" if good else got}')
    ok &= good
    return ok

def runner_cases():
    """pf_unittest_runner.py on a tiny suite: JSON statuses from the callbacks, same -v text, the judge cross-check agrees."""
    import json as _j, subprocess, textwrap
    ok = True
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 't'))
        open(os.path.join(d, 't', 'test_rs.py'), 'w').write(textwrap.dedent('''
            import sys, unittest, warnings
            class A(unittest.TestCase):
                def test_ok_with_stdout(self):
                    print("N3 PLAN SELFTEST OK (noise)"); sys.stdout.flush()
                def test_ok_with_warning(self):
                    """Docstring line."""
                    warnings.warn("unclosed file", ResourceWarning)
                def test_fail(self):
                    self.assertEqual(1, 2)
                def test_error(self):
                    raise KeyError("x")
                def test_sub(self):
                    for i in range(2):
                        with self.subTest(i=i):
                            self.assertEqual(i, 0)
                @unittest.skip("why")
                def test_skip(self):
                    pass
            class B(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    raise unittest.SkipTest("absent archive")
                def test_x(self):
                    pass
        '''))
        open(os.path.join(d, 't', 'test_broken.py'), 'w').write('import no_such_module_pf\n')
        log, js = os.path.join(d, 'run.log'), os.path.join(d, 'run.log.status.json')
        with open(log, 'w') as f:
            r = subprocess.run([sys.executable, os.path.join(here, 'pf_unittest_runner.py'), '--status-json', js, 'discover', '-s', 't'],
                               cwd=d, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        rec = _j.load(open(js))
        want = {'test_rs.A.test_ok_with_stdout': 'ok', 'test_rs.A.test_ok_with_warning': 'ok', 'test_rs.A.test_fail': 'FAIL',
                'test_rs.A.test_error': 'ERROR', 'test_rs.A.test_sub': 'FAIL', 'test_rs.A.test_skip': 'skipped why',
                'test_rs.B': 'skipped absent archive', 'unittest.loader._FailedTest.test_broken': 'ERROR'}
        good = r.returncode == 1 and rec['status'] == want and rec['ran'] == 7
        print(('ok   ' if good else 'FAIL ') + f"U1 runner JSON statuses (rc {r.returncode}, ran {rec['ran']})")
        if not good: print('   ', rec)
        ok &= good
        lr = J.load_rep(log)
        good = lr['source'] == 'json' and lr['problems'] == [] and set(lr['sigs']) == {k for k, v in want.items() if v in ('FAIL', 'ERROR')}
        print(('ok   ' if good else 'FAIL ') + f"U2 judge reads the runner output with no cross-check problem {lr['problems'][:1]}")
        ok &= good
    return ok

def record_cases():
    """R-H2: the PREFLIGHT_PASS.json writer.  Only a judge label of exactly PREFLIGHT_PASS with every field present is written as PASS."""
    import json as _json
    post, dig, img = 'a' * 40, 'b' * 64, 'reg/img@sha256:feed'
    ok = True
    with tempfile.TemporaryDirectory() as d:
        n = [0]
        def jj(label):                                   # one file per case (the cases are built before they are scored)
            n[0] += 1; p = os.path.join(d, f'judge{n[0]}.json'); _json.dump({'label': label}, open(p, 'w')); return p
        cases = [('R0 control: judge PASS', jj('PREFLIGHT_PASS'), post, dig, img, 'job', 'PREFLIGHT_PASS'),
                 ('R1 judge FAIL_NEW is never PASS', jj('PREFLIGHT_FAIL_NEW'), post, dig, img, 'job', 'PREFLIGHT_FAIL_NEW'),
                 ('R2 judge UNVERIFIABLE is never PASS', jj('PREFLIGHT_UNVERIFIABLE'), post, dig, img, 'job', 'PREFLIGHT_UNVERIFIABLE'),
                 ('R3 no judge.json', os.path.join(d, 'none.json'), post, dig, img, 'job', 'PREFLIGHT_UNVERIFIABLE'),
                 ('R4 unknown label text', jj('PREFLIGHT_PASS '), post, dig, img, 'job', 'PREFLIGHT_UNVERIFIABLE'),
                 ('R5 image without digest', jj('PREFLIGHT_PASS'), post, dig, 'reg/img:latest', 'job', 'PREFLIGHT_UNVERIFIABLE'),
                 ('R6 merge-tree digest missing', jj('PREFLIGHT_PASS'), post, None, img, 'job', 'PREFLIGHT_UNVERIFIABLE'),
                 ('R7 short post sha', jj('PREFLIGHT_PASS'), 'dc06c870', dig, img, 'job', 'PREFLIGHT_UNVERIFIABLE')]
        for name, j, pc, dg, im, jn, want in cases:
            rec = J.pass_record(j, pc, dg, im, jn)
            good = rec['verdict'] == want
            print(('ok   ' if good else 'FAIL ') + f"{name}: {rec['verdict']} (want {want})")
            ok &= good
        dfile = os.path.join(d, 'declared.txt'); open(dfile, 'w').write('# d\ntest_e1_cap_vessl.T.t\n')
        import hashlib as _hh
        dsha = _hh.sha256(open(dfile, 'rb').read()).hexdigest()
        def jd(label, sha, applied=True):
            n[0] += 1; p = os.path.join(d, f'judge{n[0]}.json')
            _json.dump({'label': label, 'conclusion_scope': J.SCOPE_CPU,
                        'declared_cpu_timing': {'file': 'declared.txt', 'sha256': sha, 'applied': applied, 'ids': ['test_e1_cap_vessl.T.t'],
                                                'disclosure': J.UNVERIFIED_CPU}}, open(p, 'w')); return p
        for name, j, df, want in (('R9 declaration sha matches -> PASS with the sha in its own field', jd('PREFLIGHT_PASS', dsha), dfile, 'PREFLIGHT_PASS'),
                                  ('R10 declaration sha differs from judge.json -> not PASS', jd('PREFLIGHT_PASS', 'f' * 64), dfile, 'PREFLIGHT_UNVERIFIABLE'),
                                  ('R11 judge applied a declaration, record not given the file -> not PASS', jd('PREFLIGHT_PASS', dsha), None, 'PREFLIGHT_UNVERIFIABLE'),
                                  ('R12 record given the file, judge.json silent -> not PASS', jj('PREFLIGHT_PASS'), dfile, 'PREFLIGHT_UNVERIFIABLE')):
            rec = J.pass_record(j, post, dig, img, 'job', declared_file=df)
            good = rec['verdict'] == want and (want != 'PREFLIGHT_PASS' or (rec['declared_cpu_timing']['sha256'] == dsha
                                                                           and rec['record_notes'] == [] and rec['conclusion_scope'] == J.SCOPE_CPU))
            print(('ok   ' if good else 'FAIL ') + f"{name}: {rec['verdict']} (want {want})")
            ok &= good
        out = os.path.join(d, 'PREFLIGHT_PASS.json')
        raw = J.write_pass_record(out, J.pass_record(jj('PREFLIGHT_PASS'), post, dig, img, 'job'))
        import hashlib as _h
        good = open(out + '.sha256').read().split()[0] == _h.sha256(raw).hexdigest() and _json.loads(raw)['verdict'] == 'PREFLIGHT_PASS'
        print(('ok   ' if good else 'FAIL ') + 'R8 written bytes match the .sha256')
        ok &= good
    return ok

if __name__ == '__main__':
    sys.exit(main())
