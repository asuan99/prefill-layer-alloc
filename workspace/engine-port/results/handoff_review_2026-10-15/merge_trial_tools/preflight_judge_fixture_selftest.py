#!/usr/bin/env python3
"""Fixture self-test of the r2 judge on the REAL container logs of pdmux-preflight-cl3-68f386b-20261016114954 (local, CPU).

usage: preflight_judge_fixture_selftest.py [FIXTURE_DIR]   (default: results/preflight_cl3/pdmux-preflight-cl3-68f386b-20261016114954)
  FIXTURE_DIR/logs/{pre,post}.rep{1,2}.log are read-only inputs; every mutation is applied to a copy in a temp dir.
  rc 0 iff every property holds; rc 2 if the fixture is absent (never a pass); rc 1 otherwise.

THIS IS NOT A RE-JUDGEMENT OF THAT RUN.  Its label stays PREFLIGHT_FAIL_NEW as recorded (DECISIONS_Q1 sec 8: the run is not
re-judged with the revised judge).  The script never prints the label of the unmutated logs; it checks properties only:
  (i)   mixed output is parsed: every header has exactly one status, counts equal each log's summary line, the two container
        shapes (stdout before 'ok' in test_plan_selftest, ResourceWarnings before 'ok' in the holb target) read 'ok', and the
        judge treats test_plan_selftest as an L2 reproduction and the holb target as verified (no NEW, no HOLE from them);
  (ii)  after masking, the 8 arm-path-only signature differences are gone (same signature pre/post, none CHANGED);
  (iii) a real new failure injected into post is FAIL_NEW; an arm-path failure pointing at another file is CHANGED;
  (iv)  a timing failure outside the 5 declared ids is still FAIL_NEW on the CPU spec;
  (v)   on a GPU spec the declaration is not applied (L2 judged on 12 ids, the PRE-failing n3 ids are L2_UNVERIFIED holes,
        a declared id failing only in post is NEW); on the CPU spec it is applied (9 L2 ids, no L2 hole, the same id excluded).
"""
import os, re, shutil, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import preflight_judge as J, sigcmp as S

DEFAULT = os.path.normpath(os.path.join(here, '..', '..', 'preflight_cl3', 'pdmux-preflight-cl3-68f386b-20261016114954'))
L2 = J.read_list(os.path.join(here, 'preflight_cl3_l2_ids.txt'))
TG = J.read_targets(os.path.join(here, 'preflight_cl3_targets.txt'))
DECL = J.read_declared(os.path.join(here, 'preflight_cl3_cpu_timing_declared.txt'))
PLAN = 'test_n3_capacity.TestRegistration.test_plan_selftest'
HOLB = 'test_holb_probe.TestSchedulerHooksInstalled.test_patch_reapplies_the_hooks_and_nothing_else'
ARM8 = ['test_lambda0_prereg.TestArrivalRealization.test_analyzer_records_the_drain_decomposition',
        'test_lambda0_prereg.TestArrivalRealization.test_realized_denominator_removes_the_artifact_on_archived_cells',
        'test_lambda0_prereg.TestDrainIdentity.test_drain_identity_on_archived_cells',
        'test_lambda0_prereg.TestShippedSelftests.test_lambda_inf_predicate_selftest',
        'test_lambda_star_per_shape.CliSurfaces.test_workloads_cli_optin_writes_the_caveats_into_the_header',
        'test_lambda_star_per_shape.FailClosed.test_shipped_template_loads_and_is_entirely_unmeasured',
        'test_n3_capacity.TestBootBranchSnippets.test_t_h1_sticky',
        'test_n3_capacity.TestBootBranchSnippets.test_t_h3_floor']
FRESH = 'test_n3_capacity.TestRegistration.test_config_passes_and_fixed_dsm_control'          # ok in all four logs
UNDECLARED_TIMING = 'test_e1_cap_vessl.TestBudgetEnforcer.test_control_both_kill_layers_removed_overruns'   # ok x4, wall >= 60
E1_DECLARED = 'test_e1_cap_vessl.TestBudgetEnforcer.test_overrunning_run_is_cut_at_the_deadline'
N3_PRE_FAIL = ['test_n3_capacity.TestBudgetEnforcer.test_overrunning_unit_is_cut',
               'test_n3_capacity.TestBudgetEnforcer.test_control_both_kill_layers_removed_overruns']
SEP = '=' * 70
RAN_SEP = '\n' + '-' * 70 + '\nRan '

def set_ok_to_fail(text, tid, exc_line):
    """header ' ... ok' -> ' ... FAIL' and a FAIL block appended before the final 'Ran' separator."""
    name = tid.rsplit('.', 1)[-1]
    pat = re.compile(r'^(' + re.escape(name) + r' \(' + re.escape(tid) + r'\)[^\n]*?) \.\.\. ok$', re.M)
    text, n = pat.subn(r'\1 ... FAIL', text, count=1)
    assert n == 1, f'header of {tid} not found as a one-line "... ok"'
    blk = (f'\n{SEP}\nFAIL: {name} ({tid})\n{"-" * 70}\nTraceback (most recent call last):\n  File "t.py", line 9, in {name}\n'
           f'{exc_line}\n')
    i = text.rindex(RAN_SEP)
    return text[:i] + blk + text[i:]

def set_fail_to_ok(text, tid):
    """remove the failure block of tid and turn its header status into ok (a declared id that passed in this arm)."""
    name = tid.rsplit('.', 1)[-1]
    text, n = re.subn(r'\n={60,}\n(?:FAIL|ERROR): ' + re.escape(name) + r' \(' + re.escape(tid) + r'\)\n.*?(?=\n={60,}\n|\n-{60,}\nRan )',
                      '', text, count=1, flags=re.S)
    assert n == 1, f'block of {tid} not found'
    text, n = re.subn(r'^(' + re.escape(name) + r' \(' + re.escape(tid) + r'\)[^\n]*?) \.\.\. (?:FAIL|ERROR)$', r'\1 ... ok', text,
                      count=1, flags=re.M)
    assert n == 1, f'header of {tid} not found'
    return text

class Fixture:
    def __init__(self, src):
        self.src = src
        self.text = {a: open(os.path.join(src, 'logs', f'{a}.log'), encoding='utf-8', errors='replace').read()
                     for a in ('pre.rep1', 'pre.rep2', 'post.rep1', 'post.rep2')}

    def judge(self, mutate=None, cpu_spec=True):
        """mutate: {arm-rep: fn(text)->text}; returns the judge result dict (never printed as a verdict)."""
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for a, t in self.text.items():
                if mutate and a in mutate: t = mutate[a](t)
                paths[a] = os.path.join(d, a + '.log'); open(paths[a], 'w', encoding='utf-8').write(t)
            rc, res = J.judge([paths['pre.rep1'], paths['pre.rep2']], [paths['post.rep1'], paths['post.rep2']], L2, TG,
                              declared=DECL, cpu_spec=cpu_spec)
            return res

def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name + ('' if cond else f'   {detail}'))
    return bool(cond)

def main(argv):
    src = argv[1] if len(argv) > 1 else DEFAULT
    if not os.path.isfile(os.path.join(src, 'logs', 'pre.rep1.log')):
        print(f'FIXTURE ABSENT: {src}'); return 2
    F = Fixture(src)
    ok = True
    # (i) parsing
    for a, t in F.text.items():
        st = S.parse_status(t)
        heads = {h for h in (S.is_header(l) for l in t.split('\n')) if h}
        summ = dict((k, int(v)) for k, v in re.findall(r'(failures|errors|skipped)=(\d+)', re.search(r'FAILED \(([^)]*)\)', t).group(1)))
        kinds = {k: sum(1 for v in st.values() if J.kind_of(v) == k) for k in ('FAIL', 'ERROR', 'skipped')}
        ok &= check(f'(i) {a}: every header read, exactly one status ({len(st)}/{len(heads)})', heads == set(st), sorted(heads - set(st))[:3])
        ok &= check(f'(i) {a}: FAIL/ERROR/skipped counts == summary line {summ}', kinds == {'FAIL': summ.get('failures', 0),
                    'ERROR': summ.get('errors', 0), 'skipped': summ.get('skipped', 0)}, kinds)
        ok &= check(f'(i) {a}: holb target (ResourceWarnings before the token) reads ok', st.get(HOLB) == 'ok', st.get(HOLB))
        if a.startswith('pre'):
            ok &= check(f'(i) {a}: test_plan_selftest (stdout before the token) reads ok', st.get(PLAN) == 'ok', st.get(PLAN))
    base_cpu, base_gpu = F.judge(cpu_spec=True), F.judge(cpu_spec=False)
    for tag, r in (('CPU', base_cpu), ('GPU', base_gpu)):
        ok &= check(f'(i) {tag}: test_plan_selftest is not NEW and pre status ok (L2 reproduction)',
                    PLAN not in r['new_failures'] and r['l2_detail'][PLAN]['pre'] == 'ok', (r['new_failures'], r['l2_detail'].get(PLAN)))
        ok &= check(f'(i) {tag}: holb target is verified (no HOLE)', not any(HOLB in h for h in r['holes']), r['holes'])
        ok &= check(f'(i) {tag}: no INVALID from parsing', r['invalid'] == [], r['invalid'])
    # (ii) masking
    pre1 = S.parse(os.path.join(src, 'logs', 'pre.rep1.log'))[1]; post1 = S.parse(os.path.join(src, 'logs', 'post.rep1.log'))[1]
    raw_pre = F.text['pre.rep1']
    ok &= check('(ii) the fixture carries arm roots in failure text (pf_work/pre in pre, pf_work/post in post)',
                '/pf_work/pre/' in raw_pre and '/pf_work/post/' in F.text['post.rep1'])
    for tid in ARM8:
        ok &= check(f'(ii) {tid.split(".", 1)[1][:60]}: same signature pre/post, not CHANGED',
                    pre1.get(tid) == post1.get(tid) and pre1.get(tid) is not None and tid not in base_cpu['signature_changed']
                    and tid not in base_gpu['signature_changed'], (pre1.get(tid), post1.get(tid)))
    ok &= check('(ii) CPU spec: no signature change remains at all', base_cpu['signature_changed'] == [], base_cpu['signature_changed'])
    # (iii) a real new failure / a different file are still caught
    inj = lambda t: set_ok_to_fail(t, FRESH, 'AssertionError: config A-agn rejected: unexpected dsm 54')
    r = F.judge({'post.rep1': inj, 'post.rep2': inj})
    ok &= check('(iii) injected new failure in post -> NEW, label FAIL_NEW', FRESH in r['new_failures'] and r['label'] == 'PREFLIGHT_FAIL_NEW',
                (r['label'], r['new_failures'], r['invalid'][:1]))
    other = lambda t: t.replace('probe_6group_2026-10-05/srv_A2_warm.log', 'probe_6group_2026-10-05/srv_B9_cold.log')
    r = F.judge({'post.rep1': other, 'post.rep2': other})
    ok &= check('(iii) arm-path failure pointing at another file in post -> CHANGED, label FAIL_NEW',
                ARM8[6] in r['signature_changed'] and r['label'] == 'PREFLIGHT_FAIL_NEW', (r['label'], r['signature_changed']))
    # (iv) timing failure outside the declaration
    tim = lambda t: set_ok_to_fail(t, UNDECLARED_TIMING, 'AssertionError: 38.3 not greater than or equal to 60')
    r = F.judge({'post.rep1': tim, 'post.rep2': tim}, cpu_spec=True)
    ok &= check('(iv) undeclared timing failure, CPU spec -> NEW, label FAIL_NEW',
                UNDECLARED_TIMING in r['new_failures'] and r['label'] == 'PREFLIGHT_FAIL_NEW', (r['label'], r['new_failures']))
    # (v) the declaration applies on the CPU spec only
    dc, dg = base_cpu['declared_cpu_timing'], base_gpu['declared_cpu_timing']
    ok &= check('(v) CPU: declaration applied, 5 ids disclosed with per-repetition status, file sha recorded',
                dc['applied'] and len(dc['status']) == 5 and dc['sha256'] == DECL['sha256'] and dc['disclosure'] == J.UNVERIFIED_CPU, dc)
    ok &= check('(v) CPU: L2 judged on 9 ids, no L2_UNVERIFIED hole', base_cpu['l2_expected'] == 9 and base_cpu['l2_unverified'] == [],
                (base_cpu['l2_expected'], base_cpu['l2_unverified']))
    ok &= check('(v) GPU: declaration recorded, NOT applied', (not dg['applied']) and dg['sha256'] == DECL['sha256'], dg)
    ok &= check('(v) GPU: L2 judged on 12 ids, the PRE-failing n3 ids are L2_UNVERIFIED holes',
                base_gpu['l2_expected'] == 12 and sorted(base_gpu['l2_unverified']) == sorted(N3_PRE_FAIL), base_gpu['l2_unverified'])
    ok &= check('(v) scope text follows the spec', base_cpu['conclusion_scope'] == J.SCOPE_CPU and base_gpu['conclusion_scope'] == J.SCOPE_GPU)
    pre_ok = lambda t: set_fail_to_ok(t, E1_DECLARED)
    rc_, rg_ = F.judge({'pre.rep1': pre_ok, 'pre.rep2': pre_ok}, cpu_spec=True), F.judge({'pre.rep1': pre_ok, 'pre.rep2': pre_ok}, cpu_spec=False)
    ok &= check('(v) declared e1 id failing only in post: CPU -> excluded (not NEW); GPU -> NEW',
                E1_DECLARED not in rc_['new_failures'] and E1_DECLARED in rg_['new_failures'], (rc_['new_failures'], rg_['new_failures']))
    print('FIXTURE_SELFTEST', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
