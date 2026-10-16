#!/usr/bin/env python3
"""Preflight judge (merge CL-3): pair PRE/POST unittest logs from ONE container and label the result.

usage: preflight_judge.py --pre LOG[+LOG..] [LOG2[+..]] --post LOG[+LOG..] [..] \
          [--l2 preflight_cl3_l2_ids.txt] [--targets preflight_cl3_targets.txt] [--invalid REASON ...] [--json OUT] \
          [--require-structured] [--declared-cpu-timing preflight_cl3_cpu_timing_declared.txt] [--cpu-spec]
       preflight_judge.py spec --gpu-seen 0|1     (r2: the CPU-spec rule; reads PDMUX_CPU_ONLY / PDMUX_SPEC; rc 0 CPU, 1 not, 4 INVALID)
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

r2 (2026-10-17; PREFLIGHT_FAILNEW_ANALYSIS_2026-10-16 sec 7 F3/F5/F6, user decision DECISIONS_Q1 sec 8; design doc sec 10):
 F3 status  = the structured runner's JSON (LOG.status.json written by pf_unittest_runner.py), cross-checked against the
              strict text status (sigcmp.parse_status) and the failure blocks.  With --require-structured (the Job) a
              repetition without its JSON is INVALID.  Without JSON (re-judging old logs) the strict text status is used;
              an unread status is never 'ok', so it can only end in a HOLE.  Any disagreement between JSON, text status,
              failure blocks or the Ran count -> INVALID.
 F5 mask    = sigcmp.mask also masks the arm root (.../pf_work/pre|post) -> one cause, one signature in both arms.
 F6 L2      = an L2 id counts as reproduced only if its pre status is exactly 'ok'.  An L2 id whose pre status is a
              failure, a skip or unread is a HOLE 'L2_UNVERIFIED' (it used to be absorbed or labelled NEW).
 declared   = --declared-cpu-timing FILE lists ids registered BEFORE the next run as "CPU-spec known timing failures".  They
              are removed from every rule (pairing, L2, flakiness) ONLY when --cpu-spec is given (the shell script decides
              that from PDMUX_CPU_ONLY / PDMUX_SPEC and the absence of a GPU).  judge.json records them with their status
              per repetition and the file sha256 as UNVERIFIED (CPU timing).  On a GPU spec the file is read and recorded
              but not applied.  A declared id that is a CL-3 target (or in a target module) -> INVALID.
 scope      = judge.json 'conclusion_scope' (copied into PREFLIGHT_PASS.json): what a PASS does and does not cover.

record (M3 rev12 harness audit r1 R-H2, 2026-10-16) -- the gate artifact, written INSIDE the preflight Job, never transcribed by hand:
  preflight_judge.py record --judge-json judge.json --post-commit SHA40 --merge-manifest arm_post/manifest.sha256 \
      --engine-digest-py <POST tree>/workspace/engine-port/scripts/bootstrap/engine_digest.py --image $PDMUX_IMAGE --job-name J --out PREFLIGHT_PASS.json \
      [--declared-cpu-timing FILE]
  -> PREFLIGHT_PASS.json (+ PREFLIGHT_PASS.json.sha256) = {verdict, post_commit, merge_tree_engine_digest, image, job_name,
     judge_json_sha256, judge_label, record_notes, schema, declared_cpu_timing, conclusion_scope}; the shape mg_plan.preflight_gate (i) reads.
  verdict = the judge label read from judge.json; it is PREFLIGHT_PASS ONLY when that label is exactly PREFLIGHT_PASS AND post_commit
  (40 hex), the merge-tree digest (pathless-v1 of the POST arm's install manifest -- the ENGINE_DIGEST.txt procedure,
  engine_digest.py load_entries + digest_of_entries) and an image pinned by @sha256 are all present.  FAIL_NEW / UNVERIFIABLE are
  written as themselves; anything missing or unknown -> PREFLIGHT_UNVERIFIABLE.  rc 0 iff verdict == PREFLIGHT_PASS, else 3.
  r2: with --declared-cpu-timing, the file's sha256 recomputed here must equal the one judge.json reports (else a note -> not PASS);
  declared_cpu_timing = {file, sha256, applied, ids} is a separate field (record_notes stays the list of reasons a PASS was refused).
"""
import argparse, hashlib, importlib.util, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigcmp import parse_multi

SKIP_RE = re.compile(r'^(?:\S+) \(([\w.]+)\)[^\n]*? \.\.\. skipped (.*)$', re.M)
FAILED_KINDS = ('FAIL', 'ERROR')
UNVERIFIED_CPU = 'UNVERIFIED (CPU timing): declared before the run, excluded from the label, not a result'
SCOPE_CPU = ('CPU container (resourcespec-a100cpu, no GPU); sgl_kernel loads its sm100 variant (cc None branch); CUDA kernels '
             'launched: 0.  A PASS covers the unit suite on the merge tree only.  The GPU execution path is covered by M2g PASS '
             '(engine digest 82a5fc5b...) and the M3 Job-internal gates (B0, ENGINE_DIGEST, xcheck), and only while the final '
             'POST engine digest equals 82a5fc5b...')
SCOPE_GPU = ('GPU spec (the CPU-timing declaration is not applied).  The suite launches no CUDA kernel (fakes/mocks); the GPU '
             'execution path is covered by M2g PASS (engine digest 82a5fc5b...) and the M3 Job-internal gates (B0, ENGINE_DIGEST, '
             'xcheck), and only while the final POST engine digest equals 82a5fc5b...')

def read_list(p):
    return [l.strip() for l in open(p, encoding='utf-8') if l.strip() and not l.startswith('#')]

def read_targets(p):
    out = []
    for l in open(p, encoding='utf-8'):
        if not l.strip() or l.startswith('#'): continue
        s, k, v = l.rstrip('\n').split('\t'); out.append((s, k, v))
    return out

def kind_of(s):
    return s.split(' ', 1)[0] if s and s.startswith('skipped') else s

def load_rep(r):
    """One repetition (LOG or LOG+LOG).  -> dict(status, sigs, ran, summary, source, problems)."""
    st, sg, ran, summ = parse_multi(r)
    problems = []
    parts = r.split('+')
    for path in parts:
        # unittest also prints skipped "..." (setUpClass skips); scan the raw text so a skip can never read as "not seen -> fine"
        for m in SKIP_RE.finditer(open(path, encoding='utf-8', errors='replace').read()):
            st.setdefault(m.group(1), 'skipped ' + m.group(2).strip()[:80])
    js = [p + '.status.json' for p in parts]
    have = [os.path.isfile(j) for j in js]
    if not any(have):
        return {'status': st, 'sigs': sg, 'ran': ran, 'summary': summ, 'source': 'text', 'problems': problems}
    if not all(have):
        problems.append(f'{r}: status JSON present for some parts only')
        return {'status': {}, 'sigs': sg, 'ran': ran, 'summary': summ, 'source': 'json', 'problems': problems}
    jst, jran = {}, 0
    for j in js:
        try:
            d = json.load(open(j, encoding='utf-8'))
            assert d.get('schema') == 'pf_status/v1' and isinstance(d.get('status'), dict) and isinstance(d.get('ran'), int)
        except (ValueError, AssertionError, OSError) as exc:
            problems.append(f'{j}: unreadable status JSON ({exc!r})'[:200]); continue
        jst.update(d['status']); jran += d['ran']
    if problems:
        return {'status': {}, 'sigs': sg, 'ran': ran, 'summary': summ, 'source': 'json', 'problems': problems}
    if jran != ran:
        problems.append(f'{r}: status JSON ran={jran} != text Ran {ran}')
    for tid, s in st.items():                                   # the strict text status, where readable, must agree
        if tid not in jst:
            problems.append(f'{r}: {tid} has a text status ({s[:20]}) but no JSON status'); break
        if kind_of(jst[tid]) != kind_of(s):
            problems.append(f'{r}: {tid} text status {s[:20]!r} != JSON {jst[tid][:20]!r}'); break
    failed = {t for t, s in jst.items() if s in FAILED_KINDS}
    if set(sg) != failed:
        extra, miss = sorted(set(sg) - failed), sorted(failed - set(sg))
        problems.append(f'{r}: failure blocks != JSON failures (blocks only: {extra[:2]}, JSON only: {miss[:2]})')
    return {'status': jst, 'sigs': sg, 'ran': ran, 'summary': summ, 'source': 'json', 'problems': problems}

def load_arm(reps):
    return [load_rep(r) for r in reps]

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

def declared_conflicts(ids, targets):
    tid = {v for s, k, v in targets if k == 'id'}
    tmod = {v for s, k, v in targets if k == 'module'}
    return sorted(i for i in ids if i in tid or i.split('.', 1)[0] in tmod)

def judge(pre_reps, post_reps, l2, targets, invalid=(), declared=None, cpu_spec=False, require_structured=False):
    """declared = None or {'file': path, 'sha256': hex, 'ids': [...]} (read_declared)."""
    pre, post = load_arm(pre_reps), load_arm(post_reps)
    inv = list(invalid)
    excl = set(declared['ids']) if (declared and cpu_spec) else set()
    if declared:
        bad = declared_conflicts(declared['ids'], targets)
        if bad: inv.append(f'declared CPU-timing ids overlap the CL-3 targets: {bad[:3]}')
        if not declared['ids']: inv.append('declared CPU-timing file lists no id')
    for name, arm in (('pre', pre), ('post', post)):
        for i, r in enumerate(arm):
            if not r['ran']: inv.append(f'{name} rep{i+1}: no "Ran N tests" line (run killed or log truncated)')
            if require_structured and r['source'] != 'json':
                inv.append(f'{name} rep{i+1}: status JSON absent (structured runner output required)')
            inv.extend(f'{name} rep{i+1}: {p}' for p in r['problems'])
        for i, r in enumerate(arm[1:], 2):
            a, b = arm[0]['sigs'], r['sigs']
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in excl)
            if diff: inv.append(f'{name} rep1 vs rep{i} disagree on {len(diff)} failing id(s) (flaky), e.g. {diff[0]}')
    # use rep1 as the arm's record only after the agreement check above
    drop = lambda d: {k: v for k, v in d.items() if k not in excl}
    ps, qs = drop(pre[0]['sigs']), drop(post[0]['sigs'])
    pst, qst = drop(pre[0]['status']), drop(post[0]['status'])
    l2e = [t for t in l2 if t not in excl]
    new, changed, l2_hit, l2_holes, l2_detail = [], [], [], [], {}
    for tid in l2e:                                    # F6: an L2 id is evidence only when it passed before
        l2_detail[tid] = {'pre': pst.get(tid, 'unread'), 'post': qst.get(tid, 'unread')}
        if pst.get(tid) != 'ok':
            l2_holes.append(f'L2_UNVERIFIED:{tid} (pre status {pst.get(tid, "unread")[:40]})')
    for tid, sig in sorted(qs.items()):
        if tid in l2e:
            if pst.get(tid) == 'ok': l2_hit.append(tid)
            continue                                   # pre not ok: already a hole above
        if tid in ps:
            if ps[tid] != sig: changed.append(tid)
        else:
            new.append(tid)
    l2_missing = sorted(t for t in l2e if t not in qs)
    if l2e and not l2_hit: inv.append('L2 never reproduced: post tree does not look different from pre (wrong tree?)')
    fixed = sorted(set(ps) - set(qs))
    holes, cov = [], {}
    for s, k, v in targets:
        ok, det = target_state(qst, qs, k, v)
        d = cov.setdefault(s, {'verified': 0, 'total': 0, 'items': {}}); d['total'] += 1; d['items'][v] = det
        if ok: d['verified'] += 1
        else: holes.append(f'{s}:{v}')
    holes += l2_holes
    fail_new = sorted(new + changed)
    if inv: label = 'PREFLIGHT_UNVERIFIABLE'; rc = 3
    elif fail_new: label = 'PREFLIGHT_FAIL_NEW'; rc = 1
    elif holes: label = 'PREFLIGHT_UNVERIFIABLE'; rc = 3
    else: label = 'PREFLIGHT_PASS'; rc = 0
    dec = None
    if declared:
        per = {t: {'pre': [r['status'].get(t, 'unread') for r in pre], 'post': [r['status'].get(t, 'unread') for r in post]}
               for t in declared['ids']}
        dec = {'file': os.path.basename(declared['file']), 'sha256': declared['sha256'], 'cpu_spec': bool(cpu_spec),
               'applied': bool(excl), 'ids': list(declared['ids']), 'status': per,
               'disclosure': UNVERIFIED_CPU if excl else 'not applied (not a CPU spec): these ids were judged by the normal rules'}
    return rc, {'label': label, 'invalid': inv, 'new_failures': new, 'signature_changed': changed,
                'l2_reproduced': len(l2_hit), 'l2_expected': len(l2e), 'l2_not_failing_in_post': l2_missing,
                'l2_unverified': [h.split(' ', 1)[0][len('L2_UNVERIFIED:'):] for h in l2_holes], 'l2_detail': l2_detail,
                'fixed_in_post': fixed, 'holes': holes, 'coverage': {k: {'verified': v['verified'], 'total': v['total']} for k, v in cov.items()},
                'coverage_detail': cov, 'declared_cpu_timing': dec,
                'status_source': {'pre': [r['source'] for r in pre], 'post': [r['source'] for r in post]},
                'conclusion_scope': SCOPE_CPU if cpu_spec else SCOPE_GPU,
                'pre': {'ran': pre[0]['ran'], 'summary': pre[0]['summary'], 'failing_ids': len(ps)},
                'post': {'ran': post[0]['ran'], 'summary': post[0]['summary'], 'failing_ids': len(qs)}}

CPU_SPEC_SLUG = 'resourcespec-a100cpu'

def spec_decision(env, gpu_seen):
    """-> (cpu_spec: bool, invalid: str|None).  CPU spec = (PDMUX_CPU_ONLY == '1' or PDMUX_SPEC == resourcespec-a100cpu) AND no GPU
    visible.  A CPU flag with a visible GPU is a contradiction -> invalid (the declaration is then not applied either)."""
    flag = env.get('PDMUX_CPU_ONLY', '') == '1' or env.get('PDMUX_SPEC', '') == CPU_SPEC_SLUG
    if flag and gpu_seen:
        return False, 'CPU spec declared (PDMUX_CPU_ONLY/PDMUX_SPEC) but a GPU is visible'
    return bool(flag and not gpu_seen), None

def spec_main(argv):
    ap = argparse.ArgumentParser(prog='preflight_judge.py spec')
    ap.add_argument('--gpu-seen', choices=('0', '1'), required=True)
    a = ap.parse_args(argv)
    cpu, why = spec_decision(os.environ, a.gpu_seen == '1')
    print(f"cpu_spec={int(cpu)} PDMUX_CPU_ONLY={os.environ.get('PDMUX_CPU_ONLY', '')} PDMUX_SPEC={os.environ.get('PDMUX_SPEC', '')} "
          f"gpu_seen={a.gpu_seen} invalid={why or ''}")
    return 0 if cpu else (4 if why else 1)

def read_declared(p):
    raw = open(p, 'rb').read()
    return {'file': p, 'sha256': hashlib.sha256(raw).hexdigest(), 'ids': read_list(p)}

PASS_LABEL = 'PREFLIGHT_PASS'
LABELS = ('PREFLIGHT_PASS', 'PREFLIGHT_FAIL_NEW', 'PREFLIGHT_UNVERIFIABLE')
_HEX40 = re.compile(r'^[0-9a-f]{40}$')
_HEX64 = re.compile(r'^[0-9a-f]{64}$')

def merge_tree_digest(manifest, engine_digest_py):
    """(digest | None, note).  The ENGINE_DIGEST.txt procedure (pathless-v1) on the POST arm's install manifest."""
    try:
        spec = importlib.util.spec_from_file_location('_pf_engine_digest', engine_digest_py)
        ed = importlib.util.module_from_spec(spec); spec.loader.exec_module(ed)
        return ed.digest_of_entries(ed.load_entries(manifest=manifest)), ''
    except Exception as exc:  # noqa: BLE001 - recorded; a missing digest can never yield PASS
        return None, f'merge-tree digest not computed: {exc!r}'[:300]

def pass_record(judge_json, post_commit, merge_tree_engine_digest, image, job_name, digest_note='', declared_file=None):
    notes = [digest_note] if digest_note else []
    label, jsha, jd = None, None, {}
    if judge_json and os.path.isfile(judge_json):
        with open(judge_json, 'rb') as f: raw = f.read()
        jsha = hashlib.sha256(raw).hexdigest()
        try:
            jd = json.loads(raw)
            label = jd.get('label')
        except ValueError:
            notes.append('judge.json is not JSON')
    else:
        notes.append('judge.json absent (no paired run was judged)')
    verdict = label if label in LABELS else 'PREFLIGHT_UNVERIFIABLE'
    if label not in LABELS:
        notes.append(f'judge label {label!r} unknown')
    if not _HEX40.match(post_commit or ''):
        notes.append(f'post_commit {post_commit!r} is not a full sha')
    if not _HEX64.match(merge_tree_engine_digest or ''):
        notes.append('merge_tree_engine_digest absent')
    if '@sha256:' not in (image or ''):
        notes.append(f'image {image!r} not pinned by @sha256 (PDMUX_IMAGE)')
    if not job_name:
        notes.append('job_name absent')
    # r2: the CPU-timing declaration (sha recomputed here == the one the judge applied/recorded) and the conclusion scope
    jdec = jd.get('declared_cpu_timing') if isinstance(jd, dict) else None
    dec = None
    if declared_file is not None:
        try:
            dsha = hashlib.sha256(open(declared_file, 'rb').read()).hexdigest()
        except OSError:
            dsha = None; notes.append(f'declared CPU-timing file {declared_file!r} unreadable')
        if not isinstance(jdec, dict):
            notes.append('judge.json does not report the CPU-timing declaration')
        elif jdec.get('sha256') != dsha:
            notes.append('declared CPU-timing sha256 != the one judge.json reports')
        if isinstance(jdec, dict):
            dec = {'file': jdec.get('file'), 'sha256': dsha, 'applied': bool(jdec.get('applied')), 'ids': jdec.get('ids'),
                   'disclosure': jdec.get('disclosure')}
    elif isinstance(jdec, dict) and jdec.get('applied'):
        notes.append('judge applied a CPU-timing declaration but record was not given the file')
    if notes and verdict == PASS_LABEL:
        verdict = 'PREFLIGHT_UNVERIFIABLE'
    return {'schema': 'preflight_pass/v2', 'verdict': verdict, 'judge_label': label, 'post_commit': post_commit,
            'merge_tree_engine_digest': merge_tree_engine_digest, 'image': image, 'job_name': job_name,
            'judge_json_sha256': jsha, 'record_notes': notes, 'declared_cpu_timing': dec,
            'conclusion_scope': jd.get('conclusion_scope') if isinstance(jd, dict) else None}

def write_pass_record(out, rec):
    raw = (json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    with open(out, 'wb') as f: f.write(raw)
    with open(out + '.sha256', 'w', encoding='ascii') as f: f.write(f'{hashlib.sha256(raw).hexdigest()}  {os.path.basename(out)}\n')
    return raw

def record_main(argv):
    ap = argparse.ArgumentParser(prog='preflight_judge.py record')
    for k in ('--judge-json', '--post-commit', '--merge-manifest', '--engine-digest-py', '--out'): ap.add_argument(k, required=True)
    ap.add_argument('--image', default=''); ap.add_argument('--job-name', default='')
    ap.add_argument('--declared-cpu-timing', default=None)
    a = ap.parse_args(argv)
    dig, note = merge_tree_digest(a.merge_manifest, a.engine_digest_py)
    rec = pass_record(a.judge_json, a.post_commit, dig, a.image, a.job_name, note, declared_file=a.declared_cpu_timing)
    write_pass_record(a.out, rec)
    print(f"PREFLIGHT_PASS.json verdict={rec['verdict']} judge_label={rec['judge_label']} digest={str(dig)[:16]} notes={rec['record_notes']}")
    return 0 if rec['verdict'] == PASS_LABEL else 3

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['record']:
        return record_main(argv[1:])
    if argv[:1] == ['spec']:                      # r2: rc 0 = CPU spec, 1 = not a CPU spec, 4 = contradiction (INVALID)
        return spec_main(argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', nargs='+', required=True); ap.add_argument('--post', nargs='+', required=True)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--l2', default=os.path.join(here, 'preflight_cl3_l2_ids.txt'))
    ap.add_argument('--targets', default=os.path.join(here, 'preflight_cl3_targets.txt'))
    ap.add_argument('--invalid', action='append', default=[]); ap.add_argument('--json')
    ap.add_argument('--require-structured', action='store_true')
    ap.add_argument('--declared-cpu-timing', default=None); ap.add_argument('--cpu-spec', action='store_true')
    a = ap.parse_args(argv)
    dec = read_declared(a.declared_cpu_timing) if a.declared_cpu_timing else None
    rc, res = judge(a.pre, a.post, read_list(a.l2), read_targets(a.targets), a.invalid, declared=dec, cpu_spec=a.cpu_spec,
                    require_structured=a.require_structured)
    print(res['label'])
    print(f"pre ran={res['pre']['ran']} failing={res['pre']['failing_ids']} | post ran={res['post']['ran']} failing={res['post']['failing_ids']}")
    print(f"L2 reproduced {res['l2_reproduced']}/{res['l2_expected']}; new={len(res['new_failures'])} sig_changed={len(res['signature_changed'])} fixed_in_post={len(res['fixed_in_post'])}")
    for k, v in res['coverage'].items(): print(f"  coverage {k}: {v['verified']}/{v['total']}")
    for x in res['invalid']: print('  INVALID', x)
    for x in res['new_failures']: print('  NEW    ', x)
    for x in res['signature_changed']: print('  CHANGED', x)
    for x in res['holes'][:40]: print('  HOLE   ', x)
    d = res['declared_cpu_timing']
    if d: print(f"  DECLARED CPU-timing ({d['sha256'][:12]}, applied={d['applied']}): {len(d['ids'])} id(s) -- {d['disclosure']}")
    if a.json: json.dump(res, open(a.json, 'w'), indent=1, ensure_ascii=False)
    return rc

if __name__ == '__main__':
    sys.exit(main())
