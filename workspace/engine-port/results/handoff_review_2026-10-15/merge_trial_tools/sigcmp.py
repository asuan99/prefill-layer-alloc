#!/usr/bin/env python3
"""C-R2: failure signature (id, signature) extraction + pre/post comparison.
usage: sigcmp.py PRE_LOG POST_LOG [--json OUT]
signature = (kind FAIL|ERROR, exception type, masked first message line, keyword set, DRIFT-line digest)
r2 (2026-10-17, judge revision F3/F5; PREFLIGHT_FAILNEW_ANALYSIS_2026-10-16 sec 7):
  F5  mask() also replaces a preflight arm root ".../pf_work/pre" or ".../pf_work/post" (container /data/runs/<job>/pf_work/<arm>)
      by "<arm>" and keeps the rest of the path, so one cause gives one signature in both arms while a different file still differs.
  F3  status from the -v text is read STRICTLY: the segment of a test runs from its header line "<name> (<id>)" to the next
      header / separator / "Ran N" line, and its status is read only if the segment holds EXACTLY ONE status token (" ... <tok>"
      at a line end, or a line that is only "<tok>").  Zero or several tokens -> status unread (absent from the dict); the
      judge turns unread into a HOLE, never into a pass.  Interleaved stdout / warnings between header and token are allowed.
      The authoritative status is the runner's JSON (pf_unittest_runner.py); the text status is a cross-check.
"""
import re, sys, json, hashlib
KEYS = ["ABORT_REG_DRIFT","ABORT_CONFIG","DIGESTS DRIFT","mirror_check","No module named","undefined symbol",
        "FileNotFoundError","Not enough SMs","child never became ready","runtime tree absent","sha256","unbound variable"]
ARM_RE = re.compile(r'(?:/[^\s/\'"]+)*/pf_work/(?:pre|post)(?=[/\s\'":]|$)')
def mask(s):
    s = ARM_RE.sub('<arm>', s)
    s = re.sub(r'/(tmp|home)/\S+','<path>',s)
    s = re.sub(r'\bline \d+','line N',s)
    s = re.sub(r'\d+\.\d+','N.N',s); s = re.sub(r'\b(?=[0-9a-f]*[a-f])[0-9a-f]{12,64}\b','<hex>',s); s = re.sub(r'\d{5,}','N',s)
    return s.strip()[:200]
HOLDERS = ('setUpClass', 'tearDownClass', 'setUpModule', 'tearDownModule')
HEAD_RE = re.compile(r'^(\S+) \(([\w.]+)\)')
TOK = r"(ok|FAIL|ERROR|skipped (?:'[^\n]*'|\"[^\n]*\")|expected failure|unexpected success)"
TOK_END = re.compile(r' \.\.\. ' + TOK + r'\s*$')
TOK_ALONE = re.compile(r'^' + TOK + r'\s*$')
STOP_RE = re.compile(r'^(?:={60,}|-{60,}|Ran \d+ tests?\b)')
def is_header(line):
    """-> test id or None.  A header names a test whose last id component is its name (or a class/module fixture holder)."""
    m = HEAD_RE.match(line)
    if not m: return None
    name, tid = m.group(1), m.group(2)
    if name in HOLDERS or tid.rsplit('.', 1)[-1] == name: return tid
    return None
def parse_status(t):
    """{id: status} for every test whose segment holds exactly one status token (strict; see module doc)."""
    status, cur, toks = {}, None, []
    def close():
        if cur is not None and len(toks) == 1: status[cur] = toks[0]
        elif cur is not None: status.pop(cur, None)
    for line in t.split('\n'):
        tid = is_header(line)
        if tid is not None or STOP_RE.match(line):
            close(); cur, toks = tid, []
            if tid is None: continue
        if cur is None: continue
        m = TOK_END.search(line) or TOK_ALONE.match(line)
        if m: toks.append(m.group(1))
    close()
    return status
def parse(path):
    t = open(path, encoding='utf-8', errors='replace').read()
    status = parse_status(t)
    sigs = {}
    parts = re.split(r'\n={60,}\n', t)
    for b in parts[1:]:
        m = re.match(r'(FAIL|ERROR): (\S+) \(([\w.]+)\)', b)
        if not m: continue
        kind, tid = m.group(1), m.group(3)
        tail = b.split('\nRan ',1)[0]
        lines = tail.split('\n')
        # exception line = last line starting with a (dotted) *Error/*Exception/Failure at col 0
        idx = [i for i,l in enumerate(lines) if re.match(r'^[\w.]*(Error|Exception|Failure|Exit)\b', l)]
        if idx:
            i = idx[-1] if kind=='ERROR' else idx[0]
            first = lines[i]
            et = re.match(r'^([\w.]+)', first).group(1)
            msg = first[len(et):].lstrip(': ')
        else:
            et, msg = '?', lines[-1] if lines else ''
        keys = sorted(k for k in KEYS if k in tail)
        drift = sorted(l.strip() for l in lines if l.startswith('DRIFT '))
        dg = hashlib.sha256('\n'.join(drift).encode()).hexdigest()[:12] if drift else ''
        body = [l for l in lines if l.strip() and not l.startswith(('---','===','Ran ','FAILED','OK'))]
        tail_line = mask(body[-1]) if body else ''
        sigs[tid] = {'kind':kind,'exc':et,'msg':mask(msg),'tail':tail_line,'keys':keys,'drift':dg}
    m = re.search(r'Ran (\d+) tests?', t); ran = int(m.group(1)) if m else None
    m = re.search(r'FAILED \(([^)]*)\)', t); summ = m.group(1) if m else 'OK'
    return status, sigs, ran, summ
def compare(a, b):
    ids_a, ids_b = set(a), set(b)
    common = ids_a & ids_b
    changed = sorted(k for k in common if a[k] != b[k])
    return {'common':len(common),'new_in_post':sorted(ids_b-ids_a),'fixed_in_post':sorted(ids_a-ids_b),
            'signature_changed':changed}
def parse_multi(arg):
    S, G, R, M = {}, {}, 0, []
    for p in arg.split('+'):
        s, g, r, m = parse(p); S.update(s); G.update(g); R += r or 0; M.append(m)
    return S, G, R, ' ; '.join(M)
if __name__ == '__main__':
    sa, ga, ra, ma = parse_multi(sys.argv[1]); sb, gb, rb, mb = parse_multi(sys.argv[2])
    c = compare(ga, gb)
    print(f"pre : ran={ra} {ma} sig_ids={len(ga)}\npost: ran={rb} {mb} sig_ids={len(gb)}")
    print(f"common={c['common']} new_in_post={len(c['new_in_post'])} fixed_in_post={len(c['fixed_in_post'])} signature_changed={len(c['signature_changed'])}")
    for k in c['signature_changed']: print(' CHANGED',k,'\n   pre ',ga[k],'\n   post',gb[k])
    if '--json' in sys.argv:
        json.dump({'pre':ga,'post':gb,'cmp':c}, open(sys.argv[sys.argv.index('--json')+1],'w'), indent=1, ensure_ascii=False)
    sys.exit(1 if (c['signature_changed'] or c['new_in_post'] or c['fixed_in_post']) else 0)
