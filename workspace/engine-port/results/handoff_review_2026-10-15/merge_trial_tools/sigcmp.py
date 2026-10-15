#!/usr/bin/env python3
"""C-R2: failure signature (id, signature) extraction + pre/post comparison.
usage: sigcmp.py PRE_LOG POST_LOG [--json OUT]
signature = (kind FAIL|ERROR, exception type, masked first message line, keyword set, DRIFT-line digest)
"""
import re, sys, json, hashlib
KEYS = ["ABORT_REG_DRIFT","ABORT_CONFIG","DIGESTS DRIFT","mirror_check","No module named","undefined symbol",
        "FileNotFoundError","Not enough SMs","child never became ready","runtime tree absent","sha256","unbound variable"]
def mask(s):
    s = re.sub(r'/(tmp|home)/\S+','<path>',s)
    s = re.sub(r'\bline \d+','line N',s)
    s = re.sub(r'\d+\.\d+','N.N',s); s = re.sub(r'\b(?=[0-9a-f]*[a-f])[0-9a-f]{12,64}\b','<hex>',s); s = re.sub(r'\d{5,}','N',s)
    return s.strip()[:200]
def parse(path):
    t = open(path, encoding='utf-8', errors='replace').read()
    status = {}
    # per-test status lines (-v). allow a docstring line between name and "... status"
    for m in re.finditer(r"^(\S+) \(([\w.]+)\)(?:\n[^\n]*?)? \.\.\. (ok|FAIL|ERROR|skipped '[^\n]*'|expected failure|unexpected success)\s*$", t, re.M):
        status[m.group(2)] = m.group(3)
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
