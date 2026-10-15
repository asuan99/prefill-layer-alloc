#!/usr/bin/env python3
"""C-R4: pin the one-time-exception value of check_citation_stops on the merge diff.
usage: cs_fixed.py LOG   -> exit 0 iff (total, per-pattern counts, path/ext whitelist) equal the pinned values."""
import re, sys, collections
PIN_TOTAL = 1230
PIN_ADDED = 11285275
PIN_PAT = {  # (pattern) -> count
 r'870295|870296|870297|870301': 35,  # [CS-OK] pattern key, not a cited value
 r'20\.39': 386,
 r'7\.24\s*s': 4,
 r'19\.49': 464,
 r'13\.711|13\.71\s*ms': 52,
 r'33\.04': 289,
}
PATH_RE = re.compile(r'^workspace/engine-port/results/multigpu/.*\.(csv|json)$')
def check(text):
    errs = []
    cnt = collections.Counter(); bad = []
    for l in text.split('\n'):
        m = re.match(r'^(.*?):(\d+): CITATION_STOP /(.*?)/ ', l)
        if not m: continue
        path, pat = m.group(1), m.group(3)
        cnt[pat] += 1
        if not PATH_RE.match(path): bad.append(path)
    tot = sum(cnt.values())
    m = re.search(r'--- (\d+) added line\(s\), (\d+) violation', text)
    if not m or int(m.group(1)) != PIN_ADDED: errs.append(f'added lines {m and m.group(1)} != {PIN_ADDED}')
    if tot != PIN_TOTAL: errs.append(f'total {tot} != {PIN_TOTAL}')
    if m and int(m.group(2)) != tot: errs.append('summary line != parsed violations')
    if dict(cnt) != PIN_PAT: errs.append(f'pattern counts {dict(cnt)} != pinned')
    if bad: errs.append(f'{len(bad)} violations outside results/multigpu/**/*.csv|json, e.g. {bad[0]}')
    return errs
if __name__ == '__main__':
    e = check(open(sys.argv[1], encoding='utf-8', errors='replace').read())
    print('PINNED_OK' if not e else 'PIN_MISMATCH:\n  ' + '\n  '.join(e)); sys.exit(1 if e else 0)
