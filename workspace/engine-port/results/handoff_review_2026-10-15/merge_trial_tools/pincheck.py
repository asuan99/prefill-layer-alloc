#!/usr/bin/env python3
"""Known-failure pin: exact (id -> signature) table. usage:
   pincheck.py --make PIN.json LOG[+LOG..]     write the pin from a run
   pincheck.py PIN.json LOG[+LOG..]            rc 0 iff failing-id set and every signature equal the pin
"""
import sys, json
from sigcmp import parse_multi
def sigs(arg):
    _, g, _, _ = parse_multi(arg); return g
if sys.argv[1] == '--make':
    json.dump(sigs(sys.argv[3]), open(sys.argv[2], 'w'), indent=1, sort_keys=True, ensure_ascii=False); sys.exit(0)
pin = json.load(open(sys.argv[1])); cur = sigs(sys.argv[2])
extra = sorted(set(cur) - set(pin)); gone = sorted(set(pin) - set(cur))
chg = sorted(k for k in set(cur) & set(pin) if cur[k] != pin[k])
print(f'pinned={len(pin)} current={len(cur)} unexpected_new_failures={len(extra)} pinned_but_absent={len(gone)} signature_changed={len(chg)}')
for k in extra: print('  NEW    ', k)
for k in gone: print('  ABSENT ', k)
for k in chg: print('  CHANGED', k, '\n     pin:', pin[k], '\n     cur:', cur[k])
sys.exit(1 if (extra or gone or chg) else 0)
