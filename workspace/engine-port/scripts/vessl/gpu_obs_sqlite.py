#!/usr/bin/env python3
"""Row counts and top kernels from an nsys-exported sqlite (direct evidence that a lane has data)."""
import sqlite3
import sys

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
tables = [r[0] for r in db.execute("select name from sqlite_master where type='table' order by 1")]
for t in tables:
    if t.startswith(("CUPTI_", "NVTX", "OSRT", "GPU_METRICS", "TARGET_INFO_GPU", "DIAGNOSTIC", "COMPOSITE")):
        print(t, db.execute(f'select count(*) from "{t}"').fetchone()[0])
if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
    q = """select s.value, count(*), sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on k.shortName=s.id group by 1 order by 3 desc limit 10"""
    for name, n, ms in db.execute(q):
        print(f"kernel {name} n={n} total_ms={ms:.3f}")
if "DIAGNOSTIC_EVENT" in tables:
    for r in db.execute("select severity, text from DIAGNOSTIC_EVENT limit 40"):
        print("diag", r)
