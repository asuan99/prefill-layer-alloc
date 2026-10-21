import json, sys
out = sys.argv[sys.argv.index("--out") + 1]
open(out, "w").write(json.dumps({"verdict": __import__("os").environ.get("FAKE_CALIB", "PASS")}))
