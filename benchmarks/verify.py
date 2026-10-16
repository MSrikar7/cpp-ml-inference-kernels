"""Validate benchmark output; not a flaky fixed-speed CI assertion."""
import csv
import math
import sys
from pathlib import Path
path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(newline="")))
assert {r["method"] for r in rows} == {"naive", "tiled", "tiled_parallel"}, rows
for row in rows:
    for key in ("median_ms", "max_abs_error", "gflops"):
        assert math.isfinite(float(row[key])), row
    assert float(row["median_ms"]) > 0, row
    assert float(row["max_abs_error"]) < 0.002, row
print("Benchmark output and numerical validation passed.")
