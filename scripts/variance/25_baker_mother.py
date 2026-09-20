# -*- coding: utf-8 -*-
"""B4 mother. Maternal-form variance component in the NEW decomposition.

The rationale for a fixed tester (Methods 2.5, decision D008) rests on
sigma2(Mother) being indistinguishable from zero. Previous numbers came from the
old decomposition (without a Year × Line term); here the same quantity is
recomputed from the sequential decomposition now in use.

sigma2(Mother) = [MS(Mother) - MS(Mother × Line) - MS(Year × Mother) + MS(Year × Mother × Line)]
               / (r * y * l)
"""
import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import DEPOSIT, WORK, CALLING, OUT, INTERMEDIATE, MODELS, use_models, work, calling, en_table, en_ids, en_id, is_line_id
use_models()
import sys, pathlib
import numpy as np
from baker_v4 import moments, load_repo

repo = load_repo()
print("GCA component of maternal testers under the new decomposition")
print(f"{'trait':18s} {'σ²Mother':>12s} {'SE':>12s} {'vs own SE':>15s}")
for key, name in [("seed_yield", "seed_yield"), ("oil_content", "oil_content"),
                  ("seed_weight_1000", "seed_weight_1000")]:
    d = repo[repo.trait == key].reset_index(drop=True)
    mo = moments(d)
    A = mo["anova"]; r = mo["r"]
    ny = float(d["year"].nunique()); nl = float(d["line"].nunique())
    ms = {k: A[k]["MS"] for k in A}; df = {k: A[k]["df"] for k in A}
    num = (ms["mother"] - ms["mother_x_line"] - ms["year_x_mother"] + ms["year_x_mother_x_line"])
    s2m = num / (r * ny * nl)
    var = sum(2 * ms[k] ** 2 / df[k] for k in
              ("mother", "mother_x_line", "year_x_mother", "year_x_mother_x_line"))
    se = np.sqrt(var) / (r * ny * nl)
    print(f"{name:18s} {s2m:12.5f} {se:12.5f} {abs(s2m)/se:15.2f}")
