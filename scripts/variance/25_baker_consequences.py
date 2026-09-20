# -*- coding: utf-8 -*-
"""B4 consequences. Component shares for Fig. 1B and additive-prediction ceilings
under two specifications (Year × Tester in X and without it)."""
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
import baker_v4  # noqa: F401  (reconfigures stdout to utf-8)

r, y, m = 3.0, 2.0, 3.0

SPEC = {
    "WITHOUT Year×Tester in X (BAKER_MODEL_AUDIT numbers)": {
        "seed_yield":      dict(L=0.020035, S=0.050678, YL=0.075968, YTL=0.029730, E=0.257847),
        "oil_content":      dict(L=3.991902, S=0.340644, YL=0.162506, YTL=0.115809, E=2.129650),
        "seed_weight_1000": dict(L=73.814580, S=3.729230, YL=0.000000, YTL=14.823480, E=25.377750),
    },
    "Year×Tester IN X (fixed)": {
        "seed_yield":      dict(L=0.019976, S=0.050948, YL=0.075819, YTL=0.029534, E=0.257847),
        "oil_content":      dict(L=3.987538, S=0.352095, YL=0.168852, YTL=0.090900, E=2.129857),
        "seed_weight_1000": dict(L=74.038310, S=7.608240, YL=0.351426, YTL=7.074950, E=25.383180),
    },
}
OLD = dict(L=0.057721, S=0.016481, GY=0.097957, E=0.268928)   # currently printed

for spec, traits in SPEC.items():
    print("\n" + "=" * 92)
    print("### " + spec)
    for name, d in traits.items():
        t = d["L"] + d["S"] + d["YL"] + d["YTL"] + d["E"]
        bak = 2 * d["L"] / (2 * d["L"] + d["S"])
        print(f"  {name:18s} Baker's ratio={bak:.4f}  nonadd.={1-bak:.4f}")
        print(f"      shares: GCA {100*d['L']/t:5.1f} %  SCA {100*d['S']/t:5.1f} %  "
              f"G×L {100*d['YL']/t:5.1f} %  G×T×L {100*d['YTL']/t:5.1f} %  "
              f"error {100*d['E']/t:5.1f} %   (sum {t:.5f})")
        print(f"      genetic part {d['L']+d['S']:.5f}; "
              f"error / genetics = {d['E']/(d['L']+d['S']):.2f}")
    d = traits["seed_yield"]
    vh = d["L"] + d["S"] + d["YL"] / y + d["YTL"] / y + d["E"] / (y * r)
    vg = d["L"] + d["S"] / m + d["YL"] / y + d["YTL"] / (m * y) + d["E"] / (m * y * r)
    print(f"  additive prediction ceiling for yield:")
    print(f"      hybrid, two-year mean: share of σ²Line = {d['L']/vh:.4f}  ->  r <= {(d['L']/vh)**.5:.3f}"
          f"   (observed 0.19)")
    print(f"      paternal GCA, 18 plots:     share of σ²Line = {d['L']/vg:.4f}  ->  r <= {(d['L']/vg)**.5:.3f}"
          f"   (observed 0.43)")
    cor = d["L"] / (d["L"] + d["YL"] + d["S"] / m + d["YTL"] / m + d["E"] / (m * r))
    print(f"      predicted GCA year-to-year correlation = {cor:.3f}   (observed 0.24)")

print("\n" + "=" * 92)
print("### currently printed (baker_v3, without Year×Line term) — for comparison")
t = OLD["L"] + OLD["S"] + OLD["GY"] + OLD["E"]
print(f"  yield  Baker's ratio={2*OLD['L']/(2*OLD['L']+OLD['S']):.4f}")
print(f"      shares: GCA {100*OLD['L']/t:5.1f} %  SCA {100*OLD['S']/t:5.1f} %  "
      f"G×year {100*OLD['GY']/t:5.1f} %  error {100*OLD['E']/t:5.1f} %")
vh = OLD["L"] + OLD["S"] + OLD["GY"] / y + OLD["E"] / (y * r)
vg = OLD["L"] + OLD["S"] / m + OLD["GY"] / (m * y) + OLD["E"] / (m * y * r)
print(f"      ceiling: hybrid r <= {(OLD['L']/vh)**.5:.3f}; GCA r <= {(OLD['L']/vg)**.5:.3f}")
cor = OLD["L"] / (OLD["L"] + OLD["S"] / m + OLD["GY"] / m + OLD["E"] / (m * r))
print(f"      predicted GCA year-to-year correlation = {cor:.3f}   (observed 0.24)")
