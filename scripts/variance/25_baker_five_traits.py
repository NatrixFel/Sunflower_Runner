# -*- coding: utf-8 -*-
"""FIX D1 (consequence). Baker's ratio, variance components, plot
repeatability and H² of the combination mean for FIVE traits on the corrected plot-level table.

Before the fix, plot-level height and diameter existed for only 20 combinations
and were unreliable, so both quantities were computed for three traits only.
After the rebuild (fix01) they are available for all 162 combinations.

The model and REML engine are those adopted by decision D020 (scripts/baker_v4.py):
    y = mu + Year + Mother + Rep(Year) + Year×Mother  (fixed)
        + Line + Tester×Line + Year×Line + Year×Tester×Line  (random) + error
    Baker's ratio = 2 s2(Line) / (2 s2(Line) + s2(Tester×Line))

Input:  deposit 22_hybrid_plot_level.parquet
Output: 25_baker_five_traits.csv
        fix03_baker_five_traits.txt
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
from pathlib import Path
import sys, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.dont_write_bytecode = True
from baker_v4 import reml_fit, moments                      # noqa: E402

PARQ = OUT / "22_hybrid_plot_level.parquet"
OUT_CSV = OUT / "25_baker_five_traits.csv"
OUT_TXT = OUT / "fix03_baker_five_traits.txt"
TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter"]

H = (en_table(pd.read_parquet(PARQ))
       .rename(columns={"mother": "mother", "father": "line", "year": "year",
                        "rep": "replicate", "trait": "trait",
                        "value": "value"})
       .dropna(subset=["value"]))

rows, txt = [], []
for tr in TRAITS:
    d = H[H["trait"] == tr].reset_index(drop=True)
    f = reml_fit(d, with_yl=True, hess=True)
    mo = moments(d)
    s = f["s2"]; e = f["s2e"]
    ny = float(d["year"].nunique()); nm = float(d["mother"].nunique())
    r = float(d.groupby(["year", "mother", "line"]).size().mean())
    G = s["line"] + s["tester_x_line"]; GY = s["year_x_line"] + s["year_x_tester_x_line"]
    var_plot = G + GY + e
    var_mean = G + GY / ny + e / (ny * r)
    rows.append(dict(trait=tr, n=len(d), n_combinations=d.groupby(["mother", "line"]).ngroups,
                     plots_per_combination=r,
                     s2_line=s["line"], s2_tester_x_line=s["tester_x_line"],
                     s2_year_x_line=s["year_x_line"], s2_year_x_tester_x_line=s["year_x_tester_x_line"],
                     s2_error=e,
                     baker_REML=f["baker"], se=f.get("se_baker", np.nan),
                     baker_moments=mo["baker"],
                     plot_repeatability=G / var_plot,
                     H2_of_combination_mean=G / var_mean))
    txt.append(f"### {tr}: n={len(d)} plots, {rows[-1]['n_combinations']} combinations, r={r:.3f}")
    txt.append(f"    Baker's ratio = {f['baker']:.4f} +- {f.get('se_baker', float('nan')):.4f}"
               f"   (moments {mo['baker']:.4f})")
    txt.append(f"    shares: Line {s['line']/var_plot*100:5.1f} %  Tester×Line "
               f"{s['tester_x_line']/var_plot*100:5.1f} %  Year×Line {s['year_x_line']/var_plot*100:5.1f} %"
               f"  Year×Tester×Line {s['year_x_tester_x_line']/var_plot*100:5.1f} %"
               f"  error {e/var_plot*100:5.1f} %")
    txt.append(f"    plot repeatability = {G/var_plot:.4f};  H² of combination mean = {G/var_mean:.4f}")
    txt.append("")

T = pd.DataFrame(rows)
T.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
OUT_TXT.write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
