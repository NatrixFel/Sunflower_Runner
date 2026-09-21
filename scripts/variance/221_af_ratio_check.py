# -*- coding: utf-8 -*-
"""Comparison: two seed counts versus their ratio (autofertility index).

Question: does the ratio reveal associations that the counts do not. Checked with the
same tool used for the paper: merged-panel, EMMAX with three
principal components and the K matrix (scripts\235_gwas_merged_panel.py, model unchanged).

Phenotypes:
  autofertility_self  — seeds under the isolator, mean over heads, then over years
  autofertility_open  — seeds at open flowering, same way
  autofertility_ratio    — (mean SO / mean SC) × 100 per line-year, then mean over years

Heritability is computed on the same structure for all three: year fixed,
line random, on line-year means (the ratio has no plot-level values
in principle — the denominator is not paired with the numerator).
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
import sys, pathlib, re, numpy as np, pandas as pd
sys.dont_write_bytecode = True
R = DEPOSIT
OUT = pathlib.Path.cwd()

src = (MODELS.parent / "gwas" / "235_gwas_merged_panel.py").read_text(encoding="utf-8")
src = src.replace('ROOT = DEPOSIT',
                  f'ROOT = pathlib.Path(r"{R}")')
src = re.sub(r'SCR = pathlib\.Path\(r"[^"]+"\)', f'SCR = pathlib.Path(r"{OUT}")', src)
src = src.replace("from blink import blink", "blink = None")
head = src[:src.index("rows_panel, rows_gwas, rows_loci")]
ns = {"__name__": "gmp"}
exec(compile(head, "gwas_merged_panel_head", "exec"), ns)
G, samp, chrom, pos = ns["G"], ns["samp"], ns["chrom"], ns["pos"]
lines54, lines52, CONF = ns["lines54"], ns["lines52"], ns["CONF"]
build_panel, pca, vanraden, emmax = ns["build_panel"], ns["pca"], ns["vanraden"], ns["emmax"]

# ---- phenotypes -----------------------------------------------------------------
ph = en_table(pd.read_parquet(OUT / "221_lines_tidy.parquet"))
cnt = (ph[ph["trait"].isin(["autofertility_self", "autofertility_open"])]
       .groupby(["genotype", "year", "trait"])["value"].mean().unstack())
cnt["autofertility_ratio"] = cnt["autofertility_self"] / cnt["autofertility_open"] * 100
LY = cnt.reset_index()                      # line-year means, three traits
TR = ["autofertility_self", "autofertility_open", "autofertility_ratio"]
print(f"line-years: {len(LY)}, lines {LY['genotype'].nunique()}, years {LY['year'].nunique()}")
print("ratio: median %.1f %%, range %.1f–%.1f, above 100 %%: %d"
      % (LY["autofertility_ratio"].median(), LY["autofertility_ratio"].min(), LY["autofertility_ratio"].max(),
         int((LY["autofertility_ratio"] > 100).sum())))

# ---- heritability on one structure ----------------------------------------
sys.path.insert(0, str(R / "scripts"))
from baker_v4 import REML, dummies

def h2_means(tr):
    d = LY[~LY["genotype"].isin(CONF)].dropna(subset=[tr])
    y = d[tr].to_numpy(float)
    X = np.hstack([np.ones((len(y), 1)), dummies(d["year"].astype(str).tolist())])
    f = REML(y, X, [dummies(d["genotype"].astype(str).tolist())], ["G"]).fit(
        [[0.3], [0.05], [1.0], [3.0]])
    s2g, s2e = f["s2"]["G"], f["s2e"]
    ny = d["year"].nunique()
    return len(y), s2g / (s2g + s2e / ny), s2g, s2e

# ---- scan --------------------------------------------------------------------
rows = []
for lset, lname in ((lines52, "52_lines"), (lines54, "54_lines")):
    P = build_panel(lset, 0.9, False)
    Gi, m = P["G"], P["m"]
    PCs, var = pca(Gi, True)
    K = vanraden(Gi)
    conf = np.array([1.0 if l in CONF else 0.0 for l in lset])
    for tr in TR:
        s = LY.groupby("genotype")[tr].mean()
        y = np.array([s.get(l, np.nan) for l in lset])
        pv, lam = emmax(y, K, Gi, PCs[:, :3])
        n, H2, s2g, s2e = h2_means(tr)
        ok = ~np.isnan(y)
        rows.append(dict(n_lines=lname, trait=tr, n_markers=m, H2=H2, n_records=n,
                         lam=lam, n_sig_5e8=int(np.nansum(pv < 5e-8)),
                         n_sig_bonf=int(np.nansum(pv < 0.05 / m)),
                         min_p=float(np.nanmin(pv)),
                         corr_PC1=abs(np.corrcoef(y[ok], PCs[ok, 0])[0, 1]),
                         corr_type=abs(np.corrcoef(y[ok], conf[ok])[0, 1]) if conf[ok].std() > 0 else np.nan))
        print(f"  {lname} {tr:12s} m={m} H²={H2:.3f} λ={lam:.3f} "
              f"min p={np.nanmin(pv):.3e} <5e-8: {int(np.nansum(pv<5e-8))} "
              f"<bonf: {int(np.nansum(pv<0.05/m))}", flush=True)

T = pd.DataFrame(rows)
T.to_csv(OUT / "af_ratio_vs_counts.csv", index=False, encoding="utf-8-sig")

# ---- how much new information the ratio carries ------------------------------------------
means = LY.groupby("genotype")[TR].mean()
print("\ncorrelations of line means (54):")
print(means.corr().round(3).to_string())
print("\nfraction of ratio variance explained by the two counts (linear): "
      f"{np.corrcoef(means['autofertility_ratio'], np.linalg.lstsq(np.column_stack([np.ones(len(means)), means[['autofertility_self','autofertility_open']].to_numpy()]), means['autofertility_ratio'].to_numpy(), rcond=None)[0] @ np.column_stack([np.ones(len(means)), means[['autofertility_self','autofertility_open']].to_numpy()]).T)[0,1]**2:.3f}")
