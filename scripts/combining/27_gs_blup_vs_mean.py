"""
Honest recompute of script 58 (GS on LINES: colleagues’ BLUP phenotypes vs my means).
The original read var2 with a fill=0 bug -> 250k phantom markers.
Here: 23_merged_genotypes.npz (missing values stored as -1), panel cr>=0.9 & MAF>=0.05.
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
import numpy as np, pandas as pd
from scipy import stats, optimize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
HERE = OUT
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}

SEQ = {f"{i}.0": f"LI{i}" for i in range(1, 33)}
SEQ.update({"33.0": "LI34", "34.0": "LI35"})
SEQ.update({f"{i}.0": f"LI{i+2}" for i in range(35, 53)})
SEQ.update({"53.0": "LI33", "54.0": "LI36"})
def seqid(s):
    try: return SEQ.get(f"{int(float(s))}.0", str(s))
    except Exception: return str(s)

TRAITS = [("oil_content_pct", "oil_content"), ("seed_weight_1000_g", "seed_weight_1000"),
          ("height_cm_mean", "plant_height"), ("head_diameter_cm_mean", "head_diameter"),
          ("hull_pct", "hull_content"), ("emergence_to_flowering_days", "days_emergence_flowering"),
          ("central_to_side_flowering_days", "days_central_to_side"),
          ("self_fertility_self_pollination_mean", "autofertility_open"),
          ("self_fertility_open_pollination_mean", "autofertility_self")]

z = np.load(NPZ, allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str)
samples = en_ids(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc)
chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
lines = [s for s in samples if is_line_id(s)]
oil52 = [l for l in lines if l not in ("LI29", "LI30")]
col = [samples.index(l) for l in oil52]
G = nalt[:, col]
cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2; maf = np.minimum(af, 1 - af)
keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
Gi = G[keep]; Gi = np.where(np.isnan(Gi), np.nanmean(Gi, axis=1, keepdims=True), Gi)
print(f"lines: {len(oil52)};  HONEST panel: {Gi.shape[0]:,} SNP "
      f"(the original script 58 had ~250,000 phantoms)")

p = Gi.mean(1) / 2; W = Gi - 2 * p[:, None]
K = (W.T @ W) / (2 * np.sum(p * (1 - p)))

lines_v2 = en_table(pd.read_parquet(OUT / "221_lines_tidy.parquet"))
mean_ph = lines_v2.groupby(["genotype", "trait"])["value"].mean().unstack()
blup = en_table(pd.read_csv(INTERMEDIATE / "gapit_phenotypes.csv", dtype={"Taxa": str}))
blup["line"] = blup["Taxa"].apply(seqid); blup = blup.set_index("line")

def reml_vc(y, Ks):
    m = ~np.isnan(y); yv = y[m]; Kp = Ks[np.ix_(m, m)]
    ev, U = np.linalg.eigh(Kp); ev = np.maximum(ev, 1e-9)
    yr = U.T @ yv; Xr = U.T @ np.ones((len(yv), 1))
    def nll(th):
        sg, se = np.exp(th); D = sg * ev + se
        if (D <= 0).any(): return 1e10
        Wd = 1 / D; XtWX = Xr.T @ (Xr * Wd[:, None])
        b = (Xr.T @ (yr * Wd)) / XtWX[0, 0]; r = yr - Xr @ b
        return 0.5 * (np.log(D).sum() + (r ** 2 * Wd).sum() + np.log(XtWX[0, 0]))
    vy = np.var(yv)
    r = optimize.minimize(nll, np.log([.5 * vy, .5 * vy]), method="Nelder-Mead")
    sg, se = np.exp(r.x); return sg, se, m

def gblup_loo(y, Ks, sg, se, mask):
    idx = np.where(mask)[0]; yv = y[mask]; nn = len(yv)
    Ksub = Ks[np.ix_(idx, idx)]; pred = np.full(nn, np.nan)
    for j in range(nn):
        tr = np.array([k for k in range(nn) if k != j])
        Vtr = sg * Ksub[np.ix_(tr, tr)] + se * np.eye(nn - 1)
        Vi = np.linalg.inv(Vtr); one = np.ones(nn - 1)
        mu = (one @ Vi @ yv[tr]) / (one @ Vi @ one)
        pred[j] = mu + sg * Ksub[j, tr] @ Vi @ (yv[tr] - mu)
    return np.corrcoef(pred, yv)[0, 1]

rows = []
for eng, rus in TRAITS:
    res = {"trait": rus}
    y_mean = mean_ph[rus].reindex(oil52).values.astype(float) if rus in mean_ph.columns else np.full(len(oil52), np.nan)
    y_blup = blup[eng].reindex(oil52).values.astype(float) if eng in blup.columns else np.full(len(oil52), np.nan)
    for tag, y in (("mean", y_mean), ("BLUP", y_blup)):
        if np.sum(~np.isnan(y)) < 20:
            res[f"h2_{tag}"] = None; res[f"GS_{tag}"] = None; continue
        sg, se, m = reml_vc(y, K)
        res[f"h2_{tag}"] = round(float(sg / (sg + se)), 2)
        res[f"GS_{tag}"] = round(float(gblup_loo(y, K, sg, se, m)), 3)
    if res.get("GS_mean") is not None and res.get("GS_BLUP") is not None:
        res["Δ"] = round(res["GS_BLUP"] - res["GS_mean"], 3)
    rows.append(res)

df = pd.DataFrame(rows)
print("\n=== GS on lines on the HONEST panel (merged) ===")
print(df.to_string(index=False))
ok = df.dropna(subset=["GS_mean", "GS_BLUP"])
print(f"\nMean accuracy: mean={ok['GS_mean'].mean():.3f}  BLUP={ok['GS_BLUP'].mean():.3f}  "
      f"(Δ={ok['GS_BLUP'].mean()-ok['GS_mean'].mean():+.3f})")
w = stats.wilcoxon(ok["GS_mean"], ok["GS_BLUP"])
print(f"Wilcoxon (mean vs BLUP), n={len(ok)}: p={w.pvalue:.3f}")
print("\nCOMPARISON with the old (bug, var2 ~250k phantoms):")
old = en_table(pd.read_csv(MODELS / "27_gs_blup_vs_mean_var2.csv"))
print(f"  old mean: mean={old['GSacc_mean'].mean():.3f}  BLUP={old['GSacc_BLUP'].mean():.3f}")
df.to_csv(HERE / "27_gs_blup_vs_mean.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: 27_gs_blup_vs_mean.csv")
