"""
CANONICAL GWAS on paternal lines — source of the numbers in the “Additive GWAS” section.
Replaces the external phase18 (45_phase18_lines_final.py), which is not in the paper materials.

Model — as declared in Methods: EMMAX, VanRaden K + the first three principal components,
computed on CENTERED AND STANDARDIZED genotypes of the same panel (D-3-ter).
Panel — merged, cr>=0.9, MAF>=0.05.
The no-PC model is also printed for comparison.

Output: 261_canonical_lines_gwas.csv
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
HERE = OUT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
CONF = ["LI29", "LI30"]
TRAITS = ["oil_content", "seed_weight_1000", "hull_content", "head_diameter", "plant_height",
          "days_emergence_flowering", "days_central_to_side", "autofertility_self", "autofertility_open"]

z = np.load(NPZ, allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str)
pos = z["pos"]; samples = en_ids(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc)
chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
ALLF = [s for s in samples if is_line_id(s)]

ph = en_table(pd.read_parquet(OUT / "221_lines_tidy.parquet"))
mean_ph = ph.groupby(["genotype", "trait"])["value"].mean().unstack()

def emmax(y, Gm, Kk, PC=None):
    m = ~np.isnan(y); yv = y[m]; Ks = Kk[np.ix_(m, m)]; Gs = Gm[:, m]
    ev, U = np.linalg.eigh(Ks); ev = np.maximum(ev, 1e-9)
    X = np.ones((len(yv), 1))
    if PC is not None: X = np.column_stack([X, PC[m]])
    yr = U.T @ yv; Xr = U.T @ X
    def nll(t):
        sg, se = np.exp(t); D = sg * ev + se
        W = 1 / D; XtWX = Xr.T @ (Xr * W[:, None])
        try: inv = np.linalg.inv(XtWX)
        except Exception: return 1e10
        b = inv @ (Xr.T @ (yr * W)); r = yr - Xr @ b
        _, ld = np.linalg.slogdet(XtWX)
        return 0.5 * (np.log(D).sum() + (r ** 2 * W).sum() + ld)
    vy = np.var(yv)
    o = optimize.minimize(nll, np.log([.5 * vy, .5 * vy]), method="Nelder-Mead")
    sg, se = np.exp(o.x); sw = np.sqrt(1 / (sg * ev + se))
    yw = (U.T @ yv) * sw; Xw = (U.T @ X) * sw[:, None]
    Q, _ = np.linalg.qr(Xw)
    yt = yw - Q @ (Q.T @ yw)
    Gw = (U.T @ Gs.T) * sw[:, None]
    Gt = Gw - Q @ (Q.T @ Gw)
    gg = np.einsum('ij,ij->j', Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    b = (Gt.T @ yt) / gg
    df = len(yv) - X.shape[1] - 1
    sse = (yt @ yt) - b ** 2 * gg
    sse = np.where(sse > 1e-12, sse, np.nan)
    t = b / np.sqrt(sse / df / gg)
    p = np.full(Gm.shape[0], np.nan); ok = np.isfinite(t)
    p[ok] = 2 * stats.t.sf(np.abs(t[ok]), df)
    return p, sg / (sg + se)

rows = []
for tag, lines in (("54_with_confectionery", ALLF),
                   ("52_oilseed_only", [f for f in ALLF if f not in CONF])):
    col = [samples.index(l) for l in lines]
    G = nalt[:, col]
    cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2
    maf = np.minimum(af, 1 - af)
    keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
    Gi = G[keep]; Gi = np.where(np.isnan(Gi), np.nanmean(Gi, axis=1, keepdims=True), Gi)
    chf = chrn[keep]; pf = pos[keep]
    p_al = Gi.mean(1) / 2; W = Gi - 2 * p_al[:, None]
    K = (W.T @ W) / (2 * np.sum(p_al * (1 - p_al)))
    Xc = Gi.T - Gi.T.mean(0); Xc = Xc / (Gi.T.std(0) + 1e-9)     # standardization — canonical
    Us, Ss, _ = np.linalg.svd(Xc, full_matrices=False)
    PC3 = Us[:, :3] * Ss[:3]
    frac_imp = np.isnan(G[keep]).mean()
    print(f"\n### {tag}: {len(lines)} lines, {Gi.shape[0]:,} SNP, "
          f"imputed {frac_imp*100:.2f} % of cells")
    for model, PC in (("3_PC_canon", PC3), ("no_PC", None)):
        lams, n5 = [], 0
        for tr in TRAITS:
            if tr not in mean_ph.columns: continue
            y = mean_ph[tr].reindex(lines).values.astype(float)
            p, h2 = emmax(y, Gi, K, PC)
            vi = np.isfinite(p)
            lam = np.median(stats.chi2.isf(p[vi], 1)) / stats.chi2.ppf(0.5, 1)
            k = int((p[vi] < 5e-8).sum())
            ti = int(np.nanargmin(np.where(vi, p, np.inf)))
            lams.append(lam); n5 += k
            rows.append({"panel": tag, "model": model, "trait": tr,
                         "n_SNP": Gi.shape[0], "λ": round(lam, 2), "h²_SNP": round(h2, 3),
                         "n_p<5e-8": k, "top": f"chr{int(chf[ti])}:{int(pf[ti])}",
                         "top_p": p[ti]})
        print(f"  {model:14s} λ = {min(lams):.2f}–{max(lams):.2f};  "
              f"significant at 5e-8 across all {len(lams)} traits: {n5}")

df = pd.DataFrame(rows)
df.to_csv(HERE / "261_canonical_lines_gwas.csv", index=False, encoding="utf-8-sig")
print("\nWritten: 261_canonical_lines_gwas.csv")
