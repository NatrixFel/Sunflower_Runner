"""
[VERSION WITH 3 PRINCIPAL COMPONENTS — as declared in Methods]
Honest counters for the confectionery artefact: GWAS on LINES, 54 (with LI29/LI30) vs 52,
on the MAIN merged panel (cr>=0.9, MAF>=0.05), not on the reduced var2.

EMMAX model (VanRaden K, intercept). Plus an empirical genome-wide threshold
via parametric bootstrap — so as not to rely on 5e-8, which is uncalibrated
for n=52–54 and strong LD.
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
import warnings, time
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
HERE = OUT
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
CONF = ["LI29", "LI30"]
TRAITS = ["oil_content", "hull_content", "seed_weight_1000", "head_diameter", "plant_height",
          "days_emergence_flowering", "autofertility_self", "autofertility_open"]
NBOOT = 200

z = np.load(NPZ, allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str)
pos = z["pos"]; samples = en_ids(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc)
chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
lines_all = [s for s in samples if is_line_id(s)]
print(f"lines in merged: {len(lines_all)}")

ph = en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
mean_ph = ph.groupby(["genotype", "trait"])["value"].mean().unstack()

def build_panel(lines):
    col = [samples.index(l) for l in lines]
    G = nalt[:, col]
    cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2
    maf = np.minimum(af, 1 - af)
    keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
    Gi = G[keep]; Gi = np.where(np.isnan(Gi), np.nanmean(Gi, axis=1, keepdims=True), Gi)
    return Gi, chrn[keep], pos[keep]

def emmax_scan(y, Gi, K, PC=None):
    m = ~np.isnan(y); yv = y[m]; Ks = K[np.ix_(m, m)]; G = Gi[:, m]
    ev, U = np.linalg.eigh(Ks); ev = np.maximum(ev, 1e-9)
    X = np.ones((len(yv), 1))
    if PC is not None:
        X = np.column_stack([X, PC[m]])          # + first 3 principal components
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
    rr = optimize.minimize(nll, np.log([.5 * vy, .5 * vy]), method="Nelder-Mead")
    sg, se = np.exp(rr.x); sw = np.sqrt(1 / (sg * ev + se))
    yw = (U.T @ yv) * sw; Xw = (U.T @ X) * sw[:, None]
    Q, _ = np.linalg.qr(Xw)
    yt = yw - Q @ (Q.T @ yw)
    Gw = (U.T @ G.T).T if False else (U.T @ G.T)      # (n x m)
    Gw = Gw * sw[:, None]
    Gt = Gw - Q @ (Q.T @ Gw)
    gg = np.einsum('ij,ij->j', Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    gy = Gt.T @ yt
    b = gy / gg
    df = len(yv) - X.shape[1] - 1
    sse = (yt @ yt) - b ** 2 * gg
    sse = np.where(sse > 1e-12, sse, np.nan)
    t = b / np.sqrt(sse / df / gg)
    p = np.full(Gi.shape[0], np.nan)
    ok = np.isfinite(t)
    p[ok] = 2 * stats.t.sf(np.abs(t[ok]), df)
    return p, U, sw, Q, Gt, gg, df, yv, sg, se, ev, m

rows = []
for tag, lines in (("54_with_confectionery", lines_all),
                   ("52_oilseed_only", [l for l in lines_all if l not in CONF])):
    Gi, chf, pf = build_panel(lines)
    p_al = Gi.mean(1) / 2; W = Gi - 2 * p_al[:, None]
    K = (W.T @ W) / (2 * np.sum(p_al * (1 - p_al)))
    Xc = Gi.T - Gi.T.mean(0); Xc /= (Xc.std(0) + 1e-9)
    Usvd, Ssvd, _ = np.linalg.svd(Xc, full_matrices=False)
    PC3 = (Usvd[:, :3] * Ssvd[:3])
    PC3 = (PC3 - PC3.mean(0)) / (PC3.std(0) + 1e-12)
    print(f"\n### {tag}: {len(lines)} lines, panel {Gi.shape[0]:,} SNP", flush=True)
    for tr in TRAITS:
        if tr not in mean_ph.columns: continue
        y = mean_ph[tr].reindex(lines).values.astype(float)
        if np.sum(~np.isnan(y)) < 20: continue
        p, U, sw, Q, Gt, gg, df, yv, sg, se, ev, msk = emmax_scan(y, Gi, K, PC3)
        vi = np.isfinite(p)
        lam = np.median(stats.chi2.isf(p[vi], 1)) / stats.chi2.ppf(0.5, 1)
        n5e8 = int((p[vi] < 5e-8).sum())
        # empirical threshold by bootstrap under the null model
        Ks = K[np.ix_(msk, msk)]
        A = sg * Ks + se * np.eye(len(yv))
        Lc = np.linalg.cholesky(A + 1e-9 * np.eye(len(yv)))
        mu = np.mean(yv)
        rng = np.random.default_rng(777)
        mins = []
        for _ in range(NBOOT):
            ys = mu + Lc @ rng.standard_normal(len(yv))
            yfull = np.full(len(lines), np.nan); yfull[msk] = ys
            pb, *_ = emmax_scan(yfull, Gi, K, PC3)
            mins.append(np.nanmin(pb))
        thr = np.quantile(mins, 0.05)
        nsig = int((p[vi] < thr).sum())
        top = int(np.nanargmin(np.where(vi, p, np.inf)))
        rows.append({"callset": tag, "trait": tr, "n_SNP": Gi.shape[0], "λ": round(lam, 2),
                     "n_p<5e-8": n5e8, "emp_threshold_5pct": thr, "n_below_emp": nsig,
                     "top": f"chr{int(chf[top])}:{int(pf[top])}", "top_p": p[top]})
        print(f"  {tr:26s} λ={lam:.2f}  p<5e-8: {n5e8}  emp.thr={thr:.1e} -> significant {nsig}"
              f"  top chr{int(chf[top])}:{int(pf[top])} p={p[top]:.1e}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(HERE / "26_artifact_merged_pc.csv", index=False, encoding="utf-8-sig")
print("\n=== SUMMARY (confectionery artefact on merged) ===")
piv = df.pivot(index="trait", columns="callset", values=["n_p<5e-8", "n_below_emp"])
print(piv.to_string())
print("\nWritten: 26_artifact_merged_pc.csv")
