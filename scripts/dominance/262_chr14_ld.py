# -*- coding: utf-8 -*-
"""B1. Literal reproduction of the calculation that produced chr17:50 245 155.

Model code is copied from the archived overdominance scan.
(the overdominance_gwas function and construction of K_hyb); input is the same VCF
4_vcf/23_no_imputation.vcf and the same phenotypes phase1_output/01_hybrids_tidy.parquet.
External files are opened read-only.
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
import numpy as np, pandas as pd, pathlib, sys
from scipy import stats, optimize
import warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

EXT = DEPOSIT
VCF = DEPOSIT / "23_no_imputation.vcf"

CONF = {"LI29", "LI30"}
LONG_TO_VCF = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
SEQ = {i: "LI%d" % i for i in range(1, 33)}
SEQ.update({33: "LI34", 34: "LI35"}); SEQ.update({i: "LI%d" % (i + 2) for i in range(35, 53)})
SEQ.update({53: "LI33", 54: "LI36", 55: "VA761", 56: "VK101", 57: "VK934"})

# ------------------------------------------------ read VCF with a custom parser
codes = {}
names, ch, po, rows = None, [], [], []
with open(VCF, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if line.startswith("##"): continue
        p = line.rstrip("\n").split("\t")
        if line.startswith("#CHROM"):
            names = p[9:]; continue
        try: c = int(p[0].replace("CM00", "").replace(".2", "")) - 7889
        except Exception: c = -1
        ch.append(c); po.append(int(p[1]))
        g = []
        for x in p[9:]:
            gt = x.split(":")[0]
            codes[gt] = codes.get(gt, 0) + 1
            g.append({"0/0": 0.0, "0/1": 1.0, "1/0": 1.0, "1/1": 2.0}.get(gt, np.nan))
        rows.append(g)
gt = np.array(rows); chrn = np.array(ch); pos = np.array(po, np.int64)
samples = np.array([SEQ.get(int(float(s)), str(s)) for s in names])
print("genotype codes in 23_no_imputation.vcf:", codes)
print("variants %d x samples %d; missing %.4f %%" % (gt.shape[0], gt.shape[1], np.isnan(gt).mean() * 100))
col_means = np.nanmean(gt, axis=1, keepdims=True)
gt = np.where(np.isnan(gt), col_means, gt)

# ------------------------------------------------ phenotypes, as in phase6c
hybrids_df = en_table(pd.read_parquet(INTERMEDIATE / "01_hybrids_tidy.parquet"))
HYB_TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter", "days_emergence_flowering"]
hyb_means = (hybrids_df[hybrids_df["trait"].isin(HYB_TRAITS)]
             .groupby(["mother_long", "father", "trait"])["value"]
             .mean().unstack().reset_index())
hyb_means["mom_vcf"] = hyb_means["mother_long"].map(LONG_TO_VCF)
sample_set = set(samples)
hyb_means = hyb_means[hyb_means["mom_vcf"].isin(sample_set) & hyb_means["father"].isin(sample_set)].reset_index(drop=True)
sample_to_col = {s: i for i, s in enumerate(samples)}
n_pairs = len(hyb_means)
print("in silico hybrids: %d, SNP: %d" % (n_pairs, len(pos)))
print("of them with confectionery fathers:", int(hyb_means["father"].isin(CONF).sum()))

hyb_geno = np.zeros((n_pairs, len(pos)))
for i, row in hyb_means.iterrows():
    hyb_geno[i] = (gt[:, sample_to_col[row["mom_vcf"]]] + gt[:, sample_to_col[row["father"]]]) / 2
is_het = (np.abs(hyb_geno - 1.0) < 0.25).astype(int)
useful_snps = is_het.sum(axis=0) >= 5
print("SNP with n_het >= 5:", int(useful_snps.sum()))

af = hyb_geno.mean(axis=0) / 2
W_hyb = (hyb_geno - 2 * af).astype(float)
K_hyb = W_hyb @ W_hyb.T / (2 * np.sum(af * (1 - af)))

def reml_loglik(theta_log, y_rot, X_rot, eigvals):
    sig2g = np.exp(theta_log[0]); sig2e = np.exp(theta_log[1])
    Dv = sig2g * eigvals + sig2e
    if (Dv <= 0).any(): return 1e10
    W = 1.0 / Dv
    XW = X_rot * W[:, None]; XtWX = X_rot.T @ XW
    try: XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError: return 1e10
    beta = XtWX_inv @ (X_rot.T @ (y_rot * W))
    resid = y_rot - X_rot @ beta
    rss = (resid ** 2 * W).sum()
    _, log_det_XtWX = np.linalg.slogdet(XtWX)
    return 0.5 * (np.log(Dv).sum() + rss + log_det_XtWX)

TRAIT = "seed_weight_1000"
y = hyb_means[TRAIT].values.astype(float)
mask = ~np.isnan(y)
y_v = y[mask]; is_het_v = is_het[mask]; geno_v = hyb_geno[mask]
K_sub = K_hyb[np.ix_(mask, mask)]
eig_sub, U_sub = np.linalg.eigh(K_sub); eig_sub = np.maximum(eig_sub, 1e-9)
y_rot = U_sub.T @ y_v
X_const = np.ones((len(y_v), 1)); X_rot = U_sub.T @ X_const
var_y = np.var(y_v)
res = optimize.minimize(reml_loglik, np.log([0.5 * var_y, 0.5 * var_y]),
                        args=(y_rot, X_rot, eig_sub), method="L-BFGS-B",
                        bounds=[(np.log(1e-8), np.log(1e8))] * 2)
sig2g, sig2e = np.exp(res.x)
print("n=%d, h2=%.3f (sig2g=%.3f sig2e=%.3f)" % (len(y_v), sig2g / (sig2g + sig2e), sig2g, sig2e))
weights = 1.0 / (sig2g * eig_sub + sig2e); sqrt_w = np.sqrt(weights)
Xw_const = X_rot * sqrt_w[:, None]; y_w = y_rot * sqrt_w
df_resid = len(y_v) - 3
pv_dom = np.full(len(pos), np.nan); b_dom = np.full(len(pos), np.nan)
pv_add = np.full(len(pos), np.nan)
for i in np.where(useful_snps)[0]:
    g = geno_v[:, i]; het_i = is_het_v[:, i].astype(float)
    if np.std(het_i) < 1e-6 or np.std(g) < 1e-6: continue
    Xf = np.column_stack([Xw_const, (U_sub.T @ g) * sqrt_w, (U_sub.T @ het_i) * sqrt_w])
    try:
        beta, *_ = np.linalg.lstsq(Xf, y_w, rcond=None)
        resid = y_w - Xf @ beta
        mse = (resid ** 2).sum() / df_resid
        XtX_inv = np.linalg.inv(Xf.T @ Xf)
        se_d = np.sqrt(mse * XtX_inv[2, 2]); se_a = np.sqrt(mse * XtX_inv[1, 1])
        if se_d > 0:
            pv_dom[i] = 2 * (1 - stats.t.cdf(abs(beta[2] / se_d), df_resid)); b_dom[i] = beta[2]
        if se_a > 0:
            pv_add[i] = 2 * (1 - stats.t.cdf(abs(beta[1] / se_a), df_resid))
    except (np.linalg.LinAlgError, ValueError):
        continue

vi = np.isfinite(pv_dom)
lam = np.median(stats.chi2.isf(pv_dom[vi], 1)) / stats.chi2.ppf(0.5, 1)
print("tested %d, lambda=%.3f, Bonferroni threshold 0.05/m = %.3e" % (vi.sum(), lam, 0.05 / vi.sum()))
order = np.argsort(np.where(vi, pv_dom, np.inf))[:10]
print("top-10 by p_dom:")
for j in order:
    print("   chr%d:%d p_dom=%.6e beta_dom=%+.4f p_add=%.4f n_het=%d"
          % (chrn[j], pos[j], pv_dom[j], b_dom[j], pv_add[j], is_het[:, j].sum()))
for tgt in (50245155, 50459845):
    i = np.where((chrn == 17) & (pos == tgt))[0]
    if len(i) == 0:
        print("chr17:%d — marker ABSENT from this panel" % tgt)
    else:
        print("chr17:%d p_dom=%.6e beta=%+.4f" % (tgt, pv_dom[i[0]], b_dom[i[0]]))

# ---------- LD around chr17:50 245 155 and chr14:169 211 003 on the merged panel
print("\n=== LD (r2) on the merged call rate >= 0.9 panel, 54 lines ===")
D = DEPOSIT
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
Gm = z["genotypes"].astype(float); Gm[Gm < 0] = np.nan
sm = [str(s) for s in z["samples"]]
chm = np.array([int(str(c).replace("CM00", "").replace(".2", "")) - 7889 for c in z["chrom"]])
pm = np.asarray(z["pos"]).astype(np.int64)
li = [i for i, s in enumerate(sm) if is_line_id(s)]
Gl = Gm[:, li]
cr = np.mean(~np.isnan(Gl), 1); afm = np.nanmean(Gl, 1) / 2; mafm = np.minimum(afm, 1 - afm)
keep = (cr >= 0.9) & (mafm >= 0.05)
Gk = Gl[keep]; Gk = np.where(np.isnan(Gk), np.nanmean(Gk, 1, keepdims=True), Gk)
chk = chm[keep]; pk = pm[keep]
print("panel:", Gk.shape)

def ld_profile(c, p0, half=3_000_000):
    i = np.where((chk == c) & (pk == p0))[0]
    if not len(i):
        print("  marker chr%d:%d absent from the panel" % (c, p0)); return None
    i = i[0]
    w = np.where((chk == c) & (np.abs(pk - p0) <= half))[0]
    x = Gk[i] - Gk[i].mean()
    Y = Gk[w] - Gk[w].mean(1, keepdims=True)
    r = (Y @ x) / np.sqrt(np.einsum("ij,ij->i", Y, Y) * (x @ x) + 1e-300)
    r2 = r ** 2
    d = np.abs(pk[w] - p0)
    print("  chr%d:%d — markers within +-%.1f Mb: %d" % (c, p0, half / 1e6, len(w)))
    for thr in (0.8, 0.5, 0.2, 0.1):
        s = r2 >= thr
        if s.sum():
            print("     r2 >= %.1f: %4d markers, position span %d-%d (%.0f kb), max distance %.0f kb"
                  % (thr, int(s.sum()), int(pk[w][s].min()), int(pk[w][s].max()),
                     (pk[w][s].max() - pk[w][s].min()) / 1e3, d[s].max() / 1e3))
        else:
            print("     r2 >= %.1f: none" % thr)
    for lo, hi in ((0, 50e3), (50e3, 200e3), (200e3, 500e3), (500e3, 1e6), (1e6, 3e6)):
        s = (d > lo) & (d <= hi)
        if s.sum():
            print("     distance %5.0f-%5.0f kb: n=%4d, mean r2=%.3f, fraction r2>=0.2 = %.3f"
                  % (lo / 1e3, hi / 1e3, int(s.sum()), r2[s].mean(), (r2[s] >= 0.2).mean()))
    return r2, d, pk[w]

ld_profile(17, 50245155)
ld_profile(17, 50436485)
ld_profile(14, 169211003)
