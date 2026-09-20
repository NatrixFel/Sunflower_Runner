# -*- coding: utf-8 -*-
"""PART A. Co-author Victoria Voronezhskaya's filtering pipeline (VNIISB; a separate
run in GAPIT v3), implemented verbatim.

Order of her scheme:
  1. assign marker IDs (analog of bcftools annotate --set-id)
  2. drop markers by missingness (geno 0.2 / 0.3 / 0.5 / 0.7)
  3. MAF >= 0.05
  4. heterozygosity; two sub-variants — with and without dropping
     excessively heterozygous markers
  5. mean-imputation per marker (same as in our pipeline)
  6. repeat MAF >= 0.05 AFTER imputation

The manuscript is not edited. External VCFs are not opened for writing.
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
import numpy as np, pandas as pd, pathlib, sys, time
from scipy import stats, optimize
import warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = DEPOSIT
D = DEPOSIT
SCR = OUT
SCR.mkdir(parents=True, exist_ok=True)
from blink import blink

CONF = {"LI29", "LI30"}
TRAITS = ["oil_content", "hull_content", "seed_weight_1000", "plant_height", "head_diameter",
          "days_emergence_flowering", "days_central_to_side", "autofertility_self", "autofertility_open"]

# ---------------------------------------------------------------- data
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
samp = [str(s) for s in z["samples"]]
chrom_raw = [str(c) for c in z["chrom"]]
ref = [str(x) for x in z["ref"]]; alt = [str(x) for x in z["alt"]]
def cn(c):
    try: return int(c.replace("CM00", "").replace(".2", "")) - 7889
    except Exception: return -1
chrom = np.array([cn(c) for c in chrom_raw])
pos = np.asarray(z["pos"]).astype(np.int64)

lines54 = [s for s in samp if is_line_id(s)]
lines52 = [s for s in lines54 if s not in CONF]

ph = en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
def pheno(lines):
    out = {}
    for tr in TRAITS:
        s = ph[ph["trait"] == tr].groupby("genotype")["value"].mean()
        out[tr] = np.array([s.get(l, np.nan) for l in lines])
    return out

KEY_LOCI = [("chr8:782740", 8, 782740, 0), ("chr17:50245155", 17, 50245155, 0),
            ("chr17:50459845", 17, 50459845, 0), ("chr5:~90.15M", 5, 90150000, 200000),
            ("chr7:~144.16M", 7, 144160000, 200000)]

# ---------------------------------------------------------------- tools
def vanraden(Gm):
    p = Gm.mean(1) / 2
    Zc = Gm - 2 * p[:, None]
    return (Zc.T @ Zc) / (2 * np.sum(p * (1 - p)))

def pca(Gm, standardise=True):
    A = Gm.T.astype(float); A = A - A.mean(0, keepdims=True)
    if standardise:
        sd = A.std(0, ddof=0); sd[sd < 1e-12] = 1.0; A = A / sd
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U * S, S ** 2 / np.sum(S ** 2)

def reml_vc(y, U, eig, X):
    yr = U.T @ y; Xr = U.T @ X; n, pn = len(y), X.shape[1]
    def nll(th):
        s2g, s2e = np.exp(th)
        w = 1.0 / (s2g * eig + s2e)
        Xw = Xr * np.sqrt(w)[:, None]; yw = yr * np.sqrt(w)
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None); r = yw - Xw @ b
        _, ldx = np.linalg.slogdet(Xw.T @ Xw)
        return 0.5 * (-np.sum(np.log(w)) + ldx + (n - pn) * np.log(max(r @ r, 1e-300)))
    v = np.var(y); best = None
    for st in ([0.5*v, 0.5*v], [0.9*v, 0.1*v], [0.1*v, 0.9*v]):
        r = optimize.minimize(nll, np.log(st), method="Nelder-Mead",
                              options=dict(maxiter=600, xatol=1e-7, fatol=1e-7))
        if best is None or r.fun < best.fun: best = r
    return np.exp(best.x)

def emmax(y, K, Gm, PCs):
    ok = ~np.isnan(y)
    y = y[ok]; Ks = K[np.ix_(ok, ok)]; Gs = Gm[:, ok]
    X = np.ones((len(y), 1))
    if PCs is not None: X = np.column_stack([X, PCs[ok]])
    eig, U = np.linalg.eigh(Ks); eig = np.maximum(eig, 1e-9)
    s2g, s2e = reml_vc(y, U, eig, X)
    w = np.sqrt(1.0 / (s2g * eig + s2e)); w = w / w.mean()
    yw = (U.T @ y) * w; Xw = (U.T @ X) * w[:, None]
    Q, _ = np.linalg.qr(Xw)
    yr = yw - Q @ (Q.T @ yw)
    Gw = (Gs @ U) * w[None, :]
    Gr = Gw - (Gw @ Q) @ Q.T
    num = Gr @ yr; den = np.einsum("ij,ij->i", Gr, Gr)
    okd = den > 1e-12 * max(den.max(), 1e-300)
    beta = np.zeros(len(den)); beta[okd] = num[okd] / den[okd]
    rss = (yr @ yr) - beta ** 2 * den
    df = len(y) - X.shape[1] - 1
    se = np.sqrt(np.maximum(rss, 1e-300) / df / np.maximum(den, 1e-300))
    t = np.where(okd & (se > 0), beta / se, 0.0)
    pv = np.full(len(den), np.nan); pv[okd] = 2 * stats.t.sf(np.abs(t[okd]), df)
    v = ~np.isnan(pv)
    lam = np.median(stats.chi2.isf(pv[v], 1)) / stats.chi2.ppf(0.5, 1) if v.sum() else np.nan
    return pv, lam

# ---------------------------------------------------------------- VV pipeline
def vv_panel(lines, geno_thr, het_thr):
    """Steps 2–6 of the co-author's scheme. het_thr=None — sub-variant without a heterozygosity filter."""
    idx = [samp.index(l) for l in lines]
    Gl = G[:, idx]
    n_start = Gl.shape[0]
    # step 2: geno — missingness above threshold -> drop
    missf = np.isnan(Gl).mean(1)
    s2 = missf <= geno_thr
    # step 3: MAF >= 0.05 (on called genotypes)
    af_all = np.full(n_start, np.nan)
    with np.errstate(invalid="ignore"):
        af_all[s2] = np.nanmean(Gl[s2], 1) / 2
    maf_all = np.minimum(af_all, 1 - af_all)
    s3 = s2 & (maf_all >= 0.05)
    # step 4: heterozygosity
    Gk = Gl[s3]
    ncall = np.sum(~np.isnan(Gk), 1)
    het_m = np.sum(Gk == 1, 1) / np.maximum(ncall, 1)
    keep_het = np.ones(int(s3.sum()), bool) if het_thr is None else (het_m <= het_thr)
    s4 = s3.copy(); s4[np.where(s3)[0][~keep_het]] = False
    Gp = Gl[s4]
    miss_before_imp = float(np.isnan(Gp).mean())
    # step 5: mean-imputation per marker
    Gi = np.where(np.isnan(Gp), np.nanmean(Gp, 1, keepdims=True), Gp)
    # step 6: repeat MAF >= 0.05 AFTER imputation
    af2 = Gi.mean(1) / 2; maf2 = np.minimum(af2, 1 - af2)
    keep2 = maf2 >= 0.05
    n_drop_maf2 = int((~keep2).sum())
    # same step after rounding imputed dosages to genotypes
    # (outside her scheme; measured separately as a "hard" imputation estimate)
    Gr_ = np.rint(Gi)
    af2r = Gr_.mean(1) / 2; maf2r = np.minimum(af2r, 1 - af2r)
    n_drop_maf2_round = int((maf2r < 0.05).sum())
    s6 = s4.copy(); s6[np.where(s4)[0][~keep2]] = False
    Gf = Gi[keep2]
    return dict(mask=s6, G=Gf, lines=lines, n_start=n_start,
                drop_geno=int(n_start - s2.sum()),
                drop_maf=int(s2.sum() - s3.sum()),
                drop_het=int(s3.sum() - s4.sum()),
                drop_maf2=n_drop_maf2, drop_maf2_round=n_drop_maf2_round,
                m=int(s6.sum()), miss=miss_before_imp,
                het_marker=het_m, mask3=s3)

# ---------------------------------------------------------------- run
HET_VARIANTS = [("no_het_filter", None), ("het <= 0.10", 0.10)]
GENOS = [0.2, 0.3, 0.5, 0.7]
tag = ""
if "--het05" in sys.argv:
    HET_VARIANTS = [("het <= 0.05", 0.05)]; tag = "_het05"

rows_panel, rows_gwas, rows_loci = [], [], []
RUN_BLINK = "--noblink" not in sys.argv
only = [float(a.split("=")[1]) for a in sys.argv if a.startswith("--geno=")]
if only: GENOS = only

for gthr in GENOS:
    for hname, hthr in HET_VARIANTS:
        for lset, lname in ((lines54, "54"), (lines52, "52")):
            t0 = time.time()
            P = vv_panel(lset, gthr, hthr)
            Gi = P["G"]; m = P["m"]
            PCs, var = pca(Gi, True)
            conf = np.array([1.0 if l in CONF else 0.0 for l in lset])
            r_pc1 = abs(np.corrcoef(PCs[:, 0], conf)[0, 1]) if conf.std() > 0 else np.nan
            K = vanraden(Gi)
            rows_panel.append(dict(geno=gthr, het=hname, n_lines=lname, n_markers=m,
                                   n_start=P["n_start"], drop_geno=P["drop_geno"],
                                   drop_maf=P["drop_maf"], drop_het=P["drop_het"],
                                   drop_maf2=P["drop_maf2"], drop_maf2_round=P["drop_maf2_round"],
                                   missingness=P["miss"], PC1=var[0]*100, PC2=var[1]*100,
                                   PC3=var[2]*100, corr_PC1_type=r_pc1, bonferroni=0.05/m))
            phen = pheno(lset)
            sub_chrom = chrom[P["mask"]]; sub_pos = pos[P["mask"]]
            for tr in TRAITS:
                y = phen[tr]
                if np.sum(~np.isnan(y)) < 20: continue
                pv, lam = emmax(y, K, Gi, PCs[:, :3])
                rows_gwas.append(dict(geno=gthr, het=hname, n_lines=lname, trait=tr,
                                      model="EMMAX", n_markers=m, lam=lam,
                                      n_sig_5e8=int(np.nansum(pv < 5e-8)),
                                      n_sig_bonf=int(np.nansum(pv < 0.05/m)),
                                      min_p=float(np.nanmin(pv))))
                for nm, ch, po, win in KEY_LOCI:
                    hit = np.where((sub_chrom == ch) & (np.abs(sub_pos - po) <= win))[0]
                    rows_loci.append(dict(geno=gthr, het=hname, n_lines=lname, trait=tr,
                                          locus=nm, present=len(hit) > 0,
                                          p=float(np.nanmin(pv[hit])) if len(hit) else np.nan,
                                          markers_in_window=len(hit)))
                if RUN_BLINK:
                    try:
                        res = blink(y, Gi, sub_chrom, sub_pos, PCs[:, :3])
                        pb = res["pvals"]
                        rows_gwas.append(dict(geno=gthr, het=hname, n_lines=lname, trait=tr,
                                              model="BLINK", n_markers=m, lam=res["lambda"],
                                              n_sig_5e8=int(np.nansum(pb < 5e-8)),
                                              n_sig_bonf=int(np.nansum(pb < 0.05/m)),
                                              min_p=float(np.nanmin(pb))))
                    except Exception as e:
                        rows_gwas.append(dict(geno=gthr, het=hname, n_lines=lname, trait=tr,
                                              model="BLINK", n_markers=m, lam=np.nan,
                                              n_sig_5e8=-1, n_sig_bonf=-1, min_p=np.nan))
            print("geno=%.1f %-16s %s m=%6d miss=%5.2f%% PC1=%5.2f%% corr=%.3f | "
                  "drop g/m/h/m2=%d/%d/%d/%d | %.0f s"
                  % (gthr, hname, lname, m, P["miss"]*100, var[0]*100, r_pc1,
                     P["drop_geno"], P["drop_maf"], P["drop_het"], P["drop_maf2"],
                     time.time()-t0), flush=True)

pd.DataFrame(rows_panel).to_csv(SCR / ("vv_panels%s.csv" % tag), index=False, encoding="utf-8-sig")
pd.DataFrame(rows_gwas).to_csv(SCR / ("vv_gwas%s.csv" % tag), index=False, encoding="utf-8-sig")
pd.DataFrame(rows_loci).to_csv(SCR / ("vv_loci%s.csv" % tag), index=False, encoding="utf-8-sig")
print("\nwritten to", SCR)
