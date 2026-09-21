# -*- coding: utf-8 -*-
"""Additions to parts A and B:
  A4  — diagnostics of the repeat MAF filter after imputation at all geno thresholds;
  A4  — heterozygosity by marker and by line, distribution;
  A4  — value of setid: ID check against the panel from the separate run by
        co-author Victoria Voronezhskaya (VNIISB) in GAPIT v3;
  B3  — dominance scan on the co-author's panels (57 samples).
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

ROOT = DEPOSIT
D = DEPOSIT
EXT = ROOT / "FINAL_DATA" / "_external"
CONF = {"LI29", "LI30"}
LTV = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}

z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
samp = [str(s) for s in z["samples"]]
chrom = np.array([int(str(c).replace("CM00", "").replace(".2", "")) - 7889 for c in z["chrom"]])
pos = np.asarray(z["pos"]).astype(np.int64)
ref = np.array([str(x) for x in z["ref"]]); alt = np.array([str(x) for x in z["alt"]])
lines54 = [s for s in samp if is_line_id(s)]
lines52 = [s for s in lines54 if s not in CONF]

print("=" * 78)
print("A4-1. Repeat MAF after imputation: what it actually drops")
print("=" * 78)
for lset, lname in ((lines54, "54"), (lines52, "52")):
    idx = [samp.index(l) for l in lset]
    Gl = G[:, idx]
    missf = np.isnan(Gl).mean(1)
    for gthr in (0.2, 0.3, 0.5, 0.7):
        s2 = missf <= gthr
        af = np.full(len(missf), np.nan)
        af[s2] = np.nanmean(Gl[s2], 1) / 2
        maf1 = np.minimum(af, 1 - af)
        s3 = s2 & (maf1 >= 0.05)
        Gp = Gl[s3]
        Gi = np.where(np.isnan(Gp), np.nanmean(Gp, 1, keepdims=True), Gp)
        m1 = maf1[s3]
        af2 = Gi.mean(1) / 2; maf2 = np.minimum(af2, 1 - af2)
        drop = maf2 < 0.05
        onboundary = np.abs(m1 - 0.05) < 1e-12
        Gr = np.rint(Gi); af2r = Gr.mean(1) / 2; maf2r = np.minimum(af2r, 1 - af2r)
        dropr = maf2r < 0.05
        print("lines %s, geno<=%.1f: m=%6d | max|MAF2-MAF1|=%.2e | dropped by repeat MAF=%4d, "
              "of those exactly on the 0.05 boundary: %4d | after rounding dosages dropped %5d, "
              "of those not on the boundary: %5d"
              % (lname, gthr, int(s3.sum()), float(np.max(np.abs(maf2 - m1))), int(drop.sum()),
                 int((drop & onboundary).sum()), int(dropr.sum()), int((dropr & ~onboundary).sum())))

print()
print("=" * 78)
print("A4-2. Heterozygosity: by marker and by line")
print("=" * 78)
idx = [samp.index(l) for l in lines54]
Gl = G[:, idx]
missf = np.isnan(Gl).mean(1)
for gthr in (0.2, 0.3, 0.5, 0.7):
    s2 = missf <= gthr
    af = np.full(len(missf), np.nan); af[s2] = np.nanmean(Gl[s2], 1) / 2
    maf1 = np.minimum(af, 1 - af); s3 = s2 & (maf1 >= 0.05)
    Gk = Gl[s3]
    nc = np.sum(~np.isnan(Gk), 1)
    hm_ = np.sum(Gk == 1, 1) / np.maximum(nc, 1)
    hl = np.nansum(Gk == 1, 0) / np.maximum(np.sum(~np.isnan(Gk), 0), 1)
    print("geno<=%.1f, m=%6d: by marker median=%.4f, mean=%.4f, q75=%.4f, q90=%.4f, "
          "q99=%.4f, max=%.4f; fraction of markers with het>0.10 = %.4f"
          % (gthr, int(s3.sum()), np.median(hm_), hm_.mean(), np.quantile(hm_, .75),
             np.quantile(hm_, .9), np.quantile(hm_, .99), hm_.max(), (hm_ > 0.10).mean()))
    if gthr == 0.2:
        o = np.argsort(hl)
        print("   by line: median=%.4f, range %.4f (%s) — %.4f (%s)"
              % (np.median(hl), hl[o[0]], lines54[o[0]], hl[o[-1]], lines54[o[-1]]))
        print("   five most heterozygous lines: " +
              ", ".join("%s %.3f" % (lines54[i], hl[i]) for i in o[::-1][:5]))

print()
print("=" * 78)
print("A4-3. setid: check against the separate-run panel (co-author V. Voronezhskaya, VNIISB, GAPIT v3)")
print("=" * 78)
f = EXT / "7_ручной обсчет воронежская" / "no_sugar" / "no_sugar" / "gapit" / "input" / "gapit_genotypes.txt"
ids2, key2 = [], []
with open(f, encoding="utf-8", errors="replace") as fh:
    fh.readline()
    for line in fh:
        p = line.split("\t", 5)
        ids2.append(p[0]); key2.append((int(p[2]), int(p[3])))
print("markers in the separate run (V. Voronezhskaya, GAPIT v3): %d" % len(ids2))
# our setid in their format: CHROM_POS_REF_ALT with the original contig name
contig = np.array([str(c) for c in z["chrom"]])
our_id = np.array(["%s_%d_%s_%s" % (contig[i], pos[i], ref[i], alt[i]) for i in range(len(pos))])
our_pos = set(zip(chrom.tolist(), pos.tolist()))
setours = set(our_id.tolist())
by_pos = sum(1 for k in key2 if k in our_pos)
by_id = sum(1 for i in ids2 if i in setours)
print("matched by position (chrom, pos): %d of %d" % (by_pos, len(ids2)))
print("matched by full identifier CHROM_POS_REF_ALT: %d of %d" % (by_id, len(ids2)))
mis = [(i, k) for i, k in zip(ids2, key2) if k in our_pos and i not in setours]
print("position matches, identifier does not: %d" % len(mis))
for i, k in mis[:8]:
    j = np.where((chrom == k[0]) & (pos == k[1]))[0][0]
    print("   theirs %s   ours %s" % (i, our_id[j]))

print()
print("=" * 78)
print("B3. Dominance scan on the co-author's panels (57 samples)")
print("=" * 78)
bl = en_table(pd.read_csv(OUT / "222_hybrid_blup_phenotypes.csv"))
bl[["mom_long", "father"]] = bl["hybrid"].str.split("_", expand=True)
bl["mom"] = bl["mom_long"].map(LTV)
TRAIT = "seed_weight_1000"

def dom_scan(Gi, chf, pf, names):
    s2c = {s: i for i, s in enumerate(names)}
    d = bl.dropna(subset=[TRAIT]).copy()
    d = d[~d["father"].isin(CONF)]
    d = d[d["mom"].isin(s2c) & d["father"].isin(s2c)].reset_index(drop=True)
    y = d[TRAIT].values.astype(float); moms = d["mom"].values
    n = len(d); n_snp = Gi.shape[0]
    HG = np.empty((n, n_snp))
    for i in range(n):
        HG[i] = (Gi[:, s2c[moms[i]]] + Gi[:, s2c[d["father"].values[i]]]) / 2
    IH = (np.abs(HG - 1) < 0.25).astype(float)
    p_al = HG.mean(0) / 2
    Wa = HG - 2 * p_al; Ka = (Wa @ Wa.T) / (2 * np.sum(p_al * (1 - p_al)))
    eh = 2 * p_al * (1 - p_al); Wd = IH - eh; Kd = (Wd @ Wd.T) / np.sum(eh * (1 - eh))
    X = np.column_stack([np.ones(n), (moms == "VA761").astype(float), (moms == "VK934").astype(float)])
    useful = (IH.sum(0) >= 5) & (IH.std(0) > 1e-6) & (HG.std(0) > 1e-6)
    nX = X.shape[1]; df = n - nX - 2; I_n = np.eye(n)
    def reml_prof(h):
        ha, hd = h
        if ha < 0 or hd < 0 or ha + hd > 0.999: return 1e10
        A = ha * Ka + hd * Kd + (1 - ha - hd) * I_n
        try: L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError: return 1e10
        AiX = np.linalg.solve(A, X); XtAX = X.T @ AiX
        b = np.linalg.solve(XtAX, AiX.T @ y)
        r = y - X @ b; q = r @ np.linalg.solve(A, r)
        if q <= 0: return 1e10
        _, ldX = np.linalg.slogdet(XtAX)
        return 0.5 * (2 * np.sum(np.log(np.diag(L))) + ldX + (n - nX) * np.log(q / (n - nX)))
    best = None
    for st in ((.5, .2), (.2, .5), (.05, .05)):
        r = optimize.minimize(reml_prof, np.array(st), method="Nelder-Mead",
                              options={"maxiter": 400, "fatol": 1e-7, "xatol": 1e-5})
        if best is None or r.fun < best.fun: best = r
    ha, hd = np.clip(best.x, 0, None)
    if ha + hd > 0.999:
        s = (ha + hd) / 0.999; ha, hd = ha / s, hd / s
    A = ha * Ka + hd * Kd + (1 - ha - hd) * I_n
    ev, U = np.linalg.eigh(A); ev = np.maximum(ev, 1e-10); sw = 1 / np.sqrt(ev)
    Xw = (U.T @ X) * sw[:, None]; Q, _ = np.linalg.qr(Xw)
    yw = (U.T @ y) * sw; yt = yw - Q @ (Q.T @ yw)
    Gt = (U.T @ HG) * sw[:, None]; Gt -= Q @ (Q.T @ Gt)
    Ht = (U.T @ IH) * sw[:, None]; Ht -= Q @ (Q.T @ Ht)
    gg = np.einsum("ij,ij->j", Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    gh = np.einsum("ij,ij->j", Gt, Ht); gy = Gt.T @ yt
    Hp = Ht - Gt * (gh / gg)
    hh = np.einsum("ij,ij->j", Hp, Hp); hy = Hp.T @ yt
    ok = useful & np.isfinite(hh) & (hh > 1e-10) & np.isfinite(gg)
    b2 = hy / np.where(ok, hh, np.nan)
    sse = (yt @ yt) - (gy ** 2) / gg - b2 ** 2 * hh
    sse = np.where(sse > 1e-12, sse, np.nan)
    t = b2 / np.sqrt(sse / df / hh)
    p = np.full(n_snp, np.nan); mk = ok & np.isfinite(t)
    p[mk] = 2 * stats.t.sf(np.abs(t[mk]), df)
    return p, b2, n, np.isfinite(p).sum(), ha, hd

cr_all = np.mean(~np.isnan(G), 1)
af_all = np.nanmean(G, 1) / 2; maf_all = np.minimum(af_all, 1 - af_all)
het_all = np.nansum(G == 1, 1) / np.maximum(np.sum(~np.isnan(G), 1), 1)
rows = []
for gthr in (0.2, 0.3, 0.5, 0.7):
    for hname, hthr in (("no_het_filter", None), ("het<=0.10", 0.10)):
        keep = (1 - cr_all <= gthr) & (maf_all >= 0.05)
        if hthr is not None: keep = keep & (het_all <= hthr)
        Gp = G[keep]
        Gi = np.where(np.isnan(Gp), np.nanmean(Gp, 1, keepdims=True), Gp)
        chf = chrom[keep]; pf = pos[keep]
        p, b, n, nt, ha, hd = dom_scan(Gi, chf, pf, samp)
        vi = np.isfinite(p); bi = int(np.nanargmin(np.where(vi, p, np.inf)))
        w14 = np.where((chf == 14) & (pf >= 169_200_000) & (pf <= 169_230_000) & vi)[0]
        w17 = np.where((chf == 17) & (pf >= 50_200_000) & (pf <= 50_500_000) & vi)[0]
        i45 = np.where((chf == 17) & (pf == 50459845))[0]
        r = dict(geno=gthr, het=hname, n_markers=int(keep.sum()), n_tested=int(nt),
                 n=n, Ka=round(ha, 3), Kd=round(hd, 3),
                 bonferroni=0.05 / nt,
                 best="chr%d:%d" % (chf[bi], pf[bi]), best_p=p[bi], best_beta=b[bi],
                 chr14_n=len(w14), chr14_p=p[w14].min() if len(w14) else np.nan,
                 chr17_n=len(w17), chr17_p=p[w17].min() if len(w17) else np.nan,
                 chr17_beta=b[w17][int(np.argmin(p[w17]))] if len(w17) else np.nan,
                 chr17_top=int(pf[w17][int(np.argmin(p[w17]))]) if len(w17) else None,
                 m50459845="present" if len(i45) else "missing")
        rows.append(r)
        print("geno=%.1f %-16s m=%6d n=%3d best %s p=%.2e | chr14 p=%.2e | chr17 p=%.2e (%s) | 50459845 %s"
              % (gthr, hname, r["n_markers"], n, r["best"], r["best_p"],
                 r["chr14_p"], r["chr17_p"], r["chr17_top"], r["m50459845"]), flush=True)
SCR = OUT
pd.DataFrame(rows).to_csv(SCR / "vv_dom_scans.csv", index=False, encoding="utf-8-sig")
print("written:", SCR / "vv_dom_scans.csv")
