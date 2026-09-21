"""
CONSOLIDATED CHECK OF NUMBERS CLAIMED IN THE MANUSCRIPT.

Replaces the invalid verify_all_results.py (see DECISIONS, D-9): that file had
two defects that made the checks vacuous, and additionally read var2 with a bug.

Updated 2026-08-23 for the current decisions D020 (Baker’s ratio, fixed
“Year × Tester” term) and D021 (H² and repeatability). Before this fix, sections 1 and 2 were
themselves part of the class of defect the script is meant to catch: section 1 did not isolate
the “year × line” term (withdrawn model), and section 2 checked the “plot-level”
and “summary” tables with the same simplified formula that omitted the “year × tester” term.
The fix changes the calculation specification — the method (sequential sums of squares,
method of moments) and the check frame (`check()`, tolerances, CSV format) were not changed.
Section 1 and the “plot-level” part of section 2 now use the same
shared variance-component calculation under model D020: year, tester and year×tester
are fixed; line, tester×line, year×line, year×tester×line are random
(`baker_h2_components()`). The “summary” part of section 2 is deliberately NOT moved onto
this model: the table `01_hybrids_tidy.parquet` gives two observations per combination
(by year) with no within-year replication, so the “year × line” term is in principle
inseparable from error — exactly the limitation that is why Results 3.1 in
the current manuscript reports it as a separate “robustness check on the
year-means table” with its own expectation (0.716/0.936/0.936), not as an alternative
estimate of the same ratio. The method of moments (not REML, unlike the canonical
`scripts\\baker_v4.py` and `scripts\\222_h2_recalc.py`) is an independent, lighter
check; a third-digit discrepancy with REML is expected and is built into the tolerance.

Quantities recalculated here that had no dedicated script in the package:
  1. Heritability H² and plot repeatability (hybrids, plot-level raw, D021).
  2. Baker’s ratio (D020) — on plot-level raw; separately, with its own expectation —
     robustness on the year-means table (independence-of-source check).
  3. GCA of maternal lines.
  4. Power analysis (detectable PVE and required n).
  5. λ and number of significant SNPs for hybrid EMMAX — with and without components.
  6. Genomic-selection accuracy (check against the stored table).

Each output row: claimed in the text / recalculated / matched.
Output: verify_reported_numbers_<date>.csv
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
import scipy.linalg as sla
from pathlib import Path
from scipy import stats, optimize
from datetime import date
import warnings
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
HERE = OUT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
CONF = {"LI29", "LI30"}
LTV = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
rows = []
def check(name, claimed, got, tol=0.015):
    ok = (claimed is None) or (abs(got - claimed) <= tol * max(1e-9, abs(claimed)))
    rows.append({"value": name, "in_text": claimed, "recomputed": round(float(got), 4),
                 "matched": "—" if claimed is None else ("YES" if ok else "NO")})
    flag = "" if claimed is None else ("  ✓" if ok else "  ✗ MISMATCH")
    print(f"  {name:52s} text={claimed!s:>8}  calc={got:.4f}{flag}")

# ---------- helper: variance components under model D020/D021 ----------
def dummies(labels):
    u = sorted(set(labels)); idx = {l: i for i, l in enumerate(u)}
    Z = np.zeros((len(labels), len(u)))
    for i, l in enumerate(labels):
        Z[i, idx[l]] = 1.0
    return Z

def inter(A, B):
    return np.einsum("ij,ik->ijk", A, B).reshape(A.shape[0], -1)

def seq_anova(y, terms, tol=1e-9):
    """Sequential (Type I) sums of squares by block orthogonalization —
    the same principle as in scripts\\baker_v4.py, but by method of moments, not REML."""
    n = len(y); Q = np.ones((n, 1)) / np.sqrt(n); rows = {}
    for name, M in terms:
        Rm = M - Q @ (Q.T @ M)
        q_, r_, _ = sla.qr(Rm, mode="economic", pivoting=True)
        d_ = np.abs(np.diag(r_)) if r_.size else np.array([])
        k = int(np.sum(d_ > tol * max(d_[0], np.linalg.norm(M)))) if d_.size else 0
        if k > 0:
            Qn = q_[:, :k]; Qn = Qn - Q @ (Q.T @ Qn); Qn, _ = sla.qr(Qn, mode="economic")
            rows[name] = dict(SS=float(np.sum((Qn.T @ y) ** 2)), df=k)
            Q = np.hstack([Q, Qn])
        else:
            rows[name] = dict(SS=0.0, df=0)
    rows["error"] = dict(SS=float(y @ y) - float(np.sum((Q.T @ y) ** 2)), df=n - Q.shape[1])
    for v in rows.values():
        v["MS"] = v["SS"] / v["df"] if v["df"] > 0 else np.nan
    return rows

def baker_h2_components(d):
    """D020/D021 specification on plot-level raw data: year, tester, year×tester
    are fixed (swept sequentially before the random terms); line,
    tester×line, year×line, year×tester×line are random, method of moments.
    Returns a dict of components (l=Line, ml=Tester×Line, yl=Year×Line,
    yml=Year×Tester×Line, e=error), r (mean number of plots per cell),
    ny (number of years), nm (number of testers)."""
    y = d["value"].to_numpy(float)
    Y = dummies(d["year"].astype(str).tolist())
    R = dummies((d["year"].astype(str) + "_" + d["rep"].astype(str)).tolist())
    M = dummies(d["mother"].astype(str).tolist())
    L = dummies(d["father"].astype(str).tolist())
    ML, YM, YL = inter(M, L), inter(Y, M), inter(Y, L)
    YML = inter(YM, L)
    A = seq_anova(y, [("year", Y), ("rep_in_year", R), ("tester", M), ("year_x_tester", YM),
                      ("line", L), ("tester_x_line", ML), ("year_x_line", YL),
                      ("year_x_tester_x_line", YML)])
    r = float(d.groupby(["year", "mother", "father"]).size().mean())
    ny, nm = float(d["year"].nunique()), float(d["mother"].nunique())
    ms = {k: A[k]["MS"] for k in A}
    e, yml, yl, ml, l = (ms["error"], ms["year_x_tester_x_line"], ms["year_x_line"],
                         ms["tester_x_line"], ms["line"])
    return dict(e=e,
                yml=max(0.0, (yml - e) / r),
                yl=max(0.0, (yl - yml) / (r * nm)),
                ml=max(0.0, (ml - yml) / (r * ny)),
                l=max(0.0, (l - ml - yl + yml) / (r * ny * nm))), r, ny, nm

# ---------- 1. H² and repeatability (D021) ----------
print("\n1. HERITABILITY AND REPEATABILITY (plot-level raw, D021 specification)")
plot = en_table(pd.read_parquet(OUT / "221_hybrid_plot_level.parquet"))
def h2_rep(trait):
    d = plot[plot["trait"] == trait].dropna(subset=["value"]).reset_index(drop=True)
    s2, r, ny, nm = baker_h2_components(d)
    s2_gen = s2["l"] + s2["ml"]; s2_gy = s2["yl"] + s2["yml"]
    h2 = s2_gen / (s2_gen + s2_gy / ny + s2["e"] / (ny * r))
    rep = s2_gen / (s2_gen + s2_gy + s2["e"])
    return h2, rep
for tr, claim_h2 in (("oil_content", 0.90), ("seed_weight_1000", 0.91), ("seed_yield", 0.43)):
    h2, rep = h2_rep(tr)
    check(f"H² of the combination mean — {tr}", claim_h2, h2, tol=0.06)
    if tr == "seed_yield":
        check("plot repeatability — seed yield", 0.16, rep, tol=0.20)

# ---------- 2. Baker’s ratio (D020) ----------
print("\n2. BAKER'S RATIO (plot-level — current D020 specification; "
      "year-means table — robustness check on a different source)")
hyb1 = en_table(pd.read_parquet(INTERMEDIATE / "01_hybrids_tidy.parquet"))
K1 = "mother_long" if "mother_long" in hyb1.columns else "mother"
def baker_pooled(tab):
    """Pooled estimate on the means table (2 observations per combination — by year,
    no within-year replication): year×line is inseparable from error and SCA on this
    source in principle, so the formula here still does not isolate that
    term — not the withdrawn D008 model, but a standalone check stated explicitly in the text
    (Results 3.1) as a robustness check with its own expectation."""
    g = np.nanmean(tab.values)
    gf = np.nanmean(tab.values, 1) - g; gm = np.nanmean(tab.values, 0) - g
    sca = tab.values - g - gf[:, None] - gm[None, :]
    vg = np.nanvar(gf, ddof=1); vs = np.nanvar(sca, ddof=1)
    return 2 * vg / (2 * vg + vs)
MAP = {"seed_yield": ("seed_yield", 0.440), "oil_content": ("oil_content", 0.958),
       "seed_weight_1000": ("seed_weight_1000", 0.951)}
MAP_TABLE = {"seed_yield": ("seed_yield", 0.716), "oil_content": ("oil_content", 0.936),
             "seed_weight_1000": ("seed_weight_1000", 0.936)}
for t14, (t1, claim) in MAP.items():
    d = plot[plot["trait"] == t14].dropna(subset=["value"]).reset_index(drop=True)
    s2, r, ny, nm = baker_h2_components(d)
    den = 2 * s2["l"] + s2["ml"]
    baker_full = 2 * s2["l"] / den if den > 0 else np.nan
    check(f"baker {t14} (plot-level, D020)", claim, baker_full, tol=0.03)
    _, claim_tab = MAP_TABLE[t14]
    b = hyb1[hyb1["trait"] == t1].groupby(["father", K1])["value"].mean().unstack()
    check(f"baker {t14} (year-means table, robustness)", claim_tab,
          baker_pooled(b), tol=0.03)

# ---------- 3. Maternal GCA ----------
print("\n3. MATERNAL-LINE GCA")
gm = en_table(pd.read_csv(OUT / "25_gca_mothers.csv"))
mc = "mother_long" if "mother_long" in gm.columns else [c for c in gm.columns if "mother" in c][0]
gm = gm.set_index(mc)
for mother, col, claim in (("VK101A", "oil_content", 1.20), ("VK934A", "seed_weight_1000", 10.9),
                           ("VK934A", "plant_height", 8.4)):
    if col in gm.columns and mother in gm.index:
        check(f"GCA {mother} — {col}", claim, float(gm.loc[mother, col]), tol=0.05)

# ---------- 4. Power ----------
print("\n4. POWER ANALYSIS (Visscher NCP, alpha=5e-8)")
crit = stats.chi2.isf(5e-8, 1)
def power(n, pve): return stats.ncx2.sf(crit, 1, n * pve / (1 - pve))
check("detectable PVE at n=54 and 80% power", 42.0,
      optimize.brentq(lambda p: power(54, p) - 0.8, 0.01, 0.99) * 100, tol=0.03)
for pve, claim in ((0.15, 225), (0.10, 355), (0.05, 750)):
    check(f"n for 80% power at PVE {pve*100:.0f} %", claim,
          optimize.brentq(lambda n: power(n, pve) - 0.8, 10, 20000), tol=0.03)

# ---------- 5. Hybrid EMMAX ----------
print("\n5. HYBRID EMMAX (λ and significant SNPs)")
z = np.load(NPZ, allow_pickle=True)
nalt = np.where(z["genotypes"].astype(float) < 0, np.nan, z["genotypes"].astype(float))
chrn = np.array([CHRMAP.get(c, -1) for c in z["chrom"].astype(str)])
samples = en_ids(z["samples"].astype(str))
cr = np.mean(~np.isnan(nalt), 1); af = np.nanmean(nalt, 1) / 2
keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (np.minimum(af, 1 - af) >= 0.05)
gt = nalt[keep]; gt = np.where(np.isnan(gt), np.nanmean(gt, 1, keepdims=True), gt)
s2c = {s: i for i, s in enumerate(samples)}
bl = en_table(pd.read_csv(OUT / "222_hybrid_blup_phenotypes.csv"))
bl[["mom_long", "father"]] = bl["hybrid"].str.split("_", expand=True)
bl["mom"] = bl["mom_long"].map(LTV)
d = bl.dropna(subset=["seed_weight_1000"])
d = d[~d["father"].isin(CONF)]
d = d[d["mom"].isin(s2c) & d["father"].isin(s2c)].reset_index(drop=True)
HG = np.empty((len(d), gt.shape[0]))
for i, r in d.iterrows(): HG[i] = (gt[:, s2c[r["mom"]]] + gt[:, s2c[r["father"]]]) / 2
p_al = HG.mean(0) / 2; W = HG - 2 * p_al
K = (W @ W.T) / (2 * np.sum(p_al * (1 - p_al)))
Xm = np.column_stack([np.ones(len(d)), (d["mom"] == "VA761").astype(float),
                      (d["mom"] == "VK934").astype(float)])
Xc = HG - HG.mean(0); Xc = Xc / (HG.std(0) + 1e-9)
U_, S_, _ = np.linalg.svd(Xc, full_matrices=False)
PC3 = U_[:, :3] * S_[:3]
y = d["seed_weight_1000"].values.astype(float)
def emmax_lam(Xf):
    ev, U = np.linalg.eigh(K); ev = np.maximum(ev, 1e-9)
    yr = U.T @ y; Xr = U.T @ Xf
    def nll(t):
        sg, se = np.exp(t); D = sg * ev + se; Wt = 1 / D
        XtWX = Xr.T @ (Xr * Wt[:, None])
        try: inv = np.linalg.inv(XtWX)
        except Exception: return 1e10
        b = inv @ (Xr.T @ (yr * Wt)); r = yr - Xr @ b
        _, ld = np.linalg.slogdet(XtWX)
        return 0.5 * (np.log(D).sum() + (r ** 2 * Wt).sum() + ld)
    vy = np.var(y)
    o = optimize.minimize(nll, np.log([.5 * vy, .5 * vy]), method="Nelder-Mead")
    sg, se = np.exp(o.x); sw = np.sqrt(1 / (sg * ev + se))
    yw = (U.T @ y) * sw; Xw = (U.T @ Xf) * sw[:, None]
    Q, _ = np.linalg.qr(Xw); yt = yw - Q @ (Q.T @ yw)
    Gw = (U.T @ HG) * sw[:, None]; Gt = Gw - Q @ (Q.T @ Gw)
    gg = np.einsum('ij,ij->j', Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    b = (Gt.T @ yt) / gg; df = len(y) - Xf.shape[1] - 1
    sse = (yt @ yt) - b ** 2 * gg; sse = np.where(sse > 1e-12, sse, np.nan)
    t = b / np.sqrt(sse / df / gg)
    p = 2 * stats.t.sf(np.abs(t[np.isfinite(t)]), df)
    return np.median(stats.chi2.isf(p, 1)) / stats.chi2.ppf(0.5, 1), int((p < 5e-8).sum())
for lab, Xf in (("mothers", Xm), ("mothers_plus_3PC", np.column_stack([Xm, PC3]))):
    lam, n5 = emmax_lam(Xf)
    check(f"hybrid EMMAX λ ({lab})", None, lam)
    check(f"significant SNP at 5e-8, hybrids ({lab})", 0, n5, tol=0)

# ---------- 6. GS accuracy ----------
print("\n6. GENOMIC-SELECTION ACCURACY (check against the stored table)")
gs = en_table(pd.read_csv(OUT / "27_gs_accuracy_blup.csv")).set_index("trait")
for tr, claim in (("seed_weight_1000", 0.84), ("oil_content", 0.65), ("oil_yield", 0.42),
                  ("seed_yield", 0.19)):
    check(f"GS accuracy — {tr}", claim, float(gs.loc[tr, "GS_accuracy_CV1"]), tol=0.03)

df = pd.DataFrame(rows)
OUT_NAME = f"verify_reported_numbers_{date.today().isoformat()}.csv"
df.to_csv(HERE / OUT_NAME, index=False, encoding="utf-8-sig")
bad = df[df["matched"] == "NO"]
print(f"\n{'='*70}")
print(f"checks: {len(df)};  matched: {(df['matched']=='YES').sum()};  "
      f"mismatches: {len(bad)};  informational (no claimed value): {(df['matched']=='—').sum()}")
if len(bad):
    print("\nMISMATCHES:")
    print(bad.to_string(index=False))
print(f"\nSaved: {OUT_NAME}")
