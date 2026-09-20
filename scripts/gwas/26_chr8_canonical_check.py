"""
Status check of chr8:782 740 on the CANONICAL panel and in the declared model.

The marker was previously uncheckable: in var2 its missingness exceeded 90 %.
In merged it is fully genotyped, so it can now be evaluated honestly.

Computed: p and genome-wide rank under EMMAX and BLINK at 54 and 52 lines,
plus the overlap of the panel from the separate run by co-author Victoria
Voronezhskaya (VNIISB) in GAPIT v3 with the canonical panel.
Output: 26_chr8_canonical_check.csv
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
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats, optimize
import warnings
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
sys.path.insert(0, str(ROOT / "phase8_blink_python"))
from blink import blink

HERE = OUT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
GAPIT = ROOT / "7_ручной обсчет воронежская" / "no_sugar" / "no_sugar" / "gapit" / "input" / "gapit_genotypes.txt"
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
CONF = ["LI29", "LI30"]
TARGET = (8, 782740)

z = np.load(NPZ, allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str)
pos = z["pos"]; samples = en_ids(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc)
chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
ALLF = [s for s in samples if is_line_id(s)]
ph = en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
mean_ph = ph.groupby(["genotype", "trait"])["value"].mean().unstack()

def emmax(y, Gm, Kk, PC):
    m = ~np.isnan(y); yv = y[m]; Ks = Kk[np.ix_(m, m)]; Gs = Gm[:, m]
    ev, U = np.linalg.eigh(Ks); ev = np.maximum(ev, 1e-9)
    X = np.column_stack([np.ones(len(yv)), PC[m]])
    yr = U.T @ yv; Xr = U.T @ X
    def nll(t):
        sg, se = np.exp(t); D = sg*ev + se; W = 1/D
        XtWX = Xr.T @ (Xr*W[:, None])
        try: inv = np.linalg.inv(XtWX)
        except Exception: return 1e10
        b = inv @ (Xr.T @ (yr*W)); r = yr - Xr @ b
        _, ld = np.linalg.slogdet(XtWX)
        return 0.5*(np.log(D).sum() + (r**2*W).sum() + ld)
    vy = np.var(yv)
    o = optimize.minimize(nll, np.log([.5*vy, .5*vy]), method="Nelder-Mead")
    sg, se = np.exp(o.x); sw = np.sqrt(1/(sg*ev + se))
    yw = (U.T@yv)*sw; Xw = (U.T@X)*sw[:, None]
    Q, _ = np.linalg.qr(Xw); yt = yw - Q@(Q.T@yw)
    Gw = (U.T@Gs.T)*sw[:, None]; Gt = Gw - Q@(Q.T@Gw)
    gg = np.einsum('ij,ij->j', Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    b = (Gt.T@yt)/gg; df = len(yv) - X.shape[1] - 1
    sse = (yt@yt) - b**2*gg; sse = np.where(sse > 1e-12, sse, np.nan)
    t = b/np.sqrt(sse/df/gg)
    p = np.full(Gm.shape[0], np.nan); ok = np.isfinite(t)
    p[ok] = 2*stats.t.sf(np.abs(t[ok]), df)
    return p, b

rows = []
for tag, lines in (("54_with_confectionery", ALLF),
                   ("52_oilseed_only", [f for f in ALLF if f not in CONF])):
    col = [samples.index(l) for l in lines]
    G = nalt[:, col]
    cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1)/2
    maf = np.minimum(af, 1-af)
    keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
    Gi = G[keep]; Gi = np.where(np.isnan(Gi), np.nanmean(Gi, axis=1, keepdims=True), Gi)
    chf = chrn[keep].astype(int); pf = pos[keep]
    p_al = Gi.mean(1)/2; W = Gi - 2*p_al[:, None]
    K = (W.T@W)/(2*np.sum(p_al*(1-p_al)))
    Xc = Gi.T - Gi.T.mean(0); Xc = Xc/(Gi.T.std(0) + 1e-9)
    Us, Ss, _ = np.linalg.svd(Xc, full_matrices=False)
    PC3 = Us[:, :3]*Ss[:3]
    idx = np.where((chf == TARGET[0]) & (pf == TARGET[1]))[0]
    print(f"\n### {tag}: {Gi.shape[0]:,} SNP")
    if not len(idx):
        print("  chr8:782740 is absent from the panel"); continue
    i = idx[0]
    raw = G[keep][i]
    print(f"  chr8:782740 — call rate {np.mean(~np.isnan(raw)):.3f}, "
          f"MAF {min(np.nanmean(raw)/2, 1-np.nanmean(raw)/2):.3f}")
    y = mean_ph["oil_content"].reindex(lines).values.astype(float)
    pe, be = emmax(y, Gi, K, PC3)
    vi = np.isfinite(pe)
    rank_e = int((pe[vi] < pe[i]).sum()) + 1
    res = blink(y, Gi, chf, pf, PCs=PC3)
    pb = res["pvals"]
    vb = np.isfinite(pb)
    rank_b = int((pb[vb] < pb[i]).sum()) + 1
    bonf = 0.05/int(vi.sum())
    print(f"  EMMAX p = {pe[i]:.3e}  rank {rank_e} of {int(vi.sum()):,}  beta = {be[i]:+.2f}")
    print(f"  BLINK p = {pb[i]:.3e}  rank {rank_b} of {int(vb.sum()):,}")
    print(f"  Bonferroni threshold = {bonf:.2e};  significant at 5e-8: {bool(pe[i] < 5e-8)}")
    rows.append({"panel": tag, "n_SNP": Gi.shape[0], "EMMAX_p": pe[i], "EMMAX_rank": rank_e,
                 "beta": be[i], "BLINK_p": pb[i], "BLINK_rank": rank_b, "Bonferroni": bonf})

print("\n=== OVERLAP OF THE SEPARATE-RUN PANEL (co-author V. Voronezhskaya, VNIISB, GAPIT v3) WITH THE CANONICAL PANEL ===")
gp = en_table(pd.read_csv(GAPIT, sep="\t", usecols=["chrom", "pos"])).dropna()
their = set(zip(gp["chrom"].astype(int), gp["pos"].astype(int)))
for tag, lines in (("54_with_confectionery", ALLF),
                   ("52_oilseed_only", [f for f in ALLF if f not in CONF])):
    col = [samples.index(l) for l in lines]
    G = nalt[:, col]
    cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1)/2
    maf = np.minimum(af, 1-af)
    keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
    panel = set(zip(chrn[keep], pos[keep]))
    ov = their & panel
    print(f"  {tag}: {len(panel):,} SNP; overlap {len(ov):,} of {len(their):,} "
          f"({len(ov)/len(their)*100:.1f} %)")

pd.DataFrame(rows).to_csv(HERE / "26_chr8_canonical_check.csv", index=False, encoding="utf-8-sig")
print("\nWritten: 26_chr8_canonical_check.csv")
