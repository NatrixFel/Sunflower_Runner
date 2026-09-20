# -*- coding: utf-8 -*-
"""H1. Heritability and repeatability on the D020 specification.

The former H² = 0.93 / 0.96 / 0.68 and plot repeatability 0.26 were taken from
`22_variance_components_h2.csv` and obtained from the model

    y = mu + Year + Replicate(Year) + Genotype(random) + eps,

where “Genotype” is the mother × line combination, and the genotype × year
interaction is NOT a separate term and goes entirely into the residual. Decision D020
judged this specification incorrect: the Year × Tester term is significant (F = 35.5),
and the year interaction must stand on its own.

The script does three things:
  1) reproduces the old model, to confirm it is understood correctly;
  2) computes the same quantities on the adopted specification;
  3) prints all prediction ceilings side by side, with an explicit statement of
     what quantity and at which aggregation level.

Does not edit anything.
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
import sys, io, json, pathlib
import numpy as np, pandas as pd
from baker_v4 import REML, dummies, inter, build, moments, load_repo, ROOT
import warnings; warnings.filterwarnings("ignore")

TR = [("seed_yield", "seed_yield"), ("oil_content", "oil_content"),
      ("seed_weight_1000", "seed_weight_1000")]
OLD = OUT / "22_variance_components_h2.csv"


# ---------------------------------------------------------------- old model
def naive(d):
    """y ~ Year + Rep(Year) fixed + Genotype(combination) random + eps."""
    y, Y, R, M, L = build(d)
    X = np.hstack([np.ones((len(y), 1)), Y, R])
    g = (d["mother"].astype(str) + "_" + d["line"].astype(str)).tolist()
    Zg = dummies(g)
    m = REML(y, X, [Zg], ["genotype"])
    f = m.fit([[0.3], [0.05], [1.0], [3.0]])
    s2g = f["s2"]["genotype"]; s2e = f["s2e"]
    n_bar = len(y) / len(set(g))                 # mean number of plots per genotype
    return dict(s2g=s2g, s2e=s2e, n_bar=n_bar,
                rep_plot=s2g / (s2g + s2e),
                H2_mean=s2g / (s2g + s2e / n_bar),
                n=len(y), ng=len(set(g)), spread=f["spread"])


# ---------------------------------------------------------------- adopted model
def accepted(d):
    """D020 specification: year, rep(year), tester and year × tester fixed;
    line, tester × line, year × line, year × tester × line random."""
    y, Y, R, M, L = build(d)
    yr = d["year"].astype(str); mo = d["mother"].astype(str); fa = d["line"].astype(str)
    X = np.hstack([np.ones((len(y), 1)), M, R, inter(Y, M)])
    Zs = [dummies(fa.tolist()), dummies((mo + "_" + fa).tolist()),
          dummies((yr + "_" + fa).tolist()), dummies((yr + "_" + mo + "_" + fa).tolist())]
    names = ["line", "tester_x_line", "year_x_line", "year_x_tester_x_line"]
    m = REML(y, X, Zs, names)
    f = m.fit([[.2, .05, .3, .3], [.05, .05, .05, .05], [1., .2, 1., 1.], [.4, .01, .4, .05]])
    s = f["s2"]; e = f["s2e"]
    L_, ML, YL, YTL = (s["line"], s["tester_x_line"], s["year_x_line"], s["year_x_tester_x_line"])
    ny = float(d["year"].nunique()); nm = float(d["mother"].nunique())
    r = float(d.groupby(["year", "mother", "line"]).size().mean())

    # genotype = tester × line combination; the tester main-effect contribution
    # is not included: tester is declared fixed
    G = L_ + ML
    GY = YL + YTL

    var_plot = G + GY + e                                  # variance of a single plot
    var_mean = G + GY / ny + e / (ny * r)                   # variance of the combination mean
    var_gca  = L_ + ML / nm + YL / ny + YTL / (nm * ny) + e / (nm * ny * r)  # variance of the GCA estimate

    return dict(s2=s, s2e=e, r=r, ny=ny, nm=nm, spread=f["spread"],
                G=G, GY=GY,
                rep_plot=G / var_plot,
                H2_mean=G / var_mean,
                var_plot=var_plot, var_mean=var_mean, var_gca=var_gca,
                # heritability of the line GCA estimate: share of sigma2(Line) in its variance
                h2_gca=L_ / var_gca,
                # additive-model ceiling at the hybrid level
                add_share_hybrid=L_ / var_mean,
                # ceiling for the full genotypic value at the hybrid level
                broad_share_hybrid=G / var_mean)


if __name__ == "__main__":
    old_csv = en_table(pd.read_csv(OLD, encoding="utf-8-sig")).set_index("trait")
    out = {}
    print("#" * 96)
    print("# H1. HERITABILITY AND REPEATABILITY: OLD MODEL VS ADOPTED (D020)")
    print("#" * 96)
    repo = load_repo()
    for key, name in TR:
        d = repo[repo.trait == key].reset_index(drop=True)
        a = naive(d)
        b = accepted(d)
        o = old_csv.loc[key]
        print(f"\n{'='*96}\n### {name}   n = {a['n']} plots, {a['ng']} combinations, "
              f"plots per combination (mean) {a['n_bar']:.3f}")
        print("\n  -- OLD model (Genotype random, year interaction in the residual) --")
        print(f"     reproduction: σ²g = {a['s2g']:.4f} (in file {o['σ²g']}), "
              f"σ²e = {a['s2e']:.4f} (in file {o['σ²e']})")
        print(f"     plot repeatability = {a['rep_plot']:.4f} (in file {o['plot_repeatability']})")
        print(f"     H² of the mean          = {a['H2_mean']:.4f} (in file {o['H2_of_mean']})")
        print(f"     √H² = {np.sqrt(a['H2_mean']):.3f}   spread across starts {a['spread']:.1e}")
        print("\n  -- ADOPTED D020 model --")
        for k, v in b["s2"].items():
            print(f"     σ²{k:18s} = {v:12.6f}")
        print(f"     σ²{'error':18s} = {b['s2e']:12.6f}")
        print(f"     σ²(genotype)  = σ²Line + σ²Tester×Line      = {b['G']:.6f}")
        print(f"     σ²(genotype × year) = σ²G×L + σ²G×T×L         = {b['GY']:.6f}")
        print(f"     plot variance              = {b['var_plot']:.6f}")
        print(f"     combination-mean variance (2 years × {b['r']:.2f} reps) = {b['var_mean']:.6f}")
        print(f"     line GCA-estimate variance (3 testers × 2 years × {b['r']:.2f}) = {b['var_gca']:.6f}")
        print(f"     PLOT REPEATABILITY         = {b['rep_plot']:.4f}   (was {o['plot_repeatability']})")
        print(f"     H² OF THE COMBINATION MEAN = {b['H2_mean']:.4f}   (was {o['H2_of_mean']}), "
              f"√H² = {np.sqrt(b['H2_mean']):.3f}")
        print(f"     h² OF THE LINE GCA ESTIMATE = {b['h2_gca']:.4f}, √ = {np.sqrt(b['h2_gca']):.3f}")
        print("\n  -- prediction ceilings, all three side by side --")
        print(f"     additive model, hybrid mean:  share of σ²Line = "
              f"{b['add_share_hybrid']:.4f}  ->  r ≤ {np.sqrt(b['add_share_hybrid']):.3f}")
        print(f"     full genotypic value, hybrid mean: share of σ²G = "
              f"{b['broad_share_hybrid']:.4f}  ->  r ≤ {np.sqrt(b['broad_share_hybrid']):.3f}")
        print(f"     additive model, line GCA:        share of σ²Line = "
              f"{b['h2_gca']:.4f}  ->  r ≤ {np.sqrt(b['h2_gca']):.3f}")
        out[key] = dict(name=name, old=a, new={k: (v if not isinstance(v, dict) else
                                                   {kk: float(vv) for kk, vv in v.items()})
                                               for k, v in b.items()},
                        file=dict(s2g=float(o["σ²g"]), s2e=float(o["σ²e"]),
                                  rep=float(o["plot_repeatability"]),
                                  H2=float(o["H2_of_mean"])))
    (ROOT / "brain" / "22_h2_recalc.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\nsaved brain\\22_h2_recalc.json")
