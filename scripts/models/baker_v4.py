# -*- coding: utf-8 -*-
"""B4. Baker's ratio from the model with a SEPARATE Year × Line term.

Replaces baker_v3.py / baker_v3_reml.py. Differences are documented in
brain\\BAKER_MODEL_AUDIT.md:
  * the fixed-effects matrix is reduced to full rank (in v3 it was
    singular: Repeat(Year) columns span the Year column);
  * REML is parameterized by ratios gamma_i = sigma2_i / sigma2_e with an
    explicit bound gamma >= 0; sigma2_e is profiled out analytically;
  * the Year × Line term is estimated rather than set to a starting value;
  * sequential sums of squares are obtained by block orthogonalization,
    not by solving the normal equations.

Model (year, replicate, tester and year × tester fixed):
  y = mu + Year + Rep(Year) + Tester + Year × Tester
        + Line + Tester × Line + Year × Line + Year × Tester × Line + eps
Baker's ratio = 2 sigma2(Line) / (2 sigma2(Line) + sigma2(Tester × Line)).

Does not edit anything; only prints and saves components to JSON.
This is the single Baker v4 implementation (the former baker_v4_FIXED.py
copy was identical after the path migration and has been removed).
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
import sys, io, json, time, pathlib
import numpy as np, pandas as pd
import scipy.linalg as sla
from scipy import optimize, stats
import warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

ROOT = DEPOSIT
REPO = OUT / "22_hybrid_plot_level.parquet"
FIVE = INTERMEDIATE / "22_hybrids_reps_long.csv"


# ------------------------------------------------------------------ helpers
def dummies(labels):
    u = sorted(set(labels)); idx = {l: i for i, l in enumerate(u)}
    Z = np.zeros((len(labels), len(u)))
    for i, l in enumerate(labels):
        Z[i, idx[l]] = 1.0
    return Z


def inter(A, B):
    return np.einsum("ij,ik->ijk", A, B).reshape(A.shape[0], -1)


def full_rank(X, tol=1e-9):
    """Maximal linearly independent set of columns (QR with column pivoting)."""
    Q, R, piv = sla.qr(X, mode="economic", pivoting=True)
    d = np.abs(np.diag(R)); p = int(np.sum(d > tol * d[0]))
    return X[:, np.sort(piv[:p])], p


# ------------------------------------------------------------------ REML
class REML:
    """V = sigma2_e * (I + Z Gamma Z'), Gamma = diag(gamma_i) block-wise.

    H^-1 = I - Z W (I + W Z'Z W)^-1 W Z',  W = Gamma^(1/2) — a form defined
    at gamma = 0, so the boundary is attainable and visible.
    log|H| = log|I + W Z'Z W|; sigma2_e is profiled: (n-p)^-1 r' H^-1 r.
    """

    def __init__(self, y, X, Zs, names):
        self.y = np.asarray(y, float); self.names = list(names)
        self.X, self.p = full_rank(np.asarray(X, float))
        self.n = len(self.y)
        self.Zs = Zs; self.Z = np.hstack(Zs); self.sizes = [z.shape[1] for z in Zs]
        self.q = self.Z.shape[1]
        self.ZtZ = self.Z.T @ self.Z
        self.ZtX = self.Z.T @ self.X
        self.Zty = self.Z.T @ self.y
        self.nev = 0

    def _pieces(self, g):
        w = np.sqrt(np.concatenate([np.full(k, max(gi, 0.0))
                                    for k, gi in zip(self.sizes, g)]))
        K = (w[:, None] * self.ZtZ) * w[None, :] + np.eye(self.q)
        c = sla.cho_factor(K, lower=True)
        ldK = 2.0 * np.sum(np.log(np.diag(c[0])))

        def Hi(A, ZtA):
            s = sla.cho_solve(c, (w * ZtA) if ZtA.ndim == 1 else (w[:, None] * ZtA))
            return A - (self.Z * w[None, :]) @ s

        return Hi, ldK

    def negll(self, g):
        self.nev += 1
        g = np.asarray(g, float)
        if np.any(g < 0) or np.any(~np.isfinite(g)):
            return 1e12
        Hi, ldK = self._pieces(g)
        HiX = Hi(self.X, self.ZtX); Hiy = Hi(self.y, self.Zty)
        cx = sla.cho_factor(self.X.T @ HiX, lower=True)
        ldXX = 2.0 * np.sum(np.log(np.diag(cx[0])))
        b = sla.cho_solve(cx, self.X.T @ Hiy)
        r = self.y - self.X @ b
        rHir = float(r @ Hi(r, self.Zty - self.ZtX @ b))
        s2e = rHir / (self.n - self.p)
        return 0.5 * ((self.n - self.p) * np.log(s2e) + ldK + ldXX + (self.n - self.p))

    def sigma2_e(self, g):
        Hi, _ = self._pieces(np.asarray(g, float))
        HiX = Hi(self.X, self.ZtX); Hiy = Hi(self.y, self.Zty)
        cx = sla.cho_factor(self.X.T @ HiX, lower=True)
        b = sla.cho_solve(cx, self.X.T @ Hiy); r = self.y - self.X @ b
        return float(r @ Hi(r, self.Zty - self.ZtX @ b)) / (self.n - self.p)

    def fit(self, starts):
        runs = []; best = None
        bnd = [(0.0, 1e4)] * len(self.sizes)
        for st in starts:
            a = optimize.minimize(self.negll, np.asarray(st, float), method="L-BFGS-B",
                                  bounds=bnd,
                                  options=dict(maxiter=800, ftol=1e-15, gtol=1e-11))
            b = optimize.minimize(self.negll, np.clip(a.x, 0, None), method="Nelder-Mead",
                                  options=dict(maxiter=4000, maxfev=4000,
                                               xatol=1e-11, fatol=1e-11))
            r = a if a.fun <= b.fun else b
            if b.fun < a.fun:
                c = optimize.minimize(self.negll, np.clip(b.x, 0, None), method="L-BFGS-B",
                                      bounds=bnd,
                                      options=dict(maxiter=800, ftol=1e-15, gtol=1e-11))
                r = c if c.fun < b.fun else b
            runs.append(r)
            if best is None or r.fun < best.fun:
                best = r
        g = np.clip(np.asarray(best.x, float), 0, None)
        s2e = self.sigma2_e(g)
        spread = max(r.fun for r in runs) - min(r.fun for r in runs)
        return dict(gamma=g, s2={n: float(gi * s2e) for n, gi in zip(self.names, g)},
                    s2e=s2e, logL=-best.fun, spread=float(spread), runs=runs)

    def hessian(self, g, rel=2e-3):
        g = np.asarray(g, float); k = len(g)
        h = np.maximum(np.abs(g) * rel, 1e-6); H = np.zeros((k, k))
        f0 = self.negll(g)
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k); ei[i] = h[i]
                ej = np.zeros(k); ej[j] = h[j]
                if i == j:
                    H[i, i] = (self.negll(np.clip(g + 2 * ei, 0, None)) - 2 * f0
                               + self.negll(np.clip(g - 2 * ei, 0, None))) / (4 * h[i] ** 2)
                else:
                    a1 = self.negll(np.clip(g + ei + ej, 0, None))
                    a2 = self.negll(np.clip(g + ei - ej, 0, None))
                    a3 = self.negll(np.clip(g - ei + ej, 0, None))
                    a4 = self.negll(np.clip(g - ei - ej, 0, None))
                    H[i, j] = H[j, i] = (a1 - a2 - a3 + a4) / (4 * h[i] * h[j])
        return H


# ------------------------------------------------------------------ moments
def seq_anova(y, terms, tol=1e-9):
    """Sequential (Type I) SS by orthogonalizing each block to the space
    already accumulated. Block rank is the number of QR pivots above the threshold."""
    n = len(y); Q = np.ones((n, 1)) / np.sqrt(n); rows = {}
    for name, M in terms:
        R = M - Q @ (Q.T @ M)
        q_, r_, _ = sla.qr(R, mode="economic", pivoting=True)
        d = np.abs(np.diag(r_))
        k = int(np.sum(d > tol * max(d[0], np.linalg.norm(M))))
        if k > 0:
            Qn = q_[:, :k]
            Qn = Qn - Q @ (Q.T @ Qn)
            Qn, _ = sla.qr(Qn, mode="economic")
            rows[name] = dict(SS=float(np.sum((Qn.T @ y) ** 2)), df=k)
            Q = np.hstack([Q, Qn])
        else:
            rows[name] = dict(SS=0.0, df=0)
    rows["error"] = dict(SS=float(y @ y) - float(np.sum((Q.T @ y) ** 2)),
                          df=n - Q.shape[1])
    for v in rows.values():
        v["MS"] = v["SS"] / v["df"] if v["df"] > 0 else np.nan
    return rows


def build(d):
    y = d["value"].to_numpy(float)
    Y = dummies(d["year"].astype(str).tolist())
    R = dummies((d["year"].astype(str) + "_" + d["replicate"].astype(str)).tolist())
    M = dummies(d["mother"].astype(str).tolist())
    L = dummies(d["line"].astype(str).tolist())
    return y, Y, R, M, L


def moments(d):
    y, Y, R, M, L = build(d)
    ML, YM, YL = inter(M, L), inter(Y, M), inter(Y, L)
    YML = inter(YM, L)
    A = seq_anova(y, [("year", Y), ("rep_in_year", R), ("mother", M), ("year_x_mother", YM),
                      ("line", L), ("mother_x_line", ML), ("year_x_line", YL),
                      ("year_x_mother_x_line", YML)])
    r = float(d.groupby(["year", "mother", "line"]).size().mean())
    ny, nm = float(d["year"].nunique()), float(d["mother"].nunique())
    ms = {k: A[k]["MS"] for k in A}; df = {k: A[k]["df"] for k in A}
    e, yml, yl, ml, l = (ms["error"], ms["year_x_mother_x_line"], ms["year_x_line"],
                         ms["mother_x_line"], ms["line"])
    s2 = dict(e=e,
              yml=max(0.0, (yml - e) / r),
              yl=max(0.0, (yl - yml) / (r * nm)),
              ml=max(0.0, (ml - yml) / (r * ny)),
              l=max(0.0, (l - ml - yl + yml) / (r * ny * nm)))
    gy_ss = A["year_x_line"]["SS"] + A["year_x_mother_x_line"]["SS"]
    gy_df = df["year_x_line"] + df["year_x_mother_x_line"]
    gy = gy_ss / gy_df
    old = dict(gy=max(0.0, (gy - e) / r), ml=max(0.0, (ml - gy) / (r * ny)),
               l=max(0.0, (l - ml) / (r * ny * nm)))
    vm = lambda k, m_: 2 * m_ ** 2 / df[k]
    se_ml = np.sqrt(vm("mother_x_line", ml) + vm("year_x_mother_x_line", yml)) / (r * ny)
    se_l = np.sqrt(vm("line", l) + vm("mother_x_line", ml) + vm("year_x_line", yl)
                   + vm("year_x_mother_x_line", yml)) / (r * ny * nm)
    se_yl = np.sqrt(vm("year_x_line", yl) + vm("year_x_mother_x_line", yml)) / (r * nm)
    bak = 2 * s2["l"] / (2 * s2["l"] + s2["ml"]) if s2["l"] + s2["ml"] > 0 else np.nan
    bak_old = (2 * old["l"] / (2 * old["l"] + old["ml"])
               if old["l"] + old["ml"] > 0 else np.nan)
    return dict(anova=A, r=r, s2=s2, old=old, baker=bak, baker_old=bak_old,
                se=dict(l=se_l, ml=se_ml, yl=se_yl), ms=ms, df=df)


def reml_fit(d, with_yl=True, hess=True):
    y, Y, R, M, L = build(d)
    yr = d["year"].astype(str); mo = d["mother"].astype(str); fa = d["line"].astype(str)
    X = np.hstack([np.ones((len(y), 1)), M, R, inter(Y, M)])
    Zl = dummies(fa.tolist())
    Zml = dummies((mo + "_" + fa).tolist())
    Zyl = dummies((yr + "_" + fa).tolist())
    Zyml = dummies((yr + "_" + mo + "_" + fa).tolist())
    if with_yl:
        Zs = [Zl, Zml, Zyl, Zyml]
        names = ["line", "tester_x_line", "year_x_line", "year_x_tester_x_line"]
        starts = [[.2, .05, .3, .3], [.05, .05, .05, .05], [1., .2, 1., 1.],
                  [.4, .01, .4, .05], [.2, .05, 1e-9, .5]]
    else:
        Zs = [Zl, Zml, Zyml]
        names = ["line", "tester_x_line", "year_x_tester_x_line"]
        starts = [[.2, .05, .3], [.05, .05, .05], [1., .2, 1.], [.4, .01, .05]]
    m = REML(y, X, Zs, names)
    f = m.fit(starts)
    s2 = f["s2"]; den = 2 * s2["line"] + s2["tester_x_line"]
    f["baker"] = 2 * s2["line"] / den if den > 0 else np.nan
    f["p"] = m.p; f["q"] = m.q; f["n"] = m.n; f["names"] = names
    if hess:
        H = m.hessian(f["gamma"])
        try:
            C = np.linalg.inv(H)
            f["se_gamma"] = np.sqrt(np.clip(np.diag(C), 0, None))
            f["se_s2"] = f["se_gamma"] * f["s2e"]
            a, b = f["gamma"][0], f["gamma"][1]; dd = 2 * a + b
            J = np.zeros(len(f["gamma"]))
            J[0] = 2 * b / dd ** 2; J[1] = -2 * a / dd ** 2
            f["se_baker"] = float(np.sqrt(J @ C @ J))
            f["hess_eig"] = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            f["se_gamma"] = np.full(len(f["gamma"]), np.nan)
            f["se_s2"] = f["se_gamma"]
            f["se_baker"] = np.nan
            f["hess_eig"] = np.array([np.nan])
    return f


# ------------------------------------------------------------------ data
def load_repo():
    p = en_table(pd.read_parquet(REPO)).rename(columns={
        "mother": "mother", "father": "line", "year": "year", "rep": "replicate",
        "trait": "trait", "value": "value"})
    return p.dropna(subset=["value"])


def load_five():
    rp = en_table(pd.read_csv(FIVE)).dropna(subset=["value"])
    return rp.groupby(["trait", "year", "mother", "line", "replicate"],
                      as_index=False).value.mean()


# ------------------------------------------------------------------ run
def run(d, label, tag):
    t0 = time.time()
    mo = moments(d)
    rf = reml_fit(d, with_yl=True)
    rn = reml_fit(d, with_yl=False, hess=False)
    lr = 2 * (rf["logL"] - rn["logL"])
    pv = 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0
    baker_no = (2 * rn["s2"]["line"]
                / (2 * rn["s2"]["line"] + rn["s2"]["tester_x_line"]))
    A = mo["anova"]
    print("\n" + "=" * 94)
    print(f"### {label}   n={len(d)}  lines={d.line.nunique()}  "
          f"combinations={d.groupby(['mother','line']).ngroups}  r={mo['r']:.3f}  "
          f"({time.time()-t0:.0f} s)")
    print(f"  rank of X = {rf['p']}, random-effect levels q = {rf['q']}")
    print("  sequential decomposition:")
    for k in ["year", "rep_in_year", "mother", "year_x_mother", "line", "mother_x_line",
              "year_x_line", "year_x_mother_x_line", "error"]:
        print(f"      {k:16s} SS={A[k]['SS']:14.4f}  df={A[k]['df']:4d}  "
              f"MS={A[k]['MS']:12.5f}")
    print("  MOMENTS, Year×Line term separated:")
    print(f"      s2 Line           = {mo['s2']['l']:12.6f} +- {mo['se']['l']:.6f}")
    print(f"      s2 Tester×Line    = {mo['s2']['ml']:12.6f} +- {mo['se']['ml']:.6f}"
          f"   (vs own SE {mo['s2']['ml']/mo['se']['ml']:.2f})")
    print(f"      s2 Year×Line      = {mo['s2']['yl']:12.6f} +- {mo['se']['yl']:.6f}")
    print(f"      s2 Year×Test×Line = {mo['s2']['yml']:12.6f}")
    print(f"      s2 error          = {mo['s2']['e']:12.6f}")
    print(f"      Baker's ratio = {mo['baker']:.4f}   |  same raw data WITHOUT the G×L term: "
          f"{mo['baker_old']:.4f}")
    print("  REML, Year×Line term separated:")
    for nm_, gi, sei in zip(rf["names"], rf["gamma"], rf["se_s2"]):
        flag = "   <-- BOUNDARY 0" if gi <= 1e-9 else ""
        print(f"      s2 {nm_:16s} = {rf['s2'][nm_]:12.6f} +- {sei:.6f}"
              f"   (gamma={gi:.6e}){flag}")
    print(f"      s2 error           = {rf['s2e']:12.6f}")
    print(f"      Baker's ratio = {rf['baker']:.4f} +- {rf['se_baker']:.4f}    "
          f"non-additive share {1-rf['baker']:.4f}")
    print(f"      s2 SCA vs own SE = "
          f"{rf['s2']['tester_x_line']/rf['se_s2'][1]:.2f}")
    print(f"      logL={rf['logL']:.6f}, spread across {len(rf['runs'])} starts "
          f"{rf['spread']:.2e}, Hessian eigenvalues "
          f"{np.array2string(rf['hess_eig'], precision=3)}")
    print(f"      REML WITHOUT the G×L term: Baker's ratio = {baker_no:.4f}, logL={rn['logL']:.6f}")
    print(f"  LR for the Year×Line term = {lr:.4f}, p = {pv:.4g}  (chi-square mixture 0:1)")
    return dict(tag=tag, label=label, n=len(d), r=mo["r"],
                anova={k: dict(SS=A[k]["SS"], df=A[k]["df"], MS=A[k]["MS"]) for k in A},
                mom=dict(s2=mo["s2"], se=mo["se"], baker=mo["baker"],
                         baker_old=mo["baker_old"], old=mo["old"]),
                reml=dict(s2=rf["s2"], s2e=rf["s2e"],
                          se_s2=list(map(float, rf["se_s2"])),
                          baker=float(rf["baker"]), se_baker=rf["se_baker"],
                          gamma=list(map(float, rf["gamma"])), logL=rf["logL"],
                          spread=rf["spread"]),
                reml_no=dict(s2=rn["s2"], s2e=rn["s2e"], logL=rn["logL"],
                             baker=float(baker_no)),
                LR=float(lr), p=float(pv))


if __name__ == "__main__":
    out = []
    repo = load_repo()
    print("#" * 94)
    print("# PART 1. THREE TRAITS OF PAPER 1 — repository plot-level data, 54 lines / 162 combinations")
    print("#" * 94)
    for key, name in [("seed_yield", "seed_yield"), ("oil_content", "oil_content"),
                      ("seed_weight_1000", "seed_weight_1000")]:
        out.append(run(repo[repo.trait == key].reset_index(drop=True),
                       f"{name} (repository)", f"repo|{key}"))
    five = load_five()
    print("\n" + "#" * 94)
    print("# PART 2. FIVE TRAITS — five-trait plot-level data, 54 lines / 162 combinations")
    print("#" * 94)
    for key in ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter"]:
        out.append(run(five[five.trait == key].reset_index(drop=True),
                       f"{key} (five-trait plot-level data)", f"five|{key}"))
    (ROOT / "brain" / "baker_v4_components.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\ncomponents saved to brain\\baker_v4_components.json")
