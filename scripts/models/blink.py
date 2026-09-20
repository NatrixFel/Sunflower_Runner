"""
True BLINK in Python (Huang et al. 2019, GigaScience), a faithful implementation.

BLINK replaces the random kinship effect (as in MLM/EMMAX) with a set of PSEUDO-QTNs
included as fixed covariates. Two steps alternate until convergence:

  FEM (Fixed Effect Model): test each marker one at a time; covariates =
       intercept + PCs + current pseudo-QTNs.
  Pseudo-QTN selection: markers are ranked by FEM p-values; an LD filter drops
       markers in strong LD (r²>threshold) with already more significant ones;
       then BIC chooses the optimal number of pseudo-QTNs among the candidates.

Final p-values come from the last FEM. For a marker that is itself a
pseudo-QTN, the test leaves that marker out of the covariates (leave-one-out).
"""
from __future__ import annotations

import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import DEPOSIT, WORK, CALLING, OUT, INTERMEDIATE, MODELS, use_models, work, calling, en_table, en_ids, en_id, is_line_id
use_models()
import numpy as np
from scipy import stats


def _residualize(CV: np.ndarray):
    """Return the projector Q = I - CV (CV'CV)^- CV' and the number of CV columns."""
    # pseudoinverse — robust to collinearity
    CVpinv = np.linalg.pinv(CV)
    Q = np.eye(CV.shape[0]) - CV @ CVpinv
    rank = np.linalg.matrix_rank(CV)
    return Q, rank


def fem_scan(y: np.ndarray, M: np.ndarray, CV: np.ndarray,
             skip: np.ndarray | None = None):
    """
    Test each marker: y ~ CV + marker. Vectorized t-test.
    y: (n,), M: (m, n) ALT dosage (imputed), CV: (n, c) covariates including intercept.
    skip: boolean (m,) — markers to skip (NaN).
    Returns pvals, betas (m,).
    """
    n = len(y)
    Q, rank = _residualize(CV)
    ry = Q @ y
    RM = M @ Q.T            # (m, n): residuals of each marker after CV
    num = RM @ ry           # (m,)
    den = np.einsum("ij,ij->i", RM, RM)   # (m,)
    df = n - rank - 1
    yry = ry @ ry
    pvals = np.full(M.shape[0], np.nan)
    betas = np.full(M.shape[0], np.nan)
    good = den > 1e-8
    if skip is not None:
        good &= ~skip
    if df <= 0:
        return pvals, betas
    beta = num[good] / den[good]
    rss = yry - beta * num[good]
    rss = np.maximum(rss, 1e-12)
    mse = rss / df
    se = np.sqrt(mse / den[good])
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    pvals[good] = p
    betas[good] = beta
    return pvals, betas


def _bic_select(y: np.ndarray, CV0: np.ndarray, M: np.ndarray,
                candidates: list[int], max_qtn: int):
    """Among ordered candidates, choose the number of pseudo-QTNs by min BIC."""
    n = len(y)
    best_bic, best_k = np.inf, 0
    for k in range(0, min(max_qtn, len(candidates)) + 1):
        if k == 0:
            X = CV0
        else:
            X = np.column_stack([CV0, M[candidates[:k]].T])
        # OLS rss
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rss = float(resid @ resid)
        rss = max(rss, 1e-12)
        params = X.shape[1]
        bic = n * np.log(rss / n) + params * np.log(n)
        if bic < best_bic - 1e-9:
            best_bic, best_k = bic, k
    return candidates[:best_k]


def _ld_prune_candidates(M: np.ndarray, order: np.ndarray,
                         ld_thresh: float, max_consider: int, max_qtn: int):
    """Walk markers in significance order, keeping those not in strong LD with ones already taken."""
    chosen: list[int] = []
    Mc = M - M.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.einsum("ij,ij->i", Mc, Mc))
    for idx in order[:max_consider]:
        if norms[idx] < 1e-8:
            continue
        ok = True
        for c in chosen:
            denom = norms[idx] * norms[c]
            if denom < 1e-12:
                continue
            r = (Mc[idx] @ Mc[c]) / denom
            if r * r > ld_thresh:
                ok = False
                break
        if ok:
            chosen.append(int(idx))
        if len(chosen) >= max_qtn:
            break
    return chosen


def blink(y: np.ndarray, M: np.ndarray, chrom: np.ndarray, pos: np.ndarray,
          PCs: np.ndarray | None = None, ld_thresh: float = 0.7,
          max_qtn: int = 10, max_consider: int = 200, max_iter: int = 6,
          verbose: bool = False):
    """
    BLINK GWAS. y(n,), M(m,n) imputed ALT dosage.
    PCs(n, p) — fixed structure covariates (or None).
    Returns dict: pvals, betas, pseudo_qtn (list of idx), n_iter.
    """
    mask = ~np.isnan(y)
    y = y[mask]
    M = M[:, mask]
    n = len(y)
    intercept = np.ones((n, 1))
    if PCs is not None and PCs.shape[1] > 0:
        CV0 = np.column_stack([intercept, PCs[mask]])
    else:
        CV0 = intercept

    pseudo: list[int] = []
    for it in range(max_iter):
        # FEM with current pseudo-QTNs
        if pseudo:
            CV = np.column_stack([CV0, M[pseudo].T])
        else:
            CV = CV0
        pvals, _ = fem_scan(y, M, CV)
        order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
        cand = _ld_prune_candidates(M, order, ld_thresh, max_consider, max_qtn)
        new_pseudo = _bic_select(y, CV0, M, cand, max_qtn)
        if verbose:
            print(f"    iter {it+1}: |pseudo|={len(new_pseudo)} "
                  f"top_p={np.nanmin(pvals):.2e}")
        if set(new_pseudo) == set(pseudo):
            pseudo = new_pseudo
            break
        pseudo = new_pseudo

    # Final FEM: joint scan with pseudo-QTNs as covariates
    if pseudo:
        CV = np.column_stack([CV0, M[pseudo].T])
    else:
        CV = CV0
    pvals, betas = fem_scan(y, M, CV)
    # leave-one-out for the pseudo-QTNs themselves
    for j in pseudo:
        others = [q for q in pseudo if q != j]
        CVj = np.column_stack([CV0, M[others].T]) if others else CV0
        pj, bj = fem_scan(y, M[j:j+1], CVj)
        pvals[j], betas[j] = pj[0], bj[0]

    valid = ~np.isnan(pvals)
    if valid.sum() > 0:
        chi2 = stats.chi2.isf(pvals[valid], 1)
        lam = np.median(chi2) / stats.chi2.ppf(0.5, 1)
    else:
        lam = np.nan
    return {"pvals": pvals, "betas": betas, "pseudo_qtn": pseudo,
            "n_iter": it + 1, "lambda": float(lam),
            "mask": mask}


# ======================= SELF-TEST (simulation) =======================
if __name__ == "__main__":
    from pathlib import Path
    ROOT = DEPOSIT
    d = np.load(DEPOSIT / "intermediate_outputs" / "50_matrix_54.npz", allow_pickle=True)
    G = d["G"]; chrom = d["chr"]; pos = d["pos"]
    # mean imputation
    Gi = np.where(np.isnan(G), np.nanmean(G, axis=1, keepdims=True), G)
    n = Gi.shape[1]; m = Gi.shape[0]
    rng = np.random.RandomState(42)

    print("=== BLINK SELF-TEST: plant a known causal SNP ===")
    print(f"matrix: {m} SNP × {n} lines")
    n_ok = 0
    TRIALS = 10
    for t in range(TRIALS):
        causal = rng.randint(0, m)
        g = Gi[causal]
        if np.std(g) < 1e-6:
            continue
        gz = (g - g.mean()) / g.std()
        # phenotype: strong causal-SNP effect + polygenic noise + error
        poly = Gi[rng.choice(m, 30, replace=False)].mean(axis=0)
        poly = (poly - poly.mean()) / (poly.std() + 1e-9)
        y = 2.0 * gz + 0.5 * poly + rng.normal(0, 1.0, n)
        res = blink(y, Gi, chrom, pos, PCs=None, verbose=False)
        order = np.argsort(np.where(np.isnan(res["pvals"]), np.inf, res["pvals"]))
        top5 = order[:5]
        # causal or a marker in LD ±1 Mb on the same chromosome
        hit = (causal in top5) or any(
            (chrom[i] == chrom[causal]) and (abs(pos[i] - pos[causal]) < 1_000_000)
            for i in top5)
        n_ok += hit
        rank_causal = int(np.where(order == causal)[0][0]) + 1
        print(f"  trial {t+1}: causal chr{chrom[causal]}:{pos[causal]} "
              f"rank={rank_causal} p={res['pvals'][causal]:.2e} "
              f"|pseudo|={len(res['pseudo_qtn'])} {'✓' if hit else '✗'}")
    print(f"\nSelf-test result: {n_ok}/{TRIALS} causal SNPs in the top 5 (or LD window)")
    print("BLINK is working correctly." if n_ok >= 8 else "WARNING: low detection — check the implementation.")
