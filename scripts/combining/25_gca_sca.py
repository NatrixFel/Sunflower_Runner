"""
Phase 3c: line × tester analysis of VNIIMK hybrids.

Design: 3 mothers (testers) × 54 fathers (lines) × 2 years; 159 unique pairs.
Input: 01_hybrids_tidy.parquet + 01_lines_tidy.parquet (for paternal line phenotypes).

What is computed:
1. Paternal GCA (general combining ability) — 54 × 6 traits.
2. Maternal GCA — 3 × 6 (statistically weak, for reference only).
3. SCA (specific) — 159 × 6 traits.
4. Variance components GCA/SCA, Baker's ratio (additive share).
5. Composite selection index — best fathers by the trait combination.
6. Top-10 hybrids: by seed yield, by oil yield (yield × oil content).
7. Pseudo-heterosis: mean hybrid of LI_i minus LI_i itself as a line.
8. CSV + figures + HTML.

Methodological caveats:
- 2 years instead of the classical r ≥ 3–4 replicates — variance components have
  large uncertainty; interpretation is qualitative.
- 3 mothers — the σ²(GCA_testers) estimate is noisy.
- F and p from ANOVA on means are inflated — we report effect sizes, not significance.
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

# %%
import base64
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid", context="notebook")

ROOT = work()  # undeposited working tree
IN  = ROOT / "phase1_output"
OUT = ROOT / "phase3c_output"
OUT.mkdir(exist_ok=True)

CONFECTIONERY_LI = {"LI29", "LI30"}
MOTHER_MAP = {"M1": "VK101A", "M2": "VA761A", "M3": "VK934A"}

hybrids = en_table(pd.read_parquet(IN / "01_hybrids_tidy.parquet"))
lines   = en_table(pd.read_parquet(IN / "01_lines_tidy.parquet"))

# Hybrid traits (those available)
HYB_TRAITS = ["seed_yield", "oil_content", "seed_weight_1000",
              "plant_height", "head_diameter", "days_emergence_flowering"]
# Matching line traits (for heterosis)
HYB_TO_LIN = {
    "plant_height":          "plant_height",
    "head_diameter":         "head_diameter",
    "oil_content":     "oil_content",
    "seed_weight_1000":  "seed_weight_1000",
    "days_emergence_flowering": "days_emergence_flowering",
}
# Which traits are “more = better” (for the selection index)
DIRECTION = {
    "seed_yield":     +1,
    "oil_content":     +1,
    "seed_weight_1000":  +1,   # larger seed is usually better
    "plant_height":           0,   # neutral
    "head_diameter":         +1,   # larger head — more seed
    "days_emergence_flowering": -1,   # earlier = better (for short-season zones)
}


# %% 1. GCA and SCA — simple Kempthorne model on means

def gca_sca_per_trait(df_hyb: pd.DataFrame, trait: str) -> dict:
    sub = df_hyb[df_hyb["trait"] == trait].dropna(subset=["value"]).copy()
    # Mean over 2 years for each mother×father pair
    pair = sub.groupby(["mother", "father"])["value"].mean().reset_index().rename(columns={"value": "Y"})
    pair["mother_long"] = pair["mother"].map(MOTHER_MAP)
    mu = pair["Y"].mean()
    gca_f = pair.groupby("father")["Y"].mean() - mu        # paternal GCA
    gca_m = pair.groupby("mother")["Y"].mean() - mu        # maternal GCA
    pair["SCA"] = pair.apply(
        lambda r: r["Y"] - mu - gca_f[r["father"]] - gca_m[r["mother"]], axis=1
    )

    # ANOVA for variance components (on yearly data, no averaging)
    df_year = sub.copy()
    df_year["year"] = df_year["year"].astype("category")
    df_year["mother"] = df_year["mother"].astype("category")
    df_year["father"] = df_year["father"].astype("category")
    try:
        model = smf.ols("value ~ C(year) + C(mother) + C(father) + C(mother):C(father)",
                        data=df_year).fit()
        anv = sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        anv = pd.DataFrame()

    # Variance-component estimates (rough: method of moments from MS)
    # n_year=2, l=number of fathers in the trial, t=3
    n_y = df_year["year"].nunique()
    l   = df_year["father"].nunique()
    t   = df_year["mother"].nunique()
    vc = {}
    if not anv.empty and {"sum_sq", "df"}.issubset(anv.columns):
        ms = (anv["sum_sq"] / anv["df"]).to_dict()
        ms_l   = ms.get("C(father)", np.nan)
        ms_t   = ms.get("C(mother)", np.nan)
        ms_lt  = ms.get("C(mother):C(father)", np.nan)
        ms_err = ms.get("Residual", np.nan)
        # Method of moments (Kempthorne, simplified)
        sigma2_e   = ms_err
        sigma2_sca = max((ms_lt - ms_err) / n_y, 0) if not np.isnan(ms_lt) else np.nan
        sigma2_gcaL = max((ms_l - ms_lt) / (t * n_y), 0) if not np.isnan(ms_l) else np.nan
        sigma2_gcaT = max((ms_t - ms_lt) / (l * n_y), 0) if not np.isnan(ms_t) else np.nan
        # Baker's ratio: 2σ²GCA / (2σ²GCA + σ²SCA)
        total_gca = 2 * (sigma2_gcaL + sigma2_gcaT)
        baker = total_gca / (total_gca + sigma2_sca) if (total_gca + sigma2_sca) > 0 else np.nan
        vc = {
            "sigma2_GCA_fathers":  sigma2_gcaL,
            "sigma2_GCA_mothers": sigma2_gcaT,
            "σ²(SCA)":         sigma2_sca,
            "σ²(error)":       sigma2_e,
            "Baker_ratio":     baker,
        }

    return {
        "grand_mean": mu,
        "GCA_fathers":   gca_f,
        "GCA_mothers": gca_m,
        "SCA":         pair[["mother", "mother_long", "father", "Y", "SCA"]],
        "var_comp":    vc,
        "anova":       anv,
    }


results = {t: gca_sca_per_trait(hybrids, t) for t in HYB_TRAITS}


# %% 2. Collect paternal GCA into one table
gca_father_df = pd.concat(
    [results[t]["GCA_fathers"].rename(t) for t in HYB_TRAITS], axis=1
).reset_index()
gca_father_df = gca_father_df.sort_values("father", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
gca_father_df["confectionery"] = gca_father_df["father"].isin(CONFECTIONERY_LI)
gca_father_df.to_csv(OUT / "25_gca_fathers.csv", index=False, encoding="utf-8-sig")
print("\n=== Paternal GCA (54 lines × 6 traits) ===")
print(gca_father_df.round(2).to_string(index=False))


# %% 3. Maternal GCA (3 rows)
gca_mother_df = pd.concat(
    [results[t]["GCA_mothers"].rename(t) for t in HYB_TRAITS], axis=1
).reset_index()
gca_mother_df["mother_long"] = gca_mother_df["mother"].map(MOTHER_MAP)
gca_mother_df = gca_mother_df[["mother", "mother_long"] + HYB_TRAITS]
gca_mother_df.to_csv(OUT / "25_gca_mothers.csv", index=False, encoding="utf-8-sig")
print("\n=== Maternal GCA (3 lines × 6 traits) ===")
print(gca_mother_df.round(2).to_string(index=False))


# %% 4. SCA — collect all pairs × all traits
sca_long = []
for trait in HYB_TRAITS:
    s = results[trait]["SCA"].copy()
    s["trait"] = trait
    sca_long.append(s)
sca_all = pd.concat(sca_long, ignore_index=True)
sca_all.to_csv(OUT / "25_sca_pairs.csv", index=False, encoding="utf-8-sig")

# Top-5 SCA per trait (positive = better than the sum of parents)
print("\n=== Top-5 SCA per trait ===")
for trait in HYB_TRAITS:
    sub = sca_all[sca_all["trait"] == trait]
    top = sub.nlargest(5, "SCA")[["mother_long", "father", "Y", "SCA"]]
    print(f"\n{trait} (direction: {'more=better' if DIRECTION[trait] >= 0 else 'less=better'}):")
    print(top.round(2).to_string(index=False))


# %% 5. Variance components and Baker's ratio
vc_df = pd.DataFrame({t: results[t]["var_comp"] for t in HYB_TRAITS}).T
vc_df.to_csv(OUT / "variance_components.csv", encoding="utf-8-sig")
print("\n=== Variance components + Baker's ratio ===")
print(vc_df.round(3))


# %% 6. Composite selection index for fathers
# z-score GCA, apply direction, then aggregate
gca_z = gca_father_df.set_index("father")[HYB_TRAITS].copy()
for t in HYB_TRAITS:
    z = (gca_z[t] - gca_z[t].mean()) / gca_z[t].std(ddof=1)
    gca_z[t] = z * DIRECTION[t]
# Index weights (for oilseed sunflower)
WEIGHTS = {
    "seed_yield":     0.35,
    "oil_content":     0.30,
    "seed_weight_1000":  0.10,
    "plant_height":           0.00,
    "head_diameter":         0.10,
    "days_emergence_flowering": 0.15,
}
gca_z["selection_index"] = sum(gca_z[t] * w for t, w in WEIGHTS.items())
gca_z["confectionery"] = gca_z.index.isin(CONFECTIONERY_LI)
gca_z = gca_z.sort_values("selection_index", ascending=False)
gca_z.to_csv(OUT / "25_paternal_selection_index.csv", encoding="utf-8-sig")
print("\n=== Top-15 fathers by composite index ===")
print(gca_z.head(15).round(2).to_string())


# %% 7. Top-10 hybrids by productivity and oil yield
# Mean over 2 years for each pair
hyb_means = (
    hybrids.groupby(["mother", "mother_long", "father", "trait"])["value"]
    .mean().unstack("trait").reset_index()
)
hyb_means["oil_yield"] = hyb_means["seed_yield"] * hyb_means["oil_content"] / 100
hyb_means["confectionery_father"] = hyb_means["father"].isin(CONFECTIONERY_LI)
hyb_means.to_csv(OUT / "22_hybrid_means_delivered.csv", index=False, encoding="utf-8-sig")

print("\n=== Top-10 hybrids by oil yield (t/ha × oil content / 100) ===")
top_oil = hyb_means.nlargest(10, "oil_yield")[
    ["mother_long", "father", "seed_yield", "oil_content", "oil_yield", "days_emergence_flowering"]
]
print(top_oil.round(3).to_string(index=False))

print("\n=== Top-10 hybrids by absolute seed yield ===")
top_yield = hyb_means.nlargest(10, "seed_yield")[
    ["mother_long", "father", "seed_yield", "oil_content", "oil_yield"]
]
print(top_yield.round(3).to_string(index=False))


# %% 8. Pseudo-heterosis: mean hybrid of father LI_i minus LI_i as a line
def pseudo_heterosis(trait_hyb: str, trait_lin: str) -> pd.DataFrame:
    """F1 (mean over 3 mothers, 2 years) − paternal line (mean over 4 years)."""
    h = hybrids[hybrids["trait"] == trait_hyb].groupby("father")["value"].mean()
    p = lines[lines["trait"] == trait_lin].groupby("genotype")["value"].mean()
    p = p.reindex(h.index)
    out = pd.DataFrame({"F1_mean": h, "father_as_line": p,
                        "heterosis_abs": h - p,
                        "heterosis_pct": (h - p) / p * 100})
    return out


het_dfs = {}
for trait_h, trait_l in HYB_TO_LIN.items():
    het_dfs[trait_h] = pseudo_heterosis(trait_h, trait_l)

# Collect pseudo-heterosis into one table (% by trait)
het_wide = pd.concat({t: d["heterosis_pct"] for t, d in het_dfs.items()}, axis=1)
het_wide.to_csv(OUT / "pseudo_heterosis_percent.csv", encoding="utf-8-sig")
print("\n=== Pseudo-heterosis in % (F1 from 3 mothers minus the paternal line) ===")
print(het_wide.describe().round(2))


# %% 9. Visualizations
def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return p


# (a) Paternal GCA: scatter (seed yield vs oil content)
fig, ax = plt.subplots(figsize=(11, 8))
sub = gca_father_df.set_index("father")
colors = sub["confectionery"].map({True: "#c0392b", False: "#2c5f8a"})
ax.scatter(sub["seed_yield"], sub["oil_content"], s=80, c=colors,
           edgecolor="black", linewidth=0.4, alpha=0.75)
for li in sub.index:
    ax.annotate(li, (sub.loc[li, "seed_yield"], sub.loc[li, "oil_content"]),
                fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("GCA for seed yield (t/ha), deviation from the mean")
ax.set_ylabel("GCA for oil content (%), deviation from the mean")
ax.set_title("Paternal GCA: seed yield vs oil content (top-right = best)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, "01_gca_urozh_vs_masl.png")

# (b) SCA heatmap for seed yield (3 mothers × 54 fathers)
sca_urozh = sca_all[sca_all["trait"] == "seed_yield"].pivot_table(
    index="mother_long", columns="father", values="SCA"
)
order_otec = sorted(sca_urozh.columns, key=lambda s: int(str(s).replace("LI", "")))
sca_urozh = sca_urozh.reindex(columns=order_otec)
fig, ax = plt.subplots(figsize=(18, 3.2))
sns.heatmap(sca_urozh, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "SCA"},
            xticklabels=True, yticklabels=True, linewidths=0.3)
ax.set_title("SCA for seed yield (t/ha): pair deviation from the parental sum")
ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
plt.tight_layout()
save(fig, "02_sca_heatmap_urozh.png")

# (c) Top-15 index
fig, ax = plt.subplots(figsize=(11, 7))
top = gca_z.head(15)
colors = ["#c0392b" if li in CONFECTIONERY_LI else "#2c5f8a" for li in top.index]
ax.barh(top.index[::-1], top["selection_index"][::-1], color=colors[::-1])
ax.set_xlabel("Composite selection index (z-score, weights: yield 0.35 + oil 0.30 + earliness 0.15 + ...)")
ax.set_title("Top-15 fathers by composite GCA index")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
save(fig, "03_selection_index_top15.png")

# (d) Pseudo-heterosis histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, trait in zip(axes.flatten(), HYB_TO_LIN.keys()):
    h = het_dfs[trait]["heterosis_pct"].dropna()
    ax.hist(h, bins=20, color="#2c5f8a", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="no_heterosis")
    ax.axvline(h.median(), color="green", lw=1.5, label=f"median {h.median():.1f}%")
    ax.set_title(f"{trait}: F1 mean vs father")
    ax.set_xlabel("heterosis_pct")
    ax.set_ylabel("n_fathers")
    ax.legend(fontsize=8)
plt.suptitle("Pseudo-heterosis: F1 (mean over 3 mothers) − paternal line")
plt.tight_layout()
save(fig, "04_pseudo_heterosis_hist.png")

# (e) Variance components — barplot Baker's ratio
fig, ax = plt.subplots(figsize=(9, 5))
ratios = vc_df["Baker_ratio"].dropna()
colors = ["#2c5f8a" if r > 0.7 else "#d4a017" if r > 0.5 else "#c0392b" for r in ratios]
ax.bar(ratios.index, ratios.values, color=colors, edgecolor="black")
ax.axhline(0.5, color="gray", ls="--", label="50%: equal GCA and SCA contributions")
ax.set_ylabel("Baker's ratio = 2σ²GCA / (2σ²GCA + σ²SCA)")
ax.set_title("Share of additive (GCA) variation in genetic variance")
ax.set_ylim(0, 1.05)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
save(fig, "05_baker_ratio.png")


# %% 10. HTML report
def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


img_gca = b64(OUT / "01_gca_urozh_vs_masl.png")
img_sca = b64(OUT / "02_sca_heatmap_urozh.png")
img_idx = b64(OUT / "03_selection_index_top15.png")
img_het = b64(OUT / "04_pseudo_heterosis_hist.png")
img_bak = b64(OUT / "05_baker_ratio.png")

top15_html = gca_z.head(15)[HYB_TRAITS + ["selection_index"]].round(2).to_html(classes="data")
gca_m_html = gca_mother_df.round(2).to_html(index=False, classes="data")
top_oil_html = top_oil.round(3).to_html(index=False, classes="data")
top_yield_html = top_yield.round(3).to_html(index=False, classes="data")
vc_html = vc_df.round(3).to_html(classes="data")

# Top-10 SCA for seed yield and oil content
top_sca_urozh = sca_all[sca_all["trait"] == "seed_yield"].nlargest(10, "SCA")[
    ["mother_long", "father", "Y", "SCA"]].round(3).to_html(index=False, classes="data")
top_sca_masl = sca_all[sca_all["trait"] == "oil_content"].nlargest(10, "SCA")[
    ["mother_long", "father", "Y", "SCA"]].round(3).to_html(index=False, classes="data")

css = """<style>
body{font-family:'Segoe UI',Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#222;line-height:1.5;}
h1{border-bottom:3px solid #2c5f8a;padding-bottom:8px;}
h2{color:#2c5f8a;margin-top:30px;}
h3{margin-top:22px;}
img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:10px 0;}
table.data{border-collapse:collapse;margin:10px 0;font-size:12px;}
table.data th{background:#2c5f8a;color:#fff;padding:5px 9px;text-align:left;}
table.data td{border-bottom:1px solid #ddd;padding:4px 9px;}
.note{background:#fff5d6;border-left:4px solid #d4a017;padding:12px 16px;margin:14px 0;}
.quote{background:#f0f4fa;border-left:4px solid #2c5f8a;padding:10px 16px;margin:10px 0;font-style:italic;}
</style>"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Phase 3c: GCA / SCA</title>{css}</head><body>

<h1>Phase 3c: Combining ability of the hybrids</h1>
<p>159 hybrids (3 mothers × 54 fathers, three pairs absent) × 2 years (2022, 2023).
Kempton's line × tester (NCII) model on two-year means.</p>

<div class="note">
<b>Limits of the design:</b><br>
• two years instead of the usual ≥3 — variance components are poorly determined;<br>
• three mothers — sigma2_GCA_mothers is noisy by construction;<br>
• ANOVA on means inflates F and deflates p;<br>
so the report shows <b>effect sizes</b> (GCA, SCA, the index), not p-values.
</div>

<h2>I. Variance components and Baker's ratio</h2>
<img src="data:image/png;base64,{img_bak}">
{vc_html}

<div class="quote">
<b>Baker's ratio = 2σ²GCA / (2σ²GCA + σ²SCA)</b> — the share of additive genetic variation.
<ul>
<li>High (&gt;0.7) — the trait is mostly additive; selection on parents is effective;</li>
<li>Intermediate (0.5–0.7) — a substantial dominance/epistatic component; look for good combinations (SCA);</li>
<li>Low (&lt;0.5) — non-additive control; parental GCA poorly predicts the hybrid.</li>
</ul>
</div>

<h2>II. Maternal GCA</h2>
<p>GCA is the deviation from the grand mean. Positive: hybrids with this mother
are above average for the trait.</p>
{gca_m_html}

<h2>III. Paternal GCA</h2>
<h3>Seed yield vs oil content</h3>
<img src="data:image/png;base64,{img_gca}">
<p>Upper-right: lines that raise both yield and oil content. Lower-left: lines that
lower both. Lower-right: mass without oil; upper-left: the reverse.</p>

<h3>Top 15 fathers by the composite selection index</h3>
<p>Index = 0.35·z(yield) + 0.30·z(oil) + 0.10·z(1000-seed weight) + 0.10·z(diameter) − 0.15·z(emergence-to-flowering).
The minus on flowering prefers earliness.</p>
<img src="data:image/png;base64,{img_idx}">
{top15_html}

<h2>IV. SCA</h2>
<p>SCA is the residual after the additive model. A high SCA means that mother × father
pair outperforms the sum of the two GCA effects.</p>

<h3>SCA heatmap for seed yield (3 × 54)</h3>
<img src="data:image/png;base64,{img_sca}">
<p>Red: pairs with positive SCA; blue: the reverse. Read by column: which mother
combines best with a given father.</p>

<h3>Top 10 SCA for seed yield</h3>
{top_sca_urozh}

<h3>Top 10 SCA for oil content</h3>
{top_sca_masl}

<h2>V. Top 10 hybrids</h2>

<h3>By oil yield (t/ha × oil content / 100)</h3>
{top_oil_html}

<h3>By seed yield</h3>
{top_yield_html}

<h2>VI. Pseudo-heterosis on the paternal background</h2>
<p>Classical mid-parent heterosis needs both parents. Maternal line phenotypes
are not available, so this is heterosis relative to the <i>father only</i>:
the mean hybrid of father LI_i over 3 mothers and 2 years minus LI_i as a line.</p>
<img src="data:image/png;base64,{img_het}">

<div class="quote">
<b>Reading the pseudo-heterosis:</b><br>
• Positive median — the mother lifts the father (ordinary hybrid vigour).<br>
• Near-zero median — the paternal phenotype already predicts the hybrid.<br>
• Negative median — hybrids worse than the father (unusual; possible plasm
incompatibility or specific interactions).
</div>

<h2>VII. What this is for</h2>

<ol>
<li><b>Lines in the top-15 of the index</b> — candidates for further crosses.
Their phenotype is additive and predictable in the hybrid.</li>

<li><b>High-SCA pairs</b> — candidate commercial hybrids, especially when SCA
is positive for several traits at once (see 25_sca_pairs.csv).</li>

<li><b>Baker's ratio by trait</b> — additive traits: pick good parents;
non-additive: search for good pairs.</li>

<li><b>Top 10 by oil yield</b> — current candidates in this pool of 159 pairs,
not a claim that they are the best possible.</li>

<li><b>Paternal-background pseudo-heterosis</b> — a proxy for the maternal
contribution when maternal phenotypes are missing.</li>
</ol>

<h2>VIII. What these numbers are not</h2>
<ul>
<li>Do not quote ANOVA F and p — they are on means, without plot replication.</li>
<li>Do not treat significance as settled — independent repeats are required.</li>
<li>SCA does not replace official variety trials.</li>
<li>Baker's ratio is ordered, not precise: read “above / below one half”, not the third digit.</li>
</ul>

</body></html>"""

(OUT / "phase3c_report.html").write_text(html, encoding="utf-8")
print(f"\nHTML: {OUT/'phase3c_report.html'}")

print("\nDone. Artifacts:")
for f in sorted(OUT.glob("*")):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
