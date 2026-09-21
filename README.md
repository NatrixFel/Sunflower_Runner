# Sunflower NCII analysis pipeline

Code that reproduces the analyses in:

**What a breeding-size panel can and cannot resolve: combining ability, stratification artefacts and one dominance locus in sunflower**

Running title: *Limits of small sunflower panels*

This repository is the analysis and genotyping pipeline for a sunflower (*Helianthus annuus* L.) North Carolina Design II (NCII) panel: three CMS mother lines × 54 paternal lines (52 oilseed, 2 confectionery) and their 162 hybrids. Parents were genotyped by GBS; hybrid genotypes were formed in silico. The scripts estimate combining ability, genomic prediction, additive GWAS (EMMAX, BLINK), a dominance-deviation scan, and the figures in the paper.

The GWAS call sets are in `vcf/`: `emmax_input.vcf.gz` (the union set used by EMMAX and the Python BLINK scans), `GAPIT_input.vcf.gz` (the separately assembled, LD-pruned panel used by GAPIT v3 BLINK), and `23_no_imputation.vcf` (dominance-scan robustness check). Tables written by the scripts go to `outputs/`. The companion matrix `23_merged_genotypes.npz` is read through `SUNFLOWER_DEPOSIT` when that path is set. Machine-specific paths are read from a gitignored `.env` (see `.env.example`).

Alignment and variant calling used the [ChoCallate](https://github.com/alermol/ChoCallate) workflow.

## Abstract

To separate reproducible results from small-sample artefacts we studied a sunflower (*Helianthus annuus* L.) population of three mother lines × 54 paternal lines (52 oilseed, 2 confectionery) and their 162 hybrids, genotyped in silico from the sequenced parents (81,903 markers in the 54 paternal lines, 82,889 in the hybrids), using association analysis (EMMAX, BLINK) and genomic prediction. Plot-level variance components made additive control predominant for oil content and 1000-seed weight (Baker's ratio 0.958 and 0.951) but left it unresolved for seed yield (0.440, interval 0.00–0.82), where the estimated specific combining ability component exceeded the general one. Predictive ability was 0.84 for 1000-seed weight, 0.65 for oil content, and, for seed yield, 0.19 at the hybrid level against 0.43 for general combining ability. The variance components attributed that gap to experimental error and non-additivity together. A dominance-deviation scan over five traits returned one candidate locus, for 1000-seed weight, Chr14:169,211,003–169,211,163 (β_dom = +3.38 g, empirical genome-wide *p* = 0.002), not tested by the additive scan and reproduced on two separately assembled marker sets, not in new material. At *n* ≈ 50, EMMAX in the 52 oilseed lines reached the 5 × 10⁻⁸ threshold for no single-nucleotide polymorphism (SNP) in nine traits, though heritable (*H*² = 0.73–0.95). Two confectionery lines alone produced 177 spurious associations for oil content, hull content and 1000-seed weight without principal-component covariates, at an empirical genome-wide threshold, and 28 at the nominal 5 × 10⁻⁸; both fell to zero on removing them. Correction on three principal components weakened the artefact but did not always remove it. Under that correction the more powerful model returned more of these associations at the nominal threshold: BLINK 207 at 54 lines and 4 at 52, against none for EMMAX. A separate implementation on a separate assembly reproduced the phenomenon, its top oil-content marker falling from *p* = 1.0 × 10⁻¹² to 3.5 × 10⁻⁴; with 22% of markers shared, agreement held for the phenomenon, not for positions. Excluding the minority use type reduced it on all sixteen panels of both filtering schemes. Individual quantitative trait loci from mixed use types require caution.

**Keywords:** *Helianthus annuus*; combining ability; genomic prediction; GBLUP; population stratification; dominance deviation; statistical power; sample size.

## Numeric prefixes

File names start with the Materials and Methods subsection that the script (or the table it writes) belongs to. The digits are the subsection number **without the dots** (so 2.2.1 → `221_`, 2.6.1 → `261_`, 2.9 → `29_`).

| Prefix | Methods section | What that section does |
|---|---|---|
| `221_` | **2.2.1 Trait measurements** | Field-record rebuilds, plot-level tables, year-mean shift tests |
| `222_` | **2.2.2 Variance components and heritability** | Hybrid and line BLUPs, plot repeatability, *H*² |
| `233_` | **2.3.3 VCF-level replicate-consensus branch** | Per-sample VCF replicate merge |
| `234_` | **2.3.4 Union call set and conflict resolution** | Position-wise union of the two calling branches |
| `235_` | **2.3.5 Genotype coding, completeness, and marker panels** | Call-rate / MAF panels and completeness checks |
| `24_` | **2.4 Population structure and relationship matrices** | PCA, LD pruning, additive / dominance relationship matrices |
| `25_` | **2.5 Combining ability** | GCA, SCA, Baker's ratio, composite paternal index |
| `261_` | **2.6.1 Additive association models and independent recomputation** | EMMAX, Python BLINK, use-type artefact, GAPIT v3 BLINK |
| `262_` | **2.6.2 Dominance-deviation association analysis** | Dominance scan on Chr14 and the no-imputation check |
| `27_` | **2.7 Genomic prediction** | GBLUP, leave-one-father-out CV, learning curves |
| `28_` | **2.8 Genetic distance and hybrid performance** | Mother–father distance, heterosis checks |
| `29_` | **2.9 Software** | Figures and the check of numbers reported in the paper |

A table and the script that writes it share the prefix, for example `25_gca_sca.py` → `25_gca_fathers.csv`. Shared libraries without a Methods home (`emmax.py`, `blink.py`, `baker_v4.py`) have no prefix. The no-imputation call set keeps the filename `23_no_imputation.vcf` used in the paper.

## Layout

```
paths.py          outputs / optional raw-data roots
names.py          English ↔ Russian I/O map (gitignored; required to run)
.env.example      copy to .env
outputs/          script-generated tables and GAPIT folders
vcf/              GWAS call sets (`emmax_input.vcf.gz`, `GAPIT_input.vcf.gz`, `23_no_imputation.vcf`)
scripts/
  models/         EMMAX, BLINK, Baker REML (imported, not run first)
  phenotypes/     221_ field and year-mean rebuilds
  variance/       222_ heritability; 25_ Baker's ratio
  callsets/       234_ merge statistics; 235_ coverage
  gwas/           235_ panels; 24_ PCA; 261_ additive scans and GAPIT; 262_ LD decay
  dominance/      262_ Chr14 dominance scan
  combining/      25_ GCA/SCA/index; 27_ genomic prediction
  heterosis/      28_ distance–yield
  figures/        29_ figures and number check
genotype/         BAM/VCF assembly that produced var1, var2 and the union
```

## Local paths

Copy `.env.example` to `.env` (gitignored).

| Variable | Default | Used for |
|---|---|---|
| `SUNFLOWER_DEPOSIT` | set in `.env` | directory that holds `23_merged_genotypes.npz` |
| `SUNFLOWER_OUT` | `./outputs` | script-generated tables |
| `SUNFLOWER_WORK` | *(empty)* | undeposited Excel workbooks and raw var1/var2 |
| `SUNFLOWER_CALLING` | *(empty)* | raw VCF/BAM tree |
| `CHOCALLATE` | *(empty)* | Nextflow variant-calling checkout |
| `SUNFLOWER_REF` | *(empty)* | HanXRQr2.0 FASTA |
| `SUNFLOWER_REF_INDEX` | *(empty)* | bowtie2 index |

`paths.py` reads `.env` and re-exports `en_table` / `en_ids` from `names.py`. Deposit tables still use the original Russian headers and IDs; convert at the boundary.

Environment used in the paper: Python 3.12.10; numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, pyarrow 24.0.0, scikit-allel 1.3.13, matplotlib 3.10.9. Alignment and VCF assembly: bowtie2 2.5.5, samtools 1.23, GATK LeftAlignIndels 4.6.2.0, bcftools 1.23.1.

## Suggested run order

1. **models** — imported: `emmax.py`, `blink.py`, `baker_v4.py`
2. **phenotypes** — `221_lines_tidy.py`, `221_year_means_shift.py`, `221_rebuild_biometry.py`, `221_parse_hybrid_reps.py`, `221_pheno_means_calibration.py`
3. **variance** — `25_baker_v4_intervals.py`, `222_h2_recalc.py`, `222_h2_lines_gy.py`, `25_baker_five_traits.py`
4. **gwas** — `261_pca_artifact.py`, `261_artifact_merged.py`, `261_artifact_pc.py`, `261_canonical_lines_gwas.py`, `261_canonical_blink.py`, `235_gwas_merged_panel.py`, `261_gapit_blink.R`
5. **dominance** — `262_chr14_final.py` (`--trait`), `262_crosspanel.py`
6. **combining** — `25_gca_sca.py`, `25_index_canonical.py`, `27_gs_blup_vs_mean.py`, `222_blup_biometry.py`
7. **figures** — `29_verify_reported_numbers.py`, `29_make_figures_EN.py`

Scripts under `scripts/phenotypes/`, `scripts/callsets/` and `scripts/heterosis/` need `SUNFLOWER_WORK` if the primary Excel or var1/var2 files are not in the repository. Of the numbers printed in the paper, most reproduce from `scripts/` plus the files in `vcf/` and `outputs/`; the rest need those undeposited files (Methods 2.9).

---

## Scripts

### Shared libraries (`scripts/models/`)

| File | What it does |
|---|---|
| `emmax.py` | Compact EMMAX (Kang et al. 2010) used for the additive scans |
| `blink.py` | Python BLINK (Huang et al. 2019) used for the additive scans |
| `baker_v4.py` | REML Baker's ratio with year, tester and year × tester fixed, and a separate year × line term |

### 2.2.1 Trait measurements (`scripts/phenotypes/`)

| File | What it does |
|---|---|
| `221_lines_tidy.py` | Builds the tidy line-phenotype table from the breeders' workbooks |
| `221_parse_hybrid_reps.py` | Parses hybrid plot-level replicates (yield, oil, 1000-seed weight) |
| `221_year_means_shift.py` | Tests the one-line shift in the delivered year-means table against field records |
| `221_rebuild_biometry.py` | Rebuilds plant height and head diameter at plot level from single-plant field records |
| `221_pheno_means_calibration.py` | Checks what can be estimated when a trait arrives only as plot means (emergence-to-flowering) |
| `221_af_ratio_check.py` | Autofertility index versus the two raw seed-set counts |

### 2.2.2 Variance components and heritability (`scripts/variance/`, `scripts/combining/`)

| File | What it does |
|---|---|
| `222_h2_recalc.py` | Hybrid broad-sense heritability and plot repeatability on the declared combining-ability model |
| `222_h2_lines_gy.py` | Line *H*² over four years, with genotype × year kept when it is estimable |
| `222_h2_af_check.py` | Sensitivity of self-fertility *H*² to one documented seed-count substitution |
| `222_r2_year_fixed.py` | Year as fixed versus random in the hybrid BLUP model |
| `222_blup_biometry.py` | Hybrid BLUPs for plant height and head diameter on the rebuilt plot-level table |

### 2.3.2–2.3.4 Calling branches and union set (`genotype/` and `scripts/callsets/`)

See also `genotype/README.md`.

| File | What it does |
|---|---|
| `genotype/notebooks/main_replicates.ipynb` | BAM-level merge and re-calling (var1 branch): ChoCallate, IBS, `samtools merge`, recall |
| `genotype/233_merge_replicates_by_rules_fast.py` | VCF-level replicate merge (var2 branch): allele over missing, homozygote over heterozygote |
| `genotype/union/234_merge_vcfs.py` | Position-wise union of the two delivered call sets → `23_merged.vcf.gz` |
| `234_merge_rule.py` | Cell counts and conflict rates of the union merge rule |
| `235_coverage_full.py` | Full-coverage and call-rate panels (54 / 52 / 57 samples) |
| `235_coverage_pca.py` | Whether per-sample call rate tracks the leading principal components |
| `234_het_bias.py` | Residual-heterozygosity bias under the homozygote-over-heterozygote rule |

### 2.3.5–2.4 Marker panels and structure (`scripts/gwas/`)

| File | What it does |
|---|---|
| `235_gwas_merged_panel.py` | Analysis panels from the union set; EMMAX and BLINK on the key loci |
| `235_alt_filtering.py` | Second, independently specified filtering scheme (twelve combinations) |
| `235_vv_extra.py` | Extra checks on that second scheme |
| `24_ld_pca.py` | LD pruning, PCA on pruned sets, and genomic-control λ |

### 2.5 Combining ability (`scripts/variance/`, `scripts/combining/`)

| File | What it does |
|---|---|
| `25_baker_v4_intervals.py` | Profile-likelihood interval of Baker's ratio; writes `25_baker_v4_intervals.json` |
| `25_baker_five_traits.py` | Baker's ratio and components for the five traits that have plot-level records |
| `25_baker_xcheck.py` | Sensitivity of the ratio to the year × tester specification |
| `25_baker_mother.py` | Maternal variance component (three testers; noisy) |
| `25_baker_consequences.py` | Component shares for Figure 1B and additive-prediction ceilings |
| `25_gca_sca.py` | Paternal and maternal GCA, SCA, and the six-trait tables |
| `25_index_canonical.py` | Composite paternal selection index (pre-set weights) |
| `25_index_after_fix.py` | Index after the year-mean rebuild; stability of the leading group |
| `25_repeatability_gca.py` | Between-year Pearson correlation of paternal GCA |

### 2.6.1 Additive association and independent recomputation (`scripts/gwas/`)

| File | What it does |
|---|---|
| `261_pca_artifact.py` | Use-type artefact versus PCA covariates (EMMAX, merged panel) |
| `261_artifact_merged.py` | Artefact counts on the analysis panel without principal-component covariates |
| `261_artifact_pc.py` | Same counts under the declared three-PC model |
| `261_threshold_uncertainty.py` | Bootstrap uncertainty of the empirical-threshold counts |
| `261_canonical_lines_gwas.py` | Nine-trait line EMMAX on 52 and 54 paternal lines |
| `261_canonical_blink.py` | Nine-trait line BLINK on the same panels |
| `261_chr8_canonical_check.py` | Chr8:782,740 on the analysis panel (Figure 7, EMMAX arm) |
| `261_supp_s2_blink_selftest.py` | BLINK self-test on simulated data (Supplementary S2) |
| `261_gapit_blink.R` | GAPIT v3 BLINK on the separately assembled, LD-pruned panel |
| `261_supp_s3_compare_rerun.py` | Concordance of Python BLINK with that GAPIT run (Supplementary S3) |

### 2.6.2 Dominance-deviation association analysis (`scripts/dominance/`)

| File | What it does |
|---|---|
| `262_chr14_final.py` | Declared dominance-deviation scan (`--trait`); parametric bootstrap and permutation |
| `262_crosspanel.py` | Same locus on the other marker panels and the no-imputation call set |
| `262_chr14_ld.py` | LD window around the haplotype |
| `262_chr14_audit.py` | Extra dissection of the haplotype and neighbouring markers |
| `262_ld_decay.py` | Within-chromosome LD decay used to justify the empirical threshold |

### 2.7 Genomic prediction (`scripts/combining/`)

| File | What it does |
|---|---|
| `27_gs_blup_vs_mean.py` | GBLUP predictive ability: year means versus BLUP line phenotypes |
| `27_gca_learning_curve.py` | Predictive ability versus training-set size (10–45 paternal lines) |

### 2.8 Genetic distance (`scripts/heterosis/`, `scripts/combining/`)

| File | What it does |
|---|---|
| `28_reconcile_heterosis.py` | Mother-specific correlation of parental distance with hybrid yield |
| `28_heterosis_metric_sensitivity.py` | Sensitivity of that correlation to how distance is defined |
| `28_repro_162.py` | Reproduces the 162-hybrid (plot-level) versus 159-hybrid (year-mean) split |

### 2.9 Software and figures (`scripts/figures/`)

| File | What it does |
|---|---|
| `29_make_figures_EN.py` | Figures 1–7 (PNG, JPEG and TIFF at 600 dpi) |
| `29_verify_reported_numbers.py` | Rechecks the numerical claims printed in the manuscript |

## Outputs

`outputs/` holds the CSV / parquet / JSON tables the scripts write, plus `GAPIT_54_lines/` and `GAPIT_52_lines/`. The GWAS VCFs are in `vcf/`.
