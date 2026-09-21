# Genotyping pipeline (alignment and VCF assembly)

Code that produced the two primary GBS call sets (var1, var2), not the analysis
scripts in `scripts/`. These inputs are not in the deposit (raw reads, BAM,
per-sample VCF). Set `SUNFLOWER_CALLING`, `CHOCALLATE` and `SUNFLOWER_REF` in
`../.env`.

## Versions used in the run

| Tool | Version | Where recorded |
|---|---|---|
| bowtie2 | 2.5.5 | BAM `@PG`, ChoCallate `environment.yaml` |
| samtools | 1.23 | BAM `@PG`; `samtools merge -f` / `addreplacerg` in the notebook |
| GATK LeftAlignIndels | 4.6.2.0 | BAM `@PG` |
| bcftools merge (all lines, `-m all`) | 1.23.1 | header of `all_samples.merged.vcf` |
| bcftools merge (replicates, `-m none`) | 1.16 | same header, earlier step |
| vcftools | 0.1.17 | `--min-alleles 2` after the merge |
| bcftools +fixploidy | (same install as the merge) | after vcftools |

Reference: HanXRQr2.0-SUNRISE, bowtie2 index. Mapper: bowtie2 only.

The scripts in this folder call whatever `bcftools` is on `PATH`; they are the
run as executed, not a rewrite.

## Layout

| Path | What |
|---|---|
| `notebooks/main_replicates.ipynb` | delivered calling run: ChoCallate → IBS → `samtools merge` → recall → VCF merge (var1) |
| `233_merge_replicates_by_rules_fast.py` | replicate VCFs → one sample (`-m none` + GT rules); var2 |
| `union/234_merge_vcfs.py` | position-wise merge of the two *delivered* files → `23_merged.vcf.gz` |

There are no `var1/` or `var2/` folders: var1 is the notebook, var2 is the script above.
The earlier 120-column notebook and the site-level union shell are in
`../../07_архив_скриптов/`. The paper analyses `234_merge_vcfs.py`.

Notebook source cells read `SUNFLOWER_CALLING` / `CHOCALLATE` / `SUNFLOWER_REF`.
Stored outputs (plots, Russian logs) were stripped.

## What the notebook actually runs

`main_replicates.ipynb` is the delivered run.

1. ChoCallate on per-replicate BAMs (`nextflow run …/ChoCallate/main.nf`).
2. IBS among technical replicates → `filt_vcf/replicate_merge_plan.tsv`
   (54 lines `merge_bam`; 15 and 49 left separate).
3. BAM merge of those replicates:
   `samtools merge -f -o tmp.bam *reps` then
   `samtools addreplacerg -r @RG\tID:{line}\tSM:{line}\tPL:ILLUMINA`.
4. Recall on merged BAMs: ChoCallate with `config_merged.yaml` → `raw_vcf_merged/`.
5. `bcftools merge -m all` of `raw_vcf_merged/per_sample/*.vcf.gz`.
6. var1 as delivered: merge the BAM-merged min_2 file with the still-separate
   `15*.vcf.gz` and `49*.vcf.gz` → 60 columns, **466,165** sites after
   `--min-alleles 2`. That is the paper's var1 cell count.
7. var2 path: `233_merge_replicates_by_rules_fast.py`, then
   `bcftools merge -m all` of `outputs/merged_vcf/*.vcf.gz`, then
   vcftools `--min-alleles 2` and `bcftools +fixploidy`.

## Three VCF merges (after the BAM merge)

1. **All per-sample VCFs → one multi-sample file** — `bcftools merge -m all -Oz`,
   in the notebooks (not a standalone `.sh`).
2. **Replicates of one line → one sample** —
   `233_merge_replicates_by_rules_fast.py`: `bcftools merge -m none -Ou`
   streamed to pysam; GT rule: allele over missing, hom over het, mix > ind >
   unknown, remaining tie by the lower replicate number.
3. **Union of the two delivered files** —
   `union/234_merge_vcfs.py`: homozygote over heterozygote at shared sites
   → `23_merged.vcf.gz`. The older site-level `isec`/`concat` shell is archived.
