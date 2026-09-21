#!/usr/bin/env Rscript
# GAPIT BLINK GWAS with kinship + PCA covariates.
#
# Independent recomputation (Methods 2.6.1 / S3): BLINK, VanRaden kinship, PCA.total = 5.
# The deposited GAPIT_54_lines / GAPIT_52_lines folders are outputs of this run.
# `261_supp_s3_compare_rerun.py` compares those outputs to the Python BLINK.
#
# Paths: copy 05_GitHub/.env.example to .env. Optional first argument is a
# working tree that contains GWAS/ (original layout). Otherwise:
#   SUNFLOWER_WORK   undeposited GAPIT tree (GWAS/ or 7_ручной обсчет…/no_sugar/no_sugar)
#   SUNFLOWER_OUT     gapit_genotypes.txt / gapit_phenotypes.csv and new GAPIT output (default ./outputs)

args <- commandArgs(trailingOnly = TRUE)
script_dir <- tryCatch(
  dirname(normalizePath(sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))]))),
  error = function(e) NA_character_
)

load_dotenv <- function(start) {
  p <- if (!is.na(start) && nzchar(start)) start else getwd()
  for (i in seq_len(10)) {
    f <- file.path(p, ".env")
    if (file.exists(f)) {
      for (line in readLines(f, warn = FALSE, encoding = "UTF-8")) {
        line <- trimws(line)
        if (!nzchar(line) || startsWith(line, "#") || !grepl("=", line, fixed = TRUE)) next
        kv <- strsplit(line, "=", fixed = TRUE)[[1]]
        key <- trimws(kv[[1]])
        val <- trimws(paste(kv[-1], collapse = "="))
        val <- gsub("^[\"']|[\"']$", "", val)
        if (nzchar(key) && !nzchar(Sys.getenv(key, unset = ""))) {
          do.call(Sys.setenv, setNames(list(val), key))
        }
      }
      return(invisible(TRUE))
    }
    nxt <- dirname(p)
    if (identical(nxt, p)) break
    p <- nxt
  }
  invisible(FALSE)
}

load_dotenv(script_dir)

env_dir <- function(name) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(NA_character_)
  normalizePath(path.expand(raw), mustWork = FALSE)
}

has_gwas <- function(d) {
  !is.na(d) && dir.exists(file.path(d, "GWAS"))
}

deposit <- env_dir("SUNFLOWER_DEPOSIT")
if (is.na(deposit)) {
  deposit <- if (!is.na(script_dir)) {
    normalizePath(file.path(script_dir, "..", "..", "..", "04_Депозит"), mustWork = FALSE)
  } else {
    NA_character_
  }
}
work <- env_dir("SUNFLOWER_WORK")
out_env <- env_dir("SUNFLOWER_OUT")

root <- if (length(args) >= 1 && has_gwas(args[[1]])) {
  args[[1]]
} else if (has_gwas(work)) {
  work
} else if (!is.na(work) && dir.exists(file.path(work, "7_ручной обсчет воронежская", "no_sugar", "no_sugar"))) {
  file.path(work, "7_ручной обсчет воронежская", "no_sugar", "no_sugar")
} else if (!is.na(script_dir) && has_gwas(dirname(dirname(script_dir)))) {
  dirname(dirname(script_dir))
} else {
  NA_character_
}

if (!is.na(root) && has_gwas(root)) {
  gwas <- file.path(root, "GWAS")
  pheno_dir <- file.path(gwas, "pheno", "baseline")
  gapit_dir <- file.path(gwas, "gapit")
  out_dir <- file.path(gapit_dir, "output", "baseline_blink")
  geno_path <- file.path(gapit_dir, "input", "gapit_genotypes.txt")
  pheno_path <- file.path(pheno_dir, "gapit_phenotypes.csv")
  vcf_path <- file.path(gwas, "genotype", "snp_LD_fathers.vcf.gz")
} else {
  out_root <- if (!is.na(out_env)) {
    out_env
  } else if (!is.na(script_dir)) {
    normalizePath(file.path(script_dir, "..", "..", "outputs"), mustWork = FALSE)
  } else {
    file.path(getwd(), "outputs")
  }
  gapit_dir <- file.path(out_root, "29_gapit")
  out_dir <- file.path(gapit_dir, "output", "baseline_blink")
  inter <- if (dir.exists(out_root)) out_root else NA_character_
  geno_path <- if (!is.na(inter)) file.path(inter, "gapit_genotypes.txt") else file.path(gapit_dir, "input", "gapit_genotypes.txt")
  pheno_path <- if (!is.na(inter)) file.path(inter, "gapit_phenotypes.csv") else file.path(gapit_dir, "input", "gapit_phenotypes.csv")
  vcf_path <- NA_character_
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(geno_path), recursive = TRUE, showWarnings = FALSE)

PCA_TOTAL <- 5L

gapit_r_libs <- file.path(if (exists("gapit_dir")) gapit_dir else out_dir, "R_libs")
dir.create(gapit_r_libs, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(gapit_r_libs, .libPaths()))

load_gapit <- function() {
  if (requireNamespace("GAPIT", quietly = TRUE)) {
    suppressPackageStartupMessages(library(GAPIT))
    return(invisible(TRUE))
  }
  pkg <- file.path(gapit_dir, "GAPIT")
  if (!dir.exists(pkg)) {
    stop("GAPIT not installed and ", pkg, " missing. Clone jiabowang/GAPIT there or install GAPIT.")
  }
  status <- system2(
    "R",
    c(
      "CMD", "INSTALL", "--no-multiarch", "--no-test-load", pkg
    ),
    env = c(sprintf("R_LIBS=%s", gapit_r_libs), sprintf("R_LIBS_USER=%s", gapit_r_libs))
  )
  if (status != 0) stop("GAPIT install failed (exit ", status, ")")
  suppressPackageStartupMessages(library(GAPIT, lib.loc = gapit_r_libs))
  invisible(TRUE)
}

load_gapit()

if (!file.exists(geno_path)) {
  py <- Sys.which("python3")
  if (!nzchar(py)) stop("python3 not found for vcf_to_gapit_g.py")
  conv <- file.path(gapit_dir, "vcf_to_gapit_g.py")
  if (!file.exists(conv) || is.na(vcf_path) || !file.exists(vcf_path)) {
    stop(
      "Missing genotypes: ", geno_path,
      "\nSet SUNFLOWER_OUT (uses outputs/gapit_genotypes.txt) ",
      "or SUNFLOWER_WORK to the tree that has GWAS/gapit/input and snp_LD_fathers.vcf.gz."
    )
  }
  status <- system2(
    py,
    c(conv, "--vcf", vcf_path, "--out", geno_path)
  )
  if (status != 0) stop("Genotype conversion failed")
}

if (!file.exists(pheno_path)) {
  stop("Missing phenotypes: ", pheno_path, "\nRun: Rscript GWAS/lmer/export_baseline_blup.R")
}

G <- read.delim(geno_path, header = FALSE, check.names = FALSE, stringsAsFactors = FALSE)
Y_all <- read.csv(
  pheno_path,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  colClasses = c(Taxa = "character")
)
Y_all$Taxa <- as.character(as.numeric(Y_all$Taxa))
trait_cols <- setdiff(names(Y_all), "Taxa")

log_path <- file.path(out_dir, "run_gapit.log")
cat("", file = log_path)

old_wd <- getwd()
setwd(out_dir)
on.exit(setwd(old_wd), add = TRUE)

for (trait in trait_cols) {
  cat("trait:", trait, "\n", file = log_path, append = TRUE)
  message("GAPIT BLINK: ", trait)

  Y <- Y_all[, c("Taxa", trait)]
  Y <- Y[!is.na(Y[[trait]]), , drop = FALSE]

  if (nrow(Y) < 20) {
    cat("  skip: too few phenotypes\n", file = log_path, append = TRUE)
    next
  }

  trait_dir <- file.path(out_dir, trait)
  dir.create(trait_dir, showWarnings = FALSE)
  setwd(trait_dir)

  tryCatch(
    {
      GAPIT(
        Y = as.data.frame(Y),
        G = G,
        model = "BLINK",
        PCA.total = PCA_TOTAL,
        kinship.algorithm = "VanRaden",
        Geno.View.output = FALSE,
        file.output = TRUE
      )
      cat("  OK\n", file = log_path, append = TRUE)
    },
    error = function(e) {
      cat("  ERROR:", conditionMessage(e), "\n", file = log_path, append = TRUE)
      message("ERROR ", trait, ": ", conditionMessage(e))
    }
  )

  setwd(out_dir)
}

message("OK: ", out_dir)
