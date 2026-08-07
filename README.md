# cordiag — Cross-Omics Readiness Diagnostics (COR)

> **Package version**: cordiag 0.1.0
> **Date**: 2026-08-07

`cordiag` is a unified algorithm family for **pre-flight diagnostics of
RNA-to-protein inference**. It answers two questions before analysis proceeds:

- **zPG** (z-score of Paired Gain) — does this paired multi-omics dataset
  support individual-level cross-omics prediction? Verdict: `GO` / `INCONCLUSIVE` / `NO_GO`.
- **TG** (Transportability Gap) — does an RNA→protein model trained in one
  cohort/condition remain informative in another? Verdict: `TRANSPORTABLE` /
  `PARTIALLY_TRANSPORTABLE` / `NON_TRANSPORTABLE`.

Both share the M1 core: stratum-conditioned Ridge regression, LOOCV /
matched-subsample CV, restricted-permutation nulls, and simulation-calibrated
decision thresholds. All random generators are seeded (`default_rng(42)`); the
full pipeline is **bit-identical across 16 parallel workers** and locked by a
three-layer golden gate (source fingerprint → environment → golden CSV).

## Quick start

```bash
git clone https://github.com/gongzhuozhi/cordiag.git
cd cordiag

# 1. Environment (scipy must stay < 1.18)
conda env create -f environment.yml
conda activate cordiag

# 2. Verify (~5 min)
bash run.sh

# 3. Full golden regression (~30 min, 16 cores)
bash run.sh --full
```

## Package layout

```
cordiag/
├── run.sh                    ← one-command reproduction
├── cordiag/                  ← algorithm package
│   ├── cordiag/              ←   m1.py · zpg.py · tg.py · calibration.py · cli.py
│   └── tests/                ←   12 passed + 2 skipped
├── data/                     ← preprocessed inputs (10-mouse + TCGA-TGCT)
├── output/                   ← frozen golden outputs
├── environment.yml           ← conda env (scipy < 1.18)
├── requirements-lock.txt     ← exact pip pins
└── CHANGELOG.md
```

## Environment contract (scipy < 1.18)

Golden outputs were generated on numpy 2.2.6 / pandas 2.3.3 / scipy 1.14.1 /
sklearn 1.9.0 / statsmodels 0.14.4. scipy ≥ 1.18 breaks bit-identity for
statsmodels OLS bootstrap.

## License

MIT — see [LICENSE](LICENSE).
