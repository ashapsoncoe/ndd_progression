# ndd_progression

Code for event-based modelling of early and preclinical dementia in UK Biobank, using [SuStaIn](https://github.com/ucl-pond/pySuStaIn) (Subtype and Stage Inference) to identify distinct spatiotemporal patterns of grey-matter atrophy and structural connectivity loss that precede a dementia diagnosis.

This repository accompanies the paper *"Event-based modelling of early and preclinical dementia reveals two distinct patterns of atrophy and connectivity loss."*

## Overview

Dementia cases (and matched controls) from UK Biobank are used to:

1. Derive region-wise z-scores of grey-matter volume (and, in a secondary analysis, structural connectivity) relative to the control distribution, adjusting for age, sex, total intracranial volume (TIV) and scanning centre.
2. Fit SuStaIn models to these z-scores to infer a data-driven ordering ("event sequence") of abnormality across regions, and to identify distinct subtypes of patients who share a common ordering.
3. Characterise the resulting subtypes (stage distributions, demographic and clinical associations, ApoE4 status, cross-validated model selection) and visualise the inferred atrophy/connectivity progression.

## Requirements

- Python 3.8+
- [`pySuStaIn`](https://github.com/ucl-pond/pySuStaIn)
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`
- `matplotlib`, `seaborn`, `plotly` (for figures/visualisation)

UK Biobank data are not included in this repository and must be obtained separately under an approved application; scripts expect local CSV extracts (paths are set as variables at the top of each script and will need updating to your own environment).

## Pipeline

The scripts are intended to be run in roughly the order below. Each script has its input/output paths set as plain variables near the top — update these to match your own directory structure before running.

### 1. Data preparation

**`z_score_matched_structural_connectivity_and_atrophy_data.py`**
Builds a matched case/control dataset combining regional grey-matter atrophy and structural connectivity (fibre bundle capacity) measures, matching each dementia case to a control on sex, age, ethnicity, handedness and scanning centre. Used for the secondary, per-connection SuStaIn analysis.

**`get_z_score_tables_by_regression.py`**
Fits a regression model (Gaussian GLM by default; negative binomial/Poisson/gamma also supported) of each region's raw measure against age, age², TIV, sex and scanning centre in controls, then expresses each case's and control's value as a z-score relative to that adjusted control distribution. Includes diagnostic checks (Breusch-Pagan, normality of residuals, VIF) and produces the z-score tables consumed by the SuStaIn scripts below.

### 2. SuStaIn model fitting

**`sustain_run.py`**
Main analysis: fits Z-score SuStaIn models (1 to `N_S_max` subtypes) to the 23-region grey-matter atrophy z-scores for all dementia cases, using a single global z-score event threshold per region (default 1,000,000 MCMC iterations, 25 start points). Also runs 10-fold cross-validation to compute CVIC for model selection.

**`sustain_run_per_connection.py`**
Secondary, per-connection analysis: fits a 2-subtype SuStaIn model separately to each case/control-matched region or connection (atrophy and structural connectivity), with an option to filter which phenotypes are included by effect size (Cohen's d) between subtypes.

### 3. Post-hoc analysis and figures

**`analysis_of_dementia_subtypes.py`**
Loads the SuStaIn results and produces the main subtype-characterisation analysis: stage/subtype assignment, association of subtype with clinical and demographic variables (including ApoE4 status via chi-square/proportions tests), and summary tables and plots.

**`plot_cvic_1_vs_2_subtypes.py`**
Plots CVIC values comparing the 1- and 2-subtype per-connection SuStaIn models, and summarises which regions/connections drive the ordering in each subtype.

**`ndd_progression_mapping_with_slider.py`**
Interactive 3D visualisation (Plotly) of the inferred SuStaIn event ordering, mapping each region onto its approximate MNI coordinates with a slider to step through disease stages.

## Notes

- Scripts assume Windows-style paths (`\\`) in places; adjust separators if running on Linux/macOS.
- `Z_max` and the single-event z-score threshold used in `sustain_run.py`/`analysis_of_dementia_subtypes.py` are derived empirically from the case data (95th percentile of pooled positive z-scores, and the mean case z-score, respectively) — see the Methods section of the paper for details and a sensitivity analysis across alternative thresholds.
- Raw UK Biobank data files referenced by these scripts are not included and must be supplied locally under your own data access agreement.

## Citation

If you use this code, please cite the accompanying paper (citation details to be added on publication).
