# b-jet / τ-jet overlap studies in HH → bbττ (ATLAS ntuples)

Analysis code for studying the **overlap between reconstructed b-jets and
hadronic τ-jets** in ATLAS ntuples (`AnalysisMiniTree`), and for quantifying how
well the offline identification discriminants (GN2v01 b-tagging, tau ID score)
behave when the two object types point to the same detector region. The final
goal is to improve the identification in the overlap region (first with a manual
combination of the two scores, then with a small neural network) and to check
the impact on signal/background separation.

The physics case is the search for Higgs-boson pair production in the
**HH → bbττ** final state
([latest public ATLAS result](https://arxiv.org/abs/2404.12660)).

> **Language note.** Docstrings and this README are in English. Console output,
> plot labels/legends and notebook comments are largely in Italian (e.g.
> "Coppie", "Soglie Tau ID"). Translating them is a possible cleanup task (see
> [Extending the project](#extending-the-project)).

---

## Table of contents

1. [Project objectives](#project-objectives)
2. [Repository layout](#repository-layout)
3. [How the pieces fit together](#how-the-pieces-fit-together)
4. [Key concepts and conventions](#key-concepts-and-conventions)
5. [Script reference](#script-reference)
6. [Neural-network notebooks (objectives 3.3 and 3.4)](#neural-network-notebooks-objectives-33-and-34)
7. [Configuration (`params.py`)](#configuration-paramspy)
8. [Setup and usage](#setup-and-usage)
9. [Outputs](#outputs)
10. [Data](#data)
11. [Extending the project](#extending-the-project)
12. [Known issues / things to check](#known-issues--things-to-check)
13. [References](#references)

---

## Project objectives

The work is organised in four objectives, mirrored by the prefix of the scripts
(`obj_3_0_…`, `obj_3_1_…`, `obj_3_2_…`).

| ID | Objective | Sample | Where |
|----|-----------|--------|-------|
| **3.0** | Preliminary exploration of the ntuples: branch structure, summary statistics and distributions of the key variables. | signal + background | `obj_3_0_root_file_analysis.py` |
| **3.1** | Understand the correlation between b-jet and τ-jet identification by **quantifying the overlap** between the two reconstructed object types, and **comparing their kinematics and identification discriminants**. | signal (GGF) | `obj_3_1_*.py` |
| **3.2** | **Quantify the classification quality**, first separately for b-jets and τ-jets, then by **manually combining** the two discriminants in case of overlap. | signal (GGF) | `obj_3_2_*.py` |
| **3.3** | **Optimise the discriminant definition** for b-jets and τ-jets with a simple NN acting on overlapping (jet, τ) pairs. | signal (GGF) | `overlap_pairs_dataset_builder.py`, `notebooks/` |
| **3.4** | Study **signal vs background separation** based on the count of objects selected with the b-jet and τ-jet discriminants. | signal + tt̄ | Partly: the NN of 3.3 is evaluated on the tt̄ pair dataset (`EVAL_bkg_tt`, see the notebooks). `[TODO: object-counting study, status]` |

---

## Repository layout

```
obj_3_datas/
├── src/                                   # all analysis code (see Script reference)
│   ├── params.py                          # single source of truth for configuration
│   ├── obj_3_0_root_file_analysis.py      # 3.0  ntuple exploration
│   ├── obj_3_1_geometric_overlap.py       # 3.1  ΔR overlap study + shared utilities
│   ├── obj_3_1_overlap_kinematics.py      # 3.1  pair kinematics, all pairs within ΔR
│   ├── obj_3_1_overlap_kinematics_no_reps.py  # 3.1 same, one-to-one greedy matching
│   ├── obj_3_1_overlap_metadatas.py       # 3.1  pair metadata
│   ├── obj_3_1_overlap_met_tau.py         # 3.1  MET-based tau variables (MET projection, mT)
│   ├── obj_3_1_overlap_scores.py          # 3.1  identification scores in overlap
│   ├── obj_3_2_bjet_quantile_analysis.py  # 3.2  b-jet (GN2v01 quantile) classification quality
│   ├── obj_3_2_tau_score_analysis.py      # 3.2  tau-ID classification quality
│   ├── obj_3_2_corr_scores_overlap.py     # 3.2  b-tag vs tau-ID correlation in overlap
│   ├── overlap_pairs_dataset_builder.py   # 3.3  builds the (jet, tau) pair dataset for the NN
│   ├── other_code/                        # diagnostics and legacy helpers
│   │   ├── diagnostics_match_truth_vs_reco_labels.py
│   │   ├── diagnostics_tau_vs_taujet.py
│   │   ├── truth_vs_reco_params.py        # truth labelling + plotting helpers
│   │   └── root_file_analysis.cpp         # C++ ROOT exploration (original version)
│   └── output/                            # outputs written by the scripts in src/
│       ├── discriminance_analysis/
│       └── obj_3.1/
├── notebooks/
│   ├── overlap_pairs_preprocessing.ipynb  # preprocessing of the pair dataset
│   ├── DNN_training.ipynb                 # 3.3/3.4: MLP training, focal-loss sweep, tt̄ evaluation
│   └── Transformer_training.ipynb         # 3.3: transformer model
├── root_datasets/                         # input ntuples (not versioned, see Data)
│   ├── HH_bbtt/                           # signal: GGF, 14 files
│   └── bkg_tt/                            # background: ttbar, 9 files
├── outputs_datasets_analysis/             # curated results of the dataset studies
│   ├── outputs_HH_bbtautau/               # discriminance_analysis, root_files_analysis, taggers_performance
│   └── outputs_tt/                        # discriminance_analysis
└── outputs_training/                      # results of the NN trainings
    ├── DNN/                               # HH_bbtt_dataset, EVAL_bkg_tt
    └── Transformer/                       # HH_bbtt_dataset, EVAL_bkg_tt
```

> The scripts run as flat modules (they import each other by name, e.g.
> `from obj_3_1_geometric_overlap import …`), so run them from `src/` or put
> `src/` on `PYTHONPATH`. The `obj_` prefix is what makes the file names valid
> Python module names (a name cannot start with a digit).
>
> Descriptions of `obj_3_1_overlap_kinematics`, `_metadatas`, `_met_tau`,
> `_scores`, `overlap_pairs_dataset_builder`, `other_code/*`, the preprocessing
> and Transformer notebooks are inferred from file names and cross-references
> only (`[TODO: check]`).

---

## How the pieces fit together

```
                    root_datasets/  (signal GGF, background ttbar)
                             │
      3.0  ntuple exploration  (branches, statistics, distributions)
                             │
                             ▼
      3.1  overlap of reco jets & taus
           ├─ how many?      geometric ΔR study        (obj_3_1_geometric_overlap)
           ├─ which kinematics?  per truth category     (obj_3_1_overlap_kinematics*)
           └─ which scores?                             (obj_3_1_overlap_scores)
                             │
                             ▼
      3.2  classification quality
           ├─ b-jet only  (GN2v01 quantile)             (obj_3_2_bjet_quantile_analysis)
           ├─ tau only    (tau ID score)                (obj_3_2_tau_score_analysis)
           └─ manual combination in the overlap region  (obj_3_2_corr_scores_overlap)
                             │
                             ▼
      3.3  overlap_pairs_dataset_builder ──► notebooks/  (MLP, Transformer on pair features)
                             │
                             ▼
      3.4  evaluation on the ttbar background + signal/background separation
```

All scripts share the same building blocks:

* `params.py` for every configurable quantity (paths, branch names, thresholds,
  plot settings);
* the utilities defined in `obj_3_1_geometric_overlap.py`: `section`,
  `load_files`, `get_analysis_selection`, `delta_phi`, `delta_r`, `summarize`;
* truth labelling and plotting helpers from `truth_vs_reco_params.py`;
* the external, public package
  [`flavour_tag_ml`](https://github.com/giumont/flavour_tagging) (repository
  `flavour_tagging`), to be installed with `pip` (see
  [Setup and usage](#setup-and-usage)), for ROC curves, working points, dataset
  classes, models and training loops.

---

## Key concepts and conventions

**Analysis-level objects.** Only jets and taus flagged by
`params.JET_IS_ANALYSIS_BRANCH` / `params.TAU_IS_ANALYSIS_BRANCH` (≠ 0) are
used. In the preliminary study the tau flag is true for 100% of taus and the jet
flag for 99.541% of jets, so the impact is small. Optional extra selections
(`JET_SELECTION_MODE`, `TAU_SELECTION_MODE`) restrict jets to the GN2v01
`FixedCutBEff_85` working point and taus to the 85% working point of the tau
score (`TAU_SCORE_WP85_THRESHOLD`).

**Overlap.** A (reco jet, reco τ) pair is *overlapping* if
ΔR = √(Δη² + Δφ²) is below a threshold (`DR_THRESHOLDS` in the geometric study,
`DR_THRESHOLD_KINEMATICS`, `DR_THRESHOLD_CORR` in the later steps). The study is
purely reco-based (no truth requirement), because the downstream overlap
removal acts on reco quantities only. Two pairing strategies exist:

* *all pairs* (`obj_3_1_overlap_kinematics.py`): every combination within the
  threshold, so one jet can appear in several pairs;
* *no repetitions* (`obj_3_1_overlap_kinematics_no_reps.py`): global greedy
  matching per event — repeatedly take the pair with the absolute minimum ΔR,
  remove both objects, stop when the minimum residual ΔR is above threshold.
  Each jet and each tau is in at most one pair, and the result does not depend
  on the storage order of the objects.

**Orphans.** Selected jets in events with no selected tau (and vice versa) can
form no pair; they are counted and reported separately.

**Truth categories.** Each pair is classified by the truth of its two members
(first letter: jet, second letter: tau; T = true, F = fake):

| Key | Short | Meaning |
|-----|-------|---------|
| `a_jet_true_tau_fake`  | **TF** | true b-jet, fake τ |
| `b_jet_false_tau_true` | **FT** | fake jet, true τ |
| `c_jet_true_tau_true`  | **TT** | true b-jet, true τ |
| `d_jet_false_tau_false`| **FF** | fake jet, fake τ |

With `AGGREGATE_CATEGORIES = True`, TT and FF are merged into `other`. The NN
notebooks use the same four classes as labels: **FF=0, FT=1, TF=2, TT=3**.
A jet is a true b-jet if `JET_TRUTH_LABEL_BRANCH == JET_TRUTH_LABEL_B_VALUE`
(`HadronConeExclTruthLabelID == 5`); a tau is a true hadronic tau if its
truth-match branch says so (`TAU_TRUTH_MATCH_BRANCH`, `tau_truth_IsHadronicTau`),
or via geometric matching with truth taus, depending on `TRUTH_MODE_TAU`.

**b-tag quantile (GN2v01).** `params.JET_SCORE_BRANCH` is *not* a raw score but
a discrete quantile assigned by ATLAS from a reference calibration. Bin
*b* ∈ {1,…,6} corresponds to a nominal b-jet efficiency
`JET_QUANTILE_BIN_NOMINAL_EFF[b]` = 97, 90, 85, 80, 75, 70 % for the cut
`quantile ≥ b`; `-1` means "not available". Because of this, correlation with
the tau score is measured with the **correlation ratio η** (treats the
quantile as categorical, so `-1` is handled) together with a **Spearman ρ**
restricted to bins > 0.

**Working points.** Thresholds giving the target signal efficiencies
`TARGET_EFFICIENCIES` (%). For the tau score the threshold is the `100 − target`
percentile of the score of true hadronic taus at analysis level (not only those
in overlap); the tau-analysis and b-jet scripts obtain them from the ROC curve
via `compute_working_points` (`flavour_tag_ml.graphics`). The same family of
targets is used for both discriminants, but the numerical thresholds are
specific to each one. Score direction: higher = more signal-like (to be checked
on the plots).

---

## Script reference

### `params.py`
Central configuration. Nothing should be hard-coded elsewhere; see
[Configuration](#configuration-paramspy).

### `obj_3_0_root_file_analysis.py` — ntuple exploration (3.0)
Lists all branches grouped by family (reco jets, reco taus, truth objects,
bbττ analysis-level variables, …) using `params.BRANCH_PREFIX_GROUPS`, then
prints statistics (mean, std, min, max, NaN fraction, objects per event) and
saves a histogram for each branch in `params.EXPLORATION_KEY_BRANCHES`.
Works on both signal and background files.

```bash
python3 obj_3_0_root_file_analysis.py ../root_datasets/HH_bbtt/output_GGF_mc23a_bypass_noOR_000001.root [n_entries_cap]
python3 obj_3_0_root_file_analysis.py ../root_datasets/bkg_tt/output_TTBAR_mc23a_bypass_noOR_000001.root [n_entries_cap]
```

### `obj_3_1_geometric_overlap.py` — geometric jet ↔ tau matching (3.1)
First step of the correlation study, and the module that holds the **shared
utilities** used by the other scripts (`section`, `file_path`, `load_files`,
`get_analysis_selection`, `delta_phi`, `delta_r`, `summarize`). For every event
it builds the ΔR(jet, τ) matrix over all combinations and uses it to:

1. build the distribution of the minimum ΔR **per jet** (distance to the closest
   tau) and **per tau** (distance to the closest jet), plus the distribution
   over all pairs;
2. find where the distributions change slope (shoulder), to justify an operating
   overlap threshold: `find_perfect_overlap_threshold` returns the largest ΔR
   below which the three distributions coincide bin by bin (every pair is
   mutually the closest one), and `print_shoulder_table` prints binned densities
   and cumulative fractions;
3. count, for each candidate threshold in `params.DR_THRESHOLDS` (e.g. 0.2, 0.3,
   0.4) and for the perfect-overlap one, how many jets, taus and pairs overlap.

Internal consistency checks: files with missing branches are skipped with a
message; selected objects = orphans + objects in events with ≥ 1 partner; the
number of entries of the min-ΔR distributions equals the number of non-orphan
objects. Optionally saves η and φ histograms (`PLOT_ETA_PHI`). Outputs go to
`params.OUTPUT_DIR_DR`.

### `obj_3_1_overlap_kinematics_no_reps.py` — pair kinematics, one-to-one (3.1)
Builds one-to-one jet–τ pairs with the global greedy matching described above,
checks that no object is repeated (`check_no_repetition`), splits the pairs by
truth category and saves (i) one histogram per jet/tau/pair variable with a
curve per category and (ii) scatter plots ΔR vs jet pT and ΔR vs τ pT. Output
goes to `params.OUTPUT_DIR_NO_REPS` so that it does not overwrite the
all-pairs variant. Key functions: `greedy_match_no_reps_indices`,
`build_matched_pair_features`, `categorize_matched_pairs`,
`merge_pair_kinematics`.

### `obj_3_2_bjet_quantile_analysis.py` — b-jet classification quality (3.2)
Checks whether the *nominal* efficiencies of the GN2v01 quantile bins hold on
this dataset. Efficiency at bin *b* is P(quantile ≥ b | true b-jet). Parts:

* **A** – properties of the quantile, overall and per subsample; real
  efficiency and fake acceptance per bin;
* **B** – real vs nominal efficiency per bin, with plot
  (`jet_gn2_quantile_calibration.png`);
* **C** – independent check of the ready-made `FixedCutBEff_85` working point;
* **D** – ROC curve, recalibrated working points for `TARGET_EFFICIENCIES`,
  and evaluation plots/metrics (score distribution, background rejection,
  efficiency vs threshold, confusion matrix) at the 85% working point.

Outputs in `params.OUTPUT_DIR_BJET_TAGGER`. Requires the `flavour_tag_ml`
package (ROC curve, working points, evaluation plots).

### `obj_3_2_tau_score_analysis.py` — tau classification quality (3.2)
Counterpart of the b-jet script for the tau score
(`params.TAU_SCORE_ANALYSIS_BRANCH`). Signal = analysis-level taus with
`TAU_TRUTH_MATCH_BRANCH != 0` (true hadronic tau); "fake" is every other
analysis-level tau (it may contain real taus not truth-matched at visible level,
or jets), used only as reference population. Parts:

* **A** – properties of the score, overall and per subsample;
* **B** – efficiency vs threshold scan (`params.N_SCAN_POINTS` points), with
  the working points drawn as vertical lines;
* **C** – ROC curve, working points for `TARGET_EFFICIENCIES` (same logic as
  the `FixedCutBEff` family of the b-tagger) and evaluation plots and metrics
  at the 85% working point. The background-rejection plot uses the same
  efficiency range (0.7–1) available for the b-tagger, to allow a direct
  comparison.

Outputs in `params.OUTPUT_DIR_TAU_TAGGER`.

### `obj_3_2_corr_scores_overlap.py` — b-tag vs tau-ID correlation (3.1/3.2)
For every (jet, τ) pair with ΔR < `DR_THRESHOLD_CORR`, relates the b-tag
quantile to the tau ID score. Pairs are split in the four truth categories; for
each, a violin plot shows the tau score distribution in every quantile bin,
with the tau working points as dashed horizontal lines and a box reporting the
number of pairs, η and Spearman ρ (bins > 0). This provides the input for the
manual combination of the two discriminants. Output:
`correlation_<score>_vs_btag_dr<ΔR>_truth_types.png` in
`params.OUTPUT_DIR_CORR_SCORES` (under `output/obj_3_2`).

### Other 3.1 scripts `[TODO: refine]`
* `obj_3_1_overlap_kinematics.py` — all-pairs variant of the kinematics study.
* `obj_3_1_overlap_metadatas.py` — metadata of overlapping pairs.
* `obj_3_1_overlap_met_tau.py` — MET-based tau variables (`compute_tau_met_proj`,
  `compute_tau_mt`), used by the kinematics scripts.
* `obj_3_1_overlap_scores.py` — identification scores of overlapping objects.

### `overlap_pairs_dataset_builder.py` (3.3) `[TODO: refine]`
Main orchestrator that builds the per-pair dataset (features + four-class truth
label) used by the notebooks; it reuses the pair-building code of the kinematics
scripts. The notebooks read its normalised output as `X_*_normalized.npy` /
`y_*_normalized.npy` (train/val/test) plus `pair_feature_names.npz`.

### `other_code/`
Diagnostics (truth vs reco labels, tau vs tau-jet), the truth/plotting helper
module `truth_vs_reco_params.py`, and the original C++ exploration
`root_file_analysis.cpp`.

---

## Neural-network notebooks (objectives 3.3 and 3.4)

The notebooks are written to run on **Google Colab** (GPU optional), reading the
datasets from Google Drive.

**Dependencies on other code.** Each notebook needs two GitHub repositories:

* [`flavour_tagging`](https://github.com/giumont/flavour_tagging) (public) —
  provides the package `flavour_tag_ml` (`data`, `merge_datasets`, `models`,
  `training`, `utils`, `graphics`, `fine_tuning`). Install it in the first cell:
  `!pip install "git+https://github.com/giumont/flavour_tagging.git"`.
* `overlap_resolver` (this project) — cloned by the notebook, whose `src/`
  folder is added to `sys.path` so that its modules are imported as flat
  top-level modules. The notebooks also push their outputs (figures,
  `config.yaml`, text report) back to this repository via `save_run_outputs`.
  A GitHub token stored as Colab secret `GITHUB_TOKEN_OVERLAP_RESOLVER` is
  needed to push, and to clone if the repository is private.

### `DNN_training.ipynb`

| Step | What it does |
|------|--------------|
| Data | Loads normalised memory-mapped arrays `X/y_{train,val,test}_normalized.npy` from `INPUT_DIR` (signal `…/np_arrays/HH_bbtt`, background `…/np_arrays/bkg_tt`). Optional balanced sub-sampling (`N_TRAIN`, `N_VAL`, `N_TEST`). |
| Problem | 4-class classification of the overlapping pair: FF=0, FT=1, TF=2, TT=3. The classes are strongly imbalanced (roughly FF 10%, FT 73%, TF 16%, TT 0.65%, as noted in the notebook). |
| Model | `MLPMultiTagger`: hidden layers (32, 16), ReLU, LayerNorm, dropout 0.1 (`MODEL_CONFIG`). |
| Loss | `CategoricalFocalLoss`, optionally with class weights computed on the training set only (`WEIGHTED_LOSS`). |
| Training | AdamW (lr 1e-3, weight decay 1e-4), `ReduceLROnPlateau` scheduler, batch size 2048, max 50 epochs, early stopping (patience 10), best model selected on validation macro average precision (`BEST_MODEL_METRIC = "ap_macro"`). Checkpoints allow resuming. |
| Evaluation | `run_evaluation_suite`: normalised confusion matrix, one-vs-rest ROC curves, multiclass score distributions, one-vs-rest working points at 70/75/80/85/90/97% efficiency, metrics report. Figures, `config.yaml` and report are pushed to the repository under `DNN/<dataset>` (→ `outputs_training/DNN/`). |
| Focal-loss sweep | Grid over `alpha_power` ∈ {0, 0.25, 0.5, 0.75, 1} × `gamma` ∈ {0, 1, 2, 3, 5} with a fixed seed (`sweep_hyperparameters`), summarised in heatmaps of validation macro AP and TT-class AP. |
| Objective 3.4 evaluation | Loads the best weights and evaluates the trained network, without any further training, on the tt̄ background pair dataset (train+val+test merged into a single external evaluation set). Same plots/metrics as above, saved with the tag `EVAL_bkg_tt` (→ `outputs_training/DNN/EVAL_bkg_tt/`). |

Every run is identified by a `RUN_TAG` built from architecture and training
hyper-parameters, used in file names of weights, checkpoints, figures and
reports, so results of different configurations do not overwrite each other.

**To reproduce or modify a run:** edit the *GLOBAL PARAMS* cell (input
directory, class names, `HIDDEN_DIMS`, loss, optimizer, `MODEL_DIR`,
`GRAPHICS_DIR`, output subfolder) and run the notebook top to bottom. To
evaluate on the background, point `INPUT_DIR` to the tt̄ arrays and run the
"objective 3.4" cells.

### `overlap_pairs_preprocessing.ipynb`, `Transformer_training.ipynb` `[TODO]`
Preprocessing of the pair dataset (normalisation, splits) and an alternative
transformer model. Presumably structured like the DNN notebook; results in
`outputs_training/Transformer/`. `[TODO: describe]`

---

## Configuration (`params.py`)

Main groups of settings (names as used in the code):

| Group | Examples |
|-------|----------|
| Input | `ROOT_DIR`, `FILE_PREFIX`, `FILE_SUFFIX`, `N_FILES`, `TREE_NAME`, `N_ENTRIES_CAP` |
| Branch names | `JET_*_BRANCH`, `TAU_*_BRANCH`, `TRUTH_TAU_*_BRANCH`, `JET_SCORE_BRANCH`, `TAU_ID_SCORE_BRANCH`, `TAU_SCORE_ANALYSIS_BRANCH`, `TAU_EFF_SCORE_BRANCH`, `JET_BTAG_BRANCH`, `JET_TRUTH_LABEL_BRANCH`, `TAU_TRUTH_MATCH_BRANCH` |
| Selections | `JET_SELECTION_MODE` (`all`/`btag85`), `TAU_SELECTION_MODE` (`all`/`score85`), `TAU_SCORE_WP85_THRESHOLD`, `TRUTH_MODE_TAU` (`label`/`geometric`) |
| Overlap | `DR_THRESHOLDS`, `DR_HIST_MIN/MAX/BINSIZE`, `DR_THRESHOLD_KINEMATICS`, `DR_THRESHOLD_CORR`, `CORR_JET_SCORE_MIN`, `CORR_TAU_SCORE_MIN`, `PLOT_ETA_PHI` |
| b-tag calibration | `JET_QUANTILE_BIN_NOMINAL_EFF`, `JET_QUANTILES_MAP`, `BTAG_WP85_NOMINAL_EFF`, `JET_TRUTH_LABEL_B_VALUE` |
| Working points | `TARGET_EFFICIENCIES`, `WP_COLORS`, `N_SCAN_POINTS` |
| Plot settings | `AGGREGATE_CATEGORIES`, `NORMALIZE_PAIR_HISTOGRAMS`, `NORMALIZE_DR_HISTOGRAMS`, `CATEGORY_*`, `*_VARIABLES_NO_REPS`, `VARIABLE_PLOT_CONFIG_NO_REPS`, `PT_HIST_MAX_NO_REPS`, `M_HIST_MAX` |
| Exploration | `BRANCH_PREFIX_GROUPS`, `EXPLORATION_KEY_BRANCHES`, `EXPLORATION_N_BINS` |
| Output paths | `OUTPUT_DIR_ROOT_EXPLORATION`, `OUTPUT_DIR_DR`, `OUTPUT_DIR_NO_REPS`, `OUTPUT_DIR_BJET_TAGGER`, `OUTPUT_DIR_TAU_TAGGER`, `OUTPUT_DIR_CORR_SCORES` |

To point the code at different files, change the input paths here; to study a
different tagger, change the score branch and the corresponding calibration
map. Input files are located as `ROOT_DIR / f"{FILE_PREFIX}{index:02d}{FILE_SUFFIX}"`
for `index = 1 … N_FILES`.

---

## Setup and usage

**Requirements:** Python 3.8, `uproot`, `awkward`, `numpy`,
`scipy`, `matplotlib`, `torch`, and the public package
[`flavour_tag_ml`](https://github.com/giumont/flavour_tagging) (ROC curves,
working points, datasets, models, training loops), which is needed by the
classification-quality scripts (`obj_3_2_bjet_quantile_analysis`,
`obj_3_2_tau_score_analysis`) and by the notebooks.

```bash
pip install uproot awkward numpy scipy matplotlib torch     # [TODO: add requirements.txt]
pip install "git+https://github.com/giumont/ML4FlavourTagging"
```

If you prefer not to install the package (or you want to edit it), clone it and
put its `src/` folder on the Python path instead:

```bash
git clone https://github.com/giumont/ML4FlavourTagging
export PYTHONPATH=$PYTHONPATH:$(pwd)/flavour_tagging/src
```

Then run the scripts from `src/`:

```bash
cd src
python3 obj_3_0_root_file_analysis.py <file.root> [n_entries_cap]
python3 obj_3_1_geometric_overlap.py
python3 obj_3_1_overlap_kinematics_no_reps.py
python3 obj_3_2_bjet_quantile_analysis.py
python3 obj_3_2_tau_score_analysis.py
python3 obj_3_2_corr_scores_overlap.py
```

Scripts other than `obj_3_0` take no command-line arguments: they read files and
settings from `params.py` and process all input files found there. Files that
miss a required branch are skipped (with a warning in most scripts).

Suggested order for a first pass: **3.0 → 3.1 → 3.2 → dataset builder →
notebooks**.

---

## Outputs

* Scripts write plots (and console summaries) to the folders defined in
  `params.py`, mostly under `src/output/`.
* `outputs_datasets_analysis/` collects curated results of the dataset studies
  (signal: `discriminance_analysis`, `root_files_analysis`,
  `taggers_performance`; background: `discriminance_analysis`).
* `outputs_training/` collects the NN training/evaluation results (figures,
  `config.yaml`, text report, saved arrays) for the DNN and the Transformer,
  on the signal dataset (`HH_bbtt_dataset`) and on the tt̄ background
  (`EVAL_bkg_tt`). The notebooks push them here automatically.

---

## Data

Ntuples (`AnalysisMiniTree`, campaign mc23a, "bypass noOR" mode):

* **Signal** – HH → bbττ, gluon–gluon fusion (GGF):
  `/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/GGF-mc23a-bypass-noOR_mode/`
  (files `output_GGF_mc23a_bypass_noOR_0000NN.root`, 14 files locally).
* **Background** – fully hadronic tt̄:
  `/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/TTBAR-mc23a-bypass-noOR_mode`
  (files `output_TTBAR_mc23a_bypass_noOR_0000NN.root`, 9 files locally).

The NN notebooks do not read the ROOT files: they use the
normalised `.npy` pair arrays (stored on Google Drive) produced by the dataset
builder and the preprocessing notebook.

---

## Extending the project

* **New variable in the overlap studies:** add the branch to `params.py`, add it
  to the variable lists (`JET_VARIABLES_NO_REPS`, …) with range, bin step and
  label, and add an entry to `VARIABLE_PLOT_CONFIG_NO_REPS` if it needs a log
  scale or a different output folder.
* **New selection or working point:** extend the `*_SELECTION_MODE` handling in
  the `main()` of the scripts and add the needed branch to the branch list;
  change `TARGET_EFFICIENCIES` for a different family of working points.
* **Different truth definition:** see `label_jets_and_taus` in
  `truth_vs_reco_params.py` and `TRUTH_MODE_TAU`.
* **New pairing algorithm:** implement it like
  `greedy_match_no_reps_indices` (return local jet/tau indices per event) and
  reuse `build_matched_pair_features` and `categorize_matched_pairs`.
* **Other tagger or tau ID:** change `JET_SCORE_BRANCH` / `TAU_ID_SCORE_BRANCH` /
  `TAU_SCORE_ANALYSIS_BRANCH` and the calibration maps in `params.py`.
* **New NN architecture or loss:** register it in `flavour_tag_ml.models` /
  `flavour_tag_ml.training` and select it through `MODEL_CONFIG` /
  `CRITERION` in the notebook; the focal-loss sweep can be reused for other
  hyper-parameters.
* **Objective 3.4:** (signal vs background from object counting) for the current choice of features and preprocessing, the models trained on the signal dataset do not generalize properly on the background dataset. **A possible direction could be training the network on mixed datasets**.

---

## Known issues / things to check

* **Two truth conventions for taus.** The tau classification script treats
  `TAU_TRUTH_MATCH_BRANCH != 0` as true tau, the correlation script uses
  `== TAU_TRUTH_TRUE_VALUE`. Make sure the two are equivalent.
* **Hard-coded labels in plots.** The tau efficiency plot legend hard-codes
  `tau_truth_IsHadronicTau`, and the b-jet calibration plot hard-codes
  `HadronConeExclTruthLabelID==5`; they do not follow the parameters.
* **MET-based variables.** `tau_met_proj` and `tau_mt` are implemented, but the
  `main()` of the no-repetition script does not pass `met`/`met_phi`, so they
  are not computed there.
* **Notebooks:** Google Drive paths (`/content/drive/MyDrive/overlap_resolver/…`)
  and the Colab secret used to clone/push `overlap_resolver` are hard-coded;
  running elsewhere requires editing the first cells. The cell that clones
  `flavour_tagging` with a token and `sys.path` should be replaced by the
  `pip install` line above.

---

## References

* ATLAS collaboration, search for HH → bbττ: <https://arxiv.org/abs/2404.12660>
* Example data-handling code: <https://gitlab.cern.ch/parodi/b-tau_studies>

## Author / contact

Giulia Montagnani
email: `giu.mont04@gmail.com`