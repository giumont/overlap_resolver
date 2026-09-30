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

> **Language note.** Docstrings and this README are in English. Console output
> and some plot labels/legends are in Italian (e.g. "Coppie", "Soglie Tau ID").
> Translating them is a possible cleanup task (see [Extending the
> project](#extending-the-project)).

---

## Table of contents

1. [Project objectives](#project-objectives)
2. [Repository layout](#repository-layout)
3. [How the pieces fit together](#how-the-pieces-fit-together)
4. [Key concepts and conventions](#key-concepts-and-conventions)
5. [Script reference](#script-reference)
6. [Configuration (`params.py`)](#configuration-paramspy)
7. [Setup and usage](#setup-and-usage)
8. [Outputs](#outputs)
9. [Data](#data)
10. [Extending the project](#extending-the-project)
11. [Known issues / things to check](#known-issues--things-to-check)
12. [References](#references)

---

## Project objectives

The work is organised in four objectives, mirrored by the numeric prefix of the
scripts (`3_0_…`, `3_1_…`, `3_2_…`).

| ID | Objective | Sample | Status in this repo |
|----|-----------|--------|---------------------|
| **3.0** | Preliminary exploration of the ntuples: branch structure, summary statistics and distributions of the key variables. | signal + background | `3_0_root_file_analysis.py` |
| **3.1** | Understand the correlation between b-jet and τ-jet identification by **quantifying the overlap** between the two reconstructed object types, and **comparing their kinematics and identification discriminants**. | signal (GGF) | `3_1_*.py` |
| **3.2** | **Quantify the classification quality**, first separately for b-jets and τ-jets, then by **manually combining** the two discriminants in case of overlap. | signal (GGF) | `3_2_*.py` |
| **3.3** | **Optimise the discriminant definition** for b-jets and τ-jets with a simple NN acting on overlapping pairs. | signal (GGF) | `overlap_pairs_dataset_builder.py`, `notebooks/` |
| **3.4** | Study **signal vs background separation** based on the count of objects selected with the b-jet and τ-jet discriminants. | signal + tt̄ | `[TODO: state where this lives / whether it is done]` |

Objective text (original wording, translated): objective 3.1 compares kinematic
characteristics and reconstructed identification discriminants of overlapping
objects; 3.2 evaluates the classification separately and then with a manual
combination; 3.3 develops a NN that acts in case of overlap; 3.4 uses object
counting for signal/background separation.

---

## Repository layout

```
obj_3_datas/
├── src/                              # all analysis code (see Script reference)
│   ├── params.py                     # single source of truth for configuration
│   ├── 3_0_root_file_analysis.py     # obj 3.0: ntuple exploration
│   ├── 3_1_geometric_overlap.py      # obj 3.1: overlap counting (ΔR)
│   ├── 3_1_overlap_kinematics.py     # obj 3.1: pair kinematics, all pairs within ΔR
│   ├── 3_1_overlap_kinematics_no_reps.py  # obj 3.1: same, one-to-one greedy matching
│   ├── 3_1_overlap_metadatas.py      # obj 3.1: pair metadata
│   ├── 3_1_overlap_met_tau.py        # obj 3.1: MET-based tau variables (MET projection, mT)
│   ├── 3_1_overlap_scores.py         # obj 3.1: identification scores in overlap
│   ├── 3_2_bjet_quantile_analysis.py # obj 3.2: b-jet (GN2v01 quantile) classification quality
│   ├── 3_2_tau_score_analysis.py     # obj 3.2: tau-ID classification quality
│   ├── 3_2_corr_scores_overlap.py    # obj 3.2: b-tag vs tau-ID correlation in overlap
│   ├── overlap_pairs_dataset_builder.py   # obj 3.3: builds the (jet, tau) pair dataset for the NN
│   ├── other_code/                   # diagnostics and legacy helpers
│   │   ├── diagnostics_match_truth_vs_reco_labels.py
│   │   ├── diagnostics_tau_vs_taujet.py
│   │   ├── truth_vs_reco_params.py   # truth labelling + plotting helpers
│   │   └── root_file_analysis.cpp    # C++ ROOT exploration (original version)
│   └── output/                       # outputs written by the scripts in src/
│       ├── discriminance_analysis/
│       └── obj_3.1/
├── notebooks/
│   ├── overlap_pairs_preprocessing.ipynb  # preprocessing of the pair dataset
│   ├── DNN_training.ipynb                 # obj 3.3: dense NN
│   └── Transformer_training.ipynb         # obj 3.3: transformer model
├── root_datasets/                    # input ntuples (not versioned, see Data)
│   ├── HH_bbtt/                      # signal: GGF, 14 files
│   └── bkg_tt/                       # background: ttbar, 9 files
├── outputs_datasets_analysis/        # curated results of the dataset studies
│   ├── outputs_HH_bbtautau/          # discriminance_analysis, root_files_analysis, taggers_performance
│   └── outputs_tt/                   # discriminance_analysis
└── outputs_training/                 # results of the NN trainings
    ├── DNN/                          # HH_bbtt_dataset, EVAL_bkg_tt
    └── Transformer/                  # HH_bbtt_dataset, EVAL_bkg_tt
```

> Descriptions marked with objective numbers for scripts I did not have at
> hand while writing this README (`3_1_geometric_overlap`, `3_1_overlap_metadatas`,
> `3_1_overlap_scores`, `3_2_tau_score_analysis`, `overlap_pairs_dataset_builder`,
> `other_code/*`, the notebooks) come from file names and cross-references
> only. `[TODO]` Please check and refine them.

---

## How the pieces fit together

```
                    root_datasets/  (signal GGF, background ttbar)
                             │
                   ┌─────────┴──────────┐
                   ▼                    │
      3_0  ntuple exploration           │
                   │                    │
                   ▼                    │
      3_1  overlap of reco jets & taus  │
           ├─ how many? (ΔR)            │
           ├─ which kinematics?         │
           └─ which scores?             │
                   │                    │
                   ▼                    │
      3_2  classification quality       │
           ├─ b-jet only  (GN2v01 quantile)
           ├─ tau only    (tau ID score)
           └─ manual combination in the overlap region
                   │                    │
                   ▼                    ▼
      3.3  overlap_pairs_dataset_builder ──► notebooks/ (DNN, Transformer)
                   │
                   ▼
      3.4  signal vs background from object counting
```

All scripts share the same building blocks:

* `params.py` for every configurable quantity (paths, branch names, thresholds,
  plot settings);
* a small utility module (imported as `obj_3_1`) providing `section`,
  `load_files`, `get_analysis_selection`, `delta_r` and `summarize`;
* truth labelling and plotting helpers from `truth_vs_reco_params.py`.

---

## Key concepts and conventions

**Analysis-level objects.** Only jets and taus flagged by
`params.JET_IS_ANALYSIS_BRANCH` / `params.TAU_IS_ANALYSIS_BRANCH` (≠ 0) are
used. Optional extra selections (`JET_SELECTION_MODE`, `TAU_SELECTION_MODE`)
can additionally require the ready-made 85% b-tag working point or a tau score
threshold.

**Overlap.** A (reco jet, reco τ) pair is *overlapping* if
ΔR = √(Δη² + Δφ²) is below a threshold (`DR_THRESHOLD_KINEMATICS`,
`DR_THRESHOLD_CORR`, …). Two pairing strategies exist:

* *all pairs* (`3_1_overlap_kinematics.py`): every combination within the
  threshold, so one jet can appear in several pairs;
* *no repetitions* (`3_1_overlap_kinematics_no_reps.py`): global greedy
  matching per event — repeatedly take the pair with the absolute minimum ΔR,
  remove both objects, stop when the minimum residual ΔR is above threshold.
  Each jet and each tau is in at most one pair, and the result does not depend
  on the storage order of the objects.

**Truth categories.** Each pair is classified by the truth of its two members
(jet true b-jet? tau true hadronic tau?):

| Key | Short | Meaning |
|-----|-------|---------|
| `a_jet_true_tau_fake`  | **TF** | true b-jet, fake τ |
| `b_jet_false_tau_true` | **FT** | fake jet, true τ |
| `c_jet_true_tau_true`  | **TT** | true b-jet, true τ |
| `d_jet_false_tau_false`| **FF** | fake jet, fake τ |

With `AGGREGATE_CATEGORIES = True`, TT and FF are merged into `other`.
A jet is a true b-jet if `JET_TRUTH_LABEL_BRANCH == JET_TRUTH_LABEL_B_VALUE`
(`HadronConeExclTruthLabelID == 5`); a tau is a true hadronic tau if
`TAU_TRUTH_MATCH_BRANCH == TAU_TRUTH_TRUE_VALUE` (or via geometric matching
with truth taus, depending on `TRUTH_MODE_TAU`).

**b-tag quantile (GN2v01).** `params.JET_SCORE_BRANCH` is *not* a raw score but
a discrete quantile assigned by ATLAS from a reference calibration. Bin
*b* ∈ {1,…,6} corresponds to a nominal b-jet efficiency
`JET_QUANTILE_BIN_NOMINAL_EFF[b]` = 97, 90, 85, 80, 75, 70 % for the cut
`quantile ≥ b`; `-1` means "not available". Because of this, correlation with
the tau score is measured with the **correlation ratio η** (treats the
quantile as categorical, so `-1` is handled) together with a **Spearman ρ**
restricted to bins > 0.

**Tau working points.** Score thresholds giving target signal efficiencies
(`TARGET_EFFICIENCIES`, in %) on *all* true hadronic taus at analysis level (not
only those in overlap): threshold = the `100 − target` percentile of the true-τ
score.

---

## Script reference

### `params.py`
Central configuration. Nothing should be hard-coded elsewhere; see
[Configuration](#configuration-paramspy).

### `3_0_root_file_analysis.py` — ntuple exploration (obj. 3.0)
Lists all branches grouped by family (reco jets, reco taus, truth objects,
bbττ analysis-level variables, …) using `params.BRANCH_PREFIX_GROUPS`, then
prints statistics (mean, std, min, max, NaN fraction, objects per event) and
saves a histogram for each branch in `params.EXPLORATION_KEY_BRANCHES`.
Works on both signal and background files.

```bash
python3 3_0_root_file_analysis.py ../root_datasets/HH_bbtt/output_GGF_mc23a_bypass_noOR_000001.root [n_entries_cap]
python3 3_0_root_file_analysis.py ../root_datasets/bkg_tt/output_TTBAR_mc23a_bypass_noOR_000001.root [n_entries_cap]
```

### `3_1_overlap_kinematics_no_reps.py` — pair kinematics, one-to-one (obj. 3.1)
Builds one-to-one jet–τ pairs with the global greedy matching described above,
checks that no object is repeated (`check_no_repetition`), splits the pairs by
truth category and saves (i) one histogram per jet/tau/pair variable with a
curve per category and (ii) scatter plots ΔR vs jet pT and ΔR vs τ pT. Output
goes to `params.OUTPUT_DIR_NO_REPS` so that it does not overwrite the
all-pairs variant. Key functions: `greedy_match_no_reps_indices`,
`build_matched_pair_features`, `categorize_matched_pairs`,
`merge_pair_kinematics`.

### `3_2_bjet_quantile_analysis.py` — b-jet classification quality (obj. 3.2)
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

Depends on the external `flavour_tagging` project
(`params.FLAVOUR_TAGGING_DIR`, module `src.graphics`) for `plot_roc_curve`,
`compute_working_points`, etc.

### `3_2_tau_score_analysis.py` — tau classification quality (obj. 3.2)
Counterpart of the previous script for the tau ID score. `[TODO: confirm scope]`

### `3_2_corr_scores_overlap.py` — b-tag vs tau-ID correlation (objectives 3.1/3.2)
For every (jet, τ) pair with ΔR < `DR_THRESHOLD_CORR`, relates the b-tag
quantile to the tau ID score. Pairs are split in the four truth categories; for
each, a violin plot shows the tau score distribution in every quantile bin,
with the tau working points as dashed horizontal lines and a box reporting the
number of pairs, η and Spearman ρ (bins > 0). This provides the input for the
manual combination of the two discriminants. Output:
`correlation_<score>_vs_btag_dr<ΔR>_truth_types.png` in
`params.OUTPUT_DIR_CORR_SCORES` (under `output/obj_3_2`).

### Other objective 3.1 scripts `[TODO: refine]`
* `3_1_geometric_overlap.py` — geometric quantification of the overlap.
* `3_1_overlap_kinematics.py` — all-pairs variant of the kinematics study.
* `3_1_overlap_metadatas.py` — metadata of overlapping pairs.
* `3_1_overlap_met_tau.py` — MET-based tau variables (`compute_tau_met_proj`,
  `compute_tau_mt`), used by the kinematics scripts.
* `3_1_overlap_scores.py` — identification scores of overlapping objects.

### `overlap_pairs_dataset_builder.py` and `notebooks/` (obj. 3.3) `[TODO: refine]`
The builder produces the per-pair dataset used for the NN; the notebooks
handle preprocessing (`overlap_pairs_preprocessing.ipynb`) and training of a
dense network (`DNN_training.ipynb`) and a transformer
(`Transformer_training.ipynb`). Results are saved in `outputs_training/`,
including an evaluation on the tt̄ background (`EVAL_bkg_tt`).

### `other_code/`
Diagnostics (truth vs reco labels, tau vs tau-jet), the truth/plotting helper
module `truth_vs_reco_params.py`, and the original C++ exploration
`root_file_analysis.cpp`.

---

## Configuration (`params.py`)

Main groups of settings (names as used in the code):

| Group | Examples |
|-------|----------|
| Input | `TREE_NAME`, input directories, `N_ENTRIES_CAP` |
| Branch names | `JET_*_BRANCH`, `TAU_*_BRANCH`, `TRUTH_TAU_*_BRANCH`, `JET_SCORE_BRANCH`, `TAU_ID_SCORE_BRANCH`, `JET_BTAG_BRANCH`, `JET_TRUTH_LABEL_BRANCH`, `TAU_TRUTH_MATCH_BRANCH` |
| Selections | `JET_SELECTION_MODE` (`all`/`btag85`), `TAU_SELECTION_MODE` (`all`/`score85`), `TAU_SCORE_WP85_THRESHOLD`, `TRUTH_MODE_TAU` (`label`/`geometric`) |
| Overlap | `DR_THRESHOLD_KINEMATICS`, `DR_THRESHOLD_CORR`, `CORR_JET_SCORE_MIN`, `CORR_TAU_SCORE_MIN` |
| b-tag calibration | `JET_QUANTILE_BIN_NOMINAL_EFF`, `JET_QUANTILES_MAP`, `BTAG_WP85_NOMINAL_EFF`, `JET_TRUTH_LABEL_B_VALUE` |
| Working points | `TARGET_EFFICIENCIES`, `WP_COLORS` |
| Plot settings | `AGGREGATE_CATEGORIES`, `NORMALIZE_PAIR_HISTOGRAMS`, `CATEGORY_*`, `*_VARIABLES_NO_REPS`, `VARIABLE_PLOT_CONFIG_NO_REPS`, `PT_HIST_MAX_NO_REPS`, `M_HIST_MAX` |
| Exploration | `BRANCH_PREFIX_GROUPS`, `EXPLORATION_KEY_BRANCHES`, `EXPLORATION_N_BINS` |
| Output paths | `OUTPUT_DIR_ROOT_EXPLORATION`, `OUTPUT_DIR_NO_REPS`, `OUTPUT_DIR_BJET_TAGGER`, `OUTPUT_DIR_CORR_SCORES` |
| External code | `FLAVOUR_TAGGING_DIR` |

To point the code at different files, change the input paths here; to study a
different tagger, change the score branch and the corresponding calibration
map.

---

## Setup and usage

**Requirements:** Python 3 (`[TODO: version]`), `uproot`, `awkward`, `numpy`,
`scipy`, `matplotlib` (plus the dependencies of the notebooks, e.g. a deep
learning framework `[TODO: specify]`).

```bash
pip install uproot awkward numpy scipy matplotlib     # [TODO: add requirements.txt]
cd src
python3 3_0_root_file_analysis.py <file.root> [n_entries_cap]
python3 3_1_overlap_kinematics_no_reps.py
python3 3_2_bjet_quantile_analysis.py
python3 3_2_corr_scores_overlap.py
```

Scripts other than `3_0` take no command-line arguments: they read files and
settings from `params.py` and process all input files found there. Files that
miss a required branch are skipped (with a warning in most scripts).

Suggested order for a first pass: **3_0 → 3_1 → 3_2 → dataset builder →
notebooks**.

---

## Outputs

* Scripts write plots (and console summaries) to the folders defined in
  `params.py`, mostly under `src/output/`.
* `outputs_datasets_analysis/` collects curated results of the dataset studies
  (signal: `discriminance_analysis`, `root_files_analysis`,
  `taggers_performance`; background: `discriminance_analysis`).
* `outputs_training/` collects the NN training/evaluation results for the DNN
  and the Transformer, on the signal dataset and on the tt̄ background.

---

## Data

Ntuples (`AnalysisMiniTree`, campaign mc23a, "bypass noOR" mode):

* **Signal** – HH → bbττ, gluon–gluon fusion (GGF):
  `/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/GGF-mc23a-bypass-noOR_mode/`
  (files `output_GGF_mc23a_bypass_noOR_0000NN.root`, 14 files locally).
* **Background** – fully hadronic tt̄:
  `/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/TTBAR-mc23a-bypass-noOR_mode`
  (files `output_TTBAR_mc23a_bypass_noOR_0000NN.root`, 9 files locally).

The ROOT files are not part of the repository; copy them under
`root_datasets/HH_bbtt/` and `root_datasets/bkg_tt/`, or change the paths in
`params.py`. Reference code for handling these data:
<https://gitlab.cern.ch/parodi/b-tau_studies>.

---

## Extending the project

* **New variable in the overlap studies:** add the branch to `params.py`, add it
  to the variable lists (`JET_VARIABLES_NO_REPS`, …) with range, bin step and
  label, and add an entry to `VARIABLE_PLOT_CONFIG_NO_REPS` if it needs a log
  scale or a different output folder.
* **New selection or working point:** extend the `*_SELECTION_MODE` handling in
  the `main()` of the scripts and add the needed branch to the branch list.
* **Different truth definition:** see `label_jets_and_taus` in
  `truth_vs_reco_params.py` and `TRUTH_MODE_TAU`.
* **New pairing algorithm:** implement it like
  `greedy_match_no_reps_indices` (return local jet/tau indices per event) and
  reuse `build_matched_pair_features` and `categorize_matched_pairs`.
* **Other tagger or tau ID:** change `JET_SCORE_BRANCH` / `TAU_ID_SCORE_BRANCH`
  and the calibration maps in `params.py`.
* **Objective 3.4** (signal vs background from object counting) `[TODO: current
  status and suggested next steps]`.
* **Housekeeping ideas:** translate console/plot text to English, add a
  `requirements.txt`, add tests for the matching and the efficiency helpers.

---

## Known issues / things to check

* Scripts import their helpers as `from obj_3_1 import …` and
  `from truth_vs_reco_params import …`, and some docstrings refer to
  `overlap_kinematics.py`, `tau_score_analysis.py`, `root_file_analysis.py`
  (without the numeric prefix). Python cannot import module names starting with
  a digit, so make sure the shared utility module is actually available under
  the name `obj_3_1` (it is not among the files listed in the tree) and that the
  file names used in docstrings/usage messages match the real ones.
* `3_2_bjet_quantile_analysis.py` needs the external `flavour_tagging`
  repository (`params.FLAVOUR_TAGGING_DIR`) on disk.
* The no-repetition kinematics script reads the tau score through
  `TAU_SCORE_BRANCH`, while the correlation script uses `TAU_ID_SCORE_BRANCH`:
  confirm both point to the intended branch.
* The MET-based variables (`tau_met_proj`, `tau_mt`) are implemented, but the
  current `main()` of the no-repetition script does not pass `met`/`met_phi`,
  so they are not computed there.

---

## References

* ATLAS collaboration, search for HH → bbττ: <https://arxiv.org/abs/2404.12660>
* Example data-handling code: <https://gitlab.cern.ch/parodi/b-tau_studies>

## Author / contact

Giulia Montagnani
email: `giu.mont04@gmail.com`
