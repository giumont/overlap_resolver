# b-jet / τ-jet overlap studies in HH → bbττ (ATLAS ntuples)

Analysis code to study the **overlap between reconstructed b-jets and hadronic τ-jets** in ATLAS ntuples (`AnalysisMiniTree`) for the search for Higgs-boson pair production in the **HH → bbττ** final state ([latest public ATLAS result](https://arxiv.org/abs/2404.12660)), and to quantify and improve the behaviour of the offline discriminants (GN2v01 b-tagging, τ-ID scores) when the two object types point to the same detector region.

> 📄 **Documentation.** Motivation, methods, results, open issues and suggestions for extending the work are in [`docs/report.pdf`](docs/report.pdf). Additional, less organised notes are in the logbook ([`docs/logbook.md`](docs/logbook.md), suggested, or [`docs/logbook.pdf`](docs/logbook.pdf)). This README only covers the practical aspects of using the repository.

---

## Objectives and corresponding code

| ID | Objective | Where |
|----|-----------|-------|
| 3.0 | Exploration of the ntuples | `src/obj_3_0_root_file_analysis.py` |
| 3.1 | Overlap between b-jets and τ-jets; kinematics and discriminants | `src/obj_3_1_*.py` |
| 3.2 | Classification quality of the taggers; correlation of the scores | `src/obj_3_2_*.py` |
| 3.3 | Neural network acting on the overlapping pairs | `src/overlap_pairs_dataset_builder.py`, `notebooks/` |
| 3.4 | Signal vs background | `outputs_training/*/EVAL_bkg_tt`, `outputs_datasets_analysis/outputs_tt` (counting study not done yet) |

See the report for the description of each objective, the workflow and the results.

---

## Repository layout

```
├── src/                          # analysis code (flat modules, run from here)
│   ├── params.py                 # single source of configuration
│   ├── obj_3_0_*.py              # 3.0  ntuple exploration
│   ├── obj_3_1_*.py              # 3.1  overlap, pair kinematics, MET variables, scores
│   ├── obj_3_2_*.py              # 3.2  tagger quality, score correlations
│   ├── overlap_pairs_dataset_builder.py   # 3.3  builds the pair dataset for the NN
│   ├── other_code/               # diagnostics, truth labelling helpers, legacy C++
│   └── output/                   # raw outputs of the scripts
├── notebooks/                    # preprocessing, DNN and Transformer training (Colab)
├── root_datasets/                # input ntuples: HH_bbtt/ (signal), bkg_tt/ (background)
├── outputs_datasets_analysis/    # curated results of the dataset studies
├── outputs_training/             # results of the NN trainings and evaluations
└── docs/                         # report and logbook
```

---

## Setup

Requirements: Python 3.8+, `uproot`, `awkward`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `torch`, plus the external package `flavour_tag_ml` (ROC curves, working points, models, training loops), needed by `obj_3_2_bjet_quantile_analysis.py`, `obj_3_2_tau_score_analysis.py` and the notebooks.

```bash
git clone https://github.com/giumont/overlap_resolver
cd overlap_resolver

pip install uproot awkward numpy scipy matplotlib scikit-learn torch

git clone https://github.com/giumont/ML4FlavourTagging
cd ML4FlavourTagging
pip install -e .
```

Set `FLAVOUR_TAGGING_DIR` in `src/params.py` to the folder where `ML4FlavourTagging` was cloned.

---

## Data

Input ntuples (`AnalysisMiniTree`, campaign mc23a, "bypass noOR" mode) are expected in `root_datasets/`:

| Sample | Process | Folder | Files |
|--------|---------|--------|-------|
| Signal | HH → bbττ, ggF | `root_datasets/HH_bbtt/` | 14 |
| Background | fully hadronic tt̄ | `root_datasets/bkg_tt/` | 9 |

Original location on EOS:

* signal: `/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/GGF-mc23a-bypass-noOR_mode/`
* background: `/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/TTBAR-mc23a-bypass-noOR_mode`

The notebooks do **not** read the ROOT files: they use the normalised `.npy` pair arrays (stored on Google Drive) produced by `overlap_pairs_dataset_builder.py` and `notebooks/overlap_pairs_preprocessing.ipynb`.

---

## Usage

The scripts run as flat modules: run them **from `src/`**. All settings (input paths, branch names, thresholds, plot options, output folders) are in `src/params.py`; only `obj_3_0` takes command-line arguments.

```bash
cd src

# 3.0  exploration (input file, optional cap on the number of entries)
python3 obj_3_0_root_file_analysis.py ../root_datasets/HH_bbtt/output_GGF_mc23a_bypass_noOR_000001.root [n_entries_cap]

# 3.1  overlap
python3 obj_3_1_geometric_overlap.py
python3 obj_3_1_overlap_kinematics.py
python3 obj_3_1_overlap_kinematics_no_reps.py

# 3.2  classification quality
python3 obj_3_2_bjet_quantile_analysis.py
python3 obj_3_2_tau_score_analysis.py
python3 obj_3_2_corr_scores_overlap.py
```

Suggested order for a first pass: **3.0 → 3.1 → 3.2 → dataset builder → notebooks**.

To switch between signal and background, or to point to different files, edit the input paths in `params.py` (files are searched as `ROOT_DIR / f"{FILE_PREFIX}{index:02d}{FILE_SUFFIX}"`, `index = 1 … N_FILES`).

### Notebooks (3.3, 3.4)

`notebooks/DNN_training.ipynb` and `notebooks/Transformer_training.ipynb` are written for **Google Colab** and read the datasets from Google Drive. In the first cells:

* install the external package: `!pip install "git+https://github.com/giumont/ML4FlavourTagging"`;
* clone this repository and add its `src/` folder to `sys.path`;
* to push the outputs back to GitHub, store a token in the Colab secret `GITHUB_TOKEN_OVERLAP_RESOLVER`.

Edit the *GLOBAL PARAMS* cell (input directory, model, loss, output folder) and run top to bottom. To evaluate on the background, point `INPUT_DIR` to the tt̄ arrays and run the "objective 3.4" cells. Each run is identified by a `RUN_TAG` encoding architecture and hyper-parameters, so different configurations do not overwrite each other.

---

## Outputs

* Scripts write plots and console summaries to the folders defined in `params.py` (mostly `src/output/`).
* `outputs_datasets_analysis/` collects curated results of the dataset studies (`outputs_HH_bbtautau/`, `outputs_tt/`).
* `outputs_training/` collects the NN results (figures, `config.yaml`, text report) for `DNN/` and `Transformer/`, on the signal (`HH_bbtt_dataset`) and on the background (`EVAL_bkg_tt`). The notebooks push them here automatically.

The report indicates, for each figure, in which folder the corresponding plot (and the related ones) can be found.

---

## Known issues

* Console output, plot labels and notebook comments are largely in Italian (e.g. "Coppie", "Soglie Tau ID").
* `tau_met_proj` and `tau_mt` are not computed by the `main()` of `obj_3_1_overlap_kinematics_no_reps.py` (it does not pass `met` / `met_phi`).
* The tau classification script treats `TAU_TRUTH_MATCH_BRANCH != 0` as true tau, the correlation script uses `== TAU_TRUTH_TRUE_VALUE`: make sure they are equivalent.
* Some plot legends hard-code the truth branch names (`tau_truth_IsHadronicTau`, `HadronConeExclTruthLabelID==5`) instead of following `params.py`.
* Notebooks: Google Drive paths (`/content/drive/MyDrive/overlap_resolver/…`) and the Colab secret are hard-coded.

Further open issues and suggestions for extensions are listed at the end of each chapter of the [report](docs/report.pdf).

---

## Contact

Giulia Montagnani, `giu.mont04@gmail.com`
