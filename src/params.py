"""
Global parameters for the HH -> bb tau tau / ttbar overlap study.

Single source of truth for the configuration shared by the analysis
scripts:

- ``obj_3_1.py``: geometric RECO jet <-> tau matching (DeltaR
  distributions, overlap thresholds).
- ``overlap_kinematics.py``: pair-level kinematics of overlapping
  (jet, tau) pairs, split by truth category.
- ``overlap_met_tau.py``: tau-MET variables (projected MET, transverse
  mass) of the overlapping pairs, split by truth category.
- ``overlap_metadatas.py``: event metadata (pile-up, primary vertices)
  of the overlapping pairs, split by truth category.
- ``overlap_scores.py``: offline identification scores (b-tag quantile,
  tau ID score) of the overlapping pairs, split by truth category.
- ``overlap_pairs_dataset_builder.py``: (jet, tau) pair dataset with
  pair-level truth index, used as input for the classification study
  (objective 3.2).
- ``root_file_analysis.py``: preliminary exploration of the ntuple
  structure and of the key branches.
- ``corr_scores_overlap.py``: correlation between b-tag quantile and tau
  ID score of the overlapping pairs, split by truth category.
- ``tau_score_analysis.py``: classification quality of a tau ID score
  (ROC, working points, efficiency vs threshold).
- ``bjet_quantile_analysis.py``: classification quality and calibration
  of the GN2v01 continuous b-tag quantile.
- ``other_code/truth_vs_reco_params.py``: truth labelling of jets and taus,
  truth-category classification of the overlapping pairs and object-level
  kinematics of overlapping vs isolated objects (objective 3.1).
- ``other_code/bjet_SF_analysis.py``: empirical efficiency calibration and
  ROC of the GN2v01 continuous b-tagging scale-factor branch (objective 3.2).

The module contains only constants (input files, branch names, selection
modes, truth-matching settings, binning, output directories, plot
configuration) and no analysis logic. Change values here; do not
redefine them in the scripts.
"""

from pathlib import Path

import numpy as np


# Folder of this file and root of every output: all results are saved under
# ../outputs_datasets_analysis/ relative to params.py, whatever the working
# directory.
PARAMS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PARAMS_DIR.parent / "outputs_datasets_analysis"


# ======================================================================
# INPUT DATA
# ======================================================================

SAMPLES = {
    "signal": {
        "root_dir": Path("../root_datasets/HH_bbtt"),
        "file_prefix": "output_GGF_mc23a_bypass_noOR_0000",
    },
    "ttbar": {
        "root_dir": Path("../root_datasets/bkg_tt"),
        "file_prefix": "output_TTBAR_mc23a_bypass_noOR_0000",
    },
}

# Sample analysed by the scripts: "signal" or "ttbar".
SAMPLE = "ttbar"

ROOT_DIR = SAMPLES[SAMPLE]["root_dir"]
FILE_PREFIX = SAMPLES[SAMPLE]["file_prefix"]
FILE_SUFFIX = ".root"

TREE_NAME = "AnalysisMiniTree"
N_FILES = 14

# Maximum number of events read from each file; None = all events.
N_ENTRIES_CAP = None


# ======================================================================
# BRANCH NAMES
# ======================================================================

JET_ETA_BRANCH = "recojet_antikt4PFlow_eta"
JET_PHI_BRANCH = "recojet_antikt4PFlow_phi"
JET_PT_BRANCH = "recojet_antikt4PFlow_pt___NOSYS"
JET_MASS_BRANCH = "recojet_antikt4PFlow_m___NOSYS"
JET_N_MUONS_BRANCH = "recojet_antikt4PFlow_n_muons___NOSYS"
JET_IS_ANALYSIS_BRANCH = "recojet_antikt4PFlow_isAnalysisJet___NOSYS"

TAU_ETA_BRANCH = "tau_eta"
TAU_PHI_BRANCH = "tau_phi"
TAU_PT_BRANCH = "tau_pt___NOSYS"
TAU_NPRONG_BRANCH = "tau_nProng"
TAU_DECAYMODE_BRANCH = "tau_decayMode"
TAU_CHARGE_BRANCH = "tau_charge"
TAU_IS_ANALYSIS_BRANCH = "tau_isAnalysisTau___NOSYS"

MET_BRANCH = "met_met___NOSYS"
MET_PHI_BRANCH = "met_phi___NOSYS"

NPV_BRANCH = "nPrimaryVertices"
ACTUAL_MU_BRANCH = "actualInteractionsPerCrossing"
AVG_MU_BRANCH = "averageInteractionsPerCrossing"


# ======================================================================
# OBJECT SELECTION
# ======================================================================

# "all": every analysis-level jet.
# "btag85": analysis-level jets passing the GN2v01 FixedCutBEff_85 WP.
JET_SELECTION_MODE = "all"
JET_BTAG_BRANCH = "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85"
VALID_JET_SELECTION_MODES = {"all", "btag85"}

# "all": every analysis-level tau.
# "score85": analysis-level taus above the 85% WP threshold on the
#            GNTauScoreSigTrans_v0prune score.
TAU_SELECTION_MODE = "all"
TAU_SCORE_BRANCH = "tau_GNTauScoreSigTrans_v0prune"
TAU_SCORE_WP85_THRESHOLD = 0.163094
VALID_TAU_SELECTION_MODES = {"all", "score85"}

if JET_SELECTION_MODE not in VALID_JET_SELECTION_MODES:
    raise ValueError(
        f"JET_SELECTION_MODE='{JET_SELECTION_MODE}' non valido. "
        f"Usare uno tra {sorted(VALID_JET_SELECTION_MODES)}."
    )

if TAU_SELECTION_MODE not in VALID_TAU_SELECTION_MODES:
    raise ValueError(
        f"TAU_SELECTION_MODE='{TAU_SELECTION_MODE}' non valido. "
        f"Usare uno tra {sorted(VALID_TAU_SELECTION_MODES)}."
    )


# ======================================================================
# TRUTH INFORMATION
# ======================================================================

JET_TRUTH_LABEL_BRANCH = "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
JET_TRUTH_LABEL_B_VALUE = 5

# "label": use the boolean truth flag stored for each reco tau.
# "geometric": match reco taus to visible truth taus within
#              DR_TRUTH_MATCH_TAU.
TRUTH_MODE_TAU = "label"
TAU_TRUTH_MATCH_BRANCH = "tau_truth_IsHadronicTau"
TRUTH_TAU_ETA_BRANCH = "truthtau_eta_vis"
TRUTH_TAU_PHI_BRANCH = "truthtau_phi_vis"
DR_TRUTH_MATCH_TAU = 0.2


# ======================================================================
# OUTPUT
# ======================================================================

OUTPUT_DIR_DR = OUTPUT_ROOT / "discriminance_analysis/combinatory_level"
OUTPUT_DIR_PAIR_KIN = OUTPUT_ROOT / "discriminance_analysis/pair_kinematics_categories/main_analysis"
OUTPUT_DIR_MET = OUTPUT_ROOT / "discriminance_analysis/pair_kinematics_categories/met_tau"
OUTPUT_DIR_METADATA = OUTPUT_ROOT / "pair_metadatas_categories"


# ======================================================================
# GEOMETRIC OVERLAP (obj_3_1.py)
# ======================================================================

# Candidate DeltaR thresholds for the overlap definition.
DR_THRESHOLDS = [0.2, 0.3, 0.4]

DR_HIST_MIN = 0.0
DR_HIST_MAX = 0.5
DR_HIST_BINSIZE = 0.015

NORMALIZE_DR_HISTOGRAMS = False

PLOT_ETA_PHI = True

ETA_HIST_MIN = -5.0
ETA_HIST_MAX = 5.0
ETA_HIST_BINSIZE = 0.1

PHI_HIST_MIN = -np.pi
PHI_HIST_MAX = np.pi
PHI_HIST_BINSIZE = 0.1


# ======================================================================
# PAIR-LEVEL KINEMATICS (overlap_kinematics.py)
# ======================================================================

PLOT_ANGULAR_VARIABLES = True

# If True, categories (TT) and (FF) are merged into "other".
AGGREGATE_CATEGORIES = False

NORMALIZE_PAIR_HISTOGRAMS = True

COMPUTE_PT_RATIO = True
COMPUTE_MET_PROJ = True
COMPUTE_MT = True

# Maximum DeltaR for a (jet, tau) pair to enter the pair-level study.
DR_THRESHOLD_KINEMATICS = 0.4

PT_HIST_MIN = 0.0
PT_HIST_MAX = 500_000.0
PT_HIST_BINSIZE = 10_000

PAIR_ETA_HIST_BINSIZE = 0.2
PAIR_PHI_HIST_BINSIZE = 0.2

M_HIST_MIN = 0.0
M_HIST_MAX = 50_000.0
M_HIST_BINSIZE = 2_000

N_MUONS_HIST_MIN = -0.5
N_MUONS_HIST_MAX = 5.5
N_MUONS_HIST_BINSIZE = 1.0

NPRONG_HIST_MIN = -0.5
NPRONG_HIST_MAX = 5.5
NPRONG_HIST_BINSIZE = 1.0

DECAYMODE_HIST_MIN = -1.5
DECAYMODE_HIST_MAX = 10.5
DECAYMODE_HIST_BINSIZE = 1.0

CHARGE_HIST_MIN = -2.5
CHARGE_HIST_MAX = 2.5
CHARGE_HIST_BINSIZE = 1.0

PAIR_ANGULAR_BINSIZE = 0.02

# Truth categories: first letter = b-jet truth, second = hadronic-tau truth
# (T = true, F = fake).
CATEGORY_LABELS = {
    "a_jet_true_tau_fake": "(TF) b-jet vero, hadr. tau fake",
    "b_jet_false_tau_true": "(FT) b-jet fake, hadr. tau vero",
    "c_jet_true_tau_true": "(TT) b-jet vero, hadr. tau vero",
    "d_jet_false_tau_false": "(FF) entrambi fake",
}
CATEGORY_COLORS = {
    "a_jet_true_tau_fake": "tab:red",
    "b_jet_false_tau_true": "tab:blue",
    "c_jet_true_tau_true": "tab:green",
    "d_jet_false_tau_false": "tab:gray",
}
CATEGORY_KEYS = list(CATEGORY_LABELS.keys())

CATEGORY_LABELS_AGG = {
    "a_jet_true_tau_fake": "(TF) b-jet vero, hadr. tau fake",
    "b_jet_false_tau_true": "(FT) b-jet fake, hadr. tau vero",
    "other": "Altro",
}
CATEGORY_COLORS_AGG = {
    "a_jet_true_tau_fake": "tab:red",
    "b_jet_false_tau_true": "tab:blue",
    "other": "tab:gray",
}
CATEGORY_KEYS_AGG = list(CATEGORY_LABELS_AGG.keys())

# y-axis scale and output directory of each pair-level plot,
# keyed by "<object>_<variable>".
VARIABLE_PLOT_CONFIG = {
    "jet_pt": {"y_scale": "log", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "jet_mass": {"y_scale": "log", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "jet_n_muons": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "jet_eta": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN / "angular"},
    "jet_phi": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN / "angular"},
    "tau_pt": {"y_scale": "log", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "tau_eta": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN / "angular"},
    "tau_phi": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN / "angular"},
    "tau_nProng": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "tau_decayMode": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "tau_charge": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "pair_pt_ratio": {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN},
    "pair_dr": {"y_scale": "log", "out_dir": OUTPUT_DIR_PAIR_KIN / "angular"},
    "pair_deta": {"y_scale": "log", "out_dir": OUTPUT_DIR_PAIR_KIN / "angular"},
    "pair_dphi": {"y_scale": "log", "out_dir": OUTPUT_DIR_PAIR_KIN / "angular"},
}

# Variables to plot: (key, hist_min, hist_max, bin_size, x-axis label).
JET_VARIABLES = [
    ("pt", PT_HIST_MIN, PT_HIST_MAX, PT_HIST_BINSIZE, "pT [MeV]"),
    ("mass", M_HIST_MIN, M_HIST_MAX, M_HIST_BINSIZE, "Massa [MeV]"),
    ("n_muons", N_MUONS_HIST_MIN, N_MUONS_HIST_MAX, N_MUONS_HIST_BINSIZE, "N. muoni soft"),
]

TAU_VARIABLES = [
    ("pt", PT_HIST_MIN, PT_HIST_MAX, PT_HIST_BINSIZE, "pT [MeV]"),
    ("nProng", NPRONG_HIST_MIN, NPRONG_HIST_MAX, NPRONG_HIST_BINSIZE, "N. Prong"),
    ("decayMode", DECAYMODE_HIST_MIN, DECAYMODE_HIST_MAX, DECAYMODE_HIST_BINSIZE, "Decay Mode"),
    ("charge", CHARGE_HIST_MIN, CHARGE_HIST_MAX, CHARGE_HIST_BINSIZE, "Carica"),
]

PAIR_VARIABLES = [
    ("pt_ratio", 0.0, 5.0, 0.1, r"$p_T^{\text{jet}} / p_T^{\text{tau}}$"),
]

if PLOT_ANGULAR_VARIABLES:
    JET_VARIABLES.extend([
        ("eta", ETA_HIST_MIN, ETA_HIST_MAX, PAIR_ETA_HIST_BINSIZE, r"$\eta$"),
        ("phi", PHI_HIST_MIN, PHI_HIST_MAX, PAIR_PHI_HIST_BINSIZE, r"$\phi$"),
    ])
    TAU_VARIABLES.extend([
        ("eta", ETA_HIST_MIN, ETA_HIST_MAX, PAIR_ETA_HIST_BINSIZE, r"$\eta$"),
        ("phi", PHI_HIST_MIN, PHI_HIST_MAX, PAIR_PHI_HIST_BINSIZE, r"$\phi$"),
    ])
    PAIR_VARIABLES.extend([
        ("dr", 0.0, DR_THRESHOLD_KINEMATICS, PAIR_ANGULAR_BINSIZE, r"$\Delta R$"),
        ("deta", -DR_THRESHOLD_KINEMATICS, DR_THRESHOLD_KINEMATICS, PAIR_ANGULAR_BINSIZE, r"$\Delta \eta$"),
        ("dphi", -DR_THRESHOLD_KINEMATICS, DR_THRESHOLD_KINEMATICS, PAIR_ANGULAR_BINSIZE, r"$\Delta \phi$"),
    ])


# ======================================================================
# TAU-MET VARIABLES (overlap_met_tau.py)
# ======================================================================

MET_PROJ_HIST_MIN = -300_000.0
MET_PROJ_HIST_MAX = 300_000.0
MET_PROJ_HIST_BINSIZE = 5_000

MT_HIST_MIN = 0.0
MT_HIST_MAX = 300_000.0
MT_HIST_BINSIZE = 5_000

VARIABLE_PLOT_CONFIG_MET = {
    "tau_met_proj": {"y_scale": "linear", "out_dir": OUTPUT_DIR_MET},
    "tau_mt": {"y_scale": "linear", "out_dir": OUTPUT_DIR_MET},
}

TAU_MET_VARIABLES = [
    ("met_proj", MET_PROJ_HIST_MIN, MET_PROJ_HIST_MAX, MET_PROJ_HIST_BINSIZE, r"MET proiettato ($MET_{\parallel}$) [MeV]"),
    ("mt", MT_HIST_MIN, MT_HIST_MAX, MT_HIST_BINSIZE, r"Massa Trasversa ($m_T$) [MeV]"),
]


# ======================================================================
# EVENT METADATA (overlap_metadatas.py)
# ======================================================================

METADATA_HIST_MIN = 0.0
METADATA_HIST_MAX = 100.0
METADATA_HIST_BINSIZE = 1.0

VARIABLE_PLOT_CONFIG_METADATA = {
    "nPrimaryVertices": {"y_scale": "linear", "out_dir": OUTPUT_DIR_METADATA},
    "actualInteractionsPerCrossing": {"y_scale": "linear", "out_dir": OUTPUT_DIR_METADATA},
    "averageInteractionsPerCrossing": {"y_scale": "linear", "out_dir": OUTPUT_DIR_METADATA},
}

METADATA_VARIABLES = [
    ("nPrimaryVertices", METADATA_HIST_MIN, METADATA_HIST_MAX, METADATA_HIST_BINSIZE, "N. Primary Vertices"),
    ("actualInteractionsPerCrossing", METADATA_HIST_MIN, METADATA_HIST_MAX, METADATA_HIST_BINSIZE, r"Actual $\mu$"),
    ("averageInteractionsPerCrossing", METADATA_HIST_MIN, METADATA_HIST_MAX, METADATA_HIST_BINSIZE, r"Average $\mu$"),
]


# ======================================================================
# IDENTIFICATION SCORES (overlap_scores.py)
# ======================================================================

OUTPUT_DIR_SCORES = OUTPUT_ROOT / "discriminance_analysis/overlap_scores"

# DeltaR below which a (jet, tau) pair is considered overlapping in the
# score study (distinct from DR_THRESHOLD_KINEMATICS).
DR_THRESHOLD_SCORES = 4.0

JET_SCORE_BRANCH = "recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous"

# Tau score to analyse; alternatives:
# "tau_GNTauScoreSigTrans_v0prune", "tau_RNNJetScoreSigTrans",
# "tau_RNNEleScoreSigTrans_v1".
TAU_ID_SCORE_BRANCH = "tau_RNNEleScoreSigTrans_v1"

# Signal efficiencies (%) of the tau working points drawn on the plots.
TARGET_EFFICIENCIES = [70, 75, 80, 85, 90, 97]
WP_COLORS = ["tab:blue", "cornflowerblue", "lightsteelblue", "peachpuff", "coral", "tab:red"]

# Meaning of the values of the continuous b-tag quantile branch.
JET_QUANTILES_MAP = {
    -1: "N/A\n(-1)", 1: "97%\n(1)", 2: "90%\n(2)", 3: "85%\n(3)",
    4: "80%\n(4)", 5: "75%\n(5)", 6: "70%\n(6)",
}

# ======================================================================
# PAIR DATASET BUILDER (overlap_pairs_dataset_builder.py)
# ======================================================================

# Pair-level truth index: first letter = b-jet truth, second = hadronic-tau
# truth (T = true, F = fake).
PAIR_LABEL_INDEX = {"FF": 0, "FT": 1, "TF": 2, "TT": 3}

# DeltaR below which a (jet, tau) pair is stored in the dataset.
PAIR_DATASET_DR_THRESHOLD = 0.4

# Features always stored; optional ones (pt ratio, MET projection, mT) and
# extra branches are appended by the builder.
PAIR_BASE_FEATURE_KEYS = [
    "pair_dr", "pair_deta", "pair_dphi",
    "jet_pt", "jet_eta", "jet_phi",
    "tau_pt", "tau_eta", "tau_phi",
]

# Event-grouped train/val/test split.
SPLIT_TRAIN_FRAC = 0.70
SPLIT_VAL_FRAC = 0.15
SPLIT_TEST_FRAC = 0.15
SPLIT_SEED = 42


# ======================================================================
# NTUPLE EXPLORATION (root_file_analysis.py)
# ======================================================================

OUTPUT_DIR_ROOT_EXPLORATION = OUTPUT_ROOT / "root_files_analysis"
EXPLORATION_N_BINS = 60

# (branch-name prefix, group description), checked in order.
BRANCH_PREFIX_GROUPS = [
    ("bbtt_", "Livello analisi (oggetti/coppie gia' selezionate dall'algoritmo bbtt)"),
    ("recojet_antikt4PFlow_", "Jet ricostruiti (reco, anti-kt R=0.4 PFlow)"),
    ("tau_", "Tau adronici ricostruiti (reco)"),
    ("el_", "Elettroni ricostruiti (reco)"),
    ("mu_", "Muoni ricostruiti (reco)"),
    ("truthjet_antikt4_", "Jet di verita' (truth)"),
    ("truthtau_", "Tau di verita' (truth)"),
    ("truthelectron_", "Elettroni di verita' (truth)"),
    ("truthmuon_", "Muoni di verita' (truth)"),
    ("truthmet_", "MET di verita' (truth)"),
    ("truth_", "Verita' a livello di evento/generatore (Higgs, HH, PDF info)"),
    ("trigPassed_", "Decisioni di trigger HLT specifiche"),
]
BRANCH_GROUP_DEFAULT = "Metadati di evento / varie"

# Branches summarised and plotted by root_file_analysis.py.
EXPLORATION_KEY_BRANCHES = [
    "recojet_antikt4PFlow_pt___NOSYS",
    "recojet_antikt4PFlow_eta",
    "recojet_antikt4PFlow_phi",
    "recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous",
    "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
    "tau_pt___NOSYS",
    "tau_eta",
    "tau_phi",
    "tau_RNNJetScoreSigTrans",
    "tau_GNTauScoreSigTrans_v0prune",
    "bbtt_HH_m___NOSYS",
    "bbtt_H_bb_m___NOSYS",
    "bbtt_mmc_m___NOSYS",
]


# ======================================================================
# SCORE CORRELATION (corr_scores_overlap.py)
# ======================================================================

OUTPUT_DIR_CORR_SCORES = OUTPUT_ROOT / "obj_3_2/corr_scores_overlap"

# DeltaR below which a (jet, tau) pair is considered overlapping in the
# correlation study.
DR_THRESHOLD_CORR = 0.4

# Value of the tau truth flag that identifies a true hadronic tau.
TAU_TRUTH_TRUE_VALUE = 1

# Validity cuts on the scores of the overlapping pairs: jet quantile
# >= CORR_JET_SCORE_MIN (-1 is the "N/A" bin), tau score > CORR_TAU_SCORE_MIN.
CORR_JET_SCORE_MIN = -1.0
CORR_TAU_SCORE_MIN = -10.0


# ======================================================================
# TAGGER PERFORMANCE (tau_score_analysis.py, bjet_quantile_analysis.py)
# ======================================================================

# Location of the flavour_tagging repository (provides src.graphics).
FLAVOUR_TAGGING_DIR = Path("../flavour_tagging")

# Number of thresholds scanned between the minimum and maximum score.
N_SCAN_POINTS = 400

# Tau score analysed by tau_score_analysis.py; alternatives:
# "tau_GNTauScoreSigTrans_v0prune", "tau_RNNEleScoreSigTrans_v1".
TAU_SCORE_ANALYSIS_BRANCH = "tau_RNNJetScoreSigTrans"

OUTPUT_DIR_TAU_TAGGER = OUTPUT_ROOT / "obj_3_2/tau_tagger" / TAU_SCORE_ANALYSIS_BRANCH
OUTPUT_DIR_BJET_TAGGER = OUTPUT_ROOT / "obj_3_2/b_tagger"

# Nominal signal efficiency (%) of the ready-made FixedCutBEff_85 working point.
BTAG_WP85_NOMINAL_EFF = 85

# Nominal b-jet efficiency (%) associated with each discrete bin of the
# continuous b-tag quantile; a higher bin is a tighter selection
# (bin >= b).
JET_QUANTILE_BIN_NOMINAL_EFF = {1: 97.0, 2: 90.0, 3: 85.0, 4: 80.0, 5: 75.0, 6: 70.0}


# ======================================================================
# TRUTH VS RECO (other_code/truth_vs_reco_params.py)
# ======================================================================

OUTPUT_DIR_TRUTH = OUTPUT_ROOT / "obj_3_1/truth"

# DeltaR thresholds and binning of the full DeltaR(jet, tau) distribution.
TRUTH_DR_THRESHOLDS = [0.2, 0.4]
TRUTH_DR_HIST_MIN = 0
TRUTH_DR_HIST_MAX = 4.0
TRUTH_DR_HIST_BINSIZE = 0.02

# Object-level kinematics; pT in MeV. The eta/phi ranges are shared with
# ETA_HIST_MIN/MAX and PHI_HIST_MIN/MAX.
TRUTH_PT_HIST_MIN = 0.0
TRUTH_PT_HIST_MAX = 1_000_000.0
TRUTH_PT_HIST_BINSIZE = 5_000
TRUTH_ETA_HIST_BINSIZE = 0.2
TRUTH_PHI_HIST_BINSIZE = 0.2

NORMALIZE_TRUTH_HISTOGRAMS = True

# Variables to plot: (key, hist_min, hist_max, bin_size, x-axis label).
OBJECT_KINEMATICS_VARIABLES = [
    ("pt", TRUTH_PT_HIST_MIN, TRUTH_PT_HIST_MAX, TRUTH_PT_HIST_BINSIZE, "pT [MeV]"),
    ("eta", ETA_HIST_MIN, ETA_HIST_MAX, TRUTH_ETA_HIST_BINSIZE, r"$\eta$"),
    ("phi", PHI_HIST_MIN, PHI_HIST_MAX, TRUTH_PHI_HIST_BINSIZE, r"$\phi$"),
]


# ======================================================================
# B-JET CONTINUOUS SF BRANCH (other_code/bjet_SF_analysis.py)
# ======================================================================

JET_SF_CONTINUOUS_BRANCH = "recojet_antikt4PFlow_ftag_effSF_GN2v01_Continuous___NOSYS"

OUTPUT_DIR_BJET_SF = OUTPUT_ROOT / "obj_3_2/b_tagger_continuous"

BJET_SF_N_SCAN_POINTS = 100

# True: (score >= threshold) selects signal. Set to False if the efficiency
# curves of the scan come out inverted.
BJET_SF_HIGHER_IS_BETTER = True