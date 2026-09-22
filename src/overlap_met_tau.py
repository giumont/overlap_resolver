"""
================================================================================
Analisi cinematica avanzata a livello di COPPIA (pair-level).
Include il calcolo esplicito della correlazione fisica tra il tau candidato
e il MET dell'evento (MET proiettato e Massa Trasversa m_T).
================================================================================
"""

from pathlib import Path
import numpy as np
import awkward as ak

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from obj_3_1 import (
    ROOT_DIR, FILE_PREFIX, FILE_SUFFIX, TREE_NAME, N_ENTRIES_CAP,
    section, load_files, get_analysis_selection, delta_r
)

from truth_vs_reco_params import (
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)

# ======================================================================
# CONFIGURAZIONE
# ======================================================================

AGGREGATE_CATEGORIES = False

JET_SELECTION_MODE = "all" # "all" oppure "btag85"
JET_BTAG_BRANCH = "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85"
JET_PT_BRANCH = "recojet_antikt4PFlow_pt___NOSYS"
JET_MASS_BRANCH = "recojet_antikt4PFlow_m___NOSYS"
JET_N_MUONS_BRANCH = "recojet_antikt4PFlow_n_muons___NOSYS"

TAU_SELECTION_MODE = "all" # "all" oppure "score85"
TAU_SCORE_BRANCH = "tau_GNTauScoreSigTrans_v0prune"
TAU_SCORE_WP85_THRESHOLD = 0.163094
TAU_PT_BRANCH = "tau_pt___NOSYS"

DR_THRESHOLD_KINEMATICS = 0.4
OUTPUT_DIR = Path("output/obj_3.1/pair_kinematics_categories/met_tau")

# Limiti istogrammi aggiuntivi
MET_PROJ_HIST_MIN = -300_000.0
MET_PROJ_HIST_MAX = 300_000.0
MET_PROJ_HIST_BINSIZE = 5_000

MT_HIST_MIN = 0.0
MT_HIST_MAX = 300_000.0
MT_HIST_BINSIZE = 5_000

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

NORMALIZE_HISTOGRAMS = True

VARIABLE_PLOT_CONFIG = {
    "tau_met_proj": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "tau_mt": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
}

TAU_MET_VARIABLES = [
    ("met_proj", MET_PROJ_HIST_MIN, MET_PROJ_HIST_MAX, MET_PROJ_HIST_BINSIZE, r"MET proiettato ($MET_{\parallel}$) [MeV]"),
    ("mt", MT_HIST_MIN, MT_HIST_MAX, MT_HIST_BINSIZE, r"Massa Trasversa ($m_T$) [MeV]"),
]

# ======================================================================
# COSTRUZIONE COPPIE E VARIABILI MET
# ======================================================================

def build_pair_met_kinematics(
    jet_eta, jet_phi, jet_label,
    tau_eta, tau_phi, tau_label, 
    tau_met_proj, tau_mt
):
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    
    _, tau_met_proj_ct = ak.unzip(ak.cartesian([jet_eta, tau_met_proj], nested=True))
    _, tau_mt_ct = ak.unzip(ak.cartesian([jet_eta, tau_mt], nested=True))
    
    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)

    return {
        "pair_dr": dr_matrix,
        "tau_met_proj": tau_met_proj_ct,
        "tau_mt": tau_mt_ct,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }

def get_overlapping_pairs_met(pair_info, dr_thr):
    overlap_mask = pair_info["pair_dr"] < dr_thr

    def flat_overlap(key):
        flat_arr = ak.to_numpy(ak.flatten(pair_info[key], axis=None))
        flat_m = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat_arr[flat_m]

    jet_lab = flat_overlap("jet_label").astype(bool)
    tau_lab = flat_overlap("tau_label").astype(bool)

    cat_masks = {
        "a_jet_true_tau_fake": jet_lab & ~tau_lab,
        "b_jet_false_tau_true": ~jet_lab & tau_lab,
        "c_jet_true_tau_true": jet_lab & tau_lab,
        "d_jet_false_tau_false": ~jet_lab & ~tau_lab,
    }

    res = {cat: {} for cat in cat_masks.keys()}
    variables = ["tau_met_proj", "tau_mt"]

    for cat, cmask in cat_masks.items():
        for var in variables:
            res[cat][var] = flat_overlap(var)[cmask]

    return res

def merge_met_kinematics(parts_list):
    variables = ["tau_met_proj", "tau_mt"]
    merged = {cat: {var: [] for var in variables} for cat in CATEGORY_KEYS}

    for part in parts_list:
        for cat in CATEGORY_KEYS:
            for var in variables:
                merged[cat][var].append(part[cat][var])

    for cat in CATEGORY_KEYS:
        for var in variables:
            arrays = merged[cat][var]
            merged[cat][var] = np.concatenate(arrays) if arrays else np.array([])

    return merged

# ======================================================================
# PLOT
# ======================================================================

def save_met_kinematics_plots(merged_data, dr_threshold):
    if not HAS_MPL:
        return
    
    for var_key, vmin, vmax, step, xlabel in TAU_MET_VARIABLES:
        var_id = f"tau_{var_key}"
        config = VARIABLE_PLOT_CONFIG[var_id]
        
        out_dir = Path(config["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        bins = np.arange(vmin, vmax + step, step)
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets = []
        for cat in CATEGORY_KEYS:
            values = merged_data[cat][var_id]
            datasets.append((values, CATEGORY_LABELS[cat], CATEGORY_COLORS[cat]))

        _plot_hist_curves(ax, datasets, bins)
        
        _finish_object_plot(
            ax,
            xlabel,
            f"Relazione $\\tau$-MET in coppie in overlap ($\\Delta R < {dr_threshold}$)",
            normalize=NORMALIZE_HISTOGRAMS
        )
        
        ax.set_yscale(config["y_scale"])
        fig.tight_layout()

        suffix = "_normalized" if NORMALIZE_HISTOGRAMS else ""
        out_path = out_dir / f"met_kinematics_{var_id}_jet{JET_SELECTION_MODE}_tau{TAU_SELECTION_MODE}{suffix}.png"

        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Plot salvato in: {out_path}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    loaded = load_files()
    if not loaded:
        return

    branches = [
        "tau_eta", "tau_phi", TAU_PT_BRANCH, "tau_isAnalysisTau___NOSYS",
        "recojet_antikt4PFlow_eta", "recojet_antikt4PFlow_phi",
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
        JET_BTAG_BRANCH, "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
        TAU_SCORE_BRANCH, "tau_truth_IsHadronicTau",
        "met_met___NOSYS", "met_phi___NOSYS"
    ]

    pair_met_parts = []

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]
        
        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel_analysis = get_analysis_selection(tree, "recojet_antikt4PFlow_isAnalysisJet___NOSYS", n_entries)
        tau_sel_analysis = get_analysis_selection(tree, "tau_isAnalysisTau___NOSYS", n_entries)

        if JET_SELECTION_MODE == "all":
            jet_sel = jet_sel_analysis
        elif JET_SELECTION_MODE == "btag85":
            jet_sel = jet_sel_analysis & (ak.fill_none(a[JET_BTAG_BRANCH], False) != 0)
        else:
            raise ValueError("JET_SELECTION_MODE non valido. Scegli 'all' o 'btag85'.")

        if TAU_SELECTION_MODE == "all":
            tau_sel = tau_sel_analysis
        elif TAU_SELECTION_MODE == "score85":
            tau_sel = tau_sel_analysis & (ak.fill_none(a[TAU_SCORE_BRANCH], -np.inf) >= TAU_SCORE_WP85_THRESHOLD)
        else:
            raise ValueError("TAU_SELECTION_MODE non valido. Scegli 'all' o 'score85'.")

        jet_label, tau_label, _, _ = label_jets_and_taus(a, jet_sel, tau_sel)

        jet_eta = a["recojet_antikt4PFlow_eta"][jet_sel]
        jet_phi = a["recojet_antikt4PFlow_phi"][jet_sel]

        tau_eta = a["tau_eta"][tau_sel]
        tau_phi = a["tau_phi"][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]

        met_val = a["met_met___NOSYS"]
        met_phi = a["met_phi___NOSYS"]

        dphi_met = (tau_phi - met_phi + np.pi) % (2 * np.pi) - np.pi
        
        tau_met_proj = met_val * np.cos(dphi_met)
        tau_mt = np.sqrt(np.maximum(0, 2 * tau_pt * met_val * (1 - np.cos(dphi_met))))

        pair_info = build_pair_met_kinematics(
            jet_eta, jet_phi, jet_label,
            tau_eta, tau_phi, tau_label,
            tau_met_proj, tau_mt
        )

        pair_met_parts.append(
            get_overlapping_pairs_met(pair_info, DR_THRESHOLD_KINEMATICS)
        )

    merged_met_kinematics = merge_met_kinematics(pair_met_parts)
    
    section("PLOT CORRELAZIONE FISICA TAU-MET")
    save_met_kinematics_plots(merged_met_kinematics, DR_THRESHOLD_KINEMATICS)

if __name__ == "__main__":
    main()