"""
================================================================================
Analisi dei metadati di evento (Pile-up, Vertici) per coppie jet-tau in overlap,
separata per categorie di verità.
================================================================================

Le funzioni duplicate di labeling e plotting vengono importate direttamente
da truth_vs_reco_params.py e obj_3_1.py.
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
    section, load_files, get_analysis_selection, delta_r, summarize,
    print_shoulder_table,
)

from truth_vs_reco_params import (
    match_reco_to_truth,
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)

# ======================================================================
# CONFIGURAZIONE
# ======================================================================

AGGREGATE_CATEGORIES = False

JET_SELECTION_MODE = "all"
JET_BTAG_BRANCH = "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85"
JET_TRUTH_LABEL_BRANCH = "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
JET_TRUTH_LABEL_B_VALUE = 5

TAU_SELECTION_MODE = "all"
TAU_SCORE_BRANCH = "tau_GNTauScoreSigTrans_v0prune"
TAU_SCORE_WP85_THRESHOLD = 0.163094

TRUTH_MODE_TAU = "label"
TAU_TRUTH_MATCH_BRANCH = "tau_truth_IsHadronicTau"
TRUTH_TAU_ETA_BRANCH = "truthtau_eta_vis"
TRUTH_TAU_PHI_BRANCH = "truthtau_phi_vis"

OUTPUT_DIR = Path("output/obj_3.1/pair_metadatas_categories")

JET_PT_BRANCH = "recojet_antikt4PFlow_pt___NOSYS"
TAU_PT_BRANCH = "tau_pt___NOSYS"

# --- Branch Metadati ---
NPV_BRANCH = "nPrimaryVertices"
ACTUAL_MU_BRANCH = "actualInteractionsPerCrossing"
AVG_MU_BRANCH = "averageInteractionsPerCrossing"

DR_THRESHOLD_KINEMATICS = 0.4

METADATA_HIST_MIN = 0.0
METADATA_HIST_MAX = 100.0
METADATA_HIST_BINSIZE = 1.0

# --- Categorie Standard ---
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

# --- Categorie Aggregate ---
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

NORMALIZE_HISTOGRAMS = True

VARIABLE_PLOT_CONFIG = {
    "nPrimaryVertices": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "actualInteractionsPerCrossing": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "averageInteractionsPerCrossing": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
}

METADATA_VARIABLES = [
    ("nPrimaryVertices", METADATA_HIST_MIN, METADATA_HIST_MAX, METADATA_HIST_BINSIZE, "N. Primary Vertices"),
    ("actualInteractionsPerCrossing", METADATA_HIST_MIN, METADATA_HIST_MAX, METADATA_HIST_BINSIZE, r"Actual $\mu$"),
    ("averageInteractionsPerCrossing", METADATA_HIST_MIN, METADATA_HIST_MAX, METADATA_HIST_BINSIZE, r"Average $\mu$"),
]


# ======================================================================
# COSTRUZIONE COPPIE E METADATA
# ======================================================================

def build_pair_metadata_and_labels(
    jet_pt, jet_eta, jet_phi, jet_label,
    tau_pt, tau_eta, tau_phi, tau_label,
    n_pv, actual_mu, avg_mu
):
    jet_pt_ct, tau_pt_ct = ak.unzip(ak.cartesian([jet_pt, tau_pt], nested=True))
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    # Broadcasting delle variabili di evento per matchare le dimensioni delle coppie (N_events, N_pairs)
    n_pv_ct = ak.broadcast_arrays(n_pv, jet_pt_ct)[0]
    actual_mu_ct = ak.broadcast_arrays(actual_mu, jet_pt_ct)[0]
    avg_mu_ct = ak.broadcast_arrays(avg_mu, jet_pt_ct)[0]

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)

    return {
        "pair_dr": dr_matrix,
        "nPrimaryVertices": n_pv_ct,
        "actualInteractionsPerCrossing": actual_mu_ct,
        "averageInteractionsPerCrossing": avg_mu_ct,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }


def get_overlapping_pairs_metadata(pair_info, dr_thr):
    overlap_mask = pair_info["pair_dr"] < dr_thr

    def flat_overlap(key):
        flat_arr = ak.to_numpy(ak.flatten(pair_info[key], axis=None))
        flat_m = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat_arr[flat_m]

    jet_lab = flat_overlap("jet_label").astype(bool)
    tau_lab = flat_overlap("tau_label").astype(bool)

    if AGGREGATE_CATEGORIES:
        cat_masks = {
            "a_jet_true_tau_fake": jet_lab & ~tau_lab,
            "b_jet_false_tau_true": ~jet_lab & tau_lab,
            "other": (jet_lab & tau_lab) | (~jet_lab & ~tau_lab),
        }
    else:
        cat_masks = {
            "a_jet_true_tau_fake": jet_lab & ~tau_lab,
            "b_jet_false_tau_true": ~jet_lab & tau_lab,
            "c_jet_true_tau_true": jet_lab & tau_lab,
            "d_jet_false_tau_false": ~jet_lab & ~tau_lab,
        }

    res = {cat: {} for cat in cat_masks.keys()}
    variables = [
        "pair_dr", "nPrimaryVertices", 
        "actualInteractionsPerCrossing", "averageInteractionsPerCrossing"
    ]

    for cat, cmask in cat_masks.items():
        for var in variables:
            res[cat][var] = flat_overlap(var)[cmask]

    return res


def merge_pair_metadata(parts_list):
    variables = [
        "pair_dr", "nPrimaryVertices", 
        "actualInteractionsPerCrossing", "averageInteractionsPerCrossing"
    ]
    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    
    merged = {cat: {var: [] for var in variables} for cat in cat_keys}

    for part in parts_list:
        for cat in cat_keys:
            for var in variables:
                merged[cat][var].append(part[cat][var])

    for cat in cat_keys:
        for var in variables:
            arrays = merged[cat][var]
            merged[cat][var] = np.concatenate(arrays) if arrays else np.array([])

    return merged


# ======================================================================
# PLOT
# ======================================================================

def save_metadata_plots(merged_data, dr_threshold):
    if not HAS_MPL:
        return
    
    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    for var_key, vmin, vmax, step, xlabel in METADATA_VARIABLES:
        
        config = VARIABLE_PLOT_CONFIG.get(var_key, {"y_scale": "linear", "out_dir": OUTPUT_DIR})
        out_dir = Path(config["out_dir"])
        
        if AGGREGATE_CATEGORIES:
            out_dir = out_dir / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        y_scale = config["y_scale"]

        bins = np.arange(vmin, vmax + step, step)
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets = []
        for cat in cat_keys:
            values = merged_data[cat][var_key]
            datasets.append((values, labels[cat], colors[cat]))

        _plot_hist_curves(ax, datasets, bins)
        
        _finish_object_plot(
            ax,
            xlabel,
            f"Event Metadata in coppie (pair-level) - $\\Delta R < {dr_threshold}$",
            normalize=NORMALIZE_HISTOGRAMS
        )
        
        ax.set_yscale(y_scale)
        fig.tight_layout()

        suffix = "_normalized" if NORMALIZE_HISTOGRAMS else ""
        suffix_agg = "_aggregated" if AGGREGATE_CATEGORIES else ""

        out_path = (
            out_dir 
            / f"pair_metadata_{var_key}_"
            f"{JET_SELECTION_MODE}{suffix}{suffix_agg}.png"
        )

        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Plot salvato in: {out_path}")

def save_scatter_plots_metadata(merged_data, dr_threshold):
    if not HAS_MPL:
        return
    
    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    out_dir = OUTPUT_DIR / "scatters"
    out_dir.mkdir(parents=True, exist_ok=True)

    scatters_config = [
        ("nPrimaryVertices", "N. Primary Vertices"),
        ("actualInteractionsPerCrossing", r"Actual $\mu$"),
        ("averageInteractionsPerCrossing", r"Average $\mu$")
    ]

    for var_key, ylabel in scatters_config:
        fig, ax = plt.subplots(figsize=(8, 6))

        for cat in cat_keys:
            dr_vals = merged_data[cat]["pair_dr"]
            meta_vals = merged_data[cat][var_key]
            count = len(dr_vals)
            
            label_con_conteggio = f"{labels[cat]} (N={count})"
            
            ax.scatter(
                dr_vals, 
                meta_vals, 
                label=label_con_conteggio, 
                color=colors[cat], 
                alpha=0.6, 
                s=15, 
                edgecolors='none'
            )

        ax.set_xlabel(r"$\Delta R$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Scatter: $\\Delta R$ vs {ylabel} ($\\Delta R < {dr_threshold}$)")
        
        ax.set_yscale("linear")
        
        ax.legend(title="Categorie", fontsize=9, loc='best')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"scatter_dr_vs_{var_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Scatter plot salvato in: {out_path}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
        return
    
    branches = [
        "tau_eta", "tau_phi", TAU_PT_BRANCH, "tau_isAnalysisTau___NOSYS",
        "recojet_antikt4PFlow_eta", "recojet_antikt4PFlow_phi", JET_PT_BRANCH,
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
        JET_TRUTH_LABEL_BRANCH, 
        NPV_BRANCH, ACTUAL_MU_BRANCH, AVG_MU_BRANCH
    ]

    if JET_SELECTION_MODE == "btag85":
        branches.append(JET_BTAG_BRANCH)
        
    if TAU_SELECTION_MODE == "score85":
        branches.append(TAU_SCORE_BRANCH)
        
    if TRUTH_MODE_TAU == "label":
        branches.append(TAU_TRUTH_MATCH_BRANCH)
    elif TRUTH_MODE_TAU == "geometric":
        branches.extend([TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH])

    pair_metadata_parts = []

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
        else:
            jet_sel = jet_sel_analysis & (ak.fill_none(a[JET_BTAG_BRANCH], False) != 0)

        if TAU_SELECTION_MODE == "all":
            tau_sel = tau_sel_analysis
        else:
            tau_sel = tau_sel_analysis & (ak.fill_none(a[TAU_SCORE_BRANCH], -np.inf) >= TAU_SCORE_WP85_THRESHOLD)

        jet_label, tau_label, _, _ = label_jets_and_taus(a, jet_sel, tau_sel)

        jet_eta = a["recojet_antikt4PFlow_eta"][jet_sel]
        jet_phi = a["recojet_antikt4PFlow_phi"][jet_sel]
        jet_pt = a[JET_PT_BRANCH][jet_sel]

        tau_eta = a["tau_eta"][tau_sel]
        tau_phi = a["tau_phi"][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]

        # Estrazione Metadati di evento
        n_pv = a[NPV_BRANCH]
        actual_mu = a[ACTUAL_MU_BRANCH]
        avg_mu = a[AVG_MU_BRANCH]

        pair_info = build_pair_metadata_and_labels(
            jet_pt, jet_eta, jet_phi, jet_label,
            tau_pt, tau_eta, tau_phi, tau_label,
            n_pv, actual_mu, avg_mu
        )

        pair_metadata_parts.append(
            get_overlapping_pairs_metadata(pair_info, DR_THRESHOLD_KINEMATICS)
        )
    merged_pair_metadata = merge_pair_metadata(pair_metadata_parts)
    
    section("PLOT METADATA PAIR-LEVEL")
    save_metadata_plots(merged_pair_metadata, DR_THRESHOLD_KINEMATICS)
    
    section("SCATTER PLOTS METADATA")
    save_scatter_plots_metadata(merged_pair_metadata, DR_THRESHOLD_KINEMATICS)

if __name__ == "__main__":
    main()