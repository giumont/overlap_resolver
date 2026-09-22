"""
================================================================================
Analisi cinematica a livello di COPPIA (pair-level) per jet e tau in overlap,
separata per categorie di verità. Include analisi su massa e muoni per i jet,
e nProng, decayMode, charge per i tau. Includendo metriche angolari di coppia.
================================================================================

Le funzioni duplicate di labeling e plotting vengono importate direttamente
da truth_vs_reco_params_3.py.
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

# Import delle funzioni duplicate dal file precedente
from truth_vs_reco_params import (
    match_reco_to_truth,
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)

# ======================================================================
# CONFIGURAZIONE
# ======================================================================

# --- Controlli booleani richiesti ---
PLOT_ANGULAR_VARIABLES = True
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
DR_TRUTH_MATCH_TAU = 0.2

OUTPUT_DIR = Path("output/obj_3.1/pair_kinematics_categories")

JET_PT_BRANCH = "recojet_antikt4PFlow_pt___NOSYS"
TAU_PT_BRANCH = "tau_pt___NOSYS"
JET_MASS_BRANCH = "recojet_antikt4PFlow_m___NOSYS"
JET_N_MUONS_BRANCH = "recojet_antikt4PFlow_n_muons___NOSYS"

DR_THRESHOLD_KINEMATICS = 0.4

PT_HIST_MIN = 0.0
PT_HIST_MAX = 200_000.0
PT_HIST_BINSIZE = 10_000

ETA_HIST_MIN = -5.0
ETA_HIST_MAX = 5.0
ETA_HIST_BINSIZE = 0.2

PHI_HIST_MIN = -np.pi
PHI_HIST_MAX = np.pi
PHI_HIST_BINSIZE = 0.2

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

# --- Configurazione Specifica Variabili (Dizionario) ---
VARIABLE_PLOT_CONFIG = {
    "jet_pt": {"y_scale": "log", "out_dir": OUTPUT_DIR},
    "jet_mass": {"y_scale": "log", "out_dir": OUTPUT_DIR },
    "jet_n_muons": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "jet_eta": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "jet_phi": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "tau_pt": {"y_scale": "log", "out_dir": OUTPUT_DIR },
    "tau_eta": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "tau_phi": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "tau_nProng": {"y_scale": "linear", "out_dir": OUTPUT_DIR },
    "tau_decayMode": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "tau_charge": {"y_scale": "linear", "out_dir": OUTPUT_DIR },
    "pair_pt_ratio": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "pair_dr": {"y_scale": "log", "out_dir": OUTPUT_DIR / "angular"},
    "pair_deta": {"y_scale": "log", "out_dir": OUTPUT_DIR / "angular"},
    "pair_dphi": {"y_scale": "log", "out_dir": OUTPUT_DIR / "angular"},
}

# --- Definizione dinamica delle variabili ---
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
        ("eta", ETA_HIST_MIN, ETA_HIST_MAX, ETA_HIST_BINSIZE, r"$\eta$"),
        ("phi", PHI_HIST_MIN, PHI_HIST_MAX, PHI_HIST_BINSIZE, r"$\phi$"),
    ])
    TAU_VARIABLES.extend([
        ("eta", ETA_HIST_MIN, ETA_HIST_MAX, ETA_HIST_BINSIZE, r"$\eta$"),
        ("phi", PHI_HIST_MIN, PHI_HIST_MAX, PHI_HIST_BINSIZE, r"$\phi$"),
    ])
    PAIR_VARIABLES.extend([
        ("dr", 0.0, DR_THRESHOLD_KINEMATICS, PAIR_ANGULAR_BINSIZE, r"$\Delta R$"),
        ("deta", -DR_THRESHOLD_KINEMATICS, DR_THRESHOLD_KINEMATICS, PAIR_ANGULAR_BINSIZE, r"$\Delta \eta$"),
        ("dphi", -DR_THRESHOLD_KINEMATICS, DR_THRESHOLD_KINEMATICS, PAIR_ANGULAR_BINSIZE, r"$\Delta \phi$"),
    ])


# ======================================================================
# COSTRUZIONE COPPIE
# ======================================================================

def build_pair_kinematics_and_labels(
    jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
    tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
):
    jet_pt_ct, tau_pt_ct = ak.unzip(ak.cartesian([jet_pt, tau_pt], nested=True))
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    
    jet_mass_ct, _ = ak.unzip(ak.cartesian([jet_mass, tau_pt], nested=True))
    jet_n_muons_ct, _ = ak.unzip(ak.cartesian([jet_n_muons, tau_pt], nested=True))
    
    _, tau_nProng_ct = ak.unzip(ak.cartesian([jet_pt, tau_nProng], nested=True))
    _, tau_decayMode_ct = ak.unzip(ak.cartesian([jet_pt, tau_decayMode], nested=True))
    _, tau_charge_ct = ak.unzip(ak.cartesian([jet_pt, tau_charge], nested=True))
    
    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)
    deta_matrix = jet_eta_ct - tau_eta_ct
    dphi_matrix = (jet_phi_ct - tau_phi_ct + np.pi) % (2 * np.pi) - np.pi
    pt_ratio_matrix = jet_pt_ct / tau_pt_ct

    return {
        "pair_dr": dr_matrix,
        "pair_pt_ratio": pt_ratio_matrix,
        "pair_deta": deta_matrix,
        "pair_dphi": dphi_matrix,
        "jet_pt": jet_pt_ct,
        "jet_eta": jet_eta_ct,
        "jet_phi": jet_phi_ct,
        "jet_mass": jet_mass_ct,
        "jet_n_muons": jet_n_muons_ct,
        "tau_pt": tau_pt_ct,
        "tau_eta": tau_eta_ct,
        "tau_phi": tau_phi_ct,
        "tau_nProng": tau_nProng_ct,
        "tau_decayMode": tau_decayMode_ct,
        "tau_charge": tau_charge_ct,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }


def get_overlapping_pairs_kinematics(pair_info, dr_thr):
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
        "jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_n_muons",
        "tau_pt", "tau_eta", "tau_phi", "tau_nProng", "tau_decayMode", "tau_charge",
        "pair_dr", "pair_deta", "pair_dphi", "pair_pt_ratio"
    ]

    for cat, cmask in cat_masks.items():
        for var in variables:
            res[cat][var] = flat_overlap(var)[cmask]

    return res


def merge_pair_kinematics(parts_list):
    variables = [
        "jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_n_muons",
        "tau_pt", "tau_eta", "tau_phi", "tau_nProng", "tau_decayMode", "tau_charge",
        "pair_dr", "pair_deta", "pair_dphi", "pair_pt_ratio"
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

def save_pair_kinematics_plots(merged_data, dr_threshold):
    if not HAS_MPL:
        return
    
    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    for obj_name, obj_title in (("jet", "jet"), ("tau", "tau"), ("pair", "coppia")):
        
        if obj_name == "jet":
            var_list = JET_VARIABLES
        elif obj_name == "tau":
            var_list = TAU_VARIABLES
        else:
            var_list = PAIR_VARIABLES
        
        for var_key, vmin, vmax, step, xlabel in var_list:
            
            var_id = f"{obj_name}_{var_key}"
            config = VARIABLE_PLOT_CONFIG.get(var_id, {"y_scale": "linear", "out_dir": OUTPUT_DIR / "pair_kinematics_categories"})
            
            out_dir = Path(config["out_dir"])
            if AGGREGATE_CATEGORIES:
                out_dir = out_dir / "aggregated"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            y_scale = config["y_scale"]

            bins = np.arange(vmin, vmax + step, step)
            fig, ax = plt.subplots(figsize=(8, 5))

            datasets = []
            for cat in cat_keys:
                values = merged_data[cat][var_id]
                datasets.append((values, labels[cat], colors[cat]))

            _plot_hist_curves(ax, datasets, bins)
            
            _finish_object_plot(
                ax,
                f"{obj_title} {xlabel}" if obj_name != "pair" else xlabel,
                f"Cinematica {obj_title} in coppie (pair-level) - $\\Delta R < {dr_threshold}$",
                normalize=NORMALIZE_HISTOGRAMS
            )
            
            ax.set_yscale(y_scale)

            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_HISTOGRAMS else ""
            if var_key == "pt":
                suffix_cut = f"_cut_{PT_HIST_MAX}" 
            elif var_key== "mass":
                suffix_cut = f"_cut_{M_HIST_MAX}" 
            else:
                suffix_cut = ""
            suffix_agg = "_aggregated" if AGGREGATE_CATEGORIES else ""

            out_path = (
                out_dir 
                / f"pair_kinematics_{var_id}_"
                f"{JET_SELECTION_MODE}{suffix}{suffix_cut}{suffix_agg}.png"
            )

            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[OK] Plot salvato in: {out_path}")

def save_scatter_plots(merged_data, dr_threshold):
    if not HAS_MPL:
        return
    
    # Utilizza le costanti globali originali per colori, etichette e categorie
    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    out_dir = OUTPUT_DIR / "scatters"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configurazione per i due plot richiesti
    scatters_config = [
        ("jet_pt", "pT Jet [MeV]"),
        ("tau_pt", "pT Tau [MeV]")
    ]

    for var_key, ylabel in scatters_config:
        fig, ax = plt.subplots(figsize=(8, 6))

        for cat in cat_keys:
            dr_vals = merged_data[cat]["pair_dr"]
            pt_vals = merged_data[cat][var_key]
            count = len(dr_vals)
            
            # Creazione label con conteggio degli oggetti per categoria
            label_con_conteggio = f"{labels[cat]} (N={count})"
            
            ax.scatter(
                dr_vals, 
                pt_vals, 
                label=label_con_conteggio, 
                color=colors[cat], 
                alpha=0.6, 
                s=15, 
                edgecolors='none'
            )

        ax.set_xlabel(r"$\Delta R$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Scatter: $\\Delta R$ vs {ylabel} ($\\Delta R < {dr_threshold}$)")
        
        # Mantiene la scala logaritmica per i pT come da configurazione originale
        ax.set_yscale("log")
        
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
        "tau_nProng", "tau_decayMode", "tau_charge",
        "recojet_antikt4PFlow_eta", "recojet_antikt4PFlow_phi", JET_PT_BRANCH,
        JET_MASS_BRANCH, JET_N_MUONS_BRANCH,
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
        JET_TRUTH_LABEL_BRANCH,  # <-- Includi SEMPRE questo ramo
    ]

    if JET_SELECTION_MODE == "btag85":
        branches.append(JET_BTAG_BRANCH)
        
    if TAU_SELECTION_MODE == "score85":
        branches.append(TAU_SCORE_BRANCH)
        
    if TRUTH_MODE_TAU == "label":
        branches.append(TAU_TRUTH_MATCH_BRANCH)
    elif TRUTH_MODE_TAU == "geometric":
        branches.extend([TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH])

    pair_kinematics_parts = []

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
        jet_mass = a[JET_MASS_BRANCH][jet_sel]
        jet_n_muons = a[JET_N_MUONS_BRANCH][jet_sel]

        tau_eta = a["tau_eta"][tau_sel]
        tau_phi = a["tau_phi"][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]
        tau_nProng = a["tau_nProng"][tau_sel]
        tau_decayMode = a["tau_decayMode"][tau_sel]
        tau_charge = a["tau_charge"][tau_sel]

        pair_info = build_pair_kinematics_and_labels(
            jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
            tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
        )

        pair_kinematics_parts.append(
            get_overlapping_pairs_kinematics(pair_info, DR_THRESHOLD_KINEMATICS)
        )
    merged_pair_kinematics = merge_pair_kinematics(pair_kinematics_parts)
    
    section("PLOT CINEMATICA PAIR-LEVEL")
    save_pair_kinematics_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)
    
    section("SCATTER PLOTS")
    save_scatter_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)

if __name__ == "__main__":
    main()