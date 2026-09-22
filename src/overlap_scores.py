"""
Script overlap_scores.py
Analisi a livello di coppia (pair-level) per gli score di identificazione 
di jet e tau in overlap spaziale.
Le distribuzioni sono separate per categorie di verità e 
annotate con le soglie di efficienza calcolate dinamicamente.
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
    ROOT_DIR, load_files, get_analysis_selection, delta_r, section
)
from truth_vs_reco_params import label_jets_and_taus, _plot_hist_curves

# ======================================================================
# CONFIGURAZIONE
# ======================================================================

OUTPUT_DIR = Path("output/obj_3.1/overlap_scores")
DR_THRESHOLD = 4.0

JET_SCORE_BRANCH = "recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous"
JET_TRUTH_LABEL_BRANCH = "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
JET_TRUTH_LABEL_B_VALUE = 5

# Selezionare quale score tau analizzare:
# - "tau_GNTauScoreSigTrans_v0prune"
# - "tau_RNNJetScoreSigTrans"
# - "tau_RNNEleScoreSigTrans_v1"
TAU_SCORE_BRANCH =  "tau_GNTauScoreSigTrans_v0prune"
TAU_TRUTH_MATCH_BRANCH = "tau_truth_IsHadronicTau"

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

# Configurazioni target efficienze
TARGET_EFFICIENCIES = [70, 75, 80, 85, 90, 97]
WP_COLORS = ["tab:blue", "cornflowerblue", "lightsteelblue", "peachpuff", "coral", "tab:red"]

JET_QUANTILES_MAP = {
    -1: "N/A\n(-1)", 1: "97%\n(1)", 2: "90%\n(2)", 3: "85%\n(3)",
    4: "80%\n(4)", 5: "75%\n(5)", 6: "70%\n(6)"
}

JET_PT_BRANCH = "recojet_antikt4PFlow_pt___NOSYS"
TAU_PT_BRANCH = "tau_pt___NOSYS"

# ======================================================================
# UTILITY WORKING POINTS
# ======================================================================

def compute_working_points(score_true, target_efficiencies, colors):
    """
    Calcola le soglie sullo score corrispondenti alle efficienze target fornite,
    utilizzando l'intera popolazione dei tau adronici veri.
    """
    wp_list = []
    for target, color in zip(target_efficiencies, colors):
        thr = float(np.percentile(score_true, 100 - target))
        wp_list.append((thr, f"{target}%", color))
    return wp_list

# ======================================================================
# ESTRAZIONE E MATCHING
# ======================================================================

def build_pair_scores(
    jet_eta, jet_phi, jet_pt, jet_score, jet_label,
    tau_eta, tau_phi, tau_pt, tau_score, tau_label
):
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    jet_pt_ct, tau_pt_ct = ak.unzip(ak.cartesian([jet_pt, tau_pt], nested=True))
    jet_score_ct, tau_score_ct = ak.unzip(ak.cartesian([jet_score, tau_score], nested=True))
    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)
    overlap_mask = dr_matrix < DR_THRESHOLD

    def get_flat_overlap(arr):
        flat_arr = ak.to_numpy(ak.flatten(arr, axis=None))
        flat_mask = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat_arr[flat_mask]

    j_pt = get_flat_overlap(jet_pt_ct)
    t_pt = get_flat_overlap(tau_pt_ct)
    j_score = get_flat_overlap(jet_score_ct)
    t_score = get_flat_overlap(tau_score_ct)
    j_lab = get_flat_overlap(jet_label_ct)
    t_lab = get_flat_overlap(tau_label_ct)

    cat_masks = {
        "a_jet_true_tau_fake": (j_lab == True) & (t_lab == False),
        "b_jet_false_tau_true": (j_lab == False) & (t_lab == True),
        "c_jet_true_tau_true": (j_lab == True) & (t_lab == True),
        "d_jet_false_tau_false": (j_lab == False) & (t_lab == False),
    }

    res = {cat: {} for cat in cat_masks.keys()}
    for cat, mask in cat_masks.items():
        res[cat]["jet_pt"] = j_pt[mask]
        res[cat]["tau_pt"] = t_pt[mask]
        res[cat]["jet_score"] = j_score[mask]
        res[cat]["tau_score"] = t_score[mask]
    
    return res

def merge_score_data(parts_list):
    merged = {cat: {"jet_pt": [], "tau_pt": [], "jet_score": [], "tau_score": []} for cat in CATEGORY_LABELS.keys()}
    for part in parts_list:
        for cat in CATEGORY_LABELS.keys():
            merged[cat]["jet_pt"].append(part[cat]["jet_pt"])
            merged[cat]["tau_pt"].append(part[cat]["tau_pt"])
            merged[cat]["jet_score"].append(part[cat]["jet_score"])
            merged[cat]["tau_score"].append(part[cat]["tau_score"])
            
    for cat in CATEGORY_LABELS.keys():
        merged[cat]["jet_pt"] = np.concatenate(merged[cat]["jet_pt"]) if merged[cat]["jet_pt"] else np.array([])
        merged[cat]["tau_pt"] = np.concatenate(merged[cat]["tau_pt"]) if merged[cat]["tau_pt"] else np.array([])
        merged[cat]["jet_score"] = np.concatenate(merged[cat]["jet_score"]) if merged[cat]["jet_score"] else np.array([])
        merged[cat]["tau_score"] = np.concatenate(merged[cat]["tau_score"]) if merged[cat]["tau_score"] else np.array([])
    return merged

# ======================================================================
# PLOTTING
# ======================================================================

def save_score_plots(merged_data, tau_wps):
    if not HAS_MPL:
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- PLOT 1: TAU SCORE ---
    fig_tau, ax_tau = plt.subplots(figsize=(9, 6))
    tau_bins = np.linspace(0, 1.0, 100)
    
    for cat, label in CATEGORY_LABELS.items():
        data = merged_data[cat]["tau_score"]
        color = CATEGORY_COLORS[cat]
        if len(data) > 0:
            ax_tau.hist(
                data, 
                bins=tau_bins, 
                histtype="step", 
                linewidth=1.5, 
                density=True, 
                label=f"{label} (N={len(data)})", 
                color=color
            )

    # Aggiunta soglie Tau ID dinamiche
    for threshold, eff, color in tau_wps:
        ax_tau.axvline(x=threshold, color=color, linestyle="--", linewidth=1.2, label=f"WP {eff} ({threshold:.3f})")

    ax_tau.set_xlabel(f"Tau ID Score\n({TAU_SCORE_BRANCH})")
    ax_tau.set_ylabel("Densità normalizzata")
    ax_tau.set_title(f"Distribuzione Tau ID Score per categorie di verità ($\\Delta R < {DR_THRESHOLD}$)")
    ax_tau.legend(fontsize=8, loc='upper right', ncol=2)
    ax_tau.grid(alpha=0.3)
    
    out_tau = OUTPUT_DIR / fr"tau_score_{TAU_SCORE_BRANCH}_truth_categories_branch.png"
    fig_tau.tight_layout()
    fig_tau.savefig(out_tau, dpi=150)
    plt.close(fig_tau)
    print(f"[OK] Plot score tau salvato in: {out_tau}")

    # --- PLOT 2: JET B-TAG QUANTILES ---
    fig_jet, ax_jet = plt.subplots(figsize=(9, 6))
    jet_bins = np.arange(-1.5, 7.5, 1.0)

    for cat, label in CATEGORY_LABELS.items():
        data = merged_data[cat]["jet_score"]
        color = CATEGORY_COLORS[cat]
        if len(data) > 0:
            ax_jet.hist(
                data, 
                bins=jet_bins, 
                histtype="step", 
                linewidth=1.5, 
                density=True, 
                label=f"{label} (N={len(data)})", 
                color=color
            )

    ax_jet.set_xticks([-1, 1, 2, 3, 4, 5, 6])
    ax_jet.set_xticklabels([JET_QUANTILES_MAP[k] for k in [-1, 1, 2, 3, 4, 5, 6]])
    ax_jet.set_xlabel(f"Jet b-tag Target Efficiency Quantile\n({JET_SCORE_BRANCH})")
    ax_jet.set_ylabel("Densità normalizzata")
    ax_jet.set_title(f"Distribuzione b-tag quantiles per categorie di verità ($\\Delta R < {DR_THRESHOLD}$)")
    ax_jet.legend(fontsize=9, loc='upper right')
    ax_jet.grid(alpha=0.3, axis='y')
    
    out_jet = OUTPUT_DIR / "jet_quantile_truth_categories.png"
    fig_jet.tight_layout()
    fig_jet.savefig(out_jet, dpi=150)
    plt.close(fig_jet)
    print(f"[OK] Plot score jet salvato in: {out_jet}")

def save_score_scatters(merged_data):
    if not HAS_MPL:
        return
        
    out_dir = OUTPUT_DIR / "scatters"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    scatters_config = [
        ("jet_pt", "jet_score", "pT Jet [MeV]", f"Jet b-tag Target Efficiency Quantile\n({JET_SCORE_BRANCH})"),
        ("tau_pt", "tau_score", "pT Tau [MeV]", f"Tau ID Score\n({TAU_SCORE_BRANCH})")
    ]
    
    cat_keys = list(CATEGORY_LABELS.keys())
    
    for pt_key, score_key, xlabel, ylabel in scatters_config:
        # Griglia 2x2 per separare le categorie
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        
        is_jet_score = (score_key == "jet_score")
        
        for idx, cat in enumerate(cat_keys):
            ax = axes_flat[idx]
            pt_vals = merged_data[cat][pt_key]
            score_vals = merged_data[cat][score_key]
            label = CATEGORY_LABELS[cat]
            color = CATEGORY_COLORS[cat]
            count = len(pt_vals)
            
            # Applica Jittering solo per lo score del Jet che è discreto
            if is_jet_score and count > 0:
                jitter = np.random.uniform(-0.18, 0.18, size=count)
                y_plot = score_vals + jitter
            else:
                y_plot = score_vals
                
            # Calcolo trasparenza adattiva in base al numero di punti
            alpha_val = max(0.05, min(0.5, 1000.0 / max(count, 1)))

            ax.scatter(
                pt_vals, 
                y_plot, 
                color=color,
                alpha=alpha_val, 
                s=8, 
                edgecolors='none'
            )

            ax.set_xscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            
            # Titolo per singolo subplot con il conteggio
            ax.set_title(f"{label}\n(N = {count:,})", fontsize=10, fontweight='bold')
            
            # Configurazione assi specifica per il Jet Quantile
            if is_jet_score:
                ax.set_yticks([-1, 1, 2, 3, 4, 5, 6])
                ax.set_yticklabels([JET_QUANTILES_MAP[k] for k in [-1, 1, 2, 3, 4, 5, 6]])
                ax.set_ylim(-1.6, 6.6)
            else:
                ax.set_ylim(-0.05, 1.05)

        # Labels comuni per la griglia
        for ax in axes[1, :]:
            ax.set_xlabel(xlabel, fontsize=11)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=11)
            
        fig.suptitle(f"Scatter: {ylabel.splitlines()[0]} vs {xlabel} ($\\Delta R < {DR_THRESHOLD}$)", fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if score_key == "tau_score":
            filename = f"grid_scatter_{score_key}_{TAU_SCORE_BRANCH}_vs_pt.png"
        else:
            filename = f"grid_scatter_{score_key}_vs_pt.png"
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"[OK] Grid Scatter plot salvato in: {out_path}")

# ======================================================================
# MAIN
# ======================================================================

def main():
    loaded = load_files()
    if not loaded:
        return

    # Aggiungi i rami a list branches
    branches = [
        "tau_eta", "tau_phi", TAU_PT_BRANCH, "tau_isAnalysisTau___NOSYS", 
        TAU_SCORE_BRANCH, TAU_TRUTH_MATCH_BRANCH,
        "recojet_antikt4PFlow_eta", "recojet_antikt4PFlow_phi", JET_PT_BRANCH, "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
        JET_SCORE_BRANCH, JET_TRUTH_LABEL_BRANCH
    ]

    parts = []
    global_true_tau_scores = []

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]
        
        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        # Selezione "all": solo analysis-level
        jet_sel = get_analysis_selection(tree, "recojet_antikt4PFlow_isAnalysisJet___NOSYS", n_entries)
        tau_sel = get_analysis_selection(tree, "tau_isAnalysisTau___NOSYS", n_entries)

        jet_eta = a["recojet_antikt4PFlow_eta"][jet_sel]
        jet_phi = a["recojet_antikt4PFlow_phi"][jet_sel]
        jet_score = a[JET_SCORE_BRANCH][jet_sel]
        jet_label = a[JET_TRUTH_LABEL_BRANCH][jet_sel] == JET_TRUTH_LABEL_B_VALUE

        tau_eta = a["tau_eta"][tau_sel]
        tau_phi = a["tau_phi"][tau_sel]
        tau_score = a[TAU_SCORE_BRANCH][tau_sel]
        tau_label = a[TAU_TRUTH_MATCH_BRANCH][tau_sel] == 1
        
        # Estrazione score per il calcolo globale dei working point
        tau_score_flat = ak.to_numpy(ak.flatten(tau_score, axis=None))
        tau_label_flat = ak.to_numpy(ak.flatten(tau_label, axis=None))
        valid_mask = np.isfinite(tau_score_flat)
        global_true_tau_scores.append(tau_score_flat[valid_mask & tau_label_flat])

        jet_pt = a[JET_PT_BRANCH][jet_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]
        
        parts.append(build_pair_scores(
            jet_eta, jet_phi, jet_pt, jet_score, jet_label,
            tau_eta, tau_phi, tau_pt, tau_score, tau_label
        ))

    # Calcolo working points usando l'intera popolazione dei tau veri processati
    all_true_taus = np.concatenate(global_true_tau_scores) if global_true_tau_scores else np.array([])
    if len(all_true_taus) == 0:
        print("Errore: Nessun vero tau adronico trovato nei file processati per calcolare i WP.")
        return
        
    tau_wps = compute_working_points(all_true_taus, TARGET_EFFICIENCIES, WP_COLORS)

    merged = merge_score_data(parts)
    
    section("PLOT SCORES (PAIR-LEVEL)")
    print(f"Calcolate {len(tau_wps)} soglie dinamiche per il branch {TAU_SCORE_BRANCH}.")
    save_score_plots(merged, tau_wps)

    section("SCATTER PLOTS (SCORE VS PT)")
    save_score_scatters(merged)


if __name__ == "__main__":
    main()