from pathlib import Path
import numpy as np
import awkward as ak
from scipy.stats import spearmanr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from obj_3_1 import (
    ROOT_DIR, FILE_PREFIX, FILE_SUFFIX, TREE_NAME, N_ENTRIES_CAP,
    section, load_files, get_analysis_selection, delta_r
)

# ======================================================================
# CONFIGURAZIONE
# ======================================================================
DR_OVERLAP_THRESHOLD = 0.4

# Selezionare quale score tau analizzare:
# - "tau_GNTauScoreSigTrans_v0prune"
# - "tau_RNNJetScoreSigTrans"
# - "tau_RNNEleScoreSigTrans_v1"
TAU_SCORE_BRANCH =  "tau_RNNEleScoreSigTrans_v1"

TAU_TRUTH_MATCH_BRANCH = "tau_truth_IsHadronicTau"

JET_SCORE_BRANCH = "recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous"
JET_TRUTH_LABEL_BRANCH = "recojet_antikt4PFlow_HadronConeExclTruthLabelID"

JET_TRUTH_B_VALUE = 5
TAU_TRUTH_TRUE_VALUE = 1

# Configurazioni target efficienze
TARGET_EFFICIENCIES = [70, 75, 80, 85, 90, 97]
WP_COLORS = ["tab:blue", "cornflowerblue", "lightsteelblue", "peachpuff", "coral", "tab:red"]


OUTPUT_DIR = Path(f"output/obj_3_2/corr_scores_overlap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# METRICHE STATISTICHE E SOGLIE
# ======================================================================

def compute_correlation_ratio(categories, measurements):
    """
    Calcola il Correlation Ratio (eta) tra una variabile categorica e una continua.
    Gestisce correttamente categorie non ordinali (come -1) misurando la dispersione
    delle medie di categoria rispetto alla media globale.
    """
    cats = np.array(categories)
    meas = np.array(measurements)
    
    total_mean = np.mean(meas)
    ss_total = np.sum((meas - total_mean) ** 2)
    
    if ss_total == 0:
        return 0.0
        
    ss_between = 0.0
    for cat_val in np.unique(cats):
        mask = (cats == cat_val)
        cat_mean = np.mean(meas[mask])
        n_cat = np.sum(mask)
        ss_between += n_cat * (cat_mean - total_mean) ** 2
        
    return np.sqrt(ss_between / ss_total)

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
# ESTRAZIONE E SELEZIONE
# ======================================================================

def extract_overlapping_scores(
    jet_eta, jet_phi, jet_score,
    tau_eta, tau_phi, tau_score,
    dr_threshold,
    jet_truth=None, tau_truth=None
):
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    jet_score_ct, tau_score_ct = ak.unzip(ak.cartesian([jet_score, tau_score], nested=True))
    
    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)
    overlap_mask = dr_matrix < dr_threshold
    
    def apply_mask(arr):
        flat = ak.to_numpy(ak.flatten(arr, axis=None))
        mask_flat = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat[mask_flat]

    j_scores_overlap = apply_mask(jet_score_ct)
    t_scores_overlap = apply_mask(tau_score_ct)
    
    valid_mask = (
        np.isfinite(j_scores_overlap) & 
        np.isfinite(t_scores_overlap) &
        (j_scores_overlap >= -1.0) &  # Accetta -1 o superiore per i bin
        (t_scores_overlap > -10.0)
    )
    
    res = {
        "jet_score": j_scores_overlap[valid_mask],
        "tau_score": t_scores_overlap[valid_mask]
    }

    if jet_truth is not None and tau_truth is not None:
        j_truth_ct, t_truth_ct = ak.unzip(ak.cartesian([jet_truth, tau_truth], nested=True))
        res["jet_truth"] = apply_mask(j_truth_ct)[valid_mask]
        res["tau_truth"] = apply_mask(t_truth_ct)[valid_mask]

    return res

# ======================================================================
# PLOTTING PER CATEGORIE DI VERITÀ
# ======================================================================

def save_truth_type_violin_plot(jet_scores, tau_scores, jet_truth, tau_truth, score_name, tau_wps):
    if not HAS_MPL:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    categories = {
        "(a) Jet Vero / Tau Vero": (jet_truth == JET_TRUTH_B_VALUE) & (tau_truth == TAU_TRUTH_TRUE_VALUE),
        "(b) Jet Vero / Tau Fake": (jet_truth == JET_TRUTH_B_VALUE) & (tau_truth != TAU_TRUTH_TRUE_VALUE),
        "(c) Jet Fake / Tau Vero": (jet_truth != JET_TRUTH_B_VALUE) & (tau_truth == TAU_TRUTH_TRUE_VALUE),
        "(d) Jet Fake / Tau Fake": (jet_truth != JET_TRUTH_B_VALUE) & (tau_truth != TAU_TRUTH_TRUE_VALUE),
    }

    quantile_eff_map = {
        -1: "N/A\n(-1)", 1: "97%\n(bin 1)", 2: "90%\n(bin 2)",
         3: "85%\n(bin 3)", 4: "80%\n(bin 4)", 5: "75%\n(bin 5)", 6: "70%\n(bin 6)"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (title, mask) in enumerate(categories.items()):
        ax = axes[idx]
        j_s = jet_scores[mask]
        t_s = tau_scores[mask]
        n_pairs = len(j_s)
        
        eta_ratio = np.nan
        spearman_rho = np.nan
        
        if n_pairs > 1:
            eta_ratio = compute_correlation_ratio(j_s, t_s)
            ordinal_mask = j_s > 0
            if np.sum(ordinal_mask) > 1:
                spearman_rho, _ = spearmanr(j_s[ordinal_mask], t_s[ordinal_mask])

        unique_quantiles = np.sort(np.unique(j_s))
        present_quantiles = [q for q in unique_quantiles if q in quantile_eff_map]
        
        if len(present_quantiles) > 0:
            data_by_quantile = [t_s[j_s == q] for q in present_quantiles]
            parts = ax.violinplot(data_by_quantile, positions=range(len(present_quantiles)), showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('#1f77b4')
                pc.set_alpha(0.6)
            ax.set_xticks(range(len(present_quantiles)))
            ax.set_xticklabels([quantile_eff_map[q] for q in present_quantiles])
        
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(f"Jet b-tag Target Efficiency / Quantile Bin ({JET_SCORE_BRANCH})")
        ax.set_ylabel(f"Tau Score ({score_name})")
        ax.set_title(title)

        # Disegno delle soglie di Working Point orizzontali
        for threshold, eff, color in tau_wps:
            # Aggiungiamo la label solo sul primo plot per non creare 4 legende ridondanti
            lbl = f"WP {eff} ({threshold:.3f})" if idx == 0 else None
            ax.axhline(y=threshold, color=color, linestyle="--", linewidth=1.2, alpha=0.8, zorder=2, label=lbl)
            
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right', title="Soglie Tau ID", ncol=2)

        textstr = (
            f"Coppie ($\\Delta R < {DR_OVERLAP_THRESHOLD}$): {n_pairs}\n"
            f"Corr. Ratio ($\\eta$): {eta_ratio:.3f}\n"
            f"Spearman $\\rho_s$ (bin>0): {spearman_rho:+.3f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"correlation_{score_name}_vs_btag_dr{DR_OVERLAP_THRESHOLD}_truth_types.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Plot salvato in: {out_path}")

# ======================================================================
# MAIN
# ======================================================================

def main():
    loaded = load_files()
    if not loaded:
        return

    branches = [
        "tau_eta", "tau_phi", "tau_isAnalysisTau___NOSYS", TAU_SCORE_BRANCH,
        "recojet_antikt4PFlow_eta", "recojet_antikt4PFlow_phi",
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS", JET_SCORE_BRANCH,
        JET_TRUTH_LABEL_BRANCH, TAU_TRUTH_MATCH_BRANCH
    ]

    all_jet_scores, all_tau_scores = [], []
    all_jet_truths, all_tau_truths = [], []
    global_true_tau_scores = []

    for item in loaded:
        tree, n_entries, filename = item["tree"], item["n_entries"], item["file_name"]
        if any(b not in tree.keys() for b in branches):
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel = get_analysis_selection(tree, "recojet_antikt4PFlow_isAnalysisJet___NOSYS", n_entries)
        tau_sel = get_analysis_selection(tree, "tau_isAnalysisTau___NOSYS", n_entries)

        # Estrazione score per il calcolo globale dei WP su *tutti* i tau veri a livello analysis (non solo overlap)
        tau_score_all = a[TAU_SCORE_BRANCH][tau_sel]
        tau_label_all = a[TAU_TRUTH_MATCH_BRANCH][tau_sel] == TAU_TRUTH_TRUE_VALUE
        tau_score_flat = ak.to_numpy(ak.flatten(tau_score_all, axis=None))
        tau_label_flat = ak.to_numpy(ak.flatten(tau_label_all, axis=None))
        valid_mask_wp = np.isfinite(tau_score_flat)
        global_true_tau_scores.append(tau_score_flat[valid_mask_wp & tau_label_flat])

        # Estrazione delle coppie in overlap
        extracted = extract_overlapping_scores(
            a["recojet_antikt4PFlow_eta"][jet_sel], a["recojet_antikt4PFlow_phi"][jet_sel], a[JET_SCORE_BRANCH][jet_sel],
            a["tau_eta"][tau_sel], a["tau_phi"][tau_sel], a[TAU_SCORE_BRANCH][tau_sel],
            DR_OVERLAP_THRESHOLD,
            jet_truth=a[JET_TRUTH_LABEL_BRANCH][jet_sel], tau_truth=a[TAU_TRUTH_MATCH_BRANCH][tau_sel]
        )
        
        all_jet_scores.append(extracted["jet_score"])
        all_tau_scores.append(extracted["tau_score"])
        all_jet_truths.append(extracted["jet_truth"])
        all_tau_truths.append(extracted["tau_truth"])

    # Calcolo Working Points
    all_true_taus = np.concatenate(global_true_tau_scores) if global_true_tau_scores else np.array([])
    if len(all_true_taus) == 0:
        print("Errore: Nessun vero tau adronico trovato nei file processati per calcolare i WP.")
        return
        
    tau_wps = compute_working_points(all_true_taus, TARGET_EFFICIENCIES, WP_COLORS)
    print(f"Calcolate {len(tau_wps)} soglie dinamiche per il branch {TAU_SCORE_BRANCH}.")

    j_s = np.concatenate(all_jet_scores) if all_jet_scores else np.array([])
    t_s = np.concatenate(all_tau_scores) if all_tau_scores else np.array([])
    j_t = np.concatenate(all_jet_truths) if all_jet_truths else np.array([])
    t_t = np.concatenate(all_tau_truths) if all_tau_truths else np.array([])

    section(f"ANALISI COMPARATIVA: {TAU_SCORE_BRANCH}")
    print(f"Coppie estratte in overlap (dR < {DR_OVERLAP_THRESHOLD}): {len(j_s)}")
    
    if len(j_s) > 0:
        save_truth_type_violin_plot(j_s, t_s, j_t, t_t, score_name=TAU_SCORE_BRANCH, tau_wps=tau_wps)

if __name__ == "__main__":
    main()