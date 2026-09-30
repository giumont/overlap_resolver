"""
================================================================================
Analisi della distribuzione e calibrazione per branch continui b-tagging
(es. recojet_antikt4PFlow_ftag_effSF_GN2v01_Continuous___NOSYS)
================================================================================

Questo script analizza empiricamente un branch continuo associato al b-tagging,
calcolando le efficienze reali sui veri b-jet del dataset al variare di 
una soglia continua, e mappa le soglie necessarie per ottenere efficienze target.
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

# Import utility dalla pipeline base (come nello script precedente)
from obj_3_1 import (
    ROOT_DIR, FILE_PREFIX, FILE_SUFFIX, TREE_NAME, N_ENTRIES_CAP,
    section, load_files, get_analysis_selection, summarize,
)

import sys
sys.path.append("../flavour_tagging")
from src.graphics import (
    plot_roc_curve,
    plot_background_rejection,
    plot_score_distribution,
    plot_efficiency_vs_threshold,
    plot_confusion_matrix,
    compute_classification_metrics,
    compute_working_points
)

# ======================================================================
# CONFIGURAZIONE
# ======================================================================

JET_ANALYSIS_SEL_BRANCH = "recojet_antikt4PFlow_isAnalysisJet___NOSYS"
JET_SCORE_BRANCH = "recojet_antikt4PFlow_ftag_effSF_GN2v01_Continuous___NOSYS"

JET_TRUTH_LABEL_BRANCH = "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
JET_TRUTH_LABEL_B_VALUE = 5

JET_BTAG_WP85_BRANCH = "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85"
JET_BTAG_WP85_NOMINAL_EFF = 85

OUTPUT_DIR = Path("output/obj_3_2/b_tagger_continuous")

N_SCAN_POINTS = 100
TARGET_EFFICIENCIES = [70, 75, 80, 85, 90, 97]

# Assunzione sul verso del taglio: True = (score >= soglia) seleziona il segnale.
# Se l'analisi del plot mostra curve invertite, cambiare in False.
HIGHER_IS_BETTER = True 

# ======================================================================
# UTILITY
# ======================================================================

def flatten_clean(values):
    flat = ak.flatten(values, axis=None)
    flat = ak.drop_none(flat)
    flat = ak.to_numpy(flat)
    if flat.size > 0 and np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]
    return flat

# ======================================================================
# ANALISI DISTRIBUZIONE CONTINUA
# ======================================================================

def print_continuous_properties(n_files, counts_per_event, score_all, is_true_all):
    section(f"PROPRIETA' DI {JET_SCORE_BRANCH} (jet analysis-level)")

    print(f"   n. file processati: {n_files}")
    print(f"   shape dopo flatten: {score_all.shape}")

    if score_all.size == 0:
        print("   nessun valore disponibile.")
        return

    print(f"\n   intervallo osservato:")
    print(f"      min = {np.min(score_all):.6f}")
    print(f"      max = {np.max(score_all):.6f}")

    print(f"\n   percentili sull'intera collezione:")
    print(f"      mean   = {np.mean(score_all):.6f}")
    print(f"      median = {np.median(score_all):.6f}")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"      p{p:02d}    = {np.percentile(score_all, p):.6f}")

    n_true = int(np.sum(is_true_all))
    n_fake = int(np.sum(~is_true_all))
    print(f"\n   composizione per truth-matching:")
    print(f"      vero jet-b: {n_true:8d}  ({100.0 * n_true / score_all.size:.3f}%)")
    print(f"      fake      : {n_fake:8d}  ({100.0 * n_fake / score_all.size:.3f}%)")

# ======================================================================
# EFFICIENZA REALE SU ASSE CONTINUO
# ======================================================================

def compute_efficiency_curve(score_true, score_fake, thresholds):
    n_true = max(score_true.size, 1)
    n_fake = max(score_fake.size, 1)

    if HIGHER_IS_BETTER:
        eff_true = np.array([np.sum(score_true >= thr) / n_true for thr in thresholds])
        eff_fake = np.array([np.sum(score_fake >= thr) / n_fake for thr in thresholds])
    else:
        eff_true = np.array([np.sum(score_true <= thr) / n_true for thr in thresholds])
        eff_fake = np.array([np.sum(score_fake <= thr) / n_fake for thr in thresholds])

    return eff_true, eff_fake

def save_calibration_plot(thresholds, eff_true, eff_fake):
    if not HAS_MPL:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(thresholds, eff_true, "-", color="tab:blue", linewidth=2, label=r"$\epsilon_b$ REALE (veri b)")
    ax.plot(thresholds, eff_fake, "--", color="tab:gray", linewidth=2, label=r"$\epsilon_{bkg}$ REALE (fake)")

    ax.set_xlabel(f"Valore di {JET_SCORE_BRANCH}")
    ax.set_ylabel("Efficienza")
    ax.set_title("Efficienza reale in funzione della soglia continua")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    out_path = OUTPUT_DIR / "continuous_score_efficiency.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Plot salvato in: {out_path}")

def print_empirical_working_points(score_true, score_fake, thresholds, eff_true, eff_fake):
    section("SOGLIE EMPIRICHE RICALIBRATE SUL BRANCH CONTINUO")
    print(f"   {'Target Eff.':>12} | {'Soglia stimata':>15} | {'Eff. Bkg':>12} | {'Rejection':>12}")
    print("   " + "-" * 58)

    for target in TARGET_EFFICIENCIES:
        target_frac = target / 100.0
        idx = np.argmin(np.abs(eff_true - target_frac))
        thr = thresholds[idx]
        f_eff = eff_fake[idx]
        rej = 1.0 / f_eff if f_eff > 0 else np.inf
        
        print(f"   {target:>11d}% | {thr:>15.6f} | {f_eff*100:>11.2f}% | {rej:>12.2f}")

# ======================================================================
# MAIN
# ======================================================================

def main():
    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
        return

    branches = [JET_SCORE_BRANCH, JET_ANALYSIS_SEL_BRANCH, JET_TRUTH_LABEL_BRANCH]
    score_parts, is_true_parts, counts_parts = [], [], []

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]
        
        if JET_SCORE_BRANCH not in tree.keys():
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")
        jet_sel = get_analysis_selection(tree, JET_ANALYSIS_SEL_BRANCH, n_entries)

        score_sel = a[JET_SCORE_BRANCH][jet_sel]
        truth_sel = ak.fill_none(a[JET_TRUTH_LABEL_BRANCH][jet_sel], -1) == JET_TRUTH_LABEL_B_VALUE

        counts_parts.append(ak.num(score_sel, axis=1))
        score_parts.append(score_sel)
        is_true_parts.append(truth_sel)

    if not score_parts:
        print("Branch continuo non trovato nei file.")
        return

    score_all_ak = ak.concatenate(score_parts)
    is_true_all_ak = ak.concatenate(is_true_parts)
    counts_per_event = ak.to_numpy(ak.concatenate(counts_parts))

    score_all = flatten_clean(score_all_ak)
    is_true_flat = ak.to_numpy(ak.flatten(is_true_all_ak, axis=None))
    score_all_raw = ak.to_numpy(ak.flatten(score_all_ak, axis=None))
    is_true_all = is_true_flat[np.isfinite(score_all_raw)].astype(bool)

    print_continuous_properties(len(score_parts), counts_per_event, score_all, is_true_all)

    score_true = score_all[is_true_all]
    score_fake = score_all[~is_true_all]

    if score_true.size > 0:
        scan_min, scan_max = np.min(score_all), np.max(score_all)
        thresholds = np.linspace(scan_min, scan_max, N_SCAN_POINTS)
        
        eff_true, eff_fake = compute_efficiency_curve(score_true, score_fake, thresholds)
        save_calibration_plot(thresholds, eff_true, eff_fake)
        print_empirical_working_points(score_true, score_fake, thresholds, eff_true, eff_fake)

        section("SUITE DI VALUTAZIONE ESTERNA (graphics.py)")
        target_effs_frac = [t / 100.0 for t in TARGET_EFFICIENCIES]
        
        # Gestione del verso per graphics.py
        y_score_for_roc = score_all if HIGHER_IS_BETTER else -score_all
        
        fpr, tpr, thresholds_roc, auc = plot_roc_curve(
            y_true=is_true_all, y_score=y_score_for_roc,
            save_dir=OUTPUT_DIR, filename="continuous_roc_curve"
        )
        print(f"   AUC: {auc:.4f}")

if __name__ == "__main__":
    main()