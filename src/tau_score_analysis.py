"""
================================================================================
Score tau_GNTauScoreSigTrans_v0prune: proprieta', efficienza di
riconoscimento del tau ed soglie operative comparabili al b-tagging.
================================================================================

Questo script e' pensato per essere eseguito nella stessa cartella di
obj_3_1.py, di cui riusa le funzioni di utilita' (section, load_files,
get_analysis_selection, summarize) e la configurazione dei file di
input (ROOT_DIR, FILE_PREFIX, FILE_SUFFIX, TREE_NAME, N_ENTRIES_CAP),
per restare coerente con il resto della pipeline.

RICHIESTA
---------
    1) Stampare le proprieta' della variabile
       tau_GNTauScoreSigTrans_v0prune (shape, intervallo, valori al
       suo interno).
    2) Plottare l'efficienza di riconoscimento del tau al variare di
       questo score (efficienza di segnale = frazione di tau
       adronici veri, truth-matched, che superano un dato taglio sullo
       score).
    3) Definire, di conseguenza, delle soglie di score comparabili a
       quelle gia' usate per il b-tagging in questa analisi.

CONVENZIONI ADOTTATE
---------------------
- Collezione tau: solo tau "analysis-level"
  (tau_isAnalysisTau___NOSYS != 0), la stessa selezione di base gia'
  usata in obj_3_1.py.
- "Vero tau adronico" (segnale, denominatore/numeratore
  dell'efficienza): tau_truth_IsHadronicTau != 0. E' la STESSA
  convenzione ("label") gia' usata per TRUTH_MODE_TAU in
  truth_vs_reco_params.py (li' chiamato TAU_TRUTH_MATCH_BRANCH), per
  restare coerenti con il resto della pipeline.
- "Fake" (non-segnale, usato solo come riferimento/contesto):
  tau_truth_IsHadronicTau == 0 fra i tau analysis-level. Non e' un
  vero e proprio campione di background (potrebbe includere tau reali
  non truth-matched a livello vis, oppure jet), ma serve solo a dare
  un riscontro visivo di quanto lo score separi le due popolazioni.
- Convenzione dello score: piu' alto = piu' "signal-like" (segnale =
  vero tau adronico), come per i punteggi *SigTrans* dei tagger GN*
  gia' usati per il b-tagging in questa analisi (vedi JET_BTAG_BRANCH
  in obj_3_1.py). L'efficienza di segnale a una soglia thr e' quindi
  definita come P(score >= thr | vero tau adronico). DA VERIFICARE
  guardando il segno della separazione nel plot (punto 2): se le due
  distribuzioni (vero/fake) risultassero invertite rispetto a questa
  assunzione, invertire il verso del taglio (>= thr sostituito da
  <= thr) prima di usare le soglie del punto 3.
- Working point "comparabili al b-tagging": in questa analisi il
  b-tagging usa il working point GN2v01 FixedCutBEff_85, cioe' una
  soglia sullo score definita per ottenere una fissata efficienza di
  segnale (85%) sui b-jet veri. La famiglia standard di working point
  ATLAS per i tagger GN* e' {60, 70, 77, 85, 90}% di efficienza di
  segnale (vedi TARGET_EFFICIENCIES sotto): la stessa logica
  "FixedCutBEff-like" viene qui applicata allo score del tau, per
  ottenere soglie definite allo stesso modo (stessa famiglia di target
  di efficienza), NON gli stessi valori numerici di soglia (che sono
  specifici del discriminante).

STRUTTURA
---------
PARTE A: lettura dello score sui tau analysis-level, proprieta'
         generali e per sottocampione (vero/fake) - punto 1.
PARTE B: curva di efficienza di segnale al variare della soglia sullo
         score, e relativo plot - punto 2.
PARTE C: soglie operative alle efficienze target TARGET_EFFICIENCIES,
         per percentile della distribuzione dello score sui tau veri
         - punto 3.
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

# Riuso delle utility e della configurazione file gia' scritte e
# validate in obj_3_1.py
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


# Selezionare quale score tau analizzare:
# - "tau_GNTauScoreSigTrans_v0prune"
# - "tau_RNNJetScoreSigTrans"
# - "tau_RNNEleScoreSigTrans_v1"
TAU_SCORE_BRANCH = "tau_RNNJetScoreSigTrans"

TAU_ANALYSIS_SEL_BRANCH = "tau_isAnalysisTau___NOSYS"

# Branch di truth-matching usato per definire "vero tau adronico"
# (stessa convenzione "label" gia' usata altrove nella pipeline)
TAU_TRUTH_MATCH_BRANCH = "tau_truth_IsHadronicTau"

OUTPUT_DIR = Path(f"output/obj_3_2/tau_tagger/{TAU_SCORE_BRANCH}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Numero di punti usati per campionare la curva di efficienza fra il
# minimo e il massimo osservato dello score (punto 2). Il range non e'
# fissato a priori: viene determinato dai dati (vedi PARTE A).
N_SCAN_POINTS = 400

# Working point di efficienza di segnale "comparabili" a quelli gia'
# usati per il b-tagging in questa analisi (vedi CONVENZIONI ADOTTATE
# nel docstring iniziale)
TARGET_EFFICIENCIES = [70, 75, 80, 85, 90, 97]  # percento


# ======================================================================
# UTILITY LOCALI
# ======================================================================

def flatten_clean(values):
    """
    Flatten (rimuovendo None/NaN) di un array jagged/regular verso un
    array numpy 1D, con la stessa logica gia' usata in obj_3_1.py
    dentro summarize().
    """

    flat = ak.flatten(values, axis=None)
    flat = ak.drop_none(flat)
    flat = ak.to_numpy(flat)

    if flat.size == 0:
        return flat

    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]

    return flat


# ======================================================================
# PARTE A - PROPRIETA' DELLA VARIABILE (punto 1)
# ======================================================================

def print_score_properties(n_taus_raw, counts_per_event, score_all,
                            is_true_all):
    """
    Stampa shape, intervallo e valori (percentili) dello score, sia
    sull'intera collezione di tau analysis-level sia separatamente per
    vero tau adronico / fake, per dare un riscontro immediato di
    quanto le due popolazioni siano separate dallo score.
    """

    section(f"PROPRIETA' DI {TAU_SCORE_BRANCH} (tau analysis-level)")

    print(f"   branch (ROOT): std::vector<float>, per evento")
    print(f"   n. eventi processati: {n_taus_raw}")
    print(
        f"   n. tau per evento (analysis-level) - "
        f"min={np.min(counts_per_event) if counts_per_event.size else 0}  "
        f"max={np.max(counts_per_event) if counts_per_event.size else 0}  "
        f"mean={np.mean(counts_per_event) if counts_per_event.size else 0:.4f}"
    )
    print(f"   shape dopo flatten (n. tau analysis-level totali): {score_all.shape}")

    if score_all.size == 0:
        print("   nessun valore disponibile: interrompo la stampa delle proprieta'.")
        return

    print(f"   dtype: {score_all.dtype}")
    print(f"\n   intervallo osservato:")
    print(f"      min = {np.min(score_all):.6f}")
    print(f"      max = {np.max(score_all):.6f}")

    print(f"\n   valori (percentili sull'intera collezione):")
    print(f"      mean   = {np.mean(score_all):.6f}")
    print(f"      median = {np.median(score_all):.6f}")
    print(f"      std    = {np.std(score_all):.6f}")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"      p{p:02d}    = {np.percentile(score_all, p):.6f}")

    n_true = int(np.sum(is_true_all))
    n_fake = int(np.sum(~is_true_all))
    print(
        f"\n   composizione per truth-matching "
        f"({TAU_TRUTH_MATCH_BRANCH} != 0 = vero tau adronico):"
    )
    print(f"      vero tau adronico: {n_true:8d}  ({100.0 * n_true / score_all.size:.3f}%)")
    print(f"      fake             : {n_fake:8d}  ({100.0 * n_fake / score_all.size:.3f}%)")

    section("PROPRIETA' PER SOTTOCAMPIONE (vero tau adronico vs fake)")
    summarize(ak.Array(score_all[is_true_all]), "score - vero tau adronico")
    summarize(ak.Array(score_all[~is_true_all]), "score - fake")


# ======================================================================
# PARTE B - EFFICIENZA DI RICONOSCIMENTO DEL TAU (punto 2)
# ======================================================================

def compute_efficiency_curve(score_true, score_fake, thresholds):
    """
    Per ciascuna soglia thr in thresholds, calcola:
        eff_true[i] = P(score >= thr | vero tau adronico)
                      (efficienza di riconoscimento del tau, punto 2)
        eff_fake[i] = P(score >= thr | fake)
                      (accettanza sul fake, solo come riferimento)

    Convenzione: score piu' alto = piu' signal-like (vedi
    CONVENZIONI ADOTTATE nel docstring iniziale).
    """

    n_true = max(score_true.size, 1)
    n_fake = max(score_fake.size, 1)

    eff_true = np.array(
        [np.sum(score_true >= thr) / n_true for thr in thresholds]
    )
    eff_fake = np.array(
        [np.sum(score_fake >= thr) / n_fake for thr in thresholds]
    )

    return eff_true, eff_fake


def save_efficiency_plot(thresholds, eff_true, eff_fake, wp_table):
    if not HAS_MPL:
        print("\n[WARNING] matplotlib non disponibile: skip del plot di efficienza.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        thresholds, eff_true,
        color="tab:blue", linewidth=1.8,
        label=r"$\epsilon_{\tau}$, (tau_truth_IsHadronicTau == 1)",
    )
    ax.plot(
        thresholds, eff_fake,
        color="tab:gray", linewidth=1.2, linestyle="--",
        label=r"$\epsilon_{bkg}$, (tau_truth_IsHadronicTau == 0)",
    )

    for wp in wp_table:
        thr_wp = wp["threshold"]
        target_pct = wp["target_efficiency"] * 100 if wp["target_efficiency"] <= 1.0 else wp["target_efficiency"]
        ax.axvline(thr_wp, color="tab:red", linestyle=":", linewidth=1.0)
        ax.text(
            thr_wp, 1.02, f"{target_pct:.0f}%",
            rotation=0, ha="center", va="bottom",
            fontsize=8, color="tab:red",
        )

    ax.set_xlabel(f"soglia su {TAU_SCORE_BRANCH}")
    ax.set_ylabel("efficienza")
    ax.set_ylim(0.0, 1.08)
    ax.set_title(
        "Efficienza di riconoscimento del tau al variare dello score\n"
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = OUTPUT_DIR / fr"tau_gn_score{TAU_SCORE_BRANCH}_efficiency_vs_threshold.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot salvato in: {out_path}")

# ======================================================================
# PARTE C - SOGLIE OPERATIVE COMPARABILI AL B-TAGGING (punto 3)
# ======================================================================

# def compute_working_points(score_true, score_fake, target_efficiencies):
#     """
#     Per ciascuna efficienza target (in %), definisce la soglia sullo
#     score come il percentile della distribuzione dello score sui VERI
#     tau adronici tale per cui la frazione di veri tau con
#     score >= soglia sia pari al target (stessa logica "FixedCutBEff"
#     gia' usata per il b-tagging: soglia fissata per ottenere una
#     efficienza di segnale fissata).

#     Ritorna una lista di tuple (target_eff, soglia, efficienza
#     misurata alla soglia, accettanza sul fake alla soglia), da
#     stampare e da riusare per il plot.
#     """

#     wp_table = []

#     for target in target_efficiencies:
#         # P(score >= thr) = target/100  <=>  thr = percentile(score, 100-target)
#         thr = float(np.percentile(score_true, 100 - target))

#         eff_meas = float(np.mean(score_true >= thr))
#         eff_bkg = float(np.mean(score_fake >= thr)) if score_fake.size else float("nan")
#         rej_bkg = 1 / eff_bkg if eff_bkg > 0 else float("inf")

#         wp_table.append((target, thr, eff_meas, eff_bkg, rej_bkg))

#     return wp_table


def print_working_points(wp_table):
    section(
        "SOGLIE OPERATIVE COMPARABILI AL B-TAGGING (punto 3) - "
        f"famiglia di working point {TARGET_EFFICIENCIES}% "
        "(stessa logica FixedCutBEff)"
    )

    print(
        f"   {'target eff.':>12} {'soglia score':>14} "
        f"{'eff. misurata':>14} {'bkg eff.':>16} {'bkg rejection':>15}"
    )

    for wp in wp_table:
        target_pct = wp["target_efficiency"] * 100 if wp["target_efficiency"] <= 1.0 else wp["target_efficiency"]
        rej = wp["background_rejection"]
        rej_str = f"{rej:.2f}" if np.isfinite(rej) else "inf"
        print(
            f"   {target_pct:>11.0f}% {wp['threshold']:>14.6f} "
            f"{100.0 * wp['achieved_efficiency']:>13.3f}% "
            f"{100.0 * wp['background_efficiency']:>15.3f}% {rej_str:>15}"
        )

# ======================================================================
# MAIN
# ======================================================================

def main():

    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
        return

    branches = [
        TAU_SCORE_BRANCH,
        TAU_ANALYSIS_SEL_BRANCH,
        TAU_TRUTH_MATCH_BRANCH,
    ]

    score_parts = []
    is_true_parts = []
    counts_per_event_parts = []
    n_files_used = 0

    for item in loaded:
        tree, n_entries, filename = item["tree"], item["n_entries"], item["file_name"]

        keys = set(tree.keys())
        missing = [b for b in branches if b not in keys]
        if missing:
            print(f"[WARNING] {filename}: branch mancanti: {missing}")
            print("    Verificare i PLACEHOLDER in CONFIGURAZIONE. File ignorato.")
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        tau_sel = get_analysis_selection(
            tree, TAU_ANALYSIS_SEL_BRANCH, n_entries,
        )

        score_sel = a[TAU_SCORE_BRANCH][tau_sel]
        truth_sel = ak.fill_none(a[TAU_TRUTH_MATCH_BRANCH][tau_sel], 0) != 0

        counts_per_event_parts.append(ak.num(score_sel, axis=1))
        score_parts.append(score_sel)
        is_true_parts.append(truth_sel)

        n_files_used += 1

    if not score_parts:
        print("Nessun file con i branch necessari. Interrompo.")
        return

    score_all_ak = ak.concatenate(score_parts)
    is_true_all_ak = ak.concatenate(is_true_parts)
    counts_per_event = ak.to_numpy(ak.concatenate(counts_per_event_parts))

    score_all = flatten_clean(score_all_ak)
    # is_true deve restare allineato a score_all posizione per posizione:
    # niente drop_none/finite qui, ma applichiamo la stessa maschera di
    # finitezza usata da flatten_clean su score_all, cosi' i due array
    # restano sincronizzati.
    is_true_flat = ak.to_numpy(ak.flatten(is_true_all_ak, axis=None))
    score_all_raw = ak.to_numpy(ak.flatten(score_all_ak, axis=None))
    finite_mask = np.isfinite(score_all_raw)
    is_true_all = is_true_flat[finite_mask].astype(bool)

    # ------------------------------------------------------------------
    # PARTE A - punto 1
    # ------------------------------------------------------------------
    print_score_properties(
        n_files_used, counts_per_event, score_all, is_true_all,
    )

    score_true = score_all[is_true_all]
    score_fake = score_all[~is_true_all]

    if score_true.size == 0:
        print(
            "\n[WARNING] nessun vero tau adronico trovato: impossibile "
            "calcolare l'efficienza (punto 2) e le soglie (punto 3)."
        )
        return

    # ------------------------------------------------------------------
    # PARTE B - punto 2
    # ------------------------------------------------------------------
    thresholds = np.linspace(
        np.min(score_all), np.max(score_all), N_SCAN_POINTS,
    )
    eff_true, eff_fake = compute_efficiency_curve(
        score_true, score_fake, thresholds,
    )

    section("CURVA DI EFFICIENZA (punto 2)")
    print(
        f"   soglia scansionata fra {thresholds[0]:.6f} e "
        f"{thresholds[-1]:.6f} ({N_SCAN_POINTS} punti)"
    )
    for target in TARGET_EFFICIENCIES:
        idx = int(np.argmin(np.abs(eff_true - target / 100.0)))
        print(
            f"   eff. riconoscimento tau ~= {target}%  "
            f"-> soglia score ~= {thresholds[idx]:.6f}  "
            f"(eff. misurata alla soglia scansionata = "
            f"{100.0 * eff_true[idx]:.3f}%)"
        )

    # ------------------------------------------------------------------
    # PARTE C - punto 3 & PARTE D (Integrazione graphics.py)
    # ------------------------------------------------------------------
    target_effs_frac = [t / 100.0 for t in TARGET_EFFICIENCIES]

    # 1. Calcolo ROC Curve
    fpr, tpr, thresholds_roc, auc = plot_roc_curve(
        y_true=is_true_all,
        y_score=score_all,
        save_dir=OUTPUT_DIR,
        filename="tau_roc_curve"
    )

    # 2. Estrazione Working Points con le uscite della ROC
    wp_table = compute_working_points(
        fpr, tpr, thresholds_roc,
        target_effs=target_effs_frac
    )

    # 3. Stampa e Salvataggio Plot Efficienza
    print_working_points(wp_table)
    save_efficiency_plot(thresholds, eff_true, eff_fake, wp_table)

    # 4. Diagnostic Graphics Aggiuntivi
    section("SUITE DI VALUTAZIONE (graphics.py)")
    print(f"   AUC calcolata: {auc:.4f}")

    plot_background_rejection(
        tpr, fpr,
        save_dir=OUTPUT_DIR,
        filename="tau_background_rejection",
        eff_range = (0.7, 1)   # matcho con efficienze disponibili per b-tagger

    )

    plot_score_distribution(
        y_true=is_true_all,
        y_score=score_all,
        working_points=wp_table,
        save_dir=OUTPUT_DIR,
        filename="tau_score_distribution"
    )

    plot_efficiency_vs_threshold(
        y_true=is_true_all,
        y_score=score_all,
        save_dir=OUTPUT_DIR,
        filename="tau_eff_vs_threshold"
    )

    thr_85 = next((wp["threshold"] for wp in wp_table if abs(wp["target_efficiency"] - 0.85) < 1e-4), 0.5)

    plot_confusion_matrix(
        y_true=is_true_all,
        y_score=score_all,
        threshold=thr_85,
        save_dir=OUTPUT_DIR,
        filename="tau_confusion_matrix_85"
    )

    metrics_85 = compute_classification_metrics(
        is_true_all, score_all, threshold=thr_85
    )
    
    print(f"\n   Metriche a efficienza 85% (soglia = {thr_85:.6f}):")
    for key, value in metrics_85.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.4f}")
        else:
            print(f"      {key}: {value}")

    section("FINE")


if __name__ == "__main__":
    main()