"""
================================================================================
Analisi preliminare su recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous:
verifica se le soglie di efficienza nominali (implicite nel quantile) sono
attendibili su QUESTO dataset, e derivazione di soglie ricalibrate.
================================================================================

Questo script e' pensato per essere eseguito nella stessa cartella di
obj_3_1.py, di cui riusa le funzioni di utilita' (section, load_files,
get_analysis_selection, summarize) e la configurazione dei file di
input (ROOT_DIR, FILE_PREFIX, FILE_SUFFIX, TREE_NAME, N_ENTRIES_CAP).
E' il pendant, per il jet b-tagging, di tau_score_analysis.py.

MOTIVAZIONE
-----------
recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous NON e' uno score
grezzo: e' gia' un quantile calibrato da ATLAS su un campione di
riferimento (ttbar "standard"), tale che, per costruzione, tagliare a
quantile <= x dovrebbe corrispondere a un'efficienza di segnale ~ x sul
campione di calibrazione. La domanda che questo script indirizza e':
questa calibrazione tiene anche su QUESTO dataset (stessa fisica, ma
possibili differenze di selezione/cinematica a monte)?

Lo script quindi:
  1) stampa le proprieta' del quantile sui jet analysis-level (e per
     sottocampione vero b / fake);
  2) calcola l'efficienza di segnale REALE (misurata sui b veri di
     questo dataset) al variare della soglia sul quantile, e la
     confronta punto per punto con l'efficienza NOMINALE implicita nel
     valore del quantile stesso (quantile == efficienza attesa);
  3) come controllo indipendente, verifica l'efficienza reale del
     working point gia' pronto GN2v01 FixedCutBEff_85
     (JET_BTAG_BRANCH), che nominalmente deve dare 85%;
  4) deriva, per la stessa famiglia di target TARGET_EFFICIENCIES usata
     per il tau, le soglie RICALIBRATE sul quantile (percentile della
     distribuzione del quantile sui b veri di questo dataset), da
     riusare al posto delle soglie nominali se il punto 2 mostra uno
     scostamento significativo.

CONVENZIONI ADOTTATE
---------------------
- Collezione jet: solo jet analysis-level
  (recojet_antikt4PFlow_isAnalysisJet___NOSYS != 0), stessa selezione
  di base gia' usata in obj_3_1.py / truth_vs_reco_params.py.
- "Vero jet-b" (segnale): recojet_antikt4PFlow_HadronConeExclTruthLabelID
  == 5, stessa convenzione ("label") gia' usata in
  truth_vs_reco_params.py (JET_TRUTH_LABEL_BRANCH / JET_TRUTH_LABEL_B_VALUE).
- "Fake" (solo come riferimento): jet analysis-level con truth label
  != 5.
- Convenzione del quantile: PIU' BASSO = PIU' signal-like (quantile
  piccolo = jet molto b-like). E' l'OPPOSTO della convenzione dello
  score del tau (li' piu' alto = piu' signal-like). L'efficienza di
  segnale a una soglia thr e' quindi definita come
  P(quantile <= thr | vero b). Come per il tau, il verso va comunque
  verificato guardando il segno della separazione nel plot (punto 2):
  se le due popolazioni risultassero invertite rispetto a questa
  assunzione, invertire il verso del taglio prima di usare le soglie
  del punto 4.
- Il confronto "nominale vs reale" al punto 2 usa il valore del
  quantile stesso come proxy dell'efficienza nominale attesa a quella
  soglia (thr -> efficienza nominale attesa = thr), che e' esattamente
  la definizione di un quantile calibrato correttamente.

STRUTTURA
---------
PARTE A: proprieta' del quantile sui jet analysis-level, generali e
         per sottocampione (vero b/fake).
PARTE B: efficienza REALE vs efficienza NOMINALE al variare della
         soglia sul quantile, plot di confronto (la diagonale nominale
         e' una retta a 45 gradi se la calibrazione e' perfetta).
PARTE C: controllo indipendente sul WP gia' pronto FixedCutBEff_85.
PARTE D: soglie RICALIBRATE sul quantile, per percentile della
         distribuzione del quantile sui b veri di questo dataset,
         nella stessa famiglia TARGET_EFFICIENCIES usata per il tau.
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

JET_ANALYSIS_SEL_BRANCH = "recojet_antikt4PFlow_isAnalysisJet___NOSYS"

# Branch del quantile GN2 (score gia' calibrato in efficienza)
JET_QUANTILE_BRANCH = "recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous"

# Branch di truth-matching usato per definire "vero jet-b" (stessa
# convenzione gia' usata in truth_vs_reco_params.py)
JET_TRUTH_LABEL_BRANCH = "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
JET_TRUTH_LABEL_B_VALUE = 5

# WP gia' pronto, usato come controllo indipendente al punto 3
JET_BTAG_WP85_BRANCH = (
    "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85"
)
JET_BTAG_WP85_NOMINAL_EFF = 85  # percento

OUTPUT_DIR = Path("output/obj_3_2/b_tagger")

# Numero di punti usati per campionare la curva di efficienza fra il
# minimo e il massimo osservato del quantile
N_SCAN_POINTS = 400

# Famiglia di working point a cui corrispondono le entries del branch del quantile
# (derivate guardando le percentuali di efficenze reali sul dataset e arrotondando)
TARGET_EFFICIENCIES = [70, 75, 80, 85, 90, 97]  # percento


# ======================================================================
# UTILITY LOCALI
# ======================================================================

def flatten_clean(values):
    """
    Flatten (rimuovendo None/NaN) di un array jagged/regular verso un
    array numpy 1D, stessa logica gia' usata in tau_score_analysis.py.
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
# PARTE A - PROPRIETA' DEL QUANTILE (punto 1, analogo al tau)
# ======================================================================

def print_quantile_properties(n_files, counts_per_event, quant_all,
                               is_true_all):

    section(f"PROPRIETA' DI {JET_QUANTILE_BRANCH} (jet analysis-level)")

    print(f"   branch (ROOT): std::vector<float>, per evento")
    print(f"   n. file processati: {n_files}")
    print(
        f"   n. jet per evento (analysis-level) - "
        f"min={np.min(counts_per_event) if counts_per_event.size else 0}  "
        f"max={np.max(counts_per_event) if counts_per_event.size else 0}  "
        f"mean={np.mean(counts_per_event) if counts_per_event.size else 0:.4f}"
    )
    print(f"   shape dopo flatten (n. jet analysis-level totali): {quant_all.shape}")

    if quant_all.size == 0:
        print("   nessun valore disponibile: interrompo la stampa delle proprieta'.")
        return

    print(f"   dtype: {quant_all.dtype}")
    print(f"\n   intervallo osservato:")
    print(f"      min = {np.min(quant_all):.6f}")
    print(f"      max = {np.max(quant_all):.6f}")

    print(f"\n   valori (percentili sull'intera collezione):")
    print(f"      mean   = {np.mean(quant_all):.6f}")
    print(f"      median = {np.median(quant_all):.6f}")
    print(f"      std    = {np.std(quant_all):.6f}")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"      p{p:02d}    = {np.percentile(quant_all, p):.6f}")

    n_true = int(np.sum(is_true_all))
    n_fake = int(np.sum(~is_true_all))
    print(
        f"\n   composizione per truth-matching "
        f"({JET_TRUTH_LABEL_BRANCH} == {JET_TRUTH_LABEL_B_VALUE} = vero jet-b):"
    )
    print(f"      vero jet-b: {n_true:8d}  ({100.0 * n_true / quant_all.size:.3f}%)")
    print(f"      fake      : {n_fake:8d}  ({100.0 * n_fake / quant_all.size:.3f}%)")

    section("PROPRIETA' PER SOTTOCAMPIONE (vero jet-b vs fake)")
    summarize(ak.Array(quant_all[is_true_all]), "quantile - vero jet-b")
    summarize(ak.Array(quant_all[~is_true_all]), "quantile - fake")


# ======================================================================
# PARTE B - EFFICIENZA REALE VS NOMINALE (punto 2)
# ======================================================================

def compute_efficiency_curve(quant_true, quant_fake, thresholds):
    n_true = max(quant_true.size, 1)
    n_fake = max(quant_fake.size, 1)

    # Inversione verso: il valore intero piu alto indica maggiore purita
    eff_true = np.array(
        [np.sum(quant_true >= thr) / n_true for thr in thresholds]
    )
    eff_fake = np.array(
        [np.sum(quant_fake >= thr) / n_fake for thr in thresholds]
    )

    return eff_true, eff_fake

def save_calibration_plot(thresholds, eff_true, eff_fake):
    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip del plot di calibrazione."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Efficienze nominali corrispondenti ai bin [1, 2, 3, 4, 5, 6]
    eff_nominal = np.array([0.97, 0.90, 0.85, 0.80, 0.75, 0.70])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Linee orizzontali di riferimento per ogni valore di efficienza nominale
    for i, eff_val in enumerate(eff_nominal):
        ax.axhline(
            y=eff_val,
            color="red",
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
            label="Livelli nominali WP" if i == 0 else "",
        )

    ax.plot(
        thresholds, eff_true, "o-",
        color="tab:blue", linewidth=1.8,
        label=r"$\epsilon_b$ REALE (misurata, HadronConeExclTruthLabelID==5)",
    )
    ax.plot(
        thresholds, eff_nominal, "s--",
        color="black", linewidth=1.2,
        label=r"$\epsilon_b$ NOMINALE attesa",
    )
    ax.plot(
        thresholds, eff_fake, "^:",
        color="tab:gray", linewidth=1.2,
        label=r"$\epsilon_{bkg}$ REALE (truth label $\neq$ 5)",
    )

    ax.set_xlabel(f"Bin discreto su {JET_QUANTILE_BRANCH}")
    ax.set_ylabel("Efficienza / Frazione")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(thresholds)

    # Tick del y-axis che mostrano sia i riferimenti generali sia le percentuali dei WP
    y_ticks = sorted(list(set([0.0, 0.2, 0.4, 0.6, 0.70, 0.75, 0.80, 0.85, 0.90, 0.97, 1.0])))
    ax.set_yticks(y_ticks)

    ax.set_title(
        "GN2v01: efficienza di b-tag reale vs nominale per bin discreto\n"
    )
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "jet_gn2_quantile_calibration.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot salvato in: {out_path}")

def print_nominal_vs_real(quant_true):
    section("EFFICIENZA REALE VS NOMINALE PER BIN DISCRETO (punto 2)")
    
    # Mappatura standard dei bin discreti ATLAS GN2 ai WP nominali - VEDI "=== ANALISI DISCRETA PER BIN INTERO ===" PER CONVINCERTI
    wp_nominal_map = {
        6: 70.0,  # Bin 6 -> Nominal ~70%
        5: 75.0,  # Bin 5 -> Nominal ~75%
        4: 80.0,  # Bin 4 -> Nominal ~80%
        3: 85.0,  # Bin 3 -> Nominal ~85%
        2: 90.0,  # Bin 2 -> Nominal ~90%
        1: 97.0   # Bin 1 -> Nominal ~97% (Loose)
    }

    n_true_tot = quant_true.size
    print(f"   {'Bin (>=)':>10} | {'Eff. Nominale':>15} | {'Eff. Reale':>12} | {'Scostamento (pp)':>18}")
    print("   " + "-" * 65)

    for b in [6, 5, 4, 3, 2, 1]:
        eff_real = 100.0 * np.sum(quant_true >= b) / n_true_tot
        eff_nom = wp_nominal_map[b]
        delta = eff_real - eff_nom
        print(f"   {b:>10d} | {eff_nom:>14.1f}% | {eff_real:>11.3f}% | {delta:>+17.3f}")

# ======================================================================
# PARTE C - CONTROLLO INDIPENDENTE SUL WP GIA' PRONTO (punto 3)
# ======================================================================

def check_fixed_wp85(a_parts, jet_sel_parts, is_true_ak_parts):
    """
    Controllo aggregato su tutti i file: efficienza reale del WP
    gia' pronto FixedCutBEff_85, misurata sui veri jet-b di questo
    dataset.
    """

    section(
        "CONTROLLO INDIPENDENTE: efficienza reale del WP gia' pronto "
        f"{JET_BTAG_WP85_BRANCH}"
    )

    wp_pass_all = []
    is_true_all = []

    for a, jet_sel, truth_sel in zip(a_parts, jet_sel_parts, is_true_ak_parts):
        if JET_BTAG_WP85_BRANCH not in a.fields:
            continue
        wp_pass_sel = ak.fill_none(a[JET_BTAG_WP85_BRANCH][jet_sel], False) != 0
        wp_pass_all.append(flatten_clean(ak.values_astype(wp_pass_sel, np.float64)) != 0)
        is_true_all.append(flatten_clean(ak.values_astype(truth_sel, np.float64)) != 0)

    if not wp_pass_all:
        print(f"   [WARNING] branch {JET_BTAG_WP85_BRANCH} non trovato in nessun file. Skip.")
        return

    wp_pass_flat = np.concatenate(wp_pass_all)
    is_true_flat = np.concatenate(is_true_all)

    if is_true_flat.size == 0 or wp_pass_flat.size != is_true_flat.size:
        print("   [WARNING] array non allineati o vuoti. Skip del controllo.")
        return

    n_true = int(np.sum(is_true_flat))
    if n_true == 0:
        print("   [WARNING] nessun vero jet-b trovato. Skip del controllo.")
        return

    eff_real = float(np.sum(wp_pass_flat & is_true_flat)) / n_true

    print(
        f"   efficienza nominale WP FixedCutBEff_85: "
        f"{JET_BTAG_WP85_NOMINAL_EFF}%"
    )
    print(f"   efficienza reale misurata su questo dataset: {100.0 * eff_real:.3f}%")
    print(
        f"   scostamento: {100.0 * eff_real - JET_BTAG_WP85_NOMINAL_EFF:+.3f} pp"
    )


# ======================================================================
# PARTE D - SOGLIE RICALIBRATE (punto 4, analogo al punto 3 del tau)
# ======================================================================

# def compute_recalibrated_working_points(quant_true, quant_fake, target_efficiencies):
#     wp_table = []
#     bins = np.array([6, 5, 4, 3, 2, 1])
#     n_true = max(quant_true.size, 1)
#     n_fake = max(quant_fake.size, 1)

#     # Calcolo dell'efficienza reale per ciascun bin discreto (verso >=)
#     bin_effs = np.array([np.sum(quant_true >= b) / n_true for b in bins])
#     bin_fakes = np.array([np.sum(quant_fake >= b) / n_fake for b in bins])

#     for target in target_efficiencies:
#         target_frac = target / 100.0
#         # Selezione del bin intero con l'efficienza reale piu vicina al target
#         best_idx = np.argmin(np.abs(bin_effs - target_frac))
#         best_bin = bins[best_idx]
#         eff_meas = bin_effs[best_idx]
#         fake_acc = bin_fakes[best_idx]

#         wp_table.append((target, best_bin, eff_meas, fake_acc))

#     return wp_table

def print_recalibrated_working_points(wp_table):
    section(
        "SOGLIE RICALIBRATE SUL QUANTILE (punto 4) - famiglia di working "
        f"point {TARGET_EFFICIENCIES}%"
    )

    print(
        f"   {'target eff.':>12} {'soglia ricalibrata':>20} "
        f"{'eff. misurata':>14} {'bkg eff.':>15} {'bkg rejection':>15}"
    )

    for wp in wp_table:
        target_pct = wp["target_efficiency"] * 100 if wp["target_efficiency"] <= 1.0 else wp["target_efficiency"]
        rej = wp["background_rejection"]
        rej_str = f"{rej:.2f}" if np.isfinite(rej) else "inf"
        print(
            f"   {target_pct:>11.0f}% {wp['threshold']:>20.1f} "
            f"{100.0 * wp['achieved_efficiency']:>13.3f}% "
            f"{100.0 * wp['background_efficiency']:>14.3f}% {rej_str:>15}"
        )

    print(
        "\n   NOTA: La 'soglia ricalibrata' identifica il bin discreto ottimale "
        "sui b veri di QUESTO dataset. Se si discosta dalle efficienze attese, "
        "il tagger necessita di ricalibrazione dell'asse."
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
        JET_QUANTILE_BRANCH,
        JET_ANALYSIS_SEL_BRANCH,
        JET_TRUTH_LABEL_BRANCH,
        JET_BTAG_WP85_BRANCH,
    ]

    quant_parts = []
    is_true_parts = []
    counts_per_event_parts = []
    n_files_used = 0

    # Per il controllo indipendente (Parte C) serve l'array "a" e la
    # selezione per file: li riaccumuliamo qui per semplicita', dato
    # che i file di questa pipeline sono di dimensioni gestibili.
    a_parts_for_wp_check = []
    jet_sel_parts_for_wp_check = []
    is_true_ak_parts_for_wp_check = []

    for item in loaded:
        tree, n_entries, filename = item["tree"], item["n_entries"], item["file_name"]

        keys = set(tree.keys())
        missing = [b for b in branches if b not in keys]
        if missing:
            print(f"[WARNING] {filename}: branch mancanti: {missing}")
            print("    Verificare i PLACEHOLDER in CONFIGURAZIONE. File ignorato.")
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel = get_analysis_selection(
            tree, JET_ANALYSIS_SEL_BRANCH, n_entries,
        )

        quant_sel = a[JET_QUANTILE_BRANCH][jet_sel]
        truth_sel = ak.fill_none(
            a[JET_TRUTH_LABEL_BRANCH][jet_sel], -1,
        ) == JET_TRUTH_LABEL_B_VALUE

        counts_per_event_parts.append(ak.num(quant_sel, axis=1))
        quant_parts.append(quant_sel)
        is_true_parts.append(truth_sel)

        a_parts_for_wp_check.append(a)
        jet_sel_parts_for_wp_check.append(jet_sel)
        is_true_ak_parts_for_wp_check.append(truth_sel)

        n_files_used += 1

    if not quant_parts:
        print("Nessun file con i branch necessari. Interrompo.")
        return

    quant_all_ak = ak.concatenate(quant_parts)
    is_true_all_ak = ak.concatenate(is_true_parts)
    counts_per_event = ak.to_numpy(ak.concatenate(counts_per_event_parts))

    quant_all = flatten_clean(quant_all_ak)
    print("debug", np.unique(quant_all, return_counts=True))
    # is_true deve restare allineato a quant_all posizione per posizione
    is_true_flat = ak.to_numpy(ak.flatten(is_true_all_ak, axis=None))
    quant_all_raw = ak.to_numpy(ak.flatten(quant_all_ak, axis=None))
    finite_mask = np.isfinite(quant_all_raw)
    is_true_all = is_true_flat[finite_mask].astype(bool)

    # ------------------------------------------------------------------
    # PARTE A - punto 1
    # ------------------------------------------------------------------
    print_quantile_properties(
        n_files_used, counts_per_event, quant_all, is_true_all,
    )

    quant_true = quant_all[is_true_all]
    quant_fake = quant_all[~is_true_all]

    if quant_true.size == 0:
        print(
            "\n[WARNING] nessun vero jet-b trovato: impossibile "
            "calcolare l'efficienza (punto 2) e le soglie (punto 4)."
        )
        return
    
    print("\n=== ANALISI DISCRETA PER BIN INTERO ===")
    print(f"{'Bin (>=)':>10} | {'True b passati':>15} | {'Eff. True b':>12} | {'Fake passati':>15} | {'Fake Acc.':>12}")
    print("-" * 75)

    n_true_tot = quant_true.size
    n_fake_tot = quant_fake.size

    for b in [1, 2, 3, 4, 5, 6]:
        n_t_pass = np.sum(quant_true >= b)
        n_f_pass = np.sum(quant_fake >= b)
        eff_t = n_t_pass / n_true_tot if n_true_tot else 0
        eff_f = n_f_pass / n_fake_tot if n_fake_tot else 0
        print(f"{b:>10d} | {n_t_pass:>15d} | {100.0*eff_t:>11.2f}% | {n_f_pass:>15d} | {100.0*eff_f:>11.2f}%")

    # ------------------------------------------------------------------
    # PARTE B - punto 2
    # ------------------------------------------------------------------
    # Il quantile e' per definizione in [0, 1]: la scansione copre
    # questo intervallo (eventualmente esteso al range osservato, se
    # dovesse differire per qualche ragione dai dati).
    # scan_min = min(0.0, np.min(quant_all))
    # scan_max = max(1.0, np.max(quant_all))
    # thresholds = np.linspace(scan_min, scan_max, N_SCAN_POINTS)

    thresholds = np.array([1, 2, 3, 4, 5, 6], dtype=int)

    eff_true, eff_fake = compute_efficiency_curve(
        quant_true, quant_fake, thresholds,
    )

    #print_nominal_vs_real(thresholds, eff_true)
    print_nominal_vs_real(quant_true=quant_true)
    save_calibration_plot(thresholds, eff_true, eff_fake)

    # ------------------------------------------------------------------
    # PARTE C - punto 3 (controllo indipendente sul WP85 gia' pronto)
    # ------------------------------------------------------------------
    check_fixed_wp85(
        a_parts_for_wp_check, jet_sel_parts_for_wp_check, is_true_ak_parts_for_wp_check,
    )

    # ------------------------------------------------------------------
    # PARTE D - punto 4 (Integrazione graphics.py)
    # ------------------------------------------------------------------
    target_effs_frac = [t / 100.0 for t in TARGET_EFFICIENCIES]

    # 1. Calcolo ROC Curve
    fpr, tpr, thresholds_roc, auc = plot_roc_curve(
        y_true=is_true_all,
        y_score=quant_all,
        save_dir=OUTPUT_DIR,
        filename="bjet_gn2_roc_curve"
    )

    # 2. Estrazione Working Points 
    wp_table = compute_working_points(
        fpr, tpr, thresholds_roc,
        target_effs=target_effs_frac
    )

    # 3. Stampa
    print_recalibrated_working_points(wp_table)

    # 4. Diagnostic Graphics Aggiuntivi
    section("SUITE DI VALUTAZIONE (graphics.py)")
    print(f"   AUC calcolata: {auc:.4f}")

    plot_background_rejection(
        tpr, fpr,
        save_dir=OUTPUT_DIR,
        filename="bjet_gn2_background_rejection"
    )

    plot_score_distribution(
        y_true=is_true_all,
        y_score=quant_all,
        working_points=wp_table,
        save_dir=OUTPUT_DIR,
        filename="bjet_gn2_score_distribution"
    )

    plot_efficiency_vs_threshold(
        y_true=is_true_all,
        y_score=quant_all,
        save_dir=OUTPUT_DIR,
        filename="bjet_gn2_eff_vs_threshold"
    )

    thr_85 = next((wp["threshold"] for wp in wp_table if abs(wp["target_efficiency"] - 0.85) < 1e-4), 3.0)

    plot_confusion_matrix(
        y_true=is_true_all,
        y_score=quant_all,
        threshold=thr_85,
        save_dir=OUTPUT_DIR,
        filename="bjet_gn2_confusion_matrix_85"
    )

    metrics_85 = compute_classification_metrics(
        is_true_all, quant_all, threshold=thr_85
    )
    
    print(f"\n   Metriche a efficienza 85% (soglia bin >= {thr_85:.0f}):")
    for key, value in metrics_85.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.4f}")
        else:
            print(f"      {key}: {value}")

    section("FINE")

if __name__ == "__main__":
    main()