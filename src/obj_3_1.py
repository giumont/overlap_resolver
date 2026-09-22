"""
================================================================================
OBIETTIVO 3.1 - Matching geometrico jet <-> tau a livello RECO
================================================================================

OBIETTIVO
---------
Quantificare l'overlap geometrico tra le collezioni RECO di jet
(recojet_antikt4PFlow_*) e tau (tau_*), calcolando per ogni evento la
matrice DeltaR(jet, tau) su TUTTE le combinazioni jet-tau dell'evento,
per poi:

    1) costruire la distribuzione di DeltaR minimo "per jet"
       (distanza dal tau piu' vicino) e "per tau"
       (distanza dal jet piu' vicino);
    2) ispezionare dove la distribuzione cambia pendenza (spalla/ginocchio)
       per giustificare una soglia operativa di overlap;
    3) contare, per una o piu' soglie candidate (es. 0.2 e 0.4), quante
       coppie/jet/tau risultano geometricamente sovrapposti.

Questo e' uno studio puramente RECO: non vengono applicati tagli di
verita' (truth-matching, flavour, provenienza da H, ...). Il motivo e'
che l'overlap removal (OR) a valle opera solo su quantita' reco, quindi
la quantificazione dell'overlap deve essere fatta sullo stesso livello
di informazione che l'algoritmo di OR utilizzerebbe.


MASCHERE APPLICATE E LORO MOTIVAZIONE
--------------------------------------

1) recojet_antikt4PFlow_isAnalysisJet___NOSYS
   tau_isAnalysisTau___NOSYS

   Sono le flag "analysis-level" gia' caratterizzate nello studio
   preliminare (vedi diagnostics_preliminary.py):
       - tau:  frazione True = 100.000% -> selezione pre-applicata,
               nessun impatto sul campione;
       - jet:  frazione True =  99.541% -> selezione quasi totalmente
               inclusiva, impatto trascurabile.
   Si applicano comunque per coerenza con il resto della catena di
   analisi (le stesse collezioni "analysis" saranno quelle usate a
   valle, es. per l'overlap removal vero e proprio), e perche' e' la
   definizione di "jet ricostruito" / "tau ricostruito" adottata in
   tutto il resto del progetto. Come gia' mostrato, l'effetto sul
   risultato finale e' atteso essere trascurabile.

2) JET_SELECTION_MODE e TAU_SELECTION_MODE
   Permettono di filtrare i jet (es. WP FixedCutBEff_85 di GN2v01) e i tau
   (es. WP 85% dello score GNTauScoreSigTrans_v0prune) a livello di analisi.

3) Eventi con 0 jet o 0 tau (dopo la selezione)
   In questi eventi non existe alcuna coppia jet-tau da formare: il
   prodotto cartesiano jet x tau restituisce liste vuote. Questi
   jet/tau "orfani" (nessuna controparte nell'evento) vengono
   esplicitamente CONTATI e riportati separatamente.


CONTROLLI INTERNI AGGIUNTI
---------------------------
   - verifica dei branch mancanti file per file (skip esplicito, non
     silenzioso);
   - verifica che jet totali (post-selezione) = jet orfani (evento
     senza tau) + jet con >=1 tau nell'evento, e specularmente per i
     tau (stesso controllo con ruoli invertiti);
   - verifica che il numero di entrate nella distribuzione "DeltaR
     minimo per jet" coincida esattamente con il numero di jet non
     orfani (idem per i tau);
   - verifica di simmetria: il numero di coppie (jet, tau) con
     DeltaR < soglia contate "dal lato jet" e "dal lato tau".
"""

from pathlib import Path

import numpy as np
import awkward as ak
import uproot

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ======================================================================
# CONFIGURAZIONE
# ======================================================================

ROOT_DIR = Path("HH_bbtt")

FILE_PREFIX = "output_GGF_mc23a_bypass_noOR_0000"
FILE_SUFFIX = ".root"

TREE_NAME = "AnalysisMiniTree"

# None = tutti gli eventi di ogni file
N_ENTRIES_CAP = None

# Normalizzazione dell'istogramma di DeltaR
NORMALIZE_HISTOGRAMS = False

# Soglie operative candidate di overlap, da giustificare guardando la
# distribuzione di DeltaR minimo
DR_THRESHOLDS = [0.2, 0.3, 0.4]

# ----------------------------------------------------------------------
# Selezione della collezione di jet usata nell'analisi
# ----------------------------------------------------------------------
JET_SELECTION_MODE = "all"
# "all"    : tutti i jet analysis-level
# "btag85" : solo jet analysis-level che passano il working point
#            GN2v01 FixedCutBEff_85

JET_BTAG_BRANCH = (
    "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85"
)

VALID_JET_SELECTION_MODES = {"all", "btag85"}

if JET_SELECTION_MODE not in VALID_JET_SELECTION_MODES:
    raise ValueError(
        f"JET_SELECTION_MODE='{JET_SELECTION_MODE}' non valido. "
        f"Usare uno tra {sorted(VALID_JET_SELECTION_MODES)}."
    )

# ----------------------------------------------------------------------
# Selezione della collezione di tau usata nell'analisi
# ----------------------------------------------------------------------
TAU_SELECTION_MODE = "all"
# "all"     : tutti i tau analysis-level
# "score85" : solo tau analysis-level che superano la soglia sullo score
#             GNTauScoreSigTrans_v0prune corrispondente all'85%

TAU_SCORE_BRANCH = "tau_GNTauScoreSigTrans_v0prune"
TAU_SCORE_WP85_THRESHOLD = 0.163094

VALID_TAU_SELECTION_MODES = {"all", "score85"}

if TAU_SELECTION_MODE not in VALID_TAU_SELECTION_MODES:
    raise ValueError(
        f"TAU_SELECTION_MODE='{TAU_SELECTION_MODE}' non valido. "
        f"Usare uno tra {sorted(VALID_TAU_SELECTION_MODES)}."
    )

# Binning per l'istogramma diagnostico di DeltaR minimo
DR_HIST_MIN = 0.0
DR_HIST_MAX = 0.5
DR_HIST_BINSIZE = 0.015


# Se True, salva anche gli istogrammi di eta e phi delle collezioni
# jet e tau effettivamente usate nell'analisi.
PLOT_ETA_PHI = True

# Binning per eta
ETA_HIST_MIN = -5.0
ETA_HIST_MAX = 5.0
ETA_HIST_BINSIZE = 0.1

# Binning per phi
PHI_HIST_MIN = -np.pi
PHI_HIST_MAX = np.pi
PHI_HIST_BINSIZE = 0.1

OUTPUT_DIR = Path("output/obj_3.1/combinatory_level")


# ======================================================================
# UTILITY 
# ======================================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def file_path(index):
    return ROOT_DIR / f"{FILE_PREFIX}{index:02d}{FILE_SUFFIX}"


def load_files():
    """
    Cerca i 14 file ROOT attesi e restituisce una lista di dict.
    File mancanti o non validi vengono segnalati e ignorati.
    """

    loaded = []

    section("FILE INPUT")

    for i in range(1, 15):

        path = file_path(i)

        if not path.exists():
            print(f"[MANCANTE] {path}")
            continue

        try:
            root_file = uproot.open(path)

            if TREE_NAME not in root_file:
                print(f"[ERRORE] {path}: tree '{TREE_NAME}' non trovato")
                continue

            tree = root_file[TREE_NAME]
            n_entries = tree.num_entries

            n_entries_used = (
                min(n_entries, N_ENTRIES_CAP)
                if N_ENTRIES_CAP is not None
                else n_entries
            )

            print(
                f"[OK] {path}   entries={n_entries}   usati={n_entries_used}"
            )

            loaded.append(
                {
                    "index": i,
                    "path": path,
                    "file_name": path.name,
                    "root_file": root_file,
                    "tree": tree,
                    "n_entries": n_entries_used,
                }
            )

        except Exception as exc:
            print(f"[ERRORE] {path}: {exc}")

    print(f"\nFile caricati correttamente: {len(loaded)}/14")

    return loaded


def get_analysis_selection(tree, selection_branch, n_entries):
    """
    Legge un branch isAnalysis... e lo converte esplicitamente in una
    maschera booleana. None -> False, 0 -> False, !=0 -> True.
    """

    selection = tree[selection_branch].array(
        entry_stop=n_entries,
        library="ak",
    )

    selection = ak.fill_none(selection, 0)

    return selection != 0


def delta_phi(phi1, phi2):
    """DeltaPhi riportato nell'intervallo [-pi, pi)."""

    dphi = phi1 - phi2

    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    """DeltaR = sqrt((Delta eta)^2 + (Delta phi)^2)."""

    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)

    return np.sqrt(deta ** 2 + dphi ** 2)


def summarize(values, label):
    """
    Flatten di un array e stampa n, mean, median, p10, p90.
    """

    flat = ak.flatten(values, axis=None)
    flat = ak.drop_none(flat)
    flat = ak.to_numpy(flat)

    if flat.size == 0:
        print(f"   {label}: nessun valore disponibile")
        return None

    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        print(f"   {label}: nessun valore finito disponibile")
        return None

    print(
        f"   {label}: "
        f"n={flat.size:>10d}  "
        f"mean={np.mean(flat):.4f}  "
        f"median={np.median(flat):.4f}  "
        f"p10={np.percentile(flat, 10):.4f}  "
        f"p90={np.percentile(flat, 90):.4f}"
    )

    return flat


def find_perfect_overlap_threshold(flat_jet_min, flat_tau_min, flat_all_pairs, tol_abs=0):
    """
    Calcola il valore di DeltaR massimo entro cui i conteggi binnati di
    tutti e tre gli istogrammi coincidono perfettamente (bin per bin).
    """

    if flat_jet_min is None or flat_tau_min is None or flat_all_pairs is None:
        return None, 0

    edges = np.arange(
        DR_HIST_MIN,
        DR_HIST_MAX + DR_HIST_BINSIZE,
        DR_HIST_BINSIZE,
    )

    c_jet, _ = np.histogram(flat_jet_min, bins=edges)
    c_tau, _ = np.histogram(flat_tau_min, bins=edges)
    c_all, _ = np.histogram(flat_all_pairs, bins=edges)

    max_dr = DR_HIST_MIN
    n_bins_matched = 0

    for i in range(len(c_jet)):
        diff_jet_tau = abs(c_jet[i] - c_tau[i])
        diff_jet_all = abs(c_jet[i] - c_all[i])

        if diff_jet_tau <= tol_abs and diff_jet_all <= tol_abs:
            max_dr = edges[i + 1]
            n_bins_matched += 1
        else:
            break

    return max_dr, n_bins_matched


def print_shoulder_table(flat_values, label):
    """
    Stampa una tabella binnata e la variazione relativa bin-a-bin.
    """

    if flat_values is None or flat_values.size == 0:
        print(f"   {label}: nessun valore per la tabella di spalla")
        return

    edges = np.arange(
        DR_HIST_MIN,
        DR_HIST_MAX + DR_HIST_BINSIZE,
        DR_HIST_BINSIZE,
    )

    counts, _ = np.histogram(flat_values, bins=edges)

    widths = np.diff(edges)
    density = counts / widths / max(flat_values.size, 1)

    print(f"\n   Tabella densita' binnata - {label}")
    print(
        f"   {'bin_low':>8} {'bin_high':>8} {'count':>10} "
        f"{'density':>12} {'d(density)':>12}"
    )

    prev_density = None

    for i in range(len(counts)):

        d = density[i]
        ddens = "" if prev_density is None else f"{d - prev_density:+.4f}"

        if edges[i] < 1.0:
            print(
                f"   {edges[i]:8.2f} {edges[i + 1]:8.2f} "
                f"{counts[i]:10d} {d:12.4f} {ddens:>12}"
            )

        prev_density = d

    print(f"\n   Frazione cumulativa (DeltaR < soglia) - {label}")

    for thr in sorted(set(DR_THRESHOLDS + [0.1, 0.3, 0.5, 0.6, 0.8, 1.0])):
        frac = np.mean(flat_values < thr)
        marker = "  <-- soglia candidata" if thr in DR_THRESHOLDS else ""
        print(f"      DeltaR < {thr:.2f} : {100.0 * frac:6.3f}%{marker}")


def save_combinatory_dr_plot(dr_jet_min, dr_tau_min, dr_all_pairs, perfect_overlap_thr=None):
    """
    Salva un istogramma delle tre distribuzioni di DeltaR.
    """

    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip del plot, uso solo le tabelle testuali."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bins = np.arange(DR_HIST_MIN, DR_HIST_MAX + DR_HIST_BINSIZE, DR_HIST_BINSIZE)

    fig, ax = plt.subplots(figsize=(8, 5))

    series = [
        (dr_jet_min, f"DeltaR min per jet (tau piu' vicino) (N={dr_jet_min.size if dr_jet_min is not None else 0})"),
        (dr_tau_min, f"DeltaR min per tau (jet piu' vicino) (N={dr_tau_min.size if dr_tau_min is not None else 0})"),
        (dr_all_pairs, f"DeltaR tutte le coppie jet-tau (N={dr_all_pairs.size if dr_all_pairs is not None else 0})"),
    ]

    ylabel = "Densità di probabilità" if NORMALIZE_HISTOGRAMS else "Conteggio (scala log)"

    for values, base_label in series:
        if values is not None and values.size > 0:
            label = f"{base_label}"
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                label=label,
                linewidth=1.5,
                density=NORMALIZE_HISTOGRAMS,
            )

    ax.set_yscale("log")
    ax.set_xlabel("DeltaR(jet, tau)")
    ax.set_ylabel(ylabel)
    ax.set_title("Matching geometrico reco jet <-> tau")

    ymax = ax.get_ylim()[1]

    all_thresholds = list(DR_THRESHOLDS)
    #if perfect_overlap_thr is not None and perfect_overlap_thr > DR_HIST_MIN:
        # ax.axvline(
        #     perfect_overlap_thr,
        #     color="red",
        #     linestyle="-.",
        #     linewidth=1.2,
        #     label=f"Soglia overlap (DeltaR = {perfect_overlap_thr:.3f})",
        # )

    for thr in all_thresholds:
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.0)
        ax.text(
            thr, ymax, f"{thr}",
            rotation=90, va="top", ha="right", fontsize=8, color="red",
        )

    ax.legend(fontsize=8)
    fig.tight_layout()

    file_suffix = "_normalized" if NORMALIZE_HISTOGRAMS else ""
    out_path = OUTPUT_DIR / f"dr_jet_tau_overlap_hist_{JET_SELECTION_MODE}_tau_{TAU_SELECTION_MODE}{file_suffix}_cut_{DR_HIST_MAX}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot salvato in: {out_path}")


def save_eta_phi_histograms(
    jet_eta_parts,
    tau_eta_parts,
    jet_phi_parts,
    tau_phi_parts,
):
    """
    Salva due istogrammi diagnostici: eta e phi di jet e tau.
    """

    if not PLOT_ETA_PHI:
        return

    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip dei plot eta/phi."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jet_eta_all = ak.concatenate(jet_eta_parts) if jet_eta_parts else None
    tau_eta_all = ak.concatenate(tau_eta_parts) if tau_eta_parts else None
    jet_phi_all = ak.concatenate(jet_phi_parts) if jet_phi_parts else None
    tau_phi_all = ak.concatenate(tau_phi_parts) if tau_phi_parts else None

    def flatten_to_numpy(values):
        if values is None:
            return np.array([])
        flat = ak.flatten(values, axis=None)
        flat = ak.drop_none(flat)
        flat = ak.to_numpy(flat)
        if flat.size == 0:
            return np.array([])
        if np.issubdtype(flat.dtype, np.floating):
            flat = flat[np.isfinite(flat)]
        return flat

    jet_eta_flat = flatten_to_numpy(jet_eta_all)
    tau_eta_flat = flatten_to_numpy(tau_eta_all)
    jet_phi_flat = flatten_to_numpy(jet_phi_all)
    tau_phi_flat = flatten_to_numpy(tau_phi_all)

    # PLOT ETA
    eta_bins = np.arange(ETA_HIST_MIN, ETA_HIST_MAX + ETA_HIST_BINSIZE, ETA_HIST_BINSIZE)
    fig, ax = plt.subplots(figsize=(8, 5))

    if jet_eta_flat.size > 0:
        ax.hist(jet_eta_flat, bins=eta_bins, histtype="step", label="jet", linewidth=1.5)
    if tau_eta_flat.size > 0:
        ax.hist(tau_eta_flat, bins=eta_bins, histtype="step", label="tau", linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Conteggio (scala log)")
    ax.set_title(f"Distribuzione eta - jet({JET_SELECTION_MODE}) / tau({TAU_SELECTION_MODE})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path_eta = OUTPUT_DIR / f"eta_jet_tau_{JET_SELECTION_MODE}_tau_{TAU_SELECTION_MODE}.png"
    fig.savefig(out_path_eta, dpi=150)
    plt.close(fig)

    print(f"[OK] Plot eta salvato in: {out_path_eta}")

    # PLOT PHI
    phi_bins = np.arange(PHI_HIST_MIN, PHI_HIST_MAX + PHI_HIST_BINSIZE, PHI_HIST_BINSIZE)
    fig, ax = plt.subplots(figsize=(8, 5))

    if jet_phi_flat.size > 0:
        ax.hist(jet_phi_flat, bins=phi_bins, histtype="step", label="jet", linewidth=1.5)
    if tau_phi_flat.size > 0:
        ax.hist(tau_phi_flat, bins=phi_bins, histtype="step", label="tau", linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel("Conteggio (scala log)")
    ax.set_title(f"Distribuzione phi - jet({JET_SELECTION_MODE}) / tau({TAU_SELECTION_MODE})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path_phi = OUTPUT_DIR / f"phi_jet_tau_{JET_SELECTION_MODE}_tau_{TAU_SELECTION_MODE}.png"
    fig.savefig(out_path_phi, dpi=150)
    plt.close(fig)

    print(f"[OK] Plot phi salvato in: {out_path_phi}")


# ======================================================================
# ANALISI PRINCIPALE
# ======================================================================

def analyze_jet_tau_overlap(loaded):
    """
    Matching geometrico reco jet <-> tau, aggregato su tutti i file.
    """

    branches = [
        "tau_eta",
        "tau_phi",
        "tau_isAnalysisTau___NOSYS",
        "recojet_antikt4PFlow_eta",
        "recojet_antikt4PFlow_phi",
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
    ]

    # Aggiunta del branch di b-tagging
    if JET_SELECTION_MODE != "all":
        branches.append(JET_BTAG_BRANCH)

    # Aggiunta del branch di tau score
    if TAU_SELECTION_MODE != "all":
        branches.append(TAU_SCORE_BRANCH)

    # ------------------------------------------------------------------
    # ACCUMULATORI GLOBALI
    # ------------------------------------------------------------------

    aggregate_events = 0

    aggregate_jets_total = 0
    aggregate_taus_total = 0

    aggregate_jets_orphan = 0   # jet in eventi senza alcun tau
    aggregate_taus_orphan = 0   # tau in eventi senza alcun jet

    dr_jet_min_parts = []       # DeltaR minimo per jet
    dr_tau_min_parts = []       # DeltaR minimo per tau
    dr_all_pairs_parts = []     # tutte le coppie jet-tau

    jet_eta_parts = []
    tau_eta_parts = []
    jet_phi_parts = []
    tau_phi_parts = []

    # ------------------------------------------------------------------
    # LOOP SU TUTTI I FILE
    # ------------------------------------------------------------------

    for item in loaded:

        filename = item["file_name"]
        tree = item["tree"]
        n_entries = item["n_entries"]

        print()
        print("-" * 100)
        print(f"Analisi file: {filename}")
        print(f"    selezione jet: {JET_SELECTION_MODE}")
        print(f"    selezione tau: {TAU_SELECTION_MODE}")
        print("-" * 100)

        keys = set(tree.keys())
        missing = [b for b in branches if b not in keys]

        if missing:
            print(f"[WARNING] {filename}: branch mancanti:")
            for b in missing:
                print(f"    - {b}")
            print("    File ignorato.")
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        n_events = len(a["tau_eta"])
        aggregate_events += n_events

        # ==============================================================
        # SELEZIONE TAU
        # ==============================================================
        tau_sel_analysis = get_analysis_selection(
            tree, "tau_isAnalysisTau___NOSYS", n_entries
        )

        if TAU_SELECTION_MODE == "all":
            tau_sel = tau_sel_analysis
        else:
            tau_score_sel = ak.fill_none(
                a[TAU_SCORE_BRANCH],
                -np.inf,
            ) >= TAU_SCORE_WP85_THRESHOLD
            tau_sel = tau_sel_analysis & tau_score_sel

        # ==============================================================
        # SELEZIONE JET
        # ==============================================================
        jet_sel_analysis = get_analysis_selection(
            tree, "recojet_antikt4PFlow_isAnalysisJet___NOSYS", n_entries
        )

        if JET_SELECTION_MODE == "all":
            jet_sel = jet_sel_analysis
        else:
            jet_btag_sel = ak.fill_none(
                a[JET_BTAG_BRANCH],
                False,
            ) != 0
            jet_sel = jet_sel_analysis & jet_btag_sel

        tau_eta = a["tau_eta"][tau_sel]
        tau_phi = a["tau_phi"][tau_sel]

        jet_eta = a["recojet_antikt4PFlow_eta"][jet_sel]
        jet_phi = a["recojet_antikt4PFlow_phi"][jet_sel]

        if PLOT_ETA_PHI:
            jet_eta_parts.append(jet_eta)
            tau_eta_parts.append(tau_eta)
            jet_phi_parts.append(jet_phi)
            tau_phi_parts.append(tau_phi)

        n_jet_evt = ak.num(jet_eta, axis=1)
        n_tau_evt = ak.num(tau_eta, axis=1)

        n_jets_file = int(ak.sum(n_jet_evt))
        n_taus_file = int(ak.sum(n_tau_evt))

        aggregate_jets_total += n_jets_file
        aggregate_taus_total += n_taus_file

        # ==============================================================
        # EVENTI ORFANI (0 jet o 0 tau selezionati nell'evento)
        # ==============================================================
        evt_no_tau = n_tau_evt == 0
        evt_no_jet = n_jet_evt == 0

        n_jets_orphan_file = int(ak.sum(n_jet_evt[evt_no_tau]))
        n_taus_orphan_file = int(ak.sum(n_tau_evt[evt_no_jet]))

        aggregate_jets_orphan += n_jets_orphan_file
        aggregate_taus_orphan += n_taus_orphan_file

        # ==============================================================
        # MATRICE DeltaR[evento][jet][tau] (prodotto cartesiano)
        # ==============================================================
        jet_eta_c, tau_eta_c = ak.unzip(
            ak.cartesian([jet_eta, tau_eta], nested=True)
        )
        jet_phi_c, tau_phi_c = ak.unzip(
            ak.cartesian([jet_phi, tau_phi], nested=True)
        )

        dr_matrix = delta_r(jet_eta_c, jet_phi_c, tau_eta_c, tau_phi_c)
        dr_jet_min = ak.min(dr_matrix, axis=-1)

        tau_eta_ct, jet_eta_ct = ak.unzip(
            ak.cartesian([tau_eta, jet_eta], nested=True)
        )
        tau_phi_ct, jet_phi_ct = ak.unzip(
            ak.cartesian([tau_phi, jet_phi], nested=True)
        )

        dr_matrix_t = delta_r(tau_eta_ct, tau_phi_ct, jet_eta_ct, jet_phi_ct)
        dr_tau_min = ak.min(dr_matrix_t, axis=-1)

        # ==============================================================
        # CONTROLLI INTERNI
        # ==============================================================
        n_dr_jet_min_valid = int(ak.count(dr_jet_min))
        n_dr_tau_min_valid = int(ak.count(dr_tau_min))

        expected_jets_with_tau = n_jets_file - n_jets_orphan_file
        expected_taus_with_jet = n_taus_file - n_taus_orphan_file

        if n_dr_jet_min_valid != expected_jets_with_tau:
            print(
                f"    [WARNING] {filename}: "
                f"jet con DeltaR_min valido = {n_dr_jet_min_valid}, "
                f"attesi = {expected_jets_with_tau}"
            )

        if n_dr_tau_min_valid != expected_taus_with_jet:
            print(
                f"    [WARNING] {filename}: "
                f"tau con DeltaR_min valido = {n_dr_tau_min_valid}, "
                f"attesi = {expected_taus_with_jet}"
            )

        dr_jet_min_parts.append(dr_jet_min)
        dr_tau_min_parts.append(dr_tau_min)
        dr_all_pairs_parts.append(dr_matrix)

        print(f"    eventi: {n_events}")
        print(f"    jet selezionati: {n_jets_file}")
        print(f"    tau selezionati: {n_taus_file}")
        print(f"    jet orfani (evento senza tau): {n_jets_orphan_file}")
        print(f"    tau orfani (evento senza jet): {n_taus_orphan_file}")
        print(f"    jet con DeltaR_min valido: {n_dr_jet_min_valid}")
        print(f"    tau con DeltaR_min valido: {n_dr_tau_min_valid}")

    # ==================================================================
    # RISULTATI AGGREGATI
    # ==================================================================

    section("RISULTATI AGGREGATI SU TUTTI I FILE")

    print(f"   file analizzati: {len(loaded)}")
    print(f"   selezione jet usata: {JET_SELECTION_MODE}")
    print(f"   selezione tau usata: {TAU_SELECTION_MODE}")
    print(f"   eventi totali: {aggregate_events}")
    print(f"   jet selezionati totali: {aggregate_jets_total}")
    print(f"   tau selezionati totali: {aggregate_taus_total}")

    print(
        f"\n   jet orfani (nessun tau nello stesso evento): "
        f"{aggregate_jets_orphan}  "
        f"({100.0 * aggregate_jets_orphan / max(aggregate_jets_total, 1):.3f}% "
        f"dei jet totali)"
    )
    print(
        f"   tau orfani (nessun jet nello stesso evento): "
        f"{aggregate_taus_orphan}  "
        f"({100.0 * aggregate_taus_orphan / max(aggregate_taus_total, 1):.3f}% "
        f"dei tau totali)"
    )

    dr_jet_min_all = ak.concatenate(dr_jet_min_parts) if dr_jet_min_parts else None
    dr_tau_min_all = ak.concatenate(dr_tau_min_parts) if dr_tau_min_parts else None
    dr_all_pairs_all = (
        ak.concatenate(dr_all_pairs_parts) if dr_all_pairs_parts else None
    )

    section("DISTRIBUZIONI DeltaR")

    flat_jet_min = summarize(dr_jet_min_all, "DeltaR minimo per jet (tau piu' vicino)")
    flat_tau_min = summarize(dr_tau_min_all, "DeltaR minimo per tau (jet piu' vicino)")
    flat_all_pairs = summarize(dr_all_pairs_all, "DeltaR su tutte le coppie jet-tau")

    if flat_jet_min is not None:
        expected = aggregate_jets_total - aggregate_jets_orphan
        if flat_jet_min.size != expected:
            print(
                f"\n   [WARNING] entrate in DeltaR_min per jet "
                f"({flat_jet_min.size}) != jet non orfani ({expected})"
            )
        else:
            print(
                f"\n   [OK] entrate DeltaR_min per jet = jet non orfani "
                f"({flat_jet_min.size})"
            )

    if flat_tau_min is not None:
        expected = aggregate_taus_total - aggregate_taus_orphan
        if flat_tau_min.size != expected:
            print(
                f"   [WARNING] entrate in DeltaR_min per tau "
                f"({flat_tau_min.size}) != tau non orfani ({expected})"
            )
        else:
            print(
                f"   [OK] entrate DeltaR_min per tau = tau non orfani "
                f"({flat_tau_min.size})"
            )

    # ------------------------------------------------------------------
    # ANALISI DELLA SOGLIA
    # ------------------------------------------------------------------

    section("ANALISI DELLA SOGLIA - dove la distribuzione cambia pendenza")

    perfect_overlap_thr, n_bins = find_perfect_overlap_threshold(
        flat_jet_min, flat_tau_min, flat_all_pairs
    )

    print(f"   [SOGLIA CALCOLATA] DeltaR massimo di perfetto overlap: {perfect_overlap_thr:.4f}")
    print(f"                      (coincidenza esatta su {n_bins} bin consecutivi con bin_size={DR_HIST_BINSIZE})")

    print_shoulder_table(flat_jet_min, "DeltaR minimo per jet")
    print_shoulder_table(flat_tau_min, "DeltaR minimo per tau")

    save_combinatory_dr_plot(flat_jet_min, flat_tau_min, flat_all_pairs, perfect_overlap_thr=perfect_overlap_thr)

    if PLOT_ETA_PHI:
        save_eta_phi_histograms(
            jet_eta_parts,
            tau_eta_parts,
            jet_phi_parts,
            tau_phi_parts,
        )

    # ------------------------------------------------------------------
    # CONTEGGIO OVERLAP PER LE SOGLIE CANDIDATE
    # ------------------------------------------------------------------

    section("CONTEGGIO OVERLAP PER SOGLIA")

    eval_thresholds = list(DR_THRESHOLDS)
    if perfect_overlap_thr is not None and perfect_overlap_thr not in eval_thresholds:
        eval_thresholds.append(perfect_overlap_thr)
        eval_thresholds.sort()

    for thr in eval_thresholds:

        tag = " (SOGLIA PERFETTO OVERLAP)" if thr == perfect_overlap_thr else ""
        print(f"\n   --- soglia DeltaR < {thr:.4f}{tag} ---")

        if flat_jet_min is not None:
            n_jet_ov = int(np.sum(flat_jet_min < thr))
            frac_jet_ov = n_jet_ov / flat_jet_min.size
            print(
                f"   jet con almeno un tau sovrapposto: "
                f"{n_jet_ov} / {flat_jet_min.size}  "
                f"({100.0 * frac_jet_ov:.3f}% dei jet con >=1 tau nell'evento)"
            )

        if flat_tau_min is not None:
            n_tau_ov = int(np.sum(flat_tau_min < thr))
            frac_tau_ov = n_tau_ov / flat_tau_min.size
            print(
                f"   tau con almeno un jet sovrapposto: "
                f"{n_tau_ov} / {flat_tau_min.size}  "
                f"({100.0 * frac_tau_ov:.3f}% dei tau con >=1 jet nell'evento)"
            )

        if flat_all_pairs is not None:
            n_pairs_ov = int(np.sum(flat_all_pairs < thr))
            frac_pairs_ov = n_pairs_ov / flat_all_pairs.size
            print(
                f"   coppie (jet,tau) sovrapposte: "
                f"{n_pairs_ov} / {flat_all_pairs.size}  "
                f"({100.0 * frac_pairs_ov:.3f}%)"
            )


# ======================================================================
# MAIN
# ======================================================================

def main():

    loaded = load_files()

    if not loaded:
        print(
            "\nNessun file disponibile. "
            "Controllare ROOT_DIR e i nomi dei file."
        )
        return

    analyze_jet_tau_overlap(loaded)

    section("FINE DIAGNOSTICA")


if __name__ == "__main__":
    main()