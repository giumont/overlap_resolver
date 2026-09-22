"""
================================================================================
Matching reco<->truth separato per jet e per tau, classificazione delle coppie
jet-tau overlappanti e studio cinematico a livello di OGGETTO (obiettivo 3.1).
================================================================================

Questo script e' pensato per essere eseguito DOPO obj_3_1.py e riusa le funzioni
utility della pipeline (delta_r, get_analysis_selection, summarize,
print_shoulder_table, load_files, section).

STRUTTURA DELL'ANALISI
----------------------
PARTE A - truth matching a livello di oggetto
    Ogni jet/tau selezionato riceve una label di verita' (true/fake).

PARTE B - classificazione dell'overlap a livello di COPPIA
    Per le sole coppie con DeltaR < soglia vengono definite le quattro
    categorie:
        (a) jet vero, tau fake
        (b) jet fake, tau vero
        (c) jet vero, tau vero
        (d) entrambi fake

    Questa e' la popolazione usata per rispondere alla domanda:
    "da quali tipi di coppie e' composto l'overlap osservato?".

    Anche la distribuzione completa di DeltaR(jet,tau) resta pair-level,
    perche' il suo scopo e' mostrare la forma del picco a piccolo DeltaR e
    la coda combinatoria.

PARTE C - cinematica a livello di OGGETTO (punto 6)
    Per ogni jet selezionato:
        jet_in_overlap = esiste almeno un tau con DeltaR < soglia.

    Per ogni tau selezionato:
        tau_in_overlap = esiste almeno un jet con DeltaR < soglia.

    Ogni oggetto viene quindi contato UNA SOLA VOLTA. Non viene costruito il
    prodotto cartesiano completo come unita' statistica per pT/eta/phi.

    Vengono prodotti:
      1) 6 plot "all jets / all taus": overlap vs isolato;
      2) 6 plot con la label di verita' separata: true-overlap,
         true-isolato, fake-overlap, fake-isolato.

    La precedente categoria di coppia (c) "jet vero + tau vero" non viene
    usata come definizione dell'oggetto nel punto 6, perche' un jet/tau puo'
    partecipare contemporaneamente a piu' coppie di categorie diverse.
    La categoria (c) resta correttamente definita solo nel punto 5.

PERCENTUALI EVENT-LEVEL
-----------------------
Viene inoltre stampata, per ciascuna delle quattro categorie di coppia,
la percentuale di eventi che contengono almeno una coppia della categoria
con DeltaR < DR_THRESHOLD_KINEMATICS, usando come denominatore gli eventi
che contengono almeno una coppia di quella categoria. Come controllo viene
riportata anche la percentuale pair-level corrispondente.

NOTA SUI BRANCH DI VERITA'
--------------------------
I nomi dei branch di truth matching sono configurabili qui sotto. In modalita'
"label" il branch tau_truth_IsHadronicTau viene usato direttamente; per i jet,
con JET_SELECTION_MODE="btag85", HadronConeExclTruthLabelID == 5 definisce un
jet vero b.

La modalita' geometrica per i tau e' mantenuta per compatibilita' con la
versione precedente, ma non e' necessaria con TRUTH_MODE_TAU="label".
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

# Riuso delle utility gia' scritte e validate in obj_3_1.py
from obj_3_1 import (
    ROOT_DIR, FILE_PREFIX, FILE_SUFFIX, TREE_NAME, N_ENTRIES_CAP,
    section, load_files, get_analysis_selection, delta_r, summarize,
    print_shoulder_table,
)

DR_THRESHOLDS = [0.2, 0.4]
DR_HIST_MIN = 0
DR_HIST_MAX = 4.0
DR_HIST_BINSIZE = 0.02


# ======================================================================
# CONFIGURAZIONE
# ======================================================================

# ----------------------------------------------------------------------
# Selezione dei jet
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

# --- truth label del jet, usato in modalita' btag85 ---
JET_TRUTH_LABEL_BRANCH = (
    "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
)
JET_TRUTH_LABEL_B_VALUE = 5


# ----------------------------------------------------------------------
# Selezione dei tau
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


# ----------------------------------------------------------------------
# Truth matching tau
# ----------------------------------------------------------------------

# "label"     -> branch di truth matching gia' disponibilev
# "geometric" -> matching geometrico reco <-> truth tau
TRUTH_MODE_TAU = "label"

TAU_TRUTH_MATCH_BRANCH = "tau_truth_IsHadronicTau"

# --- caso "geometric" per il tau ---
TRUTH_TAU_ETA_BRANCH = "truthtau_eta_vis"
TRUTH_TAU_PHI_BRANCH = "truthtau_phi_vis"
DR_TRUTH_MATCH_TAU = 0.2


# ----------------------------------------------------------------------
# Output e soglia cinematica
# ----------------------------------------------------------------------

OUTPUT_DIR = Path("output/obj_3.1/truth")

JET_PT_BRANCH = "recojet_antikt4PFlow_pt___NOSYS"
TAU_PT_BRANCH = "tau_pt___NOSYS"

# Soglia usata per definire un oggetto "in overlap" nel punto 6
DR_THRESHOLD_KINEMATICS = 0.4

# --- binning ---
# I branch pT sono mantenuti nella loro unita' originale (qui MeV).
PT_HIST_MIN = 0.0
PT_HIST_MAX = 1_000_000.0  # MeV
PT_HIST_BINSIZE = 5_000  # MeV

ETA_HIST_MIN = -5.0
ETA_HIST_MAX = 5.0
ETA_HIST_BINSIZE = 0.2

PHI_HIST_MIN = -np.pi
PHI_HIST_MAX = np.pi
PHI_HIST_BINSIZE = 0.2


# ----------------------------------------------------------------------
# Categorie di verita' per le coppie (punto 5)
# ----------------------------------------------------------------------

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


# True -> ogni distribuzione viene normalizzata in modo che la somma
# delle frazioni per bin sia 1. False -> conteggi assoluti.
NORMALIZE_HISTOGRAMS = True


# ======================================================================
# PARTE A - MATCHING RECO <-> TRUTH
# ======================================================================

def match_reco_to_truth(reco_eta, reco_phi, truth_eta, truth_phi, dr_max):
    """
    Matching geometrico reco <-> truth.

    Per ogni oggetto reco viene trovato il DeltaR minimo rispetto agli
    oggetti truth dello stesso evento. L'oggetto e' classificato come
    matchato se il minimo e' < dr_max.
    """

    # Primo asse = oggetti reco, secondo asse = oggetti truth.
    reco_eta_ct, truth_eta_ct = ak.unzip(
        ak.cartesian([reco_eta, truth_eta], nested=True)
    )
    reco_phi_ct, truth_phi_ct = ak.unzip(
        ak.cartesian([reco_phi, truth_phi], nested=True)
    )

    dr_matrix = delta_r(
        reco_eta_ct,
        reco_phi_ct,
        truth_eta_ct,
        truth_phi_ct,
    )

    dr_min = ak.min(dr_matrix, axis=-1)
    is_matched = ak.fill_none(dr_min < dr_max, False)

    return dr_min, is_matched


def label_jets_and_taus(a, jet_sel, tau_sel):
    """
    Restituisce le label di verita' per i jet e tau analysis-level
    gia' selezionati.
    """

    jet_eta = a["recojet_antikt4PFlow_eta"][jet_sel]
    jet_phi = a["recojet_antikt4PFlow_phi"][jet_sel]

    tau_eta = a["tau_eta"][tau_sel]
    tau_phi = a["tau_phi"][tau_sel]

    # ------------------------------------------------------------------
    # JET
    # ------------------------------------------------------------------

    flavour = a[JET_TRUTH_LABEL_BRANCH][jet_sel]
    jet_is_true = flavour == JET_TRUTH_LABEL_B_VALUE
    jet_dr_truth = None

    # ------------------------------------------------------------------
    # TAU
    # ------------------------------------------------------------------

    if TRUTH_MODE_TAU == "label":
        tau_is_true = a[TAU_TRUTH_MATCH_BRANCH][tau_sel] != 0
        tau_dr_truth = None

    elif TRUTH_MODE_TAU == "geometric":
        truth_tau_eta = a[TRUTH_TAU_ETA_BRANCH]
        truth_tau_phi = a[TRUTH_TAU_PHI_BRANCH]

        tau_dr_truth, tau_is_true = match_reco_to_truth(
            tau_eta,
            tau_phi,
            truth_tau_eta,
            truth_tau_phi,
            DR_TRUTH_MATCH_TAU,
        )

    else:
        raise RuntimeError(
            f"TRUTH_MODE_TAU='{TRUTH_MODE_TAU}' non gestito."
        )

    return (
        jet_is_true,
        tau_is_true,
        jet_dr_truth,
        tau_dr_truth,
    )


# ======================================================================
# UTILITA' COMUNE: MATRICE DeltaR E LABEL DI COPPIA
# ======================================================================

def build_pair_geometry_and_labels(
    jet_eta, jet_phi, jet_label,
    tau_eta, tau_phi, tau_label,
):
    """
    Costruisce una sola volta, per ogni evento, la matrice DeltaR[jet][tau]
    e le label di verita' replicate sulla stessa struttura [jet][tau].

    Questa funzione viene riutilizzata per:
      - classificazione pair-level del punto 5;
      - distribuzione completa di DeltaR;
      - flag object-level overlap del punto 6.
    """

    jet_eta_ct, tau_eta_ct = ak.unzip(
        ak.cartesian([jet_eta, tau_eta], nested=True)
    )
    jet_phi_ct, tau_phi_ct = ak.unzip(
        ak.cartesian([jet_phi, tau_phi], nested=True)
    )
    jet_label_ct, tau_label_ct = ak.unzip(
        ak.cartesian([jet_label, tau_label], nested=True)
    )

    dr_matrix = delta_r(
        jet_eta_ct,
        jet_phi_ct,
        tau_eta_ct,
        tau_phi_ct,
    )

    return {
        "dr": dr_matrix,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }


# ======================================================================
# PARTE B - CLASSIFICAZIONE DELLE COPPIE OVERLAPPANTI
# ======================================================================

def classify_overlapping_pairs(pair_info, dr_thr):
    """
    Usa la matrice pair-level gia' costruita e classifica le sole coppie
    con DeltaR < dr_thr nelle quattro categorie a/b/c/d.
    """

    dr_matrix = pair_info["dr"]
    jet_label_ct = pair_info["jet_label"]
    tau_label_ct = pair_info["tau_label"]

    overlap_mask = dr_matrix < dr_thr

    jet_lab_flat = ak.to_numpy(
        ak.flatten(jet_label_ct, axis=None)
    ).astype(bool)
    tau_lab_flat = ak.to_numpy(
        ak.flatten(tau_label_ct, axis=None)
    ).astype(bool)
    overlap_flat = ak.to_numpy(
        ak.flatten(overlap_mask, axis=None)
    ).astype(bool)

    jet_lab_flat = jet_lab_flat[overlap_flat]
    tau_lab_flat = tau_lab_flat[overlap_flat]

    n_total = int(np.count_nonzero(overlap_flat))

    categories = {
        "a_jet_true_tau_fake": int(np.count_nonzero(
            jet_lab_flat & ~tau_lab_flat
        )),
        "b_jet_false_tau_true": int(np.count_nonzero(
            ~jet_lab_flat & tau_lab_flat
        )),
        "c_jet_true_tau_true": int(np.count_nonzero(
            jet_lab_flat & tau_lab_flat
        )),
        "d_jet_false_tau_false": int(np.count_nonzero(
            ~jet_lab_flat & ~tau_lab_flat
        )),
    }

    return categories, n_total


def flatten_dr_by_category(pair_info):
    """
    Distribuzione COMPLETA di DeltaR(jet,tau), separata nelle quattro
    categorie di verita'. Nessun cut in DeltaR viene applicato qui.
    """

    dr_flat = ak.to_numpy(
        ak.flatten(pair_info["dr"], axis=None)
    )
    jet_lab_flat = ak.to_numpy(
        ak.flatten(pair_info["jet_label"], axis=None)
    ).astype(bool)
    tau_lab_flat = ak.to_numpy(
        ak.flatten(pair_info["tau_label"], axis=None)
    ).astype(bool)

    cat_masks = {
        "a_jet_true_tau_fake": jet_lab_flat & ~tau_lab_flat,
        "b_jet_false_tau_true": ~jet_lab_flat & tau_lab_flat,
        "c_jet_true_tau_true": jet_lab_flat & tau_lab_flat,
        "d_jet_false_tau_false": ~jet_lab_flat & ~tau_lab_flat,
    }

    return {
        cat: dr_flat[mask]
        for cat, mask in cat_masks.items()
    }


# ======================================================================
# PARTE C - CINEMATICA A LIVELLO DI OGGETTO
# ======================================================================

def classify_objects_kinematics(
    jet_pt, jet_eta, jet_phi, jet_label,
    tau_pt, tau_eta, tau_phi, tau_label,
    pair_info, dr_threshold,
):
    """
    Classificazione OBJECT-LEVEL.

    Per ciascun jet:
        jet_in_overlap = esiste almeno un tau con DeltaR < dr_threshold.

    Per ciascun tau:
        tau_in_overlap = esiste almeno un jet con DeltaR < dr_threshold.

    ATTENZIONE:
    la riduzione lungo l'asse dei partner deve preservare anche gli oggetti
    appartenenti a eventi nei quali la collezione partner e' vuota. Per questo
    jet e tau vengono ridotti separatamente partendo da due matrici con
    l'oggetto di interesse come asse esterno. Questo evita che, per esempio,
    i tau di un evento senza jet vengano persi durante il flatten.

    Ogni oggetto viene quindi contato una sola volta, indipendentemente
    dalla molteplicita' dell'evento.
    """

    # --------------------------------------------------------------
    # Flag overlap per i JET
    # struttura: [evento][jet][tau]
    # riducendo sull'asse tau otteniamo [evento][jet]
    # --------------------------------------------------------------
    jet_tau_dr = pair_info["dr"]
    jet_in_overlap = ak.any(
        jet_tau_dr < dr_threshold,
        axis=-1,
    )

    # --------------------------------------------------------------
    # Flag overlap per i TAU
    # Ricostruiamo la matrice nel verso opposto:
    # [evento][tau][jet]
    # riducendo sull'asse jet otteniamo [evento][tau].
    # Questo e' importante per preservare i tau negli eventi con 0 jet.
    # --------------------------------------------------------------
    tau_eta_ct, jet_eta_ct = ak.unzip(
        ak.cartesian([tau_eta, jet_eta], nested=True)
    )
    tau_phi_ct, jet_phi_ct = ak.unzip(
        ak.cartesian([tau_phi, jet_phi], nested=True)
    )

    tau_jet_dr = delta_r(
        tau_eta_ct,
        tau_phi_ct,
        jet_eta_ct,
        jet_phi_ct,
    )

    tau_in_overlap = ak.any(
        tau_jet_dr < dr_threshold,
        axis=-1,
    )

    def flat(x):
        return ak.to_numpy(
            ak.flatten(x, axis=None)
        )

    jet_pt_flat = flat(jet_pt)
    jet_in_overlap_flat = flat(jet_in_overlap).astype(bool)

    tau_pt_flat = flat(tau_pt)
    tau_in_overlap_flat = flat(tau_in_overlap).astype(bool)

    # Controlli espliciti: ogni flag object-level deve avere esattamente
    # una entry per oggetto selezionato. Se falliscono, l'errore e' nella
    # costruzione della matrice di overlap e non nei plot.
    if jet_in_overlap_flat.size != jet_pt_flat.size:
        raise RuntimeError(
            "Incoerenza object-level per i jet: "
            f"pt={jet_pt_flat.size}, overlap={jet_in_overlap_flat.size}"
        )

    if tau_in_overlap_flat.size != tau_pt_flat.size:
        raise RuntimeError(
            "Incoerenza object-level per i tau: "
            f"pt={tau_pt_flat.size}, overlap={tau_in_overlap_flat.size}"
        )

    return {
        "jet": {
            "pt": jet_pt_flat,
            "eta": flat(jet_eta),
            "phi": flat(jet_phi),
            "is_true": flat(jet_label).astype(bool),
            "in_overlap": jet_in_overlap_flat,
        },
        "tau": {
            "pt": tau_pt_flat,
            "eta": flat(tau_eta),
            "phi": flat(tau_phi),
            "is_true": flat(tau_label).astype(bool),
            "in_overlap": tau_in_overlap_flat,
        },
    }


def merge_object_kinematics(parts_list):
    """Concatena i dati object-level provenienti da piu' file."""

    variables = (
        "pt",
        "eta",
        "phi",
        "is_true",
        "in_overlap",
    )

    merged = {
        "jet": {var: [] for var in variables},
        "tau": {var: [] for var in variables},
    }

    for part in parts_list:
        for obj in ("jet", "tau"):
            for var in variables:
                merged[obj][var].append(part[obj][var])

    for obj in ("jet", "tau"):
        for var in variables:
            arrays = merged[obj][var]
            merged[obj][var] = (
                np.concatenate(arrays)
                if arrays
                else np.array([])
            )

    return merged


# ======================================================================
# PLOT: DISTRIBUZIONE COMPLETA DI DeltaR PER CATEGORIA (PAIR-LEVEL)
# ======================================================================

def _step_hist_with_gaps(ax, values, bins, density, label, color,
                          linewidth=1.5):
    """
    Istogramma a gradini con gap nei bin vuoti, utile su asse y logaritmico.
    """

    counts, edges = np.histogram(
        values,
        bins=bins,
        density=density,
    )

    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)

    # BRUTTO - TORNO AGLI ZERI
    # I bin vuoti vengono nascosti con NaN.
    #y = np.where(y > 0, y, np.nan)

    y = np.where(y > 0, y, 0)

    ax.plot(
        x,
        y,
        linewidth=linewidth,
        color=color,
        label=f"{label}  (n={values.size})",
    )


def save_dr_histogram_by_category(dr_by_category):
    """
    Distribuzione completa di DeltaR(jet,tau), separata per le quattro
    categorie di verita'. Questo plot resta volutamente pair-level.
    """

    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip plot DeltaR per categoria."
        )
        return
    
    dir = OUTPUT_DIR / "truth_categories"
    dir.mkdir(parents=True, exist_ok=True)

    bins = np.arange(
        DR_HIST_MIN,
        DR_HIST_MAX + DR_HIST_BINSIZE,
        DR_HIST_BINSIZE,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    plot_order = sorted(
        CATEGORY_KEYS,
        key=lambda cat: dr_by_category[cat].size,
        reverse=True,
    )

    for cat in plot_order:
        values = dr_by_category[cat]
        if values.size == 0:
            continue

        _step_hist_with_gaps(
            ax,
            values,
            bins,
            density=NORMALIZE_HISTOGRAMS,
            label=CATEGORY_LABELS[cat],
            color=CATEGORY_COLORS[cat],
        )

    ax.set_yscale("log")
    ax.set_xlabel("$\\Delta R$(jet, tau)")
    ax.set_xlim(DR_HIST_MIN, DR_HIST_MAX)

    if NORMALIZE_HISTOGRAMS:
        ax.set_ylabel("Densità di probabilità")
        ax.set_title(
            "Distribuzione normalizzata di $\\Delta R$(jet, tau) "
            "per categoria di verita' (pair-level)"
        )
    else:
        ax.set_ylabel("Conteggio")
        ax.set_title(
            "$\\Delta R$(jet, tau) per categoria di verita' (pair-level)"
        )

    for thr in DR_THRESHOLDS:
        ax.axvline(
            thr,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.text(
            thr + 0.005,
            0.95,
            fr"$\Delta R = {thr}$",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            verticalalignment='top',
            fontsize=8,
            color="black",
            alpha=0.8,
        )

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()

    suffix = f"_normalized_cut{DR_HIST_MAX}" if NORMALIZE_HISTOGRAMS else f"_cut{DR_HIST_MAX}"
    out_path = (
        dir 
        / f"dr_jet_tau_by_category_{JET_SELECTION_MODE}"
        f"{suffix}.png"
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot DeltaR per categoria salvato in: {out_path}")

# ======================================================================
# PLOT OBJECT-LEVEL
# ======================================================================

KINEMATICS_VARIABLES = [
    ("pt", PT_HIST_MIN, PT_HIST_MAX, PT_HIST_BINSIZE,
     "pT [MeV]"),
    ("eta", ETA_HIST_MIN, ETA_HIST_MAX, ETA_HIST_BINSIZE,
     r"$\eta$"),
    ("phi", PHI_HIST_MIN, PHI_HIST_MAX, PHI_HIST_BINSIZE,
     r"$\phi$"),
]


def _plot_hist_curves(ax, datasets, bins):
    """
    Disegna piu' dataset su un asse comune.

    datasets = [(values, label, color_or_None), ...]
    """

    for values, label, color in datasets:
        if values.size == 0:
            continue

        # if NORMALIZE_HISTOGRAMS:
        #     weights = np.ones(values.size) / values.size
        # else:
        #     weights = None

        kwargs = {
            "bins": bins,
            "histtype": "step",
            "linewidth": 1.7,
            #"weights": weights,
            "density": True,
            "label": f"{label}  (n={values.size})",
        }

        if color is not None:
            kwargs["color"] = color

        ax.hist(values, **kwargs)


def _finish_object_plot(ax, xlabel, title, normalize):
    """Format comune dei plot cinematici object-level."""

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        #"Frazione per bin"
        "Densità di probabilità"
        if normalize
        else "Conteggio"
    )
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)


def save_object_kinematics_all_overlap_vs_isolated(
    object_data,
    dr_threshold,
):
    """
    6 plot object-level:
      - 3 variabili del jet: tutti i jet, overlap vs isolati;
      - 3 variabili del tau: tutti i tau, overlap vs isolati.

    Ogni oggetto compare esattamente una volta.
    """

    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip dei plot object-level all overlap/non-overlap."
        )
        return
    dir = OUTPUT_DIR / "no_truth_categories"
    dir.mkdir(parents=True, exist_ok=True)

    for obj_name, obj_title in (
        ("jet", "jet"),
        ("tau", "tau"),
    ):
        for var_key, vmin, vmax, step, xlabel_suffix in KINEMATICS_VARIABLES:
            values = object_data[obj_name][var_key]
            overlap = object_data[obj_name]["in_overlap"]

            values_overlap = values[overlap]
            values_isolated = values[~overlap]

            bins = np.arange(vmin, vmax + step, step)

            fig, ax = plt.subplots(figsize=(8, 5))

            _plot_hist_curves(
                ax,
                [
                    (values_overlap, "overlap", "tab:orange"),
                    (values_isolated, "isolati", "tab:purple"),
                ],
                bins,
            )

            _finish_object_plot(
                ax,
                f"{obj_title} {xlabel_suffix}",
                f"{obj_title}: overlap vs isolati (event-level) "
                f"- $\\Delta R < {dr_threshold}$",
                normalize=NORMALIZE_HISTOGRAMS
            )

            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_HISTOGRAMS else ""
            if var_key == "pt":
                suffix_cut = f"_cut_{PT_HIST_MAX}"
            else:
                suffix_cut = ""

            out_path = (
                dir
                / f"{obj_name}_{var_key}_all_overlap_vs_isolated_"
                f"{JET_SELECTION_MODE}{suffix}{suffix_cut}.png"
            )

            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[OK] Plot salvato in: {out_path}")


def save_object_kinematics_truth_split(
    object_data,
    dr_threshold,
):
    """
    6 plot object-level con la verita' separata.

    Per ciascun oggetto (jet o tau) vengono mostrate quattro popolazioni:
      - vero + overlap
      - vero + isolato
      - fake + overlap
      - fake + isolato

    Questa e' una separazione object-level e NON usa le categorie di coppia
    (a)-(d), che restano definite esclusivamente al punto 5.
    """

    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip dei plot truth-split object-level."
        )
        return
    
    dir = OUTPUT_DIR / "truth_categories"
    dir.mkdir(parents=True, exist_ok=True)

    for obj_name, obj_title in (
        ("jet", "jet"),
        ("tau", "tau"),
    ):
        is_true = object_data[obj_name]["is_true"]
        in_overlap = object_data[obj_name]["in_overlap"]

        for var_key, vmin, vmax, step, xlabel_suffix in KINEMATICS_VARIABLES:
            values = object_data[obj_name][var_key]
            bins = np.arange(vmin, vmax + step, step)

            datasets = [
                (
                    values[is_true & in_overlap],
                    "vero + overlap",
                    "tab:green",
                ),
                (
                    values[is_true & ~in_overlap],
                    "vero + isolato",
                    "tab:blue",
                ),
                (
                    values[~is_true & in_overlap],
                    "fake + overlap",
                    "tab:red",
                ),
                (
                    values[~is_true & ~in_overlap],
                    "fake + isolato",
                    "tab:gray",
                ),
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            _plot_hist_curves(ax, datasets, bins)

            _finish_object_plot(
                ax,
                f"{obj_title} {xlabel_suffix}",
                f"{obj_title}: verita' × overlap/isolato (event-level)"
                f"- $\\Delta R < {dr_threshold}$",
                normalize=NORMALIZE_HISTOGRAMS
            )

            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_HISTOGRAMS else ""
            if var_key == "pt":
                suffix_cut = f"_cut_{PT_HIST_MAX}"
            else:
                suffix_cut = ""

            out_path = (
                dir
                / f"{obj_name}_{var_key}_all_overlap_vs_isolated_"
                f"{JET_SELECTION_MODE}{suffix}{suffix_cut}.png"
            )

            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[OK] Plot salvato in: {out_path}")


# ======================================================================
# PERCENTUALI EVENT-LEVEL DELLE CATEGORIE DI COPPIA
# ======================================================================

def count_event_level_overlap_by_category(
    pair_info,
):
    """
    Conteggio event-level per le quattro categorie di coppia.

    Per ogni categoria:
      event_total = numero di eventi che contengono almeno una coppia
                    della categoria, a qualsiasi DeltaR;
      event_overlap = numero di quegli eventi che contengono almeno una
                      coppia della categoria con DeltaR < soglia.

    Le quattro frazioni NON devono sommare a 100%: uno stesso evento puo'
    contenere contemporaneamente coppie di categorie diverse.
    """

    dr_matrix = pair_info["dr"]
    jet_label_ct = pair_info["jet_label"]
    tau_label_ct = pair_info["tau_label"]

    category_masks = {
        "a_jet_true_tau_fake": jet_label_ct & ~tau_label_ct,
        "b_jet_false_tau_true": ~jet_label_ct & tau_label_ct,
        "c_jet_true_tau_true": jet_label_ct & tau_label_ct,
        "d_jet_false_tau_false": ~jet_label_ct & ~tau_label_ct,
    }

    overlap_matrix = dr_matrix < DR_THRESHOLD_KINEMATICS

    event_totals = {}
    event_overlaps = {}

    for cat, pair_mask in category_masks.items():
        has_category = ak.any(pair_mask, axis=-1)
        has_overlap = ak.any(
            pair_mask & overlap_matrix,
            axis=-1,
        )

        event_totals[cat] = int(ak.sum(has_category))
        event_overlaps[cat] = int(ak.sum(has_overlap))

    return event_overlaps, event_totals


def print_overlap_percentages(
    dr_by_category,
    event_overlap_counts,
    event_category_counts,
    dr_threshold,
):
    """
    Stampa sia la percentuale event-level sia la corrispondente percentuale
    pair-level sotto la soglia di overlap.
    """

    section(
        "PERCENTUALI NELLA REGIONE DI OVERLAP "
        f"($\\Delta R < {dr_threshold}$)"
    )

    print(
        "   Le percentuali event-level hanno come denominatore gli eventi "
        "che contengono almeno una coppia della categoria."
    )
    print(
        "   Le percentuali pair-level hanno invece come denominatore tutte "
        "le coppie della categoria.\n"
    )

    for cat in CATEGORY_KEYS:
        n_evt_total = event_category_counts[cat]
        n_evt_overlap = event_overlap_counts[cat]

        if n_evt_total > 0:
            evt_fraction = 100.0 * n_evt_overlap / n_evt_total
            evt_text = (
                f"{evt_fraction:6.2f}% "
                f"({n_evt_overlap}/{n_evt_total} eventi)"
            )
        else:
            evt_text = "n/d"

        dr = dr_by_category[cat]
        n_pairs_total = dr.size
        n_pairs_overlap = int(np.count_nonzero(dr < dr_threshold))

        if n_pairs_total > 0:
            pair_fraction = 100.0 * n_pairs_overlap / n_pairs_total
            pair_text = (
                f"{pair_fraction:6.2f}% "
                f"({n_pairs_overlap}/{n_pairs_total} coppie)"
            )
        else:
            pair_text = "n/d"

        print(f"   {CATEGORY_LABELS[cat]:30s}: event-level = {evt_text}")
        print(f"   {'':30s}  pair-level  = {pair_text}")


# ======================================================================
# SUMMARY OBJECT-LEVEL
# ======================================================================

def print_object_level_summary(object_data, dr_threshold):
    """
    Riassunto numerico dei conteggi object-level.
    """

    section(
        f"SUMMARY OBJECT-LEVEL: OVERLAP / ISOLATO "
        f"(soglia $\\Delta R={dr_threshold}$)"
    )

    for obj_name in ("jet", "tau"):
        is_true = object_data[obj_name]["is_true"]
        in_overlap = object_data[obj_name]["in_overlap"]

        n_total = object_data[obj_name]["pt"].size
        n_overlap = int(np.count_nonzero(in_overlap))
        n_isolated = n_total - n_overlap

        print(f"\n   --- {obj_name} ---")
        print(f"   totale                 : {n_total:8d}")
        print(f"   overlap                : {n_overlap:8d}")
        print(f"   isolati                : {n_isolated:8d}")

        if n_total > 0:
            print(
                f"   frazione overlap       : "
                f"{100.0 * n_overlap / n_total:6.2f}%"
            )

        true_mask = is_true
        fake_mask = ~is_true

        for truth_name, truth_mask in (
            ("veri", true_mask),
            ("fake", fake_mask),
        ):
            n_truth = int(np.count_nonzero(truth_mask))
            n_truth_overlap = int(
                np.count_nonzero(truth_mask & in_overlap)
            )

            if n_truth > 0:
                frac = 100.0 * n_truth_overlap / n_truth
                print(
                    f"   {truth_name:24s}: "
                    f"{n_truth_overlap:8d} overlap / "
                    f"{n_truth:8d} totali "
                    f"({frac:6.2f}%)"
                )
            else:
                print(
                    f"   {truth_name:24s}: nessun oggetto"
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
        "tau_eta",
        "tau_phi",
        TAU_PT_BRANCH,
        "tau_isAnalysisTau___NOSYS",
        "recojet_antikt4PFlow_eta",
        "recojet_antikt4PFlow_phi",
        JET_PT_BRANCH,
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
        JET_TRUTH_LABEL_BRANCH,
    ]

    if JET_SELECTION_MODE == "btag85":
        branches.append(JET_BTAG_BRANCH)

    if TAU_SELECTION_MODE == "score85":
        branches.append(TAU_SCORE_BRANCH)

    if TRUTH_MODE_TAU == "label":
        branches.append(TAU_TRUTH_MATCH_BRANCH)
    elif TRUTH_MODE_TAU == "geometric":
        branches += [
            TRUTH_TAU_ETA_BRANCH,
            TRUTH_TAU_PHI_BRANCH,
        ]
    else:
        raise RuntimeError(
            f"TRUTH_MODE_TAU='{TRUTH_MODE_TAU}' non gestito."
        )

    jet_dr_truth_parts = []
    tau_dr_truth_parts = []

    # Conteggi pair-level per il punto 5.
    cat_totals = {cat: 0 for cat in CATEGORY_KEYS}
    n_overlap_totals = {
        thr: 0
        for thr in DR_THRESHOLDS
    }

    # Dati DeltaR pair-level completi per le 4 categorie.
    dr_category_parts = []

    # Dati object-level per il punto 6.
    object_kinematics_parts = []

    # Conteggi event-level delle categorie di coppia.
    event_overlap_counts = {
        cat: 0
        for cat in CATEGORY_KEYS
    }
    event_category_counts = {
        cat: 0
        for cat in CATEGORY_KEYS
    }

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]

        keys = set(tree.keys())
        missing = [b for b in branches if b not in keys]
        if missing:
            print(
                f"[WARNING] {item['file_name']}: branch mancanti: {missing}"
            )
            print(
                "    Verificare i nomi dei branch in configurazione. "
                "File ignorato."
            )
            continue

        a = tree.arrays(
            branches,
            entry_stop=n_entries,
            library="ak",
        )

        # --------------------------------------------------------------
        # Selezione analysis-level
        # --------------------------------------------------------------

        jet_sel_analysis = get_analysis_selection(
            tree,
            "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
            n_entries,
        )

        tau_sel_analysis = get_analysis_selection(
            tree,
            "tau_isAnalysisTau___NOSYS",
            n_entries,
        )

        # --------------------------------------------------------------
        # Selezione jet
        # --------------------------------------------------------------

        if JET_SELECTION_MODE == "all":
            jet_sel = jet_sel_analysis
        else:
            jet_btag_sel = ak.fill_none(
                a[JET_BTAG_BRANCH],
                False,
            ) != 0
            jet_sel = jet_sel_analysis & jet_btag_sel

        # --------------------------------------------------------------
        # Selezione tau
        # --------------------------------------------------------------

        if TAU_SELECTION_MODE == "all":
            tau_sel = tau_sel_analysis
        else:
            tau_score_sel = ak.fill_none(
                a[TAU_SCORE_BRANCH],
                -np.inf,
            ) >= TAU_SCORE_WP85_THRESHOLD
            tau_sel = tau_sel_analysis & tau_score_sel

        # --------------------------------------------------------------
        # Label di verita'
        # --------------------------------------------------------------

        (
            jet_label,
            tau_label,
            jet_dr_truth,
            tau_dr_truth,
        ) = label_jets_and_taus(
            a,
            jet_sel,
            tau_sel,
        )

        if jet_dr_truth is not None:
            jet_dr_truth_parts.append(jet_dr_truth)
        if tau_dr_truth is not None:
            tau_dr_truth_parts.append(tau_dr_truth)

        # Oggetti gia' selezionati.
        jet_eta = a["recojet_antikt4PFlow_eta"][jet_sel]
        jet_phi = a["recojet_antikt4PFlow_phi"][jet_sel]
        jet_pt = a[JET_PT_BRANCH][jet_sel]

        tau_eta = a["tau_eta"][tau_sel]
        tau_phi = a["tau_phi"][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]

        # --------------------------------------------------------------
        # Matrice comune jet x tau
        # --------------------------------------------------------------

        pair_info = build_pair_geometry_and_labels(
            jet_eta,
            jet_phi,
            jet_label,
            tau_eta,
            tau_phi,
            tau_label,
        )

        # --------------------------------------------------------------
        # Pair-level: punto 5 e distribuzione completa di DeltaR
        # --------------------------------------------------------------

        for thr in DR_THRESHOLDS:
            cats, n_tot = classify_overlapping_pairs(
                pair_info,
                thr,
            )

            n_overlap_totals[thr] += n_tot

            if thr == DR_THRESHOLDS[0]:
                for cat in CATEGORY_KEYS:
                    cat_totals[cat] += cats[cat]

        dr_category_parts.append(
            flatten_dr_by_category(pair_info)
        )

        # --------------------------------------------------------------
        # Event-level: percentuali richieste per le 4 categorie
        # --------------------------------------------------------------

        file_event_overlap, file_event_total = (
            count_event_level_overlap_by_category(pair_info)
        )

        for cat in CATEGORY_KEYS:
            event_overlap_counts[cat] += file_event_overlap[cat]
            event_category_counts[cat] += file_event_total[cat]

        # --------------------------------------------------------------
        # Object-level: punto 6
        # --------------------------------------------------------------

        object_kinematics_parts.append(
            classify_objects_kinematics(
                jet_pt,
                jet_eta,
                jet_phi,
                jet_label,
                tau_pt,
                tau_eta,
                tau_phi,
                tau_label,
                pair_info,
                DR_THRESHOLD_KINEMATICS,
            )
        )

    # ------------------------------------------------------------------
    # A2: tabelle di spalla per truth matching geometrico, se attivo
    # ------------------------------------------------------------------

    section("SOGLIA DI TRUTH-MATCHING (punto 3)")

    if jet_dr_truth_parts:
        jet_truth_summary = summarize(
            ak.concatenate(jet_dr_truth_parts),
            "DeltaR(reco jet, truth b)",
        )
        print_shoulder_table(
            jet_truth_summary,
            "DeltaR(reco jet, truth b)",
        )

    if tau_dr_truth_parts:
        tau_truth_summary = summarize(
            ak.concatenate(tau_dr_truth_parts),
            "DeltaR(reco tau, truth tau)",
        )
        print_shoulder_table(
            tau_truth_summary,
            "DeltaR(reco tau, truth tau)",
        )

    # ------------------------------------------------------------------
    # B: classificazione pair-level del punto 5
    # ------------------------------------------------------------------

    #primary_overlap_threshold = DR_THRESHOLDS[0]
    primary_overlap_threshold = DR_THRESHOLD_KINEMATICS

    section(
        "CLASSIFICAZIONE COPPIE OVERLAPPANTI "
        f"(soglia={primary_overlap_threshold}) - punto 5"
    )

    n_class_total = sum(cat_totals.values())

    for cat in CATEGORY_KEYS:
        value = cat_totals[cat]
        fraction = (
            100.0 * value / n_class_total
            if n_class_total > 0
            else 0.0
        )
        print(
            f"   {cat:30s}: {value:8d} "
            f"({fraction:6.2f}%)"
        )

    print(f"   {'TOTALE':30s}: {n_class_total:8d}")

    # ------------------------------------------------------------------
    # Controllo con obj_3_1.py
    # ------------------------------------------------------------------

    section("CONTROLLO DI CONSISTENZA CON obj_3_1.py")

    for thr in DR_THRESHOLDS:
        print(
            f"   soglia {thr}: coppie overlappanti "
            f"(questo script) = {n_overlap_totals[thr]} "
            "-> confrontare con 'coppie (jet,tau) sovrapposte' "
            "stampato da obj_3_1.py per la stessa soglia "
            "(devono coincidere esattamente)"
        )

    # ------------------------------------------------------------------
    # Merge DeltaR pair-level
    # ------------------------------------------------------------------

    dr_by_category = {
        cat: []
        for cat in CATEGORY_KEYS
    }

    for part in dr_category_parts:
        for cat in CATEGORY_KEYS:
            dr_by_category[cat].append(part[cat])

    for cat in CATEGORY_KEYS:
        arrays = dr_by_category[cat]
        dr_by_category[cat] = (
            np.concatenate(arrays)
            if arrays
            else np.array([])
        )

    # ------------------------------------------------------------------
    # Merge object-level
    # ------------------------------------------------------------------

    object_data = merge_object_kinematics(
        object_kinematics_parts
    )

    # ------------------------------------------------------------------
    # Plot completo DeltaR per categoria (PAIR-LEVEL)
    # ------------------------------------------------------------------

    save_dr_histogram_by_category(
        dr_by_category
    )

    # ------------------------------------------------------------------
    # Percentuali event-level e pair-level sotto la soglia
    # ------------------------------------------------------------------

    print_overlap_percentages(
        dr_by_category,
        event_overlap_counts,
        event_category_counts,
        DR_THRESHOLD_KINEMATICS,
    )

    # ------------------------------------------------------------------
    # Punto 6: cinematica OBJECT-LEVEL
    # ------------------------------------------------------------------

    print_object_level_summary(
        object_data,
        DR_THRESHOLD_KINEMATICS,
    )

    # 6 plot: tutti i jet/tau, overlap vs isolati.
    save_object_kinematics_all_overlap_vs_isolated(
        object_data,
        DR_THRESHOLD_KINEMATICS,
    )

    # 6 plot: verita' separata, sempre object-level.
    save_object_kinematics_truth_split(
        object_data,
        DR_THRESHOLD_KINEMATICS,
    )


if __name__ == "__main__":
    main()
