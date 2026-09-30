"""
================================================================================
Analisi cinematica a livello di COPPIA (pair-level) per jet e tau in overlap,
VARIANTE "no-repetitions" (matching greedy globale).
================================================================================

DIFFERENZA RISPETTO A overlap_kinematics.py
--------------------------------------------
In overlap_kinematics.py una coppia (jet, tau) entra nell'analisi se e solo
se DeltaR(jet, tau) < DR_THRESHOLD_KINEMATICS. Con questo criterio nulla
vieta che uno stesso jet compaia in piu' coppie (con piu' tau diversi) e
viceversa: un evento con 1 jet e 3 tau tutti entro soglia dal jet produce
3 coppie che condividono lo stesso jet.

Qui si costruisce invece un insieme di coppie SENZA ripetizioni: ogni jet
e ogni tau compaiono al piu' in UNA coppia per evento. L'algoritmo e' un
matching greedy globale (nearest-neighbor senza rimpiazzo), per evento:

    1. si calcola la matrice DeltaR(jet_i, tau_j) dell'evento;
    2. si cerca la coppia (i, j) con DeltaR minimo su TUTTA la matrice
       (non "fissato un tau", ma il minimo assoluto residuo);
    3. se questo DeltaR minimo e' < DR_THRESHOLD_KINEMATICS, la coppia
       (jet_i, tau_j) viene accantonata e sia la riga i sia la colonna j
       vengono rimosse dalla matrice (jet_i e tau_j non sono piu'
       disponibili per altre coppie);
    4. si ripete su righe/colonne rimanenti finche' il DeltaR minimo
       residuo non supera la soglia, oppure finche' non si esauriscono
       i jet o i tau disponibili nell'evento.

NOTA SULLA VERSIONE "sequenziale per tau" ORIGINARIAMENTE IPOTIZZATA
---------------------------------------------------------------------
Un algoritmo che scorre i tau nell'ordine del branch (tau1, tau2, ...) e
per ciascuno prende il jet libero piu' vicino e' order-dependent: il
risultato puo' cambiare a seconda di quale tau viene processato per
primo, perche' un jet "conteso" da due tau viene assegnato al primo
tau processato anche se e' oggettivamente piu' vicino al secondo. Il
matching greedy globale sopra descritto risolve l'ambiguita' scegliendo
sempre, ad ogni passo, la coppia col DeltaR minimo assoluto tra tutte
quelle ancora disponibili: il risultato non dipende dall'ordine di
storage di jet o tau nei branch.

Tutto il resto (selezioni, categorie di verita', variabili plottate,
struttura dei plot) e' identico a overlap_kinematics.py: cambia solo il
criterio con cui si costruiscono le coppie di analisi.

Le funzioni duplicate di labeling e plotting vengono importate
direttamente da truth_vs_reco_params.py, come nel file originale.
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

# Import delle funzioni duplicate dal file precedente (stesse di overlap_kinematics.py)
from truth_vs_reco_params import (
    match_reco_to_truth,
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)
from overlap_met_tau import compute_tau_met_proj, compute_tau_mt

# ======================================================================
# CONFIGURAZIONE
# (identica a overlap_kinematics.py, solo OUTPUT_DIR distinta per non
#  sovrascrivere i plot della versione "con ripetizioni")
# ======================================================================

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

OUTPUT_DIR = Path("../output/discriminance_analysis/no_reps/pair_kinematics_categories")

JET_PT_BRANCH = "recojet_antikt4PFlow_pt___NOSYS"
TAU_PT_BRANCH = "tau_pt___NOSYS"
JET_MASS_BRANCH = "recojet_antikt4PFlow_m___NOSYS"
JET_N_MUONS_BRANCH = "recojet_antikt4PFlow_n_muons___NOSYS"

COMPUTE_PT_RATIO = True
COMPUTE_MET_PROJ = True
COMPUTE_MT = True

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
    "jet_pt": {"y_scale": "log", "out_dir": OUTPUT_DIR},
    "jet_mass": {"y_scale": "log", "out_dir": OUTPUT_DIR},
    "jet_n_muons": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "jet_eta": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "jet_phi": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "tau_pt": {"y_scale": "log", "out_dir": OUTPUT_DIR},
    "tau_eta": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "tau_phi": {"y_scale": "linear", "out_dir": OUTPUT_DIR / "angular"},
    "tau_nProng": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "tau_decayMode": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "tau_charge": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "pair_pt_ratio": {"y_scale": "linear", "out_dir": OUTPUT_DIR},
    "pair_dr": {"y_scale": "log", "out_dir": OUTPUT_DIR / "angular"},
    "pair_deta": {"y_scale": "log", "out_dir": OUTPUT_DIR / "angular"},
    "pair_dphi": {"y_scale": "log", "out_dir": OUTPUT_DIR / "angular"},
}

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
# MATCHING GREEDY GLOBALE SENZA RIPETIZIONI
# ======================================================================

def greedy_match_no_reps_indices(jet_eta, jet_phi, tau_eta, tau_phi, dr_thr):
    """
    Per ogni evento, calcola un insieme di coppie (jet, tau) senza
    ripetizioni tramite matching greedy globale (nearest-neighbor senza
    rimpiazzo):

        - si cerca ad ogni passo la coppia (jet, tau) ancora disponibile
          col DeltaR minimo assoluto nell'evento;
        - se DeltaR_min < dr_thr si forma la coppia e si rimuovono jet e
          tau coinvolti dal pool disponibile;
        - si continua finche' il DeltaR minimo residuo non supera la
          soglia, o finche' non si esauriscono jet o tau nell'evento.

    Il matching e' quindi al piu' min(n_jet, n_tau) coppie per evento,
    e ogni jet/tau compare in al piu' UNA coppia.

    Ritorna due ak.Array jagged (uno per evento) con gli indici LOCALI
    (cioe' relativi alle collezioni jet_eta/tau_eta gia' selezionate in
    input) dei jet e dei tau accoppiati, nello stesso ordine (la coppia
    i-esima e' (matched_jet_idx[evt][i], matched_tau_idx[evt][i])).
    """

    # Conversione a liste Python: il matching e' intrinsecamente
    # iterativo per evento, quindi si lavora con NumPy sulla singola
    # matrice DeltaR dell'evento (piccola: tipicamente pochi jet/tau).
    jet_eta_list = ak.to_list(jet_eta)
    jet_phi_list = ak.to_list(jet_phi)
    tau_eta_list = ak.to_list(tau_eta)
    tau_phi_list = ak.to_list(tau_phi)

    n_events = len(jet_eta_list)

    out_jet_idx = []
    out_tau_idx = []

    for iev in range(n_events):
        j_eta = np.asarray(jet_eta_list[iev], dtype=float)
        j_phi = np.asarray(jet_phi_list[iev], dtype=float)
        t_eta = np.asarray(tau_eta_list[iev], dtype=float)
        t_phi = np.asarray(tau_phi_list[iev], dtype=float)

        n_j = j_eta.size
        n_t = t_eta.size

        if n_j == 0 or n_t == 0:
            out_jet_idx.append([])
            out_tau_idx.append([])
            continue

        deta = j_eta[:, None] - t_eta[None, :]
        dphi = (j_phi[:, None] - t_phi[None, :] + np.pi) % (2.0 * np.pi) - np.pi
        dr = np.sqrt(deta ** 2 + dphi ** 2)

        available_j = np.ones(n_j, dtype=bool)
        available_t = np.ones(n_t, dtype=bool)

        pairs_j = []
        pairs_t = []

        n_max_pairs = min(n_j, n_t)

        for _ in range(n_max_pairs):
            masked = np.where(
                available_j[:, None] & available_t[None, :], dr, np.inf
            )
            min_val = masked.min()

            if not np.isfinite(min_val) or min_val >= dr_thr:
                break

            idx_j, idx_t = np.unravel_index(np.argmin(masked), masked.shape)

            pairs_j.append(int(idx_j))
            pairs_t.append(int(idx_t))

            available_j[idx_j] = False
            available_t[idx_t] = False

        out_jet_idx.append(pairs_j)
        out_tau_idx.append(pairs_t)

    return ak.Array(out_jet_idx), ak.Array(out_tau_idx)


def check_no_repetition(matched_jet_idx, matched_tau_idx, label=""):
    """
    Controllo interno: verifica che, evento per evento, gli indici dei
    jet accoppiati siano tutti distinti tra loro, e lo stesso per i tau
    (garanzia strutturale del matching greedy, verificata qui a scopo
    diagnostico).
    """

    def has_no_duplicates_per_event(idx_array):
        as_list = ak.to_list(idx_array)
        for ev_idx in as_list:
            if len(ev_idx) != len(set(ev_idx)):
                return False
        return True

    ok_jet = has_no_duplicates_per_event(matched_jet_idx)
    ok_tau = has_no_duplicates_per_event(matched_tau_idx)

    n_pairs = int(ak.sum(ak.num(matched_jet_idx)))

    status_jet = "OK" if ok_jet else "FALLITO"
    status_tau = "OK" if ok_tau else "FALLITO"

    print(
        f"   [{label}] controllo no-ripetizioni: "
        f"jet={status_jet}  tau={status_tau}  (coppie totali: {n_pairs})"
    )


# ======================================================================
# COSTRUZIONE FEATURE DI COPPIA (a partire dalle coppie gia' individuate)
# ======================================================================

def build_matched_pair_features(
    jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
    tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
    matched_jet_idx, matched_tau_idx,
    met=None, met_phi=None,
    compute_pt_ratio=COMPUTE_PT_RATIO,
    compute_met_proj=COMPUTE_MET_PROJ,
    compute_mt=COMPUTE_MT,
):
    """
    A differenza di build_pair_kinematics_and_labels (in
    overlap_kinematics.py), qui NON si costruisce il prodotto cartesiano
    jet x tau: le coppie sono gia' state individuate dal matching greedy
    (matched_jet_idx / matched_tau_idx), quindi si usa l'indicizzazione
    "jagged" di awkward per selezionare, evento per evento, esattamente
    gli oggetti accoppiati.
    """

    jet_pt_m = jet_pt[matched_jet_idx]
    jet_eta_m = jet_eta[matched_jet_idx]
    jet_phi_m = jet_phi[matched_jet_idx]
    jet_mass_m = jet_mass[matched_jet_idx]
    jet_n_muons_m = jet_n_muons[matched_jet_idx]
    jet_label_m = jet_label[matched_jet_idx]

    tau_pt_m = tau_pt[matched_tau_idx]
    tau_eta_m = tau_eta[matched_tau_idx]
    tau_phi_m = tau_phi[matched_tau_idx]
    tau_nProng_m = tau_nProng[matched_tau_idx]
    tau_decayMode_m = tau_decayMode[matched_tau_idx]
    tau_charge_m = tau_charge[matched_tau_idx]
    tau_label_m = tau_label[matched_tau_idx]

    # Per costruzione ogni coppia ha gia' DeltaR < dr_thr: non serve
    # ricalcolare/filtrare con una maschera come nella versione originale.
    dr_pairs = delta_r(jet_eta_m, jet_phi_m, tau_eta_m, tau_phi_m)
    deta_pairs = jet_eta_m - tau_eta_m
    dphi_pairs = (jet_phi_m - tau_phi_m + np.pi) % (2.0 * np.pi) - np.pi

    features = {
        "pair_dr": dr_pairs,
        "pair_deta": deta_pairs,
        "pair_dphi": dphi_pairs,
        "jet_pt": jet_pt_m,
        "jet_eta": jet_eta_m,
        "jet_phi": jet_phi_m,
        "jet_mass": jet_mass_m,
        "jet_n_muons": jet_n_muons_m,
        "tau_pt": tau_pt_m,
        "tau_eta": tau_eta_m,
        "tau_phi": tau_phi_m,
        "tau_nProng": tau_nProng_m,
        "tau_decayMode": tau_decayMode_m,
        "tau_charge": tau_charge_m,
        "jet_label": jet_label_m,
        "tau_label": tau_label_m,
    }

    if compute_pt_ratio:
        features["pair_pt_ratio"] = jet_pt_m / tau_pt_m

    if compute_met_proj and met is not None and met_phi is not None:
        met_bcast = ak.broadcast_arrays(met, tau_pt_m)[0]
        met_phi_bcast = ak.broadcast_arrays(met_phi, tau_phi_m)[0]
        features["tau_met_proj"] = compute_tau_met_proj(tau_phi_m, met_bcast, met_phi_bcast)

    if compute_mt and met is not None and met_phi is not None:
        met_bcast = ak.broadcast_arrays(met, tau_pt_m)[0]
        met_phi_bcast = ak.broadcast_arrays(met_phi, tau_phi_m)[0]
        features["tau_mt"] = compute_tau_mt(tau_pt_m, tau_phi_m, met_bcast, met_phi_bcast)

    return features


def categorize_matched_pairs(features):
    """
    Equivalente di get_overlapping_pairs_kinematics in
    overlap_kinematics.py, ma senza maschera di soglia (gia' incorporata
    nella costruzione delle coppie a monte).
    """

    def flat(key):
        return ak.to_numpy(ak.flatten(features[key], axis=None))

    jet_lab = flat("jet_label").astype(bool)
    tau_lab = flat("tau_label").astype(bool)

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
    variables = [k for k in features.keys() if k not in ("jet_label", "tau_label")]

    for cat, cmask in cat_masks.items():
        for var in variables:
            res[cat][var] = flat(var)[cmask]

    return res


def merge_pair_kinematics(parts_list):
    if not parts_list:
        return {}

    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    variables = list(parts_list[0][cat_keys[0]].keys())

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
# (identici a overlap_kinematics.py, adattati solo nel titolo per
#  indicare la variante "no-repetitions")
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
            config = VARIABLE_PLOT_CONFIG.get(
                var_id, {"y_scale": "linear", "out_dir": OUTPUT_DIR / "pair_kinematics_categories_no_reps"}
            )

            out_dir = Path(config["out_dir"])
            if AGGREGATE_CATEGORIES:
                out_dir = out_dir / "aggregated"

            plot_dir = out_dir / f"DR_max_{DR_THRESHOLD_KINEMATICS}"
            plot_dir.mkdir(parents=True, exist_ok=True)

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
                f"Cinematica {obj_title} in coppie no-rep (pair-level) - $\\Delta R < {dr_threshold}$",
                normalize=NORMALIZE_HISTOGRAMS
            )

            ax.set_yscale(y_scale)

            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_HISTOGRAMS else ""
            if var_key == "pt":
                suffix_cut = f"_cut_{PT_HIST_MAX}"
            elif var_key == "mass":
                suffix_cut = f"_cut_{M_HIST_MAX}"
            else:
                suffix_cut = ""
            suffix_agg = "_aggregated" if AGGREGATE_CATEGORIES else ""

            out_path = (
                plot_dir
                / f"pair_kinematics_no_reps_{var_id}_"
                f"{JET_SELECTION_MODE}{suffix}{suffix_cut}{suffix_agg}.png"
            )

            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[OK] Plot salvato in: {out_path}")


def save_scatter_plots(merged_data, hreshold):
    if not HAS_MPL:
        return

    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    out_dir = OUTPUT_DIR / "scatters"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        ax.set_title(f"Scatter (no-rep): $\\Delta R$ vs {ylabel} ($\\Delta R < {DR_THRESHOLD_KINEMATICS}$)")

        ax.set_yscale("log")

        ax.legend(title="Categorie", fontsize=9, loc='best')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"scatter_no_reps_dr_vs_{var_key}.png"
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
        JET_TRUTH_LABEL_BRANCH,
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

    section("MATCHING GREEDY SENZA RIPETIZIONI")

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

        # --- costruzione coppie: matching greedy globale senza ripetizioni ---
        matched_jet_idx, matched_tau_idx = greedy_match_no_reps_indices(
            jet_eta, jet_phi, tau_eta, tau_phi, DR_THRESHOLD_KINEMATICS
        )

        check_no_repetition(matched_jet_idx, matched_tau_idx, label=item["file_name"])

        pair_features = build_matched_pair_features(
            jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
            tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
            matched_jet_idx, matched_tau_idx,
        )

        pair_kinematics_parts.append(categorize_matched_pairs(pair_features))

    merged_pair_kinematics = merge_pair_kinematics(pair_kinematics_parts)

    section("PLOT CINEMATICA PAIR-LEVEL (NO-REPS)")
    save_pair_kinematics_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)

    section("SCATTER PLOTS (NO-REPS)")
    save_scatter_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)


if __name__ == "__main__":
    main()