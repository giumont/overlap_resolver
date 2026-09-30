from pathlib import Path

import numpy as np
import awkward as ak
import uproot


# ======================================================================
# CONFIGURAZIONE
# ======================================================================

ROOT_DIR = Path("HH_bbtt")
FILE_PREFIX = "output_GGF_mc23a_bypass_noOR_0000"
FILE_SUFFIX = ".root"

TREE_NAME = "AnalysisMiniTree"

# None = tutti gli eventi
# Mettere ad esempio 10000 per una diagnostica rapida
N_ENTRIES_CAP = None

# ================================================================
# SELEZIONE OGGETTI ANALYSIS-LEVEL
# ================================================================

# True  -> tutte le statistiche di jet/tau usano solo:
#          isAnalysisJet == True
#          isAnalysisTau == True
#
# False -> vengono usati tutti gli oggetti presenti nelle collezioni
#
# NOTA:
# analyze_A() NON usa questo parametro, perché serve a studiare
# proprio la frazione di oggetti che soddisfa il flag.
USE_ANALYSIS_OBJECTS = True

# Quanti valori mostrare nelle distribuzioni aggregate
TOP_VALUES = 20

# Quanti valori unici mostrare per le parentHiggsParentsMask
TOP_MASK_VALUES = 30


# ======================================================================
# UTILITY
# ======================================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subsection(title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def flatten_to_numpy(arr):
    """
    Appiattisce un array awkward e rimuove None.
    Restituisce un numpy array 1D.
    """
    flat = ak.flatten(arr, axis=None)
    flat = ak.drop_none(flat)
    return ak.to_numpy(flat)


def get_value_counts(arr):
    """
    Restituisce:
        counts: dict {valore: conteggio}
        total: numero totale di valori validi
    """
    flat_np = flatten_to_numpy(arr)

    if flat_np.size == 0:
        return {}, 0

    vals, counts = np.unique(flat_np, return_counts=True)

    return {
        v.item() if hasattr(v, "item") else v: int(c)
        for v, c in zip(vals, counts)
    }, int(flat_np.size)


def merge_count_dicts(dicts):
    """
    Somma una lista di dizionari {valore: conteggio}.
    """
    out = {}

    for d in dicts:
        for key, count in d.items():
            out[key] = out.get(key, 0) + count

    return out


def sorted_counts(counts):
    """
    Restituisce lista [(valore, conteggio)] ordinata per conteggio decrescente.
    """
    return sorted(counts.items(), key=lambda x: -x[1])


def print_distribution(counts, total, top=20, indent="      "):
    """
    Stampa distribuzione di valori.
    """
    if total == 0:
        print(indent + "nessun valore disponibile")
        return

    for value, count in sorted_counts(counts)[:top]:
        frac = count / total
        print(
            f"{indent}{value!r:>12}  "
            f"count={count:>10d}  "
            f"({100.0 * frac:6.2f}%)"
        )

    if len(counts) > top:
        print(
            indent
            + f"... e altri {len(counts) - top} valori distinti"
        )


def print_min_max_deviation(records, aggregate_value, value_key, unit=""):
    """
    records = lista di dict, uno per file.

    value_key = quantità da confrontare con l'aggregato.

    Stampa:
      - deviazione minima firmata
      - deviazione massima firmata
      - massimo scostamento assoluto
    """
    valid = [
        r for r in records
        if r.get(value_key) is not None
    ]

    if not valid:
        print("      nessun file disponibile")
        return

    deviations = [
        (
            r[value_key] - aggregate_value,
            r["file_name"]
        )
        for r in valid
    ]

    min_dev, min_file = min(deviations, key=lambda x: x[0])
    max_dev, max_file = max(deviations, key=lambda x: x[0])
    abs_dev, abs_file = max(deviations, key=lambda x: abs(x[0]))

    print(
        f"      deviazione minima: {min_dev:+.6f}{unit} "
        f"({min_file})"
    )
    print(
        f"      deviazione massima: {max_dev:+.6f}{unit} "
        f"({max_file})"
    )
    print(
        f"      max |deviazione|: {abs_dev:+.6f}{unit} "
        f"({abs_file})"
    )

def get_analysis_selection(tree, selection_branch, n_entries):
    """
    Legge il flag di selezione analysis-level e lo converte
    esplicitamente in una maschera booleana.

    0    -> False
    != 0 -> True
    """
    selection = tree[selection_branch].array(
        entry_stop=n_entries,
        library="ak",
    )

    # Eventuali None vengono trattati come False
    selection = ak.fill_none(selection, 0)

    # IMPORTANTISSIMO:
    # il branch può essere int8/uint8, quindi convertiamo
    # esplicitamente 0/1 in False/True
    selection = selection != 0

    return selection


def apply_object_selection(
    arr,
    tree,
    selection_branch,
    n_entries,
):
    """
    Applica eventualmente il filtro analysis-level alla collezione.

    Se USE_ANALYSIS_OBJECTS == True:
        arr -> arr[selection]

    Se USE_ANALYSIS_OBJECTS == False:
        arr viene restituito invariato.

    Il filtro viene applicato evento per evento, prima del flatten.
    """
    if not USE_ANALYSIS_OBJECTS:
        return arr

    selection = get_analysis_selection(
        tree,
        selection_branch,
        n_entries,
    )

    return arr[selection]


def read_branch(
    tree,
    branch,
    n_entries,
    selection_branch=None,
):
    """
    Legge un branch e, se richiesto e attivato globalmente,
    applica il relativo filtro analysis-level.

    selection_branch:
        None -> nessuno slicing
        branch del flag -> slicing corrispondente
    """
    arr = tree[branch].array(
        entry_stop=n_entries,
        library="ak",
    )

    if selection_branch is not None:
        arr = apply_object_selection(
            arr,
            tree,
            selection_branch,
            n_entries,
        )

    return arr


# ======================================================================
# LETTURA DEI FILE
# ======================================================================

def file_path(index):
    """
    index = 1 ... 14
    """
    return (
        ROOT_DIR
        / f"{FILE_PREFIX}{index:02d}{FILE_SUFFIX}"
    )


def load_files():
    """
    Restituisce lista di dict contenenti:
        index
        path
        file_name
        tree
        n_entries

    per tutti i file esistenti.

    Se un file atteso non esiste, viene segnalato e saltato.
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
                print(
                    f"[ERRORE] {path}: "
                    f"tree '{TREE_NAME}' non trovato"
                )
                continue

            tree = root_file[TREE_NAME]

            n_entries = tree.num_entries

            if N_ENTRIES_CAP is not None:
                n_entries_used = min(
                    n_entries,
                    N_ENTRIES_CAP
                )
            else:
                n_entries_used = n_entries

            print(
                f"[OK] {path}   "
                f"entries={n_entries}   "
                f"usati={n_entries_used}"
            )

            loaded.append(
                {
                    "index": i,
                    "path": path,
                    "file_name": path.name,
                    "tree": tree,
                    "n_entries": n_entries_used,
                }
            )

        except Exception as exc:
            print(f"[ERRORE] {path}: {exc}")

    print(f"\nFile caricati correttamente: {len(loaded)}/14")

    return loaded


# ======================================================================
# (A) PRE-FILTRAGGIO
# ======================================================================

A_CHECKS = [
    (
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
        "jet",
        "recojet_antikt4PFlow_eta",
    ),
    (
        "tau_isAnalysisTau___NOSYS",
        "tau",
        "tau_eta",
    ),
    (
        "el_isAnalysisElectron___NOSYS",
        "elettrone",
        "el_eta",
    ),
    (
        "mu_isAnalysisMuon___NOSYS",
        "muone",
        "mu_eta",
    ),
]


def analyze_A(loaded):
    """
    Studia i flag isAnalysisJet/Tau/Electron/Muon per determinare
    se le relative collezioni siano già pre-filtrate a livello
    analysis.

    Questa funzione NON applica USE_ANALYSIS_OBJECTS, perché
    l'obiettivo è proprio misurare la frazione True/False.
    """

    section("(A) PRE-FILTRAGGIO DELLE COLLEZIONI")

    for flag_branch, label, ref_branch in A_CHECKS:

        subsection(
            f"{label}: {flag_branch}"
        )

        per_file = []

        aggregate_total_objects = 0
        aggregate_true_objects = 0
        aggregate_events = 0

        for item in loaded:

            tree = item["tree"]
            file_name = item["file_name"]
            n_entries = item["n_entries"]

            keys = set(tree.keys())

            if flag_branch not in keys:
                print(
                    f"  {file_name}: "
                    f"{flag_branch} NON TROVATO"
                )
                continue

            flag = tree[flag_branch].array(
                entry_stop=n_entries,
                library="ak",
            )

            flag = ak.fill_none(flag, False)

            flat_flag = flatten_to_numpy(flag)

            n_total = int(flat_flag.size)
            n_true = int(
                np.count_nonzero(
                    flat_flag.astype(bool)
                )
            )

            n_events = n_entries

            frac_true = (
                n_true / n_total
                if n_total > 0
                else np.nan
            )

            avg_total = (
                n_total / n_events
                if n_events > 0
                else np.nan
            )

            avg_true = (
                n_true / n_events
                if n_events > 0
                else np.nan
            )

            # Controllo della collezione di riferimento
            if ref_branch not in keys:
                print(
                    f"  {file_name}: "
                    f"{ref_branch} NON TROVATO"
                )
                continue

            ref = tree[ref_branch].array(
                entry_stop=n_entries,
                library="ak",
            )

            n_ref = int(
                ak.sum(
                    ak.num(ref, axis=1)
                )
            )

            if n_ref != n_total:
                print(
                    f"  ATTENZIONE {file_name}: "
                    f"{flag_branch} contiene {n_total} oggetti, "
                    f"{ref_branch} ne contiene {n_ref}"
                )

            per_file.append(
                {
                    "file_name": file_name,
                    "n_events": n_events,
                    "n_total": n_total,
                    "n_true": n_true,
                    "frac_true": frac_true,
                    "avg_total": avg_total,
                    "avg_true": avg_true,
                }
            )

            aggregate_total_objects += n_total
            aggregate_true_objects += n_true
            aggregate_events += n_events

        if aggregate_total_objects == 0:
            print("\n  Nessun dato disponibile.")
            continue

        aggregate_frac_true = (
            aggregate_true_objects
            / aggregate_total_objects
        )

        aggregate_avg_total = (
            aggregate_total_objects
            / aggregate_events
        )

        aggregate_avg_true = (
            aggregate_true_objects
            / aggregate_events
        )

        print("\n  STATISTICA AGGREGATA SU TUTTI I FILE")

        print(
            f"      eventi totali usati:       "
            f"{aggregate_events:>10d}"
        )

        print(
            f"      oggetti totali:            "
            f"{aggregate_total_objects:>10d}"
        )

        print(
            f"      oggetti con flag=True:     "
            f"{aggregate_true_objects:>10d}"
        )

        print(
            f"      frazione True:             "
            f"{aggregate_frac_true:.6f} "
            f"({100.0 * aggregate_frac_true:.3f}%)"
        )

        print(
            f"      media oggetti/evento:      "
            f"{aggregate_avg_total:.6f}"
        )

        print(
            f"      media True/evento:         "
            f"{aggregate_avg_true:.6f}"
        )

        if aggregate_frac_true > 0.98:
            print(
                "      interpretazione: quasi tutti gli oggetti "
                "sono True; la collezione sembra già "
                "pre-filtrata a livello analysis."
            )

        elif aggregate_frac_true < 0.02:
            print(
                "      interpretazione: quasi tutti gli oggetti "
                "sono False; ricontrollare la definizione del flag."
            )

        else:
            print(
                "      interpretazione: presenza significativa "
                "di True e False; la collezione contiene "
                "oggetti non ancora passati dal filtro analysis."
            )

        print(
            "\n  DEVIAZIONE DEI SINGOLI FILE "
            "RISPETTO ALL'AGGREGATO"
        )

        print("\n    Frazione True:")
        print_min_max_deviation(
            per_file,
            aggregate_frac_true,
            "frac_true",
            unit="",
        )

        print("\n    Media oggetti/evento:")
        print_min_max_deviation(
            per_file,
            aggregate_avg_total,
            "avg_total",
            unit="",
        )

        print("\n    Media oggetti con flag=True/evento:")
        print_min_max_deviation(
            per_file,
            aggregate_avg_true,
            "avg_true",
            unit="",
        )

        pp_records = [
            {
                **r,
                "frac_true": 100.0 * r["frac_true"],
            }
            for r in per_file
        ]

        print("\n    Frazione True in punti percentuali:")
        print_min_max_deviation(
            pp_records,
            100.0 * aggregate_frac_true,
            "frac_true",
            unit=" pp",
        )


# ======================================================================
# (B) TRUTH LABELS
# ======================================================================

TRUTH_LABEL_BRANCHES = [
    "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
    "recojet_antikt4PFlow_PartonTruthLabelID",
    "tau_truthType",
    "tau_truthOrigin",
    "tau_tauTruthJetLabel",
    "tau_truth_IsHadronicTau",
]

MASK_BRANCHES = [
    "recojet_antikt4PFlow_parentHiggsParentsMask",
    "tau_parentHiggsParentsMask",
]


def get_selection_branch_for_branch(branch):
    """
    Associa a ciascun branch jet/tau il relativo flag
    analysis-level.

    Jet:
        recojet_antikt4PFlow_* -> isAnalysisJet

    Tau:
        tau_* -> isAnalysisTau

    Altri branch:
        nessuno slicing.
    """

    if branch.startswith("recojet_antikt4PFlow_"):
        return "recojet_antikt4PFlow_isAnalysisJet___NOSYS"

    if branch.startswith("tau_"):
        return "tau_isAnalysisTau___NOSYS"

    return None


def analyze_distribution_branch(loaded, branch, top=20):
    """
    Analizza un branch categoriale:

      - distribuzione aggregata
      - deviazioni per ciascuno dei 14 file

    Se USE_ANALYSIS_OBJECTS=True, i branch jet/tau vengono
    filtrati rispettivamente con isAnalysisJet/isAnalysisTau.
    """

    per_file_counts = []
    aggregate_counts_list = []

    aggregate_total = 0

    selection_branch = get_selection_branch_for_branch(branch)

    if USE_ANALYSIS_OBJECTS and selection_branch is not None:
        print(
            f"\n    SELEZIONE ATTIVA: {selection_branch}"
        )
    else:
        print("\n    SELEZIONE ANALYSIS-LEVEL: DISATTIVATA")

    for item in loaded:

        tree = item["tree"]
        file_name = item["file_name"]
        n_entries = item["n_entries"]

        if branch not in set(tree.keys()):
            continue

        # Lettura + eventuale slicing analysis-level
        arr = read_branch(
            tree,
            branch,
            n_entries,
            selection_branch=selection_branch,
        )

        counts, total = get_value_counts(arr)

        per_file_counts.append(
            {
                "file_name": file_name,
                "counts": counts,
                "total": total,
            }
        )

        aggregate_counts_list.append(counts)
        aggregate_total += total

    if aggregate_total == 0:
        print("      nessun valore disponibile")
        return

    aggregate_counts = merge_count_dicts(
        aggregate_counts_list
    )

    print("\n    Distribuzione AGGREGATA:")

    print_distribution(
        aggregate_counts,
        aggregate_total,
        top=top,
    )

    print(
        "\n    DEVIAZIONI PER VALORE "
        "(frequenza file - frequenza aggregata, "
        "in punti percentuali)"
    )

    values_to_report = [
        value
        for value, _ in sorted_counts(
            aggregate_counts
        )[:top]
    ]

    all_values = set(aggregate_counts.keys())

    for record in per_file_counts:
        all_values.update(
            record["counts"].keys()
        )

    extra_values = [
        value
        for value in all_values
        if value not in values_to_report
    ]

    extra_values = sorted(
        extra_values,
        key=lambda v: -aggregate_counts.get(v, 0),
    )[: max(
        0,
        top - len(values_to_report)
    )]

    values_to_report.extend(extra_values)

    for value in values_to_report:

        agg_frac = (
            aggregate_counts.get(value, 0)
            / aggregate_total
        )

        records = []

        for record in per_file_counts:

            file_frac = (
                record["counts"].get(value, 0)
                / record["total"]
                if record["total"] > 0
                else np.nan
            )

            records.append(
                {
                    "file_name": record["file_name"],
                    "value_frac": 100.0 * file_frac,
                }
            )

        if not records:
            continue

        valid = [
            r for r in records
            if not np.isnan(r["value_frac"])
        ]

        if not valid:
            continue

        deviations = [
            (
                r["value_frac"]
                - 100.0 * agg_frac,
                r["file_name"],
            )
            for r in valid
        ]

        min_dev, min_file = min(
            deviations,
            key=lambda x: x[0]
        )

        max_dev, max_file = max(
            deviations,
            key=lambda x: x[0]
        )

        abs_dev, abs_file = max(
            deviations,
            key=lambda x: abs(x[0])
        )

        print(
            f"      valore={value!r:>12} | "
            f"agg={100.0 * agg_frac:7.3f}% | "
            f"min={min_dev:+7.3f} pp ({min_file}) | "
            f"max={max_dev:+7.3f} pp ({max_file}) | "
            f"max|dev|={abs_dev:+7.3f} pp ({abs_file})"
        )


def analyze_mask_branch(
    loaded,
    branch,
    additional_selection_branch=None,
):
    """
    Analisi specifica delle parentHiggsParentsMask:

      - valori aggregati
      - rappresentazione binaria
      - distribuzione aggregata
      - deviazioni per valore tra file

    Se USE_ANALYSIS_OBJECTS=True:
      - jet mask -> isAnalysisJet
      - tau mask -> isAnalysisTau

    Se additional_selection_branch è specificato, vengono stampate
    due tabelle:
      1) con la sola selezione analysis-level
      2) con la selezione analysis-level + additional_selection_branch
    """

    subsection(branch)

    selection_branch = get_selection_branch_for_branch(branch)

    # --------------------------------------------------------------
    # Definizione dei due casi da analizzare
    # --------------------------------------------------------------

    selection_cases = []

    # Caso base: solo analysis-level
    selection_cases.append(
        {
            "extra_branch": None,
            "label": selection_branch,
        }
    )

    # Caso aggiuntivo: analysis-level + maschera extra
    if additional_selection_branch is not None:
        selection_cases.append(
            {
                "extra_branch": additional_selection_branch,
                "label": (
                    f"{selection_branch} "
                    f"+ {additional_selection_branch}"
                ),
            }
        )

    # --------------------------------------------------------------
    # Analisi dei diversi casi
    # --------------------------------------------------------------

    for case in selection_cases:

        extra_selection_branch = case["extra_branch"]
        selection_label = case["label"]

        print()
        print(
            f"    SELEZIONE ATTIVA: {selection_label}"
        )

        per_file_counts = []
        aggregate_counts_list = []

        aggregate_total = 0

        # ----------------------------------------------------------
        # Loop sui file
        # ----------------------------------------------------------

        for item in loaded:

            tree = item["tree"]
            file_name = item["file_name"]
            n_entries = item["n_entries"]

            keys = set(tree.keys())

            if branch not in keys:
                print(
                    f"      {file_name}: {branch} NON TROVATO"
                )
                continue

            # ------------------------------------------------------
            # Lettura della mask
            # ------------------------------------------------------

            arr = tree[branch].array(
                entry_stop=n_entries,
                library="ak",
            )

            # ------------------------------------------------------
            # Selezione analysis-level
            # ------------------------------------------------------

            if (
                USE_ANALYSIS_OBJECTS
                and selection_branch is not None
            ):

                analysis_selection = (
                    get_analysis_selection(
                        tree,
                        selection_branch,
                        n_entries,
                    )
                )

                arr = arr[analysis_selection]

            # ------------------------------------------------------
            # Eventuale selezione tau truth-hadronic
            # ------------------------------------------------------

            if extra_selection_branch is not None:

                if extra_selection_branch not in keys:
                    print(
                        f"      {file_name}: "
                        f"{extra_selection_branch} NON TROVATO "
                        f"-> file ignorato per questo caso"
                    )
                    continue

                extra_selection = (
                    tree[extra_selection_branch].array(
                        entry_stop=n_entries,
                        library="ak",
                    )
                )

                # Il branch può essere int8/uint8:
                # convertiamo esplicitamente in bool.
                extra_selection = ak.fill_none(
                    extra_selection,
                    0,
                )

                extra_selection = (
                    extra_selection != 0
                )

                # IMPORTANTE:
                # la maschera viene applicata alla stessa
                # collezione prima del flatten.
                arr = arr[extra_selection]

            # ------------------------------------------------------
            # Conteggi
            # ------------------------------------------------------

            counts, total = get_value_counts(arr)

            per_file_counts.append(
                {
                    "file_name": file_name,
                    "counts": counts,
                    "total": total,
                }
            )

            aggregate_counts_list.append(counts)
            aggregate_total += total

        # ----------------------------------------------------------
        # Controllo dati
        # ----------------------------------------------------------

        if aggregate_total == 0:
            print(
                "      nessun valore disponibile"
            )
            continue

        aggregate_counts = merge_count_dicts(
            aggregate_counts_list
        )

        # ----------------------------------------------------------
        # Tabella aggregata
        # ----------------------------------------------------------

        print(
            "\n    Valori AGGREGATI e bit attivi "
            "(valore | binario | count | frazione)"
        )

        for value, count in sorted_counts(
            aggregate_counts
        )[:TOP_MASK_VALUES]:

            value_int = int(value)
            frac = count / aggregate_total

            print(
                f"      valore={value_int:>6d}  "
                f"binario={value_int:012b}  "
                f"count={count:>10d}  "
                f"({100.0 * frac:6.2f}%)"
            )

        if len(aggregate_counts) > TOP_MASK_VALUES:

            print(
                f"      ... e altri "
                f"{len(aggregate_counts) - TOP_MASK_VALUES} "
                "valori distinti"
            )

        # ----------------------------------------------------------
        # Deviazioni
        # ----------------------------------------------------------

        print(
            "\n    DEVIAZIONI PER VALORE "
            "(in punti percentuali)"
        )

        values_to_report = [
            value
            for value, _ in sorted_counts(
                aggregate_counts
            )[:TOP_MASK_VALUES]
        ]

        for value in values_to_report:

            agg_frac = (
                aggregate_counts[value]
                / aggregate_total
            )

            deviations = []

            for record in per_file_counts:

                if record["total"] == 0:
                    continue

                file_frac = (
                    record["counts"].get(value, 0)
                    / record["total"]
                )

                delta_pp = (
                    100.0 * file_frac
                    - 100.0 * agg_frac
                )

                deviations.append(
                    (
                        delta_pp,
                        record["file_name"],
                    )
                )

            if not deviations:
                continue

            min_dev, min_file = min(
                deviations,
                key=lambda x: x[0]
            )

            max_dev, max_file = max(
                deviations,
                key=lambda x: x[0]
            )

            abs_dev, abs_file = max(
                deviations,
                key=lambda x: abs(x[0])
            )

            value_int = int(value)

            print(
                f"      valore={value_int:>6d} | "
                f"agg={100.0 * agg_frac:7.3f}% | "
                f"min={min_dev:+7.3f} pp ({min_file}) | "
                f"max={max_dev:+7.3f} pp ({max_file}) | "
                f"max|dev|={abs_dev:+7.3f} pp ({abs_file})"
            )

def analyze_B(loaded):
    """
    Analizza truth label e parentHiggsParentsMask.

    Se USE_ANALYSIS_OBJECTS=True, tutte le statistiche relative
    a jet e tau sono calcolate sugli oggetti analysis-level.
    """

    section(
        "(B) DISTRIBUZIONI AGGREGATE DELLE TRUTH LABEL "
        "E CONFRONTO TRA FILE"
    )

    print(
        "\n  USE_ANALYSIS_OBJECTS = "
        f"{USE_ANALYSIS_OBJECTS}"
    )

    if USE_ANALYSIS_OBJECTS:
        print(
            "  -> jet: isAnalysisJet == True"
        )
        print(
            "  -> tau: isAnalysisTau == True"
        )
    else:
        print(
            "  -> usate tutte le collezioni ricostruite"
        )

    # --------------------------------------------------------------
    # Truth labels
    # --------------------------------------------------------------

    for branch in TRUTH_LABEL_BRANCHES:

        subsection(branch)

        analyze_distribution_branch(
            loaded,
            branch,
            top=TOP_VALUES,
        )

    # --------------------------------------------------------------
    # Bitmask
    # --------------------------------------------------------------

    for branch in MASK_BRANCHES:

        if branch == "tau_parentHiggsParentsMask":

            analyze_mask_branch(
                loaded,
                branch,
                additional_selection_branch=(
                    "tau_truth_IsHadronicTau"  # controllo deviazioni facendo slicing su tau adronici oppure no
                ),
            )

        else:

            analyze_mask_branch(
                loaded,
                branch,
            )


# ======================================================================
# CROSS-CHECK GENERICO LABEL vs HIGGS MASK
# ======================================================================

def analyze_label_vs_higgs_mask(
    loaded,
    label_value,
):
    """
    Cross-check generico tra:

      recojet_antikt4PFlow_HadronConeExclTruthLabelID == label_value
      recojet_antikt4PFlow_parentHiggsParentsMask

    Se USE_ANALYSIS_OBJECTS=True, vengono considerati solamente
    i jet con isAnalysisJet == True.

    Per i jet con il label scelto vengono riportati:

      GLOBALI
      -------
      - n. jet totali
      - n. jet con label == label_value
      - n. jet con mask != 0
      - n. jet con label AND mask != 0
      - n. jet con label AND mask == 0
      - P(mask != 0 | label)
      - P(label | mask != 0)

      DISTRIBUZIONE DELLA MASK TRA I JET CON IL LABEL
      ------------------------------------------------
      - mask = 0
      - mask = 1
      - mask = 2
      - mask = 3
      - eventuali altri valori

      DEVIAZIONI FILE-PER-FILE
      -------------------------
      Per tutte le quantità globali sopra.
      Inoltre, per ogni valore della mask:
        frazione tra i jet con label scelto.
    """

    label_branch = (
        "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
    )

    mask_branch = (
        "recojet_antikt4PFlow_parentHiggsParentsMask"
    )

    selection_branch = (
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS"
    )

    print()
    print("-" * 100)
    print(
        f"Cross-check jet: "
        f"HadronConeExclTruthLabelID == {label_value} "
        f"vs parentHiggsParentsMask"
    )
    print("-" * 100)

    if USE_ANALYSIS_OBJECTS:
        print(
            f"    SELEZIONE ATTIVA: {selection_branch}"
        )
    else:
        print(
            "    SELEZIONE ANALYSIS-LEVEL: DISATTIVATA"
        )

    # ------------------------------------------------------------------
    # Accumulatori aggregati
    # ------------------------------------------------------------------

    aggregate_total = 0
    aggregate_label = 0
    aggregate_mask_nonzero = 0
    aggregate_label_and_mask = 0
    aggregate_label_and_mask_zero = 0

    # Distribuzione della mask tra i jet con label == label_value
    aggregate_mask_counts = {}

    # Informazioni per singolo file
    per_file = []

    # ------------------------------------------------------------------
    # Loop sui file
    # ------------------------------------------------------------------

    for item in loaded:

        filename = item["file_name"]
        tree = item["tree"]
        n_entries = item["n_entries"]

        keys = set(tree.keys())

        if (
            label_branch not in keys
            or mask_branch not in keys
        ):
            print(
                f"[WARNING] {filename}: "
                "branch necessarie non trovate. "
                "File ignorato."
            )
            continue

        # --------------------------------------------------------------
        # Lettura
        # --------------------------------------------------------------

        labels = tree[label_branch].array(
            entry_stop=n_entries,
            library="ak",
        )

        masks = tree[mask_branch].array(
            entry_stop=n_entries,
            library="ak",
        )

        # --------------------------------------------------------------
        # Slicing analysis-level
        # --------------------------------------------------------------

        if USE_ANALYSIS_OBJECTS:

            analysis_selection = (
                get_analysis_selection(
                    tree,
                    selection_branch,
                    n_entries,
                )
            )

            labels = labels[analysis_selection]
            masks = masks[analysis_selection]

        # --------------------------------------------------------------
        # Flatten
        # --------------------------------------------------------------

        labels = flatten_to_numpy(labels)
        masks = flatten_to_numpy(masks)

        # --------------------------------------------------------------
        # Controllo dimensioni
        # --------------------------------------------------------------

        if len(labels) != len(masks):

            print(
                f"[WARNING] {filename}: "
                f"numero di label ({len(labels)}) "
                f"!= numero di mask ({len(masks)}). "
                "File ignorato."
            )

            continue

        # --------------------------------------------------------------
        # Quantità globali del file
        # --------------------------------------------------------------

        n_total = len(labels)

        is_label = labels == label_value
        is_mask_nonzero = masks != 0

        n_label = int(
            np.count_nonzero(is_label)
        )

        n_mask_nonzero = int(
            np.count_nonzero(is_mask_nonzero)
        )

        n_label_and_mask = int(
            np.count_nonzero(
                is_label & is_mask_nonzero
            )
        )

        n_label_and_mask_zero = int(
            np.count_nonzero(
                is_label & ~is_mask_nonzero
            )
        )

        frac_label = (
            n_label / n_total
            if n_total > 0
            else np.nan
        )

        frac_mask_nonzero = (
            n_mask_nonzero / n_total
            if n_total > 0
            else np.nan
        )

        frac_label_and_mask = (
            n_label_and_mask / n_total
            if n_total > 0
            else np.nan
        )

        frac_label_and_mask_zero = (
            n_label_and_mask_zero / n_total
            if n_total > 0
            else np.nan
        )

        p_mask_given_label = (
            n_label_and_mask / n_label
            if n_label > 0
            else np.nan
        )

        p_label_given_mask = (
            n_label_and_mask / n_mask_nonzero
            if n_mask_nonzero > 0
            else np.nan
        )

        # --------------------------------------------------------------
        # Distribuzione mask tra i jet con label scelto
        # --------------------------------------------------------------

        selected_masks = masks[is_label]

        file_mask_counts = {}

        if selected_masks.size > 0:

            values, counts = np.unique(
                selected_masks,
                return_counts=True,
            )

            file_mask_counts = {
                v.item() if hasattr(v, "item") else v: int(c)
                for v, c in zip(values, counts)
            }

        # --------------------------------------------------------------
        # Salvataggio per-file
        # --------------------------------------------------------------

        per_file.append(
            {
                "file_name": filename,
                "n_total": n_total,
                "n_label": n_label,
                "n_mask_nonzero": n_mask_nonzero,
                "n_label_and_mask": n_label_and_mask,
                "n_label_and_mask_zero": n_label_and_mask_zero,
                "frac_label": frac_label,
                "frac_mask_nonzero": frac_mask_nonzero,
                "frac_label_and_mask": frac_label_and_mask,
                "frac_label_and_mask_zero": frac_label_and_mask_zero,
                "p_mask_given_label": p_mask_given_label,
                "p_label_given_mask": p_label_given_mask,
                "mask_counts": file_mask_counts,
            }
        )

        # --------------------------------------------------------------
        # Aggiornamento aggregati
        # --------------------------------------------------------------

        aggregate_total += n_total
        aggregate_label += n_label
        aggregate_mask_nonzero += n_mask_nonzero
        aggregate_label_and_mask += n_label_and_mask
        aggregate_label_and_mask_zero += (
            n_label_and_mask_zero
        )

        for value, count in file_mask_counts.items():

            aggregate_mask_counts[value] = (
                aggregate_mask_counts.get(value, 0)
                + count
            )

    # ------------------------------------------------------------------
    # Controllo dati
    # ------------------------------------------------------------------

    if aggregate_total == 0:

        print(
            "  Nessun dato disponibile."
        )

        return

    # ------------------------------------------------------------------
    # Quantità aggregate
    # ------------------------------------------------------------------

    agg_frac_label = (
        aggregate_label
        / aggregate_total
    )

    agg_frac_mask_nonzero = (
        aggregate_mask_nonzero
        / aggregate_total
    )

    agg_frac_label_and_mask = (
        aggregate_label_and_mask
        / aggregate_total
    )

    agg_frac_label_and_mask_zero = (
        aggregate_label_and_mask_zero
        / aggregate_total
    )

    agg_p_mask_given_label = (
        aggregate_label_and_mask
        / aggregate_label
        if aggregate_label > 0
        else np.nan
    )

    agg_p_label_given_mask = (
        aggregate_label_and_mask
        / aggregate_mask_nonzero
        if aggregate_mask_nonzero > 0
        else np.nan
    )

    # ------------------------------------------------------------------
    # STATISTICA AGGREGATA
    # ------------------------------------------------------------------

    print()
    print("  STATISTICA AGGREGATA")

    print(
        f"      n. jet totali:                         "
        f"{aggregate_total:>10d}"
    )

    print(
        f"      n. jet con label=={label_value:<2d}:             "
        f"{aggregate_label:>10d}"
        f"  ({100.0 * agg_frac_label:6.3f}%)"
    )

    print(
        f"      n. jet con mask!=0:                   "
        f"{aggregate_mask_nonzero:>10d}"
        f"  ({100.0 * agg_frac_mask_nonzero:6.3f}%)"
    )

    print(
        f"      n. jet con label E mask!=0:           "
        f"{aggregate_label_and_mask:>10d}"
        f"  ({100.0 * agg_frac_label_and_mask:6.3f}%)"
    )

    print(
        f"      n. jet con label E mask==0:           "
        f"{aggregate_label_and_mask_zero:>10d}"
        f"  ({100.0 * agg_frac_label_and_mask_zero:6.3f}%)"
    )

    print()

    print(
        f"      P(mask!=0 | label=={label_value}):            "
        f"{agg_p_mask_given_label:.6f}"
        f"  ({100.0 * agg_p_mask_given_label:6.3f}%)"
    )

    print(
        f"      P(label=={label_value} | mask!=0):            "
        f"{agg_p_label_given_mask:.6f}"
        f"  ({100.0 * agg_p_label_given_mask:6.3f}%)"
    )

    # ------------------------------------------------------------------
    # DISTRIBUZIONE MASK TRA I JET CON IL LABEL
    # ------------------------------------------------------------------

    total_selected_for_mask = aggregate_label

    print()
    print(
        f"      Distribuzione di parentHiggsParentsMask "
        f"tra i jet con label=={label_value}:"
    )

    # Prima i valori 0,1,2,3 per mantenere lo stesso formato
    standard_mask_values = [0, 1, 2, 3]

    for mask_value in standard_mask_values:

        count = aggregate_mask_counts.get(
            mask_value,
            0,
        )

        fraction = (
            count / total_selected_for_mask
            if total_selected_for_mask > 0
            else np.nan
        )

        print(
            f"        mask={mask_value:<2d} "
            f"binario={mask_value:012b} "
            f"count={count:>10d} "
            f"({100.0 * fraction:6.3f}%)"
        )

    # Eventuali valori non 0,1,2,3
    extra_mask_values = sorted(
        value
        for value in aggregate_mask_counts
        if value not in standard_mask_values
    )

    for mask_value in extra_mask_values:

        count = aggregate_mask_counts[mask_value]

        fraction = (
            count / total_selected_for_mask
            if total_selected_for_mask > 0
            else np.nan
        )

        print(
            f"        mask={int(mask_value):<2d} "
            f"binario={int(mask_value):012b} "
            f"count={count:>10d} "
            f"({100.0 * fraction:6.3f}%)"
        )

    # ------------------------------------------------------------------
    # QUANTITÀ ESPLICITE DERIVATE DALLA DISTRIBUZIONE
    # ------------------------------------------------------------------

    n_mask_0 = aggregate_mask_counts.get(0, 0)
    n_mask_3 = aggregate_mask_counts.get(3, 0)

    print()

    print(
        f"      n. jet label=={label_value} con mask!=0: "
        f"{aggregate_label_and_mask:>12d} "
        f"({100.0 * agg_p_mask_given_label:6.3f}%)"
    )

    print(
        f"      n. jet label=={label_value} con mask==0: "
        f"{n_mask_0:>12d} "
        f"({100.0 * n_mask_0 / total_selected_for_mask:6.3f}%)"
    )

    print(
        f"      n. jet label=={label_value} con mask==3: "
        f"{n_mask_3:>12d} "
        f"({100.0 * n_mask_3 / total_selected_for_mask:6.3f}%)"
    )

    # ------------------------------------------------------------------
    # CHECK
    # ------------------------------------------------------------------

    print()
    print("      CHECK:")

    print(
        f"        P(mask!=0 | label=={label_value}) = "
        f"{agg_p_mask_given_label:.6f} "
        f"({100.0 * agg_p_mask_given_label:.3f}%)"
    )

    print(
        f"        P(label=={label_value} | mask!=0) = "
        f"{agg_p_label_given_mask:.6f} "
        f"({100.0 * agg_p_label_given_mask:.3f}%)"
    )

    # ------------------------------------------------------------------
    # DEVIAZIONI GLOBALI
    # ------------------------------------------------------------------

    print()
    print(
        "  DEVIAZIONI DEI SINGOLI FILE "
        "RISPETTO ALL'AGGREGATO"
    )

    metrics = [
        (
            f"Frazione jet con label=={label_value}:",
            agg_frac_label,
            "frac_label",
        ),
        (
            "Frazione jet con mask!=0:",
            agg_frac_mask_nonzero,
            "frac_mask_nonzero",
        ),
        (
            f"Frazione jet con label=={label_value} "
            "E mask!=0:",
            agg_frac_label_and_mask,
            "frac_label_and_mask",
        ),
        (
            f"Frazione jet con label=={label_value} "
            "E mask==0:",
            agg_frac_label_and_mask_zero,
            "frac_label_and_mask_zero",
        ),
        (
            f"P(mask!=0 | label=={label_value}):",
            agg_p_mask_given_label,
            "p_mask_given_label",
        ),
        (
            f"P(label=={label_value} | mask!=0):",
            agg_p_label_given_mask,
            "p_label_given_mask",
        ),
    ]

    for title, aggregate_value, key in metrics:

        print()
        print(f"    {title}")

        print_min_max_deviation(
            per_file,
            aggregate_value,
            key,
            unit="",
        )

    # ------------------------------------------------------------------
    # DEVIAZIONI DELLA DISTRIBUZIONE DELLA MASK
    # ------------------------------------------------------------------

    print()
    print(
        f"  DEVIAZIONI DELLA DISTRIBUZIONE DELLA MASK "
        f"TRA I JET CON LABEL=={label_value}"
    )

    all_mask_values = set(
        aggregate_mask_counts.keys()
    )

    for record in per_file:

        all_mask_values.update(
            record["mask_counts"].keys()
        )

    # Ordine 0,1,2,3, poi eventuali valori extra
    ordered_mask_values = [
        value
        for value in standard_mask_values
        if value in all_mask_values
    ]

    ordered_mask_values.extend(
        sorted(
            value
            for value in all_mask_values
            if value not in standard_mask_values
        )
    )

    for mask_value in ordered_mask_values:

        aggregate_count = (
            aggregate_mask_counts.get(
                mask_value,
                0,
            )
        )

        aggregate_fraction = (
            aggregate_count
            / total_selected_for_mask
            if total_selected_for_mask > 0
            else np.nan
        )

        deviations = []

        for record in per_file:

            n_label = record["n_label"]

            if n_label == 0:
                continue

            file_count = record["mask_counts"].get(
                mask_value,
                0,
            )

            file_fraction = (
                file_count
                / n_label
            )

            deviation = (
                file_fraction
                - aggregate_fraction
            )

            deviations.append(
                (
                    record["file_name"],
                    deviation,
                )
            )

        if not deviations:
            continue

        min_file, min_dev = min(
            deviations,
            key=lambda x: x[1]
        )

        max_file, max_dev = max(
            deviations,
            key=lambda x: x[1]
        )

        max_abs_file, max_abs_dev = max(
            deviations,
            key=lambda x: abs(x[1])
        )

        print()

        print(
            f"    mask={int(mask_value)} "
            f"(agg={100.0 * aggregate_fraction:.3f}%)"
        )

        print(
            f"      deviazione minima: "
            f"{min_dev:+.6f} "
            f"({100.0 * min_dev:+.3f} pp) "
            f"({min_file})"
        )

        print(
            f"      deviazione massima: "
            f"{max_dev:+.6f} "
            f"({100.0 * max_dev:+.3f} pp) "
            f"({max_file})"
        )

        print(
            f"      max |deviazione|: "
            f"{max_abs_dev:+.6f} "
            f"({100.0 * max_abs_dev:+.3f} pp) "
            f"({max_abs_file})"
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

    analyze_A(loaded)

    analyze_B(loaded)

    analyze_label_vs_higgs_mask(
        loaded,
        label_value=5,
    )

    analyze_label_vs_higgs_mask(
        loaded,
        label_value=15,
    )

    section("FINE DIAGNOSTICA")


if __name__ == "__main__":
    main()