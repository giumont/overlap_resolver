"""
Preliminary diagnostics of the truth information of the signal ntuples.

Role in the project: objectives 3.1 and 3.2 classify (jet, tau) pairs
using the truth labels of the signal sample (true b-jet:
``HadronConeExclTruthLabelID == 5``, true hadronic tau:
``tau_truth_IsHadronicTau``). This script validates those inputs on all
signal files (``params.SAMPLES["signal"]``) and checks that the files are
statistically homogeneous. Paths come from ``params.py`` and are resolved
relative to its folder, since this file lives in ``other_code/``.

Parts:

(A) Pre-filtering. For jets, taus, electrons and muons
    (`params.ANALYSIS_FLAG_CHECKS`), fraction of objects with the
    ``isAnalysis*`` flag set, to tell whether each collection is already
    filtered at analysis level. Ignores `params.DIAG_USE_ANALYSIS_OBJECTS`.

(B) Truth labels. Aggregate distribution of the categorical truth branches
    (`params.TRUTH_LABEL_BRANCHES`) and of the ``parentHiggsParentsMask``
    bitmasks (`params.PARENT_HIGGS_MASK_BRANCHES`), each with the maximum
    deviation of the single files from the aggregate. The tau mask is also
    studied restricted to true hadronic taus.

Cross-check. For each truth-label value in
    `params.DIAG_LABEL_CROSSCHECK_VALUES`, relation between the jet truth
    label and ``parentHiggsParentsMask`` (joint and conditional
    probabilities, mask distribution among the jets with that label), with
    file-by-file deviations.

If `params.DIAG_USE_ANALYSIS_OBJECTS` is True, jet and tau statistics use
only analysis-level objects.

Requirements: uproot, awkward, numpy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import awkward as ak
import numpy as np
import uproot

import params


def section(title):
    """
    Print a title between two separator lines (``=``).

    Parameters
    ----------
    title : str
        Text to print.

    Returns
    -------
    None
    """
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subsection(title):
    """
    Print a title between two separator lines (``-``).

    Parameters
    ----------
    title : str
        Text to print.

    Returns
    -------
    None
    """
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def flatten_to_numpy(arr):
    """
    Flatten an awkward array to 1D numpy, dropping None.

    Parameters
    ----------
    arr : ak.Array
        Array of any nesting.

    Returns
    -------
    numpy.ndarray
        1D array of the valid values.
    """
    flat = ak.flatten(arr, axis=None)
    flat = ak.drop_none(flat)
    return ak.to_numpy(flat)


def get_value_counts(arr):
    """
    Count the occurrences of each distinct value.

    Parameters
    ----------
    arr : ak.Array
        Array of any nesting.

    Returns
    -------
    counts : dict
        ``{value: count}``; empty if there are no valid values.
    total : int
        Number of valid values.
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
    Sum a list of ``{value: count}`` dictionaries.

    Parameters
    ----------
    dicts : list of dict
        Count dictionaries, e.g. one per file.

    Returns
    -------
    dict
        ``{value: total count}``.
    """
    out = {}

    for d in dicts:
        for key, count in d.items():
            out[key] = out.get(key, 0) + count

    return out


def sorted_counts(counts):
    """
    Sort a count dictionary by decreasing count.

    Parameters
    ----------
    counts : dict
        ``{value: count}``.

    Returns
    -------
    list of tuple
        ``(value, count)`` by decreasing count.
    """
    return sorted(counts.items(), key=lambda x: -x[1])


def print_distribution(counts, total, top=20, indent="      "):
    """
    Print the most frequent values of a distribution.

    Parameters
    ----------
    counts : dict
        ``{value: count}``.

    total : int
        Total number of values, used for the percentages.

    top : int, default=20
        Number of values printed; the number of remaining distinct values
        is reported.

    indent : str, default="      "
        Prefix of each printed line.

    Returns
    -------
    None
    """
    if total == 0:
        print(indent + "nessun valore disponibile")
        return

    for value, count in sorted_counts(counts)[:top]:
        frac = count / total
        print(f"{indent}{value!r:>12}  count={count:>10d}  ({100.0 * frac:6.2f}%)")

    if len(counts) > top:
        print(indent + f"... e altri {len(counts) - top} valori distinti")


def print_min_max_deviation(records, aggregate_value, value_key, unit=""):
    """
    Print how much the single files deviate from an aggregate value.

    Reports the minimum and maximum signed deviations
    ``record[value_key] - aggregate_value`` and the largest absolute one,
    each with the file name. Records with ``None`` value are ignored.

    Parameters
    ----------
    records : list of dict
        One dict per file, with keys ``file_name`` and `value_key`.

    aggregate_value : float
        Aggregate value over all files.

    value_key : str
        Key of the quantity to compare.

    unit : str, default=""
        Unit string appended to the printed deviations.

    Returns
    -------
    None
    """
    valid = [r for r in records if r.get(value_key) is not None]

    if not valid:
        print("      nessun file disponibile")
        return

    deviations = [(r[value_key] - aggregate_value, r["file_name"]) for r in valid]

    min_dev, min_file = min(deviations, key=lambda x: x[0])
    max_dev, max_file = max(deviations, key=lambda x: x[0])
    abs_dev, abs_file = max(deviations, key=lambda x: abs(x[0]))

    print(f"      deviazione minima: {min_dev:+.6f}{unit} ({min_file})")
    print(f"      deviazione massima: {max_dev:+.6f}{unit} ({max_file})")
    print(f"      max |deviazione|: {abs_dev:+.6f}{unit} ({abs_file})")


def get_analysis_selection(tree, selection_branch, n_entries):
    """
    Read an ``isAnalysis...`` branch as a boolean mask.

    Parameters
    ----------
    tree : uproot.TTree
        Tree containing the branch.

    selection_branch : str
        Name of the flag branch.

    n_entries : int
        Number of events to read.

    Returns
    -------
    ak.Array of bool
        True where the flag is non-zero; None and 0 map to False. The
        explicit comparison is needed because the branch may be int8/uint8.
    """
    selection = tree[selection_branch].array(entry_stop=n_entries, library="ak")
    selection = ak.fill_none(selection, 0)
    return selection != 0


def apply_object_selection(arr, tree, selection_branch, n_entries):
    """
    Apply the analysis-level filter to a collection, if enabled.

    Parameters
    ----------
    arr : ak.Array
        Jagged per-object array.

    tree : uproot.TTree
        Tree containing the selection branch.

    selection_branch : str
        Analysis flag branch of the collection.

    n_entries : int
        Number of events read.

    Returns
    -------
    ak.Array
        ``arr[selection]`` if `params.DIAG_USE_ANALYSIS_OBJECTS` is True
        (event by event, before flattening), otherwise `arr` unchanged.
    """
    if not params.DIAG_USE_ANALYSIS_OBJECTS:
        return arr

    selection = get_analysis_selection(tree, selection_branch, n_entries)

    return arr[selection]


def read_branch(tree, branch, n_entries, selection_branch=None):
    """
    Read a branch, optionally applying the analysis-level filter.

    Parameters
    ----------
    tree : uproot.TTree
        Tree containing the branch.

    branch : str
        Branch to read.

    n_entries : int
        Number of events to read.

    selection_branch : str or None, default=None
        Analysis flag used to filter the branch (see
        `apply_object_selection`); None means no filter.

    Returns
    -------
    ak.Array
    """
    arr = tree[branch].array(entry_stop=n_entries, library="ak")

    if selection_branch is not None:
        arr = apply_object_selection(arr, tree, selection_branch, n_entries)

    return arr


def file_path(index):
    """
    Path of the signal input file with a given index.

    Parameters
    ----------
    index : int
        File index, from 1 to `params.N_FILES`.

    Returns
    -------
    pathlib.Path
        ``PROJECT_DIR / root_dir / <prefix><index:02d><suffix>`` of the
        signal sample in `params.SAMPLES`.
    """
    signal = params.SAMPLES["signal"]
    return (
        params.PROJECT_DIR
        / signal["root_dir"]
        / f"{signal['file_prefix']}{index:02d}{params.FILE_SUFFIX}"
    )


def load_files():
    """
    Open the expected signal files and their tree.

    Missing or invalid files (including files without the tree
    `params.TREE_NAME`) are reported and skipped. The number of events
    used per file is capped at `params.N_ENTRIES_CAP`.

    Returns
    -------
    list of dict
        One dict per loaded file with keys ``index``, ``path``,
        ``file_name``, ``tree`` and ``n_entries`` (events actually used).
    """
    loaded = []

    section("FILE INPUT")

    for i in range(1, params.N_FILES + 1):
        path = file_path(i)

        if not path.exists():
            print(f"[MANCANTE] {path}")
            continue

        try:
            root_file = uproot.open(path)

            if params.TREE_NAME not in root_file:
                print(f"[ERRORE] {path}: tree '{params.TREE_NAME}' non trovato")
                continue

            tree = root_file[params.TREE_NAME]

            n_entries = tree.num_entries

            if params.N_ENTRIES_CAP is not None:
                n_entries_used = min(n_entries, params.N_ENTRIES_CAP)
            else:
                n_entries_used = n_entries

            print(f"[OK] {path}   entries={n_entries}   usati={n_entries_used}")

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

    print(f"\nFile caricati correttamente: {len(loaded)}/{params.N_FILES}")

    return loaded


def analyze_A(loaded):
    """
    Part (A): check whether the collections are pre-filtered.

    For each entry of `params.ANALYSIS_FLAG_CHECKS` prints, aggregated
    over all files, the number of objects, the fraction with the flag set,
    the mean number of objects per event, an interpretation, and the
    deviation of the single files from the aggregate. Warns if the flag
    branch and the reference kinematic branch have different numbers of
    objects. Does not use `params.DIAG_USE_ANALYSIS_OBJECTS`.

    Parameters
    ----------
    loaded : list of dict
        Output of `load_files`.

    Returns
    -------
    None
    """
    section("(A) PRE-FILTRAGGIO DELLE COLLEZIONI")

    for flag_branch, label, ref_branch in params.ANALYSIS_FLAG_CHECKS:

        subsection(f"{label}: {flag_branch}")

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
                print(f"  {file_name}: {flag_branch} NON TROVATO")
                continue

            flag = tree[flag_branch].array(entry_stop=n_entries, library="ak")

            flag = ak.fill_none(flag, False)

            flat_flag = flatten_to_numpy(flag)

            n_total = int(flat_flag.size)
            n_true = int(np.count_nonzero(flat_flag.astype(bool)))

            n_events = n_entries

            frac_true = n_true / n_total if n_total > 0 else np.nan
            avg_total = n_total / n_events if n_events > 0 else np.nan
            avg_true = n_true / n_events if n_events > 0 else np.nan

            if ref_branch not in keys:
                print(f"  {file_name}: {ref_branch} NON TROVATO")
                continue

            ref = tree[ref_branch].array(entry_stop=n_entries, library="ak")

            n_ref = int(ak.sum(ak.num(ref, axis=1)))

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

        aggregate_frac_true = aggregate_true_objects / aggregate_total_objects
        aggregate_avg_total = aggregate_total_objects / aggregate_events
        aggregate_avg_true = aggregate_true_objects / aggregate_events

        print("\n  STATISTICA AGGREGATA SU TUTTI I FILE")
        print(f"      eventi totali usati:       {aggregate_events:>10d}")
        print(f"      oggetti totali:            {aggregate_total_objects:>10d}")
        print(f"      oggetti con flag=True:     {aggregate_true_objects:>10d}")
        print(
            f"      frazione True:             "
            f"{aggregate_frac_true:.6f} ({100.0 * aggregate_frac_true:.3f}%)"
        )
        print(f"      media oggetti/evento:      {aggregate_avg_total:.6f}")
        print(f"      media True/evento:         {aggregate_avg_true:.6f}")

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

        print("\n  DEVIAZIONE DEI SINGOLI FILE RISPETTO ALL'AGGREGATO")

        print("\n    Frazione True:")
        print_min_max_deviation(per_file, aggregate_frac_true, "frac_true", unit="")

        print("\n    Media oggetti/evento:")
        print_min_max_deviation(per_file, aggregate_avg_total, "avg_total", unit="")

        print("\n    Media oggetti con flag=True/evento:")
        print_min_max_deviation(per_file, aggregate_avg_true, "avg_true", unit="")

        pp_records = [{**r, "frac_true": 100.0 * r["frac_true"]} for r in per_file]

        print("\n    Frazione True in punti percentuali:")
        print_min_max_deviation(
            pp_records, 100.0 * aggregate_frac_true, "frac_true", unit=" pp"
        )


def get_selection_branch_for_branch(branch):
    """
    Analysis flag associated with a jet or tau branch.

    Parameters
    ----------
    branch : str
        Branch name.

    Returns
    -------
    str or None
        `params.JET_IS_ANALYSIS_BRANCH` for branches starting with
        `params.JET_BRANCH_PREFIX`, `params.TAU_IS_ANALYSIS_BRANCH` for
        those starting with `params.TAU_BRANCH_PREFIX`, None otherwise.
    """
    if branch.startswith(params.JET_BRANCH_PREFIX):
        return params.JET_IS_ANALYSIS_BRANCH

    if branch.startswith(params.TAU_BRANCH_PREFIX):
        return params.TAU_IS_ANALYSIS_BRANCH

    return None


def analyze_distribution_branch(loaded, branch, top=20):
    """
    Aggregate distribution of a categorical branch and file deviations.

    Prints the aggregate distribution over all files and, for each of the
    most frequent values, the minimum, maximum and largest absolute
    deviation (in percentage points) of the single-file frequency from the
    aggregate. Jet and tau branches are filtered at analysis level if
    `params.DIAG_USE_ANALYSIS_OBJECTS` is True.

    Parameters
    ----------
    loaded : list of dict
        Output of `load_files`.

    branch : str
        Categorical branch to analyse; files without it are skipped.

    top : int, default=20
        Number of values reported.

    Returns
    -------
    None
    """
    per_file_counts = []
    aggregate_counts_list = []

    aggregate_total = 0

    selection_branch = get_selection_branch_for_branch(branch)

    if params.DIAG_USE_ANALYSIS_OBJECTS and selection_branch is not None:
        print(f"\n    SELEZIONE ATTIVA: {selection_branch}")
    else:
        print("\n    SELEZIONE ANALYSIS-LEVEL: DISATTIVATA")

    for item in loaded:

        tree = item["tree"]
        file_name = item["file_name"]
        n_entries = item["n_entries"]

        if branch not in set(tree.keys()):
            continue

        arr = read_branch(tree, branch, n_entries, selection_branch=selection_branch)

        counts, total = get_value_counts(arr)

        per_file_counts.append(
            {"file_name": file_name, "counts": counts, "total": total}
        )

        aggregate_counts_list.append(counts)
        aggregate_total += total

    if aggregate_total == 0:
        print("      nessun valore disponibile")
        return

    aggregate_counts = merge_count_dicts(aggregate_counts_list)

    print("\n    Distribuzione AGGREGATA:")

    print_distribution(aggregate_counts, aggregate_total, top=top)

    print(
        "\n    DEVIAZIONI PER VALORE "
        "(frequenza file - frequenza aggregata, "
        "in punti percentuali)"
    )

    values_to_report = [value for value, _ in sorted_counts(aggregate_counts)[:top]]

    all_values = set(aggregate_counts.keys())

    for record in per_file_counts:
        all_values.update(record["counts"].keys())

    extra_values = [value for value in all_values if value not in values_to_report]

    extra_values = sorted(
        extra_values,
        key=lambda v: -aggregate_counts.get(v, 0),
    )[: max(0, top - len(values_to_report))]

    values_to_report.extend(extra_values)

    for value in values_to_report:

        agg_frac = aggregate_counts.get(value, 0) / aggregate_total

        records = []

        for record in per_file_counts:

            file_frac = (
                record["counts"].get(value, 0) / record["total"]
                if record["total"] > 0
                else np.nan
            )

            records.append(
                {"file_name": record["file_name"], "value_frac": 100.0 * file_frac}
            )

        if not records:
            continue

        valid = [r for r in records if not np.isnan(r["value_frac"])]

        if not valid:
            continue

        deviations = [
            (r["value_frac"] - 100.0 * agg_frac, r["file_name"]) for r in valid
        ]

        min_dev, min_file = min(deviations, key=lambda x: x[0])
        max_dev, max_file = max(deviations, key=lambda x: x[0])
        abs_dev, abs_file = max(deviations, key=lambda x: abs(x[0]))

        print(
            f"      valore={value!r:>12} | "
            f"agg={100.0 * agg_frac:7.3f}% | "
            f"min={min_dev:+7.3f} pp ({min_file}) | "
            f"max={max_dev:+7.3f} pp ({max_file}) | "
            f"max|dev|={abs_dev:+7.3f} pp ({abs_file})"
        )


def analyze_mask_branch(loaded, branch, additional_selection_branch=None):
    """
    Analyse a ``parentHiggsParentsMask`` branch.

    Prints, aggregated over all files, the most frequent mask values with
    their binary representation (`params.PARENT_HIGGS_MASK_BITS` bits),
    then the deviation in percentage points of the single files for each
    value. Jet/tau masks are filtered at analysis level if
    `params.DIAG_USE_ANALYSIS_OBJECTS` is True. If
    `additional_selection_branch` is given, the analysis is repeated a
    second time with that extra mask applied, to check the effect of the
    extra selection.

    Parameters
    ----------
    loaded : list of dict
        Output of `load_files`.

    branch : str
        Mask branch to analyse.

    additional_selection_branch : str or None, default=None
        Extra flag branch (e.g. `params.TAU_TRUTH_MATCH_BRANCH`) used to
        select objects in the second case; values are converted to bool.
        Files lacking it are ignored for that case.

    Returns
    -------
    None
    """
    subsection(branch)

    selection_branch = get_selection_branch_for_branch(branch)
    top_mask = params.DIAG_TOP_MASK_VALUES
    n_bits = params.PARENT_HIGGS_MASK_BITS

    selection_cases = [{"extra_branch": None, "label": selection_branch}]

    if additional_selection_branch is not None:
        selection_cases.append(
            {
                "extra_branch": additional_selection_branch,
                "label": f"{selection_branch} + {additional_selection_branch}",
            }
        )

    for case in selection_cases:

        extra_selection_branch = case["extra_branch"]
        selection_label = case["label"]

        print()
        print(f"    SELEZIONE ATTIVA: {selection_label}")

        per_file_counts = []
        aggregate_counts_list = []

        aggregate_total = 0

        for item in loaded:

            tree = item["tree"]
            file_name = item["file_name"]
            n_entries = item["n_entries"]

            keys = set(tree.keys())

            if branch not in keys:
                print(f"      {file_name}: {branch} NON TROVATO")
                continue

            arr = tree[branch].array(entry_stop=n_entries, library="ak")

            if params.DIAG_USE_ANALYSIS_OBJECTS and selection_branch is not None:
                analysis_selection = get_analysis_selection(
                    tree, selection_branch, n_entries
                )
                arr = arr[analysis_selection]

            if extra_selection_branch is not None:

                if extra_selection_branch not in keys:
                    print(
                        f"      {file_name}: "
                        f"{extra_selection_branch} NON TROVATO "
                        f"-> file ignorato per questo caso"
                    )
                    continue

                extra_selection = tree[extra_selection_branch].array(
                    entry_stop=n_entries, library="ak"
                )

                extra_selection = ak.fill_none(extra_selection, 0)
                extra_selection = extra_selection != 0

                arr = arr[extra_selection]

            counts, total = get_value_counts(arr)

            per_file_counts.append(
                {"file_name": file_name, "counts": counts, "total": total}
            )

            aggregate_counts_list.append(counts)
            aggregate_total += total

        if aggregate_total == 0:
            print("      nessun valore disponibile")
            continue

        aggregate_counts = merge_count_dicts(aggregate_counts_list)

        print("\n    Valori AGGREGATI e bit attivi (valore | binario | count | frazione)")

        for value, count in sorted_counts(aggregate_counts)[:top_mask]:

            value_int = int(value)
            frac = count / aggregate_total

            print(
                f"      valore={value_int:>6d}  "
                f"binario={value_int:0{n_bits}b}  "
                f"count={count:>10d}  "
                f"({100.0 * frac:6.2f}%)"
            )

        if len(aggregate_counts) > top_mask:
            print(
                f"      ... e altri {len(aggregate_counts) - top_mask} "
                "valori distinti"
            )

        print("\n    DEVIAZIONI PER VALORE (in punti percentuali)")

        values_to_report = [
            value for value, _ in sorted_counts(aggregate_counts)[:top_mask]
        ]

        for value in values_to_report:

            agg_frac = aggregate_counts[value] / aggregate_total

            deviations = []

            for record in per_file_counts:

                if record["total"] == 0:
                    continue

                file_frac = record["counts"].get(value, 0) / record["total"]

                delta_pp = 100.0 * file_frac - 100.0 * agg_frac

                deviations.append((delta_pp, record["file_name"]))

            if not deviations:
                continue

            min_dev, min_file = min(deviations, key=lambda x: x[0])
            max_dev, max_file = max(deviations, key=lambda x: x[0])
            abs_dev, abs_file = max(deviations, key=lambda x: abs(x[0]))

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
    Part (B): truth-label and Higgs-mask distributions.

    Runs `analyze_distribution_branch` on every branch in
    `params.TRUTH_LABEL_BRANCHES` and `analyze_mask_branch` on every
    branch in `params.PARENT_HIGGS_MASK_BRANCHES`. For the tau mask the
    analysis is repeated restricted to true hadronic taus
    (`params.TAU_TRUTH_MATCH_BRANCH`).

    Parameters
    ----------
    loaded : list of dict
        Output of `load_files`.

    Returns
    -------
    None
    """
    section("(B) DISTRIBUZIONI AGGREGATE DELLE TRUTH LABEL E CONFRONTO TRA FILE")

    print(f"\n  USE_ANALYSIS_OBJECTS = {params.DIAG_USE_ANALYSIS_OBJECTS}")

    if params.DIAG_USE_ANALYSIS_OBJECTS:
        print("  -> jet: isAnalysisJet == True")
        print("  -> tau: isAnalysisTau == True")
    else:
        print("  -> usate tutte le collezioni ricostruite")

    for branch in params.TRUTH_LABEL_BRANCHES:

        subsection(branch)

        analyze_distribution_branch(loaded, branch, top=params.DIAG_TOP_VALUES)

    for branch in params.PARENT_HIGGS_MASK_BRANCHES:

        if branch == params.TAU_PARENT_HIGGS_MASK_BRANCH:

            analyze_mask_branch(
                loaded,
                branch,
                additional_selection_branch=params.TAU_TRUTH_MATCH_BRANCH,
            )

        else:

            analyze_mask_branch(loaded, branch)


def analyze_label_vs_higgs_mask(loaded, label_value):
    """
    Cross-check between the jet truth label and ``parentHiggsParentsMask``.

    Uses `params.JET_TRUTH_LABEL_BRANCH` and
    `params.JET_PARENT_HIGGS_MASK_BRANCH`, restricted to analysis-level
    jets if `params.DIAG_USE_ANALYSIS_OBJECTS` is True. Files lacking a
    branch, or with a different number of labels and masks, are skipped.

    Aggregated over all files, prints the number of jets with the label,
    with mask != 0, with both, with label and mask == 0,
    ``P(mask != 0 | label)`` and ``P(label | mask != 0)``, the mask
    distribution among the jets with the label (values
    `params.PARENT_HIGGS_MASK_STANDARD_VALUES` first), and the deviation
    of each single file from the aggregate for every quantity and for the
    mask distribution.

    Parameters
    ----------
    loaded : list of dict
        Output of `load_files`.

    label_value : int
        Value of the jet truth label to cross-check (e.g. 5 for b-jets,
        15 for tau-matched jets).

    Returns
    -------
    None
    """
    label_branch = params.JET_TRUTH_LABEL_BRANCH
    mask_branch = params.JET_PARENT_HIGGS_MASK_BRANCH
    selection_branch = params.JET_IS_ANALYSIS_BRANCH
    n_bits = params.PARENT_HIGGS_MASK_BITS
    standard_mask_values = params.PARENT_HIGGS_MASK_STANDARD_VALUES

    print()
    print("-" * 100)
    print(f"Cross-check jet: HadronConeExclTruthLabelID == {label_value} vs parentHiggsParentsMask")
    print("-" * 100)

    if params.DIAG_USE_ANALYSIS_OBJECTS:
        print(f"    SELEZIONE ATTIVA: {selection_branch}")
    else:
        print("    SELEZIONE ANALYSIS-LEVEL: DISATTIVATA")

    aggregate_total = 0
    aggregate_label = 0
    aggregate_mask_nonzero = 0
    aggregate_label_and_mask = 0
    aggregate_label_and_mask_zero = 0

    aggregate_mask_counts = {}

    per_file = []

    for item in loaded:

        filename = item["file_name"]
        tree = item["tree"]
        n_entries = item["n_entries"]

        keys = set(tree.keys())

        if label_branch not in keys or mask_branch not in keys:
            print(f"[WARNING] {filename}: branch necessarie non trovate. File ignorato.")
            continue

        labels = tree[label_branch].array(entry_stop=n_entries, library="ak")
        masks = tree[mask_branch].array(entry_stop=n_entries, library="ak")

        if params.DIAG_USE_ANALYSIS_OBJECTS:
            analysis_selection = get_analysis_selection(tree, selection_branch, n_entries)
            labels = labels[analysis_selection]
            masks = masks[analysis_selection]

        labels = flatten_to_numpy(labels)
        masks = flatten_to_numpy(masks)

        if len(labels) != len(masks):
            print(
                f"[WARNING] {filename}: "
                f"numero di label ({len(labels)}) "
                f"!= numero di mask ({len(masks)}). "
                "File ignorato."
            )
            continue

        n_total = len(labels)

        is_label = labels == label_value
        is_mask_nonzero = masks != 0

        n_label = int(np.count_nonzero(is_label))
        n_mask_nonzero = int(np.count_nonzero(is_mask_nonzero))
        n_label_and_mask = int(np.count_nonzero(is_label & is_mask_nonzero))
        n_label_and_mask_zero = int(np.count_nonzero(is_label & ~is_mask_nonzero))

        frac_label = n_label / n_total if n_total > 0 else np.nan
        frac_mask_nonzero = n_mask_nonzero / n_total if n_total > 0 else np.nan
        frac_label_and_mask = n_label_and_mask / n_total if n_total > 0 else np.nan
        frac_label_and_mask_zero = n_label_and_mask_zero / n_total if n_total > 0 else np.nan

        p_mask_given_label = n_label_and_mask / n_label if n_label > 0 else np.nan
        p_label_given_mask = n_label_and_mask / n_mask_nonzero if n_mask_nonzero > 0 else np.nan

        selected_masks = masks[is_label]

        file_mask_counts = {}

        if selected_masks.size > 0:

            values, counts = np.unique(selected_masks, return_counts=True)

            file_mask_counts = {
                v.item() if hasattr(v, "item") else v: int(c)
                for v, c in zip(values, counts)
            }

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

        aggregate_total += n_total
        aggregate_label += n_label
        aggregate_mask_nonzero += n_mask_nonzero
        aggregate_label_and_mask += n_label_and_mask
        aggregate_label_and_mask_zero += n_label_and_mask_zero

        for value, count in file_mask_counts.items():
            aggregate_mask_counts[value] = aggregate_mask_counts.get(value, 0) + count

    if aggregate_total == 0:
        print("  Nessun dato disponibile.")
        return

    agg_frac_label = aggregate_label / aggregate_total
    agg_frac_mask_nonzero = aggregate_mask_nonzero / aggregate_total
    agg_frac_label_and_mask = aggregate_label_and_mask / aggregate_total
    agg_frac_label_and_mask_zero = aggregate_label_and_mask_zero / aggregate_total

    agg_p_mask_given_label = (
        aggregate_label_and_mask / aggregate_label if aggregate_label > 0 else np.nan
    )

    agg_p_label_given_mask = (
        aggregate_label_and_mask / aggregate_mask_nonzero
        if aggregate_mask_nonzero > 0
        else np.nan
    )

    print()
    print("  STATISTICA AGGREGATA")

    print(f"      n. jet totali:                         {aggregate_total:>10d}")
    print(
        f"      n. jet con label=={label_value:<2d}:             "
        f"{aggregate_label:>10d}  ({100.0 * agg_frac_label:6.3f}%)"
    )
    print(
        f"      n. jet con mask!=0:                   "
        f"{aggregate_mask_nonzero:>10d}  ({100.0 * agg_frac_mask_nonzero:6.3f}%)"
    )
    print(
        f"      n. jet con label E mask!=0:           "
        f"{aggregate_label_and_mask:>10d}  ({100.0 * agg_frac_label_and_mask:6.3f}%)"
    )
    print(
        f"      n. jet con label E mask==0:           "
        f"{aggregate_label_and_mask_zero:>10d}  ({100.0 * agg_frac_label_and_mask_zero:6.3f}%)"
    )

    print()

    print(
        f"      P(mask!=0 | label=={label_value}):            "
        f"{agg_p_mask_given_label:.6f}  ({100.0 * agg_p_mask_given_label:6.3f}%)"
    )
    print(
        f"      P(label=={label_value} | mask!=0):            "
        f"{agg_p_label_given_mask:.6f}  ({100.0 * agg_p_label_given_mask:6.3f}%)"
    )

    total_selected_for_mask = aggregate_label

    print()
    print(
        f"      Distribuzione di parentHiggsParentsMask "
        f"tra i jet con label=={label_value}:"
    )

    for mask_value in standard_mask_values:

        count = aggregate_mask_counts.get(mask_value, 0)

        fraction = (
            count / total_selected_for_mask
            if total_selected_for_mask > 0
            else np.nan
        )

        print(
            f"        mask={mask_value:<2d} "
            f"binario={mask_value:0{n_bits}b} "
            f"count={count:>10d} "
            f"({100.0 * fraction:6.3f}%)"
        )

    extra_mask_values = sorted(
        value for value in aggregate_mask_counts if value not in standard_mask_values
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
            f"binario={int(mask_value):0{n_bits}b} "
            f"count={count:>10d} "
            f"({100.0 * fraction:6.3f}%)"
        )

    n_mask_0 = aggregate_mask_counts.get(0, 0)
    n_mask_3 = aggregate_mask_counts.get(3, 0)

    print()

    print(
        f"      n. jet label=={label_value} con mask!=0: "
        f"{aggregate_label_and_mask:>12d} ({100.0 * agg_p_mask_given_label:6.3f}%)"
    )
    print(
        f"      n. jet label=={label_value} con mask==0: "
        f"{n_mask_0:>12d} ({100.0 * n_mask_0 / total_selected_for_mask:6.3f}%)"
    )
    print(
        f"      n. jet label=={label_value} con mask==3: "
        f"{n_mask_3:>12d} ({100.0 * n_mask_3 / total_selected_for_mask:6.3f}%)"
    )

    print()
    print("      CHECK:")

    print(
        f"        P(mask!=0 | label=={label_value}) = "
        f"{agg_p_mask_given_label:.6f} ({100.0 * agg_p_mask_given_label:.3f}%)"
    )
    print(
        f"        P(label=={label_value} | mask!=0) = "
        f"{agg_p_label_given_mask:.6f} ({100.0 * agg_p_label_given_mask:.3f}%)"
    )

    print()
    print("  DEVIAZIONI DEI SINGOLI FILE RISPETTO ALL'AGGREGATO")

    metrics = [
        (f"Frazione jet con label=={label_value}:", agg_frac_label, "frac_label"),
        ("Frazione jet con mask!=0:", agg_frac_mask_nonzero, "frac_mask_nonzero"),
        (
            f"Frazione jet con label=={label_value} E mask!=0:",
            agg_frac_label_and_mask,
            "frac_label_and_mask",
        ),
        (
            f"Frazione jet con label=={label_value} E mask==0:",
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

        print_min_max_deviation(per_file, aggregate_value, key, unit="")

    print()
    print(
        f"  DEVIAZIONI DELLA DISTRIBUZIONE DELLA MASK "
        f"TRA I JET CON LABEL=={label_value}"
    )

    all_mask_values = set(aggregate_mask_counts.keys())

    for record in per_file:
        all_mask_values.update(record["mask_counts"].keys())

    ordered_mask_values = [
        value for value in standard_mask_values if value in all_mask_values
    ]

    ordered_mask_values.extend(
        sorted(value for value in all_mask_values if value not in standard_mask_values)
    )

    for mask_value in ordered_mask_values:

        aggregate_count = aggregate_mask_counts.get(mask_value, 0)

        aggregate_fraction = (
            aggregate_count / total_selected_for_mask
            if total_selected_for_mask > 0
            else np.nan
        )

        deviations = []

        for record in per_file:

            n_label = record["n_label"]

            if n_label == 0:
                continue

            file_count = record["mask_counts"].get(mask_value, 0)

            file_fraction = file_count / n_label

            deviation = file_fraction - aggregate_fraction

            deviations.append((record["file_name"], deviation))

        if not deviations:
            continue

        min_file, min_dev = min(deviations, key=lambda x: x[1])
        max_file, max_dev = max(deviations, key=lambda x: x[1])
        max_abs_file, max_abs_dev = max(deviations, key=lambda x: abs(x[1]))

        print()

        print(f"    mask={int(mask_value)} (agg={100.0 * aggregate_fraction:.3f}%)")

        print(
            f"      deviazione minima: {min_dev:+.6f} "
            f"({100.0 * min_dev:+.3f} pp) ({min_file})"
        )
        print(
            f"      deviazione massima: {max_dev:+.6f} "
            f"({100.0 * max_dev:+.3f} pp) ({max_file})"
        )
        print(
            f"      max |deviazione|: {max_abs_dev:+.6f} "
            f"({100.0 * max_abs_dev:+.3f} pp) ({max_abs_file})"
        )


def main():
    """
    Run parts (A), (B) and the label-vs-Higgs-mask cross-checks.

    Loads the signal files, then runs `analyze_A`, `analyze_B` and
    `analyze_label_vs_higgs_mask` for every value in
    `params.DIAG_LABEL_CROSSCHECK_VALUES`.

    Returns
    -------
    None
    """
    loaded = load_files()

    if not loaded:
        print("\nNessun file disponibile. Controllare ROOT_DIR e i nomi dei file.")
        return

    analyze_A(loaded)

    analyze_B(loaded)

    for label_value in params.DIAG_LABEL_CROSSCHECK_VALUES:
        analyze_label_vs_higgs_mask(loaded, label_value=label_value)

    section("FINE DIAGNOSTICA")


if __name__ == "__main__":
    main()