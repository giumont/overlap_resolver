"""
Consistency between the Higgs genealogy of tau-labelled jets and true taus.

Role in the project: objectives 3.1 and 3.2 need a reliable truth
definition for the (jet, tau) pairs of the signal sample. The jets with
``HadronConeExclTruthLabelID == 15`` are jets matched to a hadronic tau,
so they overlap with the reco tau by construction. This diagnostic checks
whether their truth genealogy (``parentHiggsParentsMask``) agrees with
that of the true hadronic taus of the same event, and whether the
disagreement can be explained geometrically. It runs on the signal sample
only (``params.SAMPLES["signal"]``), where the Higgs genealogy exists, and
aggregates all files.

STEP 0
    Within an event, do all true hadronic taus with mask != 0 share the
    same mask? Events with at least one such tau and a uniform mask are
    "valid" and provide the reference mask of the event.

STEP 1
    Subset: ``label == 15`` AND ``mask != 0``, restricted to valid events.
    Each jet is compared with the reference mask of its event: COHERENT
    (same mask) or INCOHERENT (different mask). Jets with label 15 and
    mask == 0 are excluded and only counted (baseline).

STEP 2
    On exactly the same subset as STEP 1, the minimum DeltaR to the true
    hadronic taus of the event is computed and summarised separately for
    coherent and incoherent jets, to tell the geometric component of the
    incoherence from the genealogical one.

Paths come from ``params.py`` and are resolved relative to its folder,
since this file lives in ``other_code/``.

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
    Print a title between two separator lines.

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


def file_path(index):
    """
    Path of the signal input file with a given index.

    Parameters
    ----------
    index : int
        File index, from 1 to `params.N_FILES`. Index 1 corresponds to
        ``<prefix>01<suffix>``.

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
        ``file_name``, ``root_file`` (uproot handle), ``tree`` and
        ``n_entries`` (events actually used).
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
                    "root_file": root_file,
                    "tree": tree,
                    "n_entries": n_entries_used,
                }
            )

        except Exception as exc:
            print(f"[ERRORE] {path}: {exc}")

    print(f"\nFile caricati correttamente: {len(loaded)}/{params.N_FILES}")

    return loaded


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
        True where the flag is non-zero; None and 0 map to False.
    """
    selection = tree[selection_branch].array(entry_stop=n_entries, library="ak")
    selection = ak.fill_none(selection, 0)
    return selection != 0


def clean_bool_branch(arr):
    """
    Convert a bool/int8/uint8 branch to a boolean awkward array.

    Parameters
    ----------
    arr : ak.Array
        Flag values, possibly with None.

    Returns
    -------
    ak.Array of bool
        True where the value is non-zero; None maps to False.
    """
    arr = ak.fill_none(arr, 0)
    return arr != 0


def summarize(values, label):
    """
    Print n, mean, median, p10 and p90 of an array.

    None and non-finite values are removed before computing the
    statistics.

    Parameters
    ----------
    values : ak.Array or numpy.ndarray
        Values to summarise (any nesting, flattened).

    label : str
        Label printed before the statistics.

    Returns
    -------
    None
    """
    flat = ak.flatten(values, axis=None)
    flat = ak.drop_none(flat)
    flat = ak.to_numpy(flat)

    if flat.size == 0:
        print(f"   {label}: nessun valore disponibile")
        return

    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        print(f"   {label}: nessun valore finito disponibile")
        return

    print(
        f"   {label}: "
        f"n={flat.size:>10d}  "
        f"mean={np.mean(flat):.4f}  "
        f"median={np.median(flat):.4f}  "
        f"p10={np.percentile(flat, 10):.4f}  "
        f"p90={np.percentile(flat, 90):.4f}"
    )


def delta_phi(phi1, phi2):
    """
    Azimuthal difference wrapped to [-pi, pi).

    Parameters
    ----------
    phi1, phi2 : array-like
        Azimuthal angles in radians.

    Returns
    -------
    array-like
        ``phi1 - phi2`` wrapped to [-pi, pi).
    """
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    """
    Angular distance DeltaR = sqrt(deta^2 + dphi^2).

    Parameters
    ----------
    eta1, phi1, eta2, phi2 : array-like
        Pseudorapidity and azimuth of the two objects.

    Returns
    -------
    array-like
        DeltaR, with the same structure as the inputs.
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def analyze_label15_higgs_consistency(loaded):
    """
    Run STEP 0, STEP 1 and STEP 2 (see module docstring) on all files.

    Per-file counts are printed while looping; the aggregated results
    (events valid for STEP 1/2, coherent/incoherent jets, DeltaR summaries
    and internal consistency checks) are printed at the end. Files lacking
    any required branch are skipped.

    Parameters
    ----------
    loaded : list of dict
        Output of `load_files`.

    Returns
    -------
    None
    """
    branches = [
        params.TAU_ETA_BRANCH,
        params.TAU_PHI_BRANCH,
        params.TAU_TRUTH_MATCH_BRANCH,
        params.TAU_PARENT_HIGGS_MASK_BRANCH,
        params.TAU_IS_ANALYSIS_BRANCH,
        params.JET_ETA_BRANCH,
        params.JET_PHI_BRANCH,
        params.JET_TRUTH_LABEL_BRANCH,
        params.JET_PARENT_HIGGS_MASK_BRANCH,
        params.JET_IS_ANALYSIS_BRANCH,
    ]
    label_tau = params.JET_TRUTH_LABEL_TAU_VALUE

    aggregate_events = 0
    aggregate_events_with_ref = 0
    aggregate_events_inconsistent_tau = 0
    aggregate_events_valid = 0

    aggregate_jet15_total = 0
    aggregate_jet15_mask0 = 0
    aggregate_jet15_nonzero_total = 0
    aggregate_compared = 0
    aggregate_consistent = 0
    aggregate_inconsistent = 0

    dr_consistent_parts = []
    dr_inconsistent_parts = []

    for item in loaded:
        filename = item["file_name"]
        tree = item["tree"]
        n_entries = item["n_entries"]

        print()
        print("-" * 100)
        print(f"Analisi file: {filename}")
        print("-" * 100)

        keys = set(tree.keys())
        missing = [branch for branch in branches if branch not in keys]

        if missing:
            print(f"[WARNING] {filename}: branch mancanti:")
            for branch in missing:
                print(f"    - {branch}")
            print("    File ignorato.")
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        n_events = len(a[params.TAU_ETA_BRANCH])

        tau_sel = get_analysis_selection(tree, params.TAU_IS_ANALYSIS_BRANCH, n_entries)
        jet_sel = get_analysis_selection(tree, params.JET_IS_ANALYSIS_BRANCH, n_entries)

        tau_eta = a[params.TAU_ETA_BRANCH][tau_sel]
        tau_phi = a[params.TAU_PHI_BRANCH][tau_sel]
        tau_ishad = clean_bool_branch(a[params.TAU_TRUTH_MATCH_BRANCH][tau_sel])
        tau_mask = ak.fill_none(a[params.TAU_PARENT_HIGGS_MASK_BRANCH][tau_sel], 0)

        jet_eta = a[params.JET_ETA_BRANCH][jet_sel]
        jet_phi = a[params.JET_PHI_BRANCH][jet_sel]
        jet_label = ak.fill_none(a[params.JET_TRUTH_LABEL_BRANCH][jet_sel], -999999)
        jet_mask = ak.fill_none(a[params.JET_PARENT_HIGGS_MASK_BRANCH][jet_sel], 0)

        # STEP 0
        real_tau = tau_ishad
        real_tau_mask = tau_mask[real_tau]
        nonzero_real_tau_mask = real_tau_mask[real_tau_mask != 0]

        n_real_tau_nz = ak.num(nonzero_real_tau_mask, axis=1)
        has_ref = n_real_tau_nz > 0

        min_mask = ak.min(nonzero_real_tau_mask, axis=1)
        max_mask = ak.max(nonzero_real_tau_mask, axis=1)

        consistent_tau_evt = ak.fill_none(min_mask == max_mask, True)

        valid_evt = has_ref & consistent_tau_evt
        ref_mask = min_mask

        n_evt_with_ref = int(ak.sum(has_ref))
        n_evt_inconsistent = int(ak.sum(has_ref & ~consistent_tau_evt))
        n_evt_valid = int(ak.sum(valid_evt))

        aggregate_events += n_events
        aggregate_events_with_ref += n_evt_with_ref
        aggregate_events_inconsistent_tau += n_evt_inconsistent
        aggregate_events_valid += n_evt_valid

        # STEP 1
        jet15 = jet_label == label_tau

        jet15_nonzero = jet15 & (jet_mask != 0)
        jet15_mask0 = jet15 & (jet_mask == 0)

        # ak.sum on a boolean mask counts the True values; ak.num would count list elements.
        n_jet15_total = int(ak.sum(jet15))
        n_jet15_mask0 = int(ak.sum(jet15_mask0))
        n_jet15_nonzero = int(ak.sum(jet15_nonzero))

        aggregate_jet15_total += n_jet15_total
        aggregate_jet15_mask0 += n_jet15_mask0
        aggregate_jet15_nonzero_total += n_jet15_nonzero

        valid_ref_mask = ref_mask[valid_evt]

        subset_mask_v = jet15_nonzero[valid_evt]

        jet15_mask_v = jet_mask[valid_evt][subset_mask_v]

        ref_mask_broadcast, _ = ak.broadcast_arrays(valid_ref_mask, jet15_mask_v)

        match_ok = jet15_mask_v == ref_mask_broadcast

        n_compared = int(ak.sum(ak.num(jet15_mask_v, axis=1)))
        n_consistent = int(ak.sum(match_ok))
        n_inconsistent = n_compared - n_consistent

        aggregate_compared += n_compared
        aggregate_consistent += n_consistent
        aggregate_inconsistent += n_inconsistent

        # STEP 2: same subset as STEP 1 (mask==0 jets excluded)
        real_tau_eta = tau_eta[real_tau]
        real_tau_phi = tau_phi[real_tau]

        real_tau_eta_v = real_tau_eta[valid_evt]
        real_tau_phi_v = real_tau_phi[valid_evt]

        jet_eta_v = jet_eta[valid_evt][subset_mask_v]
        jet_phi_v = jet_phi[valid_evt][subset_mask_v]

        n_step2_jets = int(ak.sum(ak.num(jet_eta_v, axis=1)))

        if n_step2_jets != n_compared:
            print(
                f"    [WARNING] {filename}: "
                f"STEP 1 ha {n_compared} jet, "
                f"STEP 2 ne ha {n_step2_jets}"
            )

        jet_eta_c, tau_eta_c = ak.unzip(
            ak.cartesian([jet_eta_v, real_tau_eta_v], nested=True)
        )
        jet_phi_c, tau_phi_c = ak.unzip(
            ak.cartesian([jet_phi_v, real_tau_phi_v], nested=True)
        )

        dr_matrix = delta_r(jet_eta_c, jet_phi_c, tau_eta_c, tau_phi_c)

        dr_min = ak.min(dr_matrix, axis=-1)

        dr_min_consistent = dr_min[match_ok]
        dr_min_inconsistent = dr_min[~match_ok]

        if ak.count(dr_min_consistent) > 0:
            dr_consistent_parts.append(dr_min_consistent)

        if ak.count(dr_min_inconsistent) > 0:
            dr_inconsistent_parts.append(dr_min_inconsistent)

        print(f"    eventi: {n_events}")
        print(f"    eventi validi STEP 0: {n_evt_valid}")
        print(f"    jet label=={label_tau}: {n_jet15_total}")
        print(f"    jet label=={label_tau} con mask==0: {n_jet15_mask0}")
        print(f"    jet label=={label_tau} con mask!=0: {n_jet15_nonzero}")

        if n_jet15_mask0 + n_jet15_nonzero != n_jet15_total:
            print(f"    [WARNING] label=={label_tau} != mask==0 + mask!=0")

        print(f"    jet confrontati STEP 1: {n_compared}")
        print(f"    coerenti: {n_consistent}")
        print(f"    incoerenti: {n_inconsistent}")
        print(f"    jet usati STEP 2: {n_step2_jets}")

    section("RISULTATI AGGREGATI SU TUTTI I FILE")

    section("STEP 0 - I tau adronici veri condividono la stessa mask entro evento?")

    print(f"   file analizzati: {len(loaded)}")
    print(f"   eventi totali: {aggregate_events}")
    print(f"   eventi con almeno un tau vero mask!=0: {aggregate_events_with_ref}")

    frac_inconsistent_tau = (
        aggregate_events_inconsistent_tau / aggregate_events_with_ref
        if aggregate_events_with_ref > 0
        else np.nan
    )

    print(
        f"   di questi, con mask NON uniforme tra i tau veri: "
        f"{aggregate_events_inconsistent_tau}"
        f"  ({100.0 * frac_inconsistent_tau:.3f}%)"
    )

    print(
        f"   eventi validi per STEP 1/2: {aggregate_events_valid}"
        f"  ({100.0 * aggregate_events_valid / max(aggregate_events, 1):.3f}%)"
    )

    print(
        "\n   -> Gli eventi con almeno un tau vero mask!=0 "
        "e mask uniforme costituiscono il campione"
    )
    print(
        "      usato per il confronto jet vs mask di riferimento "
        "dei tau veri."
    )

    section(
        f"STEP 1 - Coerenza mask jet(label=={label_tau}, mask!=0) "
        "vs tau veri dello stesso evento"
    )

    print(f"   jet label=={label_tau} totali: {aggregate_jet15_total}")

    frac_mask0 = (
        aggregate_jet15_mask0 / aggregate_jet15_total
        if aggregate_jet15_total > 0
        else np.nan
    )

    frac_mask_nonzero = (
        aggregate_jet15_nonzero_total / aggregate_jet15_total
        if aggregate_jet15_total > 0
        else np.nan
    )

    print(
        f"   di cui con mask==0 (baseline esclusa): {aggregate_jet15_mask0}"
        f"  ({100.0 * frac_mask0:.3f}%)"
    )
    print(
        f"   jet label=={label_tau} con mask!=0: {aggregate_jet15_nonzero_total}"
        f"  ({100.0 * frac_mask_nonzero:.3f}%)"
    )

    print()
    print(f"   CHECK decomposizione label=={label_tau}:")
    print(f"      mask==0 + mask!=0 = {aggregate_jet15_mask0 + aggregate_jet15_nonzero_total}")
    print(f"      label=={label_tau} totale   = {aggregate_jet15_total}")

    if aggregate_jet15_mask0 + aggregate_jet15_nonzero_total == aggregate_jet15_total:
        print("      -> OK")
    else:
        print("      -> ERRORE")

    frac_nonzero_compared = (
        aggregate_compared / aggregate_jet15_nonzero_total
        if aggregate_jet15_nonzero_total > 0
        else np.nan
    )

    print()
    print(
        f"   jet label=={label_tau} & mask!=0 confrontati in eventi validi: "
        f"{aggregate_compared}"
        f"  ({100.0 * frac_nonzero_compared:.3f}% "
        f"dei label=={label_tau} & mask!=0)"
    )

    print()
    print(f"   jet confrontati: {aggregate_compared}")

    frac_consistent = (
        aggregate_consistent / aggregate_compared
        if aggregate_compared > 0
        else np.nan
    )

    frac_inconsistent = (
        aggregate_inconsistent / aggregate_compared
        if aggregate_compared > 0
        else np.nan
    )

    print(
        f"   coerenti (stessa mask del tau vero): {aggregate_consistent}"
        f"  ({100.0 * frac_consistent:.3f}%)"
    )
    print(
        f"   INCOERENTI (mask diversa): {aggregate_inconsistent}"
        f"  ({100.0 * frac_inconsistent:.3f}%)"
    )

    section("STEP 2 - DeltaR(jet, tau vero piu' vicino) per coerenti vs incoerenti")

    print("   SUBSET USATO:")
    print(f"      HadronConeExclTruthLabelID == {label_tau}")
    print("      parentHiggsParentsMask != 0")
    print("      stesso identico subset dello STEP 1")
    print()

    if dr_consistent_parts:
        dr_consistent_all = ak.concatenate(dr_consistent_parts)
        summarize(dr_consistent_all, "DeltaR min - jet COERENTI con la mask del tau")
    else:
        print("   DeltaR min - jet COERENTI: nessun valore disponibile")

    if dr_inconsistent_parts:
        dr_inconsistent_all = ak.concatenate(dr_inconsistent_parts)
        summarize(dr_inconsistent_all, "DeltaR min - jet INCOERENTI con la mask del tau")
    else:
        print("   DeltaR min - jet INCOERENTI: nessun valore disponibile")

    n_dr_consistent = 0
    n_dr_inconsistent = 0

    if dr_consistent_parts:
        n_dr_consistent = int(ak.sum(ak.num(dr_consistent_parts, axis=1)))

    if dr_inconsistent_parts:
        n_dr_inconsistent = int(ak.sum(ak.num(dr_inconsistent_parts, axis=1)))

    print()
    print("   CHECK STEP 2:")
    print(f"      DeltaR coerenti:   {n_dr_consistent}")
    print(f"      DeltaR incoerenti: {n_dr_inconsistent}")
    print(f"      totale STEP 2:     {n_dr_consistent + n_dr_inconsistent}")
    print(f"      jet confrontati STEP 1: {aggregate_compared}")

    if n_dr_consistent + n_dr_inconsistent == aggregate_compared:
        print("      -> OK: STEP 2 usa esattamente gli stessi jet dello STEP 1.")
    else:
        print("      -> WARNING: il numero di jet non coincide.")

    print()
    print("   Interpretazione:")
    print("   - Se i due gruppi hanno DeltaR simile (entrambi piccoli),")
    print("     l'incoerenza di mask non sembra spiegabile semplicemente con una distanza geometrica maggiore.")
    print("   - Se gli incoerenti hanno DeltaR sistematicamente piu' grande,")
    print("     questo suggerisce che siano geometricamente meno correlati al tau vero piu' vicino.")
    print(f"   - Lo STEP 2 non contiene i jet label=={label_tau} con mask==0: essi sono esclusi gia' dalla definizione del subset.")
    print("   - Questa diagnostica non dimostra da sola la causa dell'incoerenza: distingue la componente geometrica da quella genealogica.")


def main():
    """
    Load the signal files and run `analyze_label15_higgs_consistency`.

    Returns
    -------
    None
    """
    loaded = load_files()

    if not loaded:
        print("\nNessun file disponibile. Controllare ROOT_DIR e i nomi dei file.")
        return

    analyze_label15_higgs_consistency(loaded)

    section("FINE DIAGNOSTICA")


if __name__ == "__main__":
    main()