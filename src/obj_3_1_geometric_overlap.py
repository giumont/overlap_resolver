"""
Objective 3.1 - Geometric RECO jet <-> tau matching.

Global goal
-----------
Quantify the geometric overlap between the RECO jet collection
(``recojet_antikt4PFlow_*``) and the RECO tau collection (``tau_*``), as
the first step of the study of the correlation between b-jet and
tau-jet identification. For every event the DeltaR(jet, tau) matrix is
computed over all jet-tau combinations, and is used to:

1. build the distribution of the minimum DeltaR "per jet" (distance to
   the closest tau) and "per tau" (distance to the closest jet);
2. inspect where the distribution changes slope (shoulder/knee) in order
   to justify an operating overlap threshold;
3. count, for one or more candidate thresholds (e.g. 0.2, 0.3, 0.4), how
   many pairs/jets/taus are geometrically overlapping.

This is a purely RECO study: no truth requirement (truth matching,
flavour, origin from H) is applied. The downstream overlap removal (OR)
acts on reco quantities only, so the overlap must be quantified with the
same information the OR algorithm would use. Truth-based categories are
handled in ``overlap_kinematics.py``.

Selections
----------
- ``recojet_antikt4PFlow_isAnalysisJet___NOSYS`` and
  ``tau_isAnalysisTau___NOSYS`` are always applied, to be consistent with
  the analysis-level collections used downstream. In the preliminary
  study the tau flag is True for 100% of the taus and the jet flag for
  99.541% of the jets, so the impact is negligible.
- ``JET_SELECTION_MODE`` / ``TAU_SELECTION_MODE`` (see ``params.py``)
  optionally restrict jets to the GN2v01 FixedCutBEff_85 working point
  and taus to the 85% working point of GNTauScoreSigTrans_v0prune.
- Events with no selected jet or no selected tau contain no jet-tau
  pair. The jets/taus in such events ("orphans") are counted and
  reported separately.

Internal checks
---------------
- files with missing branches are skipped with an explicit message;
- selected jets = orphan jets + jets in events with >= 1 tau (and
  likewise for taus);
- the number of entries in the "minimum DeltaR per jet" (per tau)
  distribution equals the number of non-orphan jets (taus).

All configuration (input files, branches, thresholds, binning, output
directory) is defined in ``params.py``.
"""

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

from params import (
    ROOT_DIR, FILE_PREFIX, FILE_SUFFIX, TREE_NAME, N_FILES, N_ENTRIES_CAP,
    JET_ETA_BRANCH, JET_PHI_BRANCH, JET_IS_ANALYSIS_BRANCH,
    TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_IS_ANALYSIS_BRANCH,
    JET_SELECTION_MODE, JET_BTAG_BRANCH,
    TAU_SELECTION_MODE, TAU_EFF_SCORE_BRANCH, TAU_SCORE_WP85_THRESHOLD,
    TRUTH_MODE_TAU, TAU_TRUTH_MATCH_BRANCH, TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH,
    DR_THRESHOLDS, DR_HIST_MIN, DR_HIST_MAX, DR_HIST_BINSIZE,
    NORMALIZE_DR_HISTOGRAMS, OUTPUT_DIR_DR, PLOT_ETA_PHI,
    ETA_HIST_MIN, ETA_HIST_MAX, ETA_HIST_BINSIZE,
    PHI_HIST_MIN, PHI_HIST_MAX, PHI_HIST_BINSIZE,
)


def section(title):
    """
    Print a title framed by separator lines.

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
    Build the path of the ``index``-th input ROOT file.

    Parameters
    ----------
    index : int
        File number, zero-padded to two digits and appended to
        ``FILE_PREFIX``.

    Returns
    -------
    pathlib.Path
        ``ROOT_DIR / f"{FILE_PREFIX}{index:02d}{FILE_SUFFIX}"``.
    """
    return ROOT_DIR / f"{FILE_PREFIX}{index:02d}{FILE_SUFFIX}"


def load_files():
    """
    Open the ``N_FILES`` expected ROOT files of the configured sample.

    Missing files, files without the tree ``TREE_NAME`` and files that
    fail to open are reported and skipped.

    Returns
    -------
    list of dict
        One entry per file opened correctly, with keys:

        - "index" : int, file number;
        - "path" : pathlib.Path;
        - "file_name" : str;
        - "root_file" : uproot file handle;
        - "tree" : uproot TTree ``TREE_NAME``;
        - "n_entries" : int, number of events to read
          (``min(tree entries, N_ENTRIES_CAP)``, or all if
          ``N_ENTRIES_CAP`` is None).
    """
    loaded = []

    section("FILE INPUT")

    for i in range(1, N_FILES + 1):
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

    print(f"\nFile caricati correttamente: {len(loaded)}/{N_FILES}")

    return loaded

def get_analysis_selection(tree, selection_branch, n_entries):
    """
    Read an ``isAnalysis...`` branch as a boolean mask.

    Parameters
    ----------
    tree : uproot.TTree
        Tree to read from.

    selection_branch : str
        Name of the per-object selection branch.

    n_entries : int
        Number of events to read (``entry_stop`` of the read).

    Returns
    -------
    awkward.Array of bool
        Jagged mask with the same structure as the branch. Missing
        values and 0 are mapped to False, any other value to True.
    """
    selection = tree[selection_branch].array(entry_stop=n_entries, library="ak")
    selection = ak.fill_none(selection, 0)

    return selection != 0


def required_branches(base, include_truth=False):
    """
    Extend a list of always-needed branches with those required by the
    configuration in ``params.py``.

    Parameters
    ----------
    base : list of str
        Branches the calling script always needs.

    include_truth : bool, default=False
        If True, also add the tau truth branches required by
        ``TRUTH_MODE_TAU`` (``TAU_TRUTH_MATCH_BRANCH`` for ``"label"``,
        ``TRUTH_TAU_ETA_BRANCH`` and ``TRUTH_TAU_PHI_BRANCH`` for
        ``"geometric"``). Only scripts that assign truth labels need
        this.

    Returns
    -------
    list of str
        ``base`` plus ``JET_BTAG_BRANCH`` if ``JET_SELECTION_MODE`` is
        ``"btag85"``, ``TAU_EFF_SCORE_BRANCH`` if ``TAU_SELECTION_MODE``
        is ``"score85"``, and the truth branches if requested. A new
        list is returned; ``base`` is not modified.
    """
    branches = list(base)

    if JET_SELECTION_MODE == "btag85":
        branches.append(JET_BTAG_BRANCH)

    if TAU_SELECTION_MODE == "score85":
        branches.append(TAU_EFF_SCORE_BRANCH)

    if include_truth:
        if TRUTH_MODE_TAU == "label":
            branches.append(TAU_TRUTH_MATCH_BRANCH)
        elif TRUTH_MODE_TAU == "geometric":
            branches.extend([TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH])

    return branches


def build_selections(a, tree, n_entries):
    """
    Build the jet and tau selection masks for the configured modes.

    Starts from the analysis-level flags and, depending on
    ``JET_SELECTION_MODE`` / ``TAU_SELECTION_MODE``, adds the b-tag
    working point requirement (``"btag85"``) and/or the tau score
    working point requirement (``"score85"``). Validation of the mode
    values is done in ``params.py``.

    Parameters
    ----------
    a : awkward.Array
        Branches read for the file; must contain the b-tag / tau score
        branch when the corresponding mode needs it (see
        `required_branches`).

    tree : uproot.TTree
        Tree the analysis-level flags are read from.

    n_entries : int
        Number of events read.

    Returns
    -------
    jet_sel, tau_sel : awkward.Array of bool
        Per-object selection masks, shape ``[event][jet]`` and
        ``[event][tau]``.
    """
    jet_sel = get_analysis_selection(tree, JET_IS_ANALYSIS_BRANCH, n_entries)
    tau_sel = get_analysis_selection(tree, TAU_IS_ANALYSIS_BRANCH, n_entries)

    if JET_SELECTION_MODE == "btag85":
        jet_sel = jet_sel & (ak.fill_none(a[JET_BTAG_BRANCH], False) != 0)

    if TAU_SELECTION_MODE == "score85":
        tau_sel = tau_sel & (
            ak.fill_none(a[TAU_EFF_SCORE_BRANCH], -np.inf) >= TAU_SCORE_WP85_THRESHOLD
        )

    return jet_sel, tau_sel


def delta_phi(phi1, phi2):
    """
    Azimuthal difference wrapped into [-pi, pi).

    Parameters
    ----------
    phi1, phi2 : array-like of float
        Azimuthal angles in radians (broadcastable against each other).

    Returns
    -------
    array-like of float
        ``phi1 - phi2`` reduced to the interval [-pi, pi).
    """
    dphi = phi1 - phi2

    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    """
    Angular distance DeltaR = sqrt(Delta_eta^2 + Delta_phi^2).

    Parameters
    ----------
    eta1, phi1 : array-like of float
        Pseudorapidity and azimuth of the first object.

    eta2, phi2 : array-like of float
        Pseudorapidity and azimuth of the second object (broadcastable
        against the first).

    Returns
    -------
    array-like of float
        DeltaR, with Delta_phi wrapped into [-pi, pi).
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)

    return np.sqrt(deta ** 2 + dphi ** 2)


def summarize(values, label):
    """
    Flatten an array, drop missing/non-finite entries and print summary
    statistics (n, mean, median, 10th and 90th percentile).

    Parameters
    ----------
    values : awkward.Array or None
        Array of any nesting depth.

    label : str
        Description printed in front of the statistics.

    Returns
    -------
    numpy.ndarray or None
        Flat array of the valid values, or None if no valid value is
        available.
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
    Find the largest DeltaR up to which the three binned distributions
    (min DeltaR per jet, min DeltaR per tau, all pairs) coincide.

    Bins are scanned from ``DR_HIST_MIN`` upwards with width
    ``DR_HIST_BINSIZE``; the scan stops at the first bin where the jet
    histogram differs from the tau or from the all-pairs histogram by
    more than ``tol_abs`` counts. Below that point every jet-tau pair is
    mutually the closest one, i.e. the overlap is "perfect".

    Parameters
    ----------
    flat_jet_min, flat_tau_min, flat_all_pairs : numpy.ndarray or None
        Flat DeltaR values of the three distributions.

    tol_abs : int, default=0
        Maximum absolute difference in bin counts still considered a
        match.

    Returns
    -------
    max_dr : float or None
        Upper edge of the last matching bin (``DR_HIST_MIN`` if the
        first bin already differs); None if any input is None.

    n_bins_matched : int
        Number of consecutive matching bins.
    """
    if flat_jet_min is None or flat_tau_min is None or flat_all_pairs is None:
        return None, 0

    edges = np.arange(DR_HIST_MIN, DR_HIST_MAX + DR_HIST_BINSIZE, DR_HIST_BINSIZE)

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
    Print the binned density of a DeltaR distribution, its bin-to-bin
    variation and the cumulative fractions at several thresholds.

    Used to locate by eye the shoulder of the distribution and to
    justify the candidate thresholds in ``DR_THRESHOLDS``, which are
    flagged in the cumulative table.

    Parameters
    ----------
    flat_values : numpy.ndarray or None
        Flat DeltaR values.

    label : str
        Description printed in the table headers.

    Returns
    -------
    None
    """
    if flat_values is None or flat_values.size == 0:
        print(f"   {label}: nessun valore per la tabella di spalla")
        return

    edges = np.arange(DR_HIST_MIN, DR_HIST_MAX + DR_HIST_BINSIZE, DR_HIST_BINSIZE)

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
    Save the histogram of the three DeltaR distributions (min per jet,
    min per tau, all jet-tau pairs), with the candidate thresholds as
    vertical lines.

    The file is written to ``OUTPUT_DIR_DR``; nothing is done if
    matplotlib is unavailable.

    Parameters
    ----------
    dr_jet_min : numpy.ndarray or None
        Minimum DeltaR to a tau, for each jet.

    dr_tau_min : numpy.ndarray or None
        Minimum DeltaR to a jet, for each tau.

    dr_all_pairs : numpy.ndarray or None
        DeltaR of all jet-tau pairs.

    perfect_overlap_thr : float or None, default=None
        Currently unused: the line marking the perfect-overlap
        threshold is disabled.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip del plot, uso solo le tabelle testuali."
        )
        return

    OUTPUT_DIR_DR.mkdir(parents=True, exist_ok=True)

    bins = np.arange(DR_HIST_MIN, DR_HIST_MAX + DR_HIST_BINSIZE, DR_HIST_BINSIZE)

    fig, ax = plt.subplots(figsize=(8, 5))

    series = [
        (dr_jet_min, f"DeltaR min per jet (tau piu' vicino) (N={dr_jet_min.size if dr_jet_min is not None else 0})"),
        (dr_tau_min, f"DeltaR min per tau (jet piu' vicino) (N={dr_tau_min.size if dr_tau_min is not None else 0})"),
        (dr_all_pairs, f"DeltaR tutte le coppie jet-tau (N={dr_all_pairs.size if dr_all_pairs is not None else 0})"),
    ]

    ylabel = "Densità di probabilità" if NORMALIZE_DR_HISTOGRAMS else "Conteggio (scala log)"

    for values, label in series:
        if values is not None and values.size > 0:
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                label=label,
                linewidth=1.5,
                density=NORMALIZE_DR_HISTOGRAMS,
            )

    ax.set_yscale("log")
    ax.set_xlabel("DeltaR(jet, tau)")
    ax.set_ylabel(ylabel)
    ax.set_title("Matching geometrico reco jet <-> tau")

    ymax = ax.get_ylim()[1]

    for thr in DR_THRESHOLDS:
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.0)
        ax.text(
            thr, ymax, f"{thr}",
            rotation=90, va="top", ha="right", fontsize=8, color="red",
        )

    ax.legend(fontsize=8)
    fig.tight_layout()

    file_suffix = "_normalized" if NORMALIZE_DR_HISTOGRAMS else ""
    out_path = (
        OUTPUT_DIR_DR
        / f"dr_jet_tau_overlap_hist_{JET_SELECTION_MODE}_tau_{TAU_SELECTION_MODE}"
        f"{file_suffix}_cut_{DR_HIST_MAX}.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot salvato in: {out_path}")


def save_eta_phi_histograms(jet_eta_parts, tau_eta_parts, jet_phi_parts, tau_phi_parts):
    """
    Save the eta and phi histograms of the selected jets and taus, as a
    check of the collections entering the overlap study.

    Two files (eta, phi) are written to ``OUTPUT_DIR_DR``. Nothing is
    done if ``PLOT_ETA_PHI`` is False or matplotlib is unavailable.

    Parameters
    ----------
    jet_eta_parts, tau_eta_parts, jet_phi_parts, tau_phi_parts : list of awkward.Array
        Per-file arrays of the selected objects, concatenated
        internally.

    Returns
    -------
    None
    """
    if not PLOT_ETA_PHI:
        return

    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip dei plot eta/phi."
        )
        return

    OUTPUT_DIR_DR.mkdir(parents=True, exist_ok=True)

    jet_eta_all = ak.concatenate(jet_eta_parts) if jet_eta_parts else None
    tau_eta_all = ak.concatenate(tau_eta_parts) if tau_eta_parts else None
    jet_phi_all = ak.concatenate(jet_phi_parts) if jet_phi_parts else None
    tau_phi_all = ak.concatenate(tau_phi_parts) if tau_phi_parts else None

    def flatten_to_numpy(values):
        """Flatten to a 1D numpy array, dropping None and non-finite values."""
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

    out_path_eta = OUTPUT_DIR_DR / f"eta_jet_tau_{JET_SELECTION_MODE}_tau_{TAU_SELECTION_MODE}.png"
    fig.savefig(out_path_eta, dpi=150)
    plt.close(fig)

    print(f"[OK] Plot eta salvato in: {out_path_eta}")

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

    out_path_phi = OUTPUT_DIR_DR / f"phi_jet_tau_{JET_SELECTION_MODE}_tau_{TAU_SELECTION_MODE}.png"
    fig.savefig(out_path_phi, dpi=150)
    plt.close(fig)

    print(f"[OK] Plot phi salvato in: {out_path_phi}")


def analyze_jet_tau_overlap(loaded):
    """
    Run the geometric jet <-> tau matching study on all input files.

    For each file: apply the analysis-level (and optional working point)
    selections, count orphan jets/taus, build the DeltaR matrix over all
    jet-tau combinations of each event, extract the minimum DeltaR per
    jet and per tau, and run the internal consistency checks. The
    results are then aggregated over the files to print the DeltaR
    summaries, compute the perfect-overlap threshold, print the shoulder
    tables, save the plots and count the overlapping jets/taus/pairs for
    each threshold in ``DR_THRESHOLDS`` (plus the perfect-overlap one).

    Parameters
    ----------
    loaded : list of dict
        Output of `load_files`. Files missing any required branch are
        skipped.

    Returns
    -------
    None
        Results are printed and plots are saved to ``OUTPUT_DIR_DR``.
    """
    branches = required_branches([
        TAU_ETA_BRANCH,
        TAU_PHI_BRANCH,
        TAU_IS_ANALYSIS_BRANCH,
        JET_ETA_BRANCH,
        JET_PHI_BRANCH,
        JET_IS_ANALYSIS_BRANCH,
    ])

    aggregate_events = 0

    aggregate_jets_total = 0
    aggregate_taus_total = 0

    aggregate_jets_orphan = 0
    aggregate_taus_orphan = 0

    dr_jet_min_parts = []
    dr_tau_min_parts = []
    dr_all_pairs_parts = []

    jet_eta_parts = []
    tau_eta_parts = []
    jet_phi_parts = []
    tau_phi_parts = []

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

        n_events = len(a[TAU_ETA_BRANCH])
        aggregate_events += n_events

        jet_sel, tau_sel = build_selections(a, tree, n_entries)

        tau_eta = a[TAU_ETA_BRANCH][tau_sel]
        tau_phi = a[TAU_PHI_BRANCH][tau_sel]

        jet_eta = a[JET_ETA_BRANCH][jet_sel]
        jet_phi = a[JET_PHI_BRANCH][jet_sel]

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

        # Orphans: selected objects in events with none of the other kind.
        evt_no_tau = n_tau_evt == 0
        evt_no_jet = n_jet_evt == 0

        n_jets_orphan_file = int(ak.sum(n_jet_evt[evt_no_tau]))
        n_taus_orphan_file = int(ak.sum(n_tau_evt[evt_no_jet]))

        aggregate_jets_orphan += n_jets_orphan_file
        aggregate_taus_orphan += n_taus_orphan_file

        # DeltaR[event][jet][tau]; the second product, with roles swapped,
        # gives DeltaR[event][tau][jet] so that the minimum is taken per tau.
        jet_eta_c, tau_eta_c = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
        jet_phi_c, tau_phi_c = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))

        dr_matrix = delta_r(jet_eta_c, jet_phi_c, tau_eta_c, tau_phi_c)
        dr_jet_min = ak.min(dr_matrix, axis=-1)

        tau_eta_ct, jet_eta_ct = ak.unzip(ak.cartesian([tau_eta, jet_eta], nested=True))
        tau_phi_ct, jet_phi_ct = ak.unzip(ak.cartesian([tau_phi, jet_phi], nested=True))

        dr_matrix_t = delta_r(tau_eta_ct, tau_phi_ct, jet_eta_ct, jet_phi_ct)
        dr_tau_min = ak.min(dr_matrix_t, axis=-1)

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
    dr_all_pairs_all = ak.concatenate(dr_all_pairs_parts) if dr_all_pairs_parts else None

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

    section("ANALISI DELLA SOGLIA - dove la distribuzione cambia pendenza")

    perfect_overlap_thr, n_bins = find_perfect_overlap_threshold(
        flat_jet_min, flat_tau_min, flat_all_pairs
    )

    print(f"   [SOGLIA CALCOLATA] DeltaR massimo di perfetto overlap: {perfect_overlap_thr:.4f}")
    print(f"                      (coincidenza esatta su {n_bins} bin consecutivi con bin_size={DR_HIST_BINSIZE})")

    print_shoulder_table(flat_jet_min, "DeltaR minimo per jet")
    print_shoulder_table(flat_tau_min, "DeltaR minimo per tau")

    save_combinatory_dr_plot(
        flat_jet_min, flat_tau_min, flat_all_pairs,
        perfect_overlap_thr=perfect_overlap_thr,
    )

    if PLOT_ETA_PHI:
        save_eta_phi_histograms(
            jet_eta_parts,
            tau_eta_parts,
            jet_phi_parts,
            tau_phi_parts,
        )

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


def main():
    """
    Entry point: load the input files and run the geometric overlap
    study (`analyze_jet_tau_overlap`).

    Returns
    -------
    None
    """
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