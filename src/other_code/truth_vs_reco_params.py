"""
Truth labelling and object-level kinematics of the jet/tau overlap.

Objective 3.1: on the signal sample, understand the correlation between
b-jet and tau-jet identification by quantifying the overlap between the two
types of reconstructed objects and comparing their kinematic properties.

This script adds the truth information to the geometric overlap measured in
``obj_3_1.py``: every selected jet and tau is labelled as true or fake, and
the overlap is studied separately for each truth population. It must be run
after ``obj_3_1.py``, whose utilities it reuses.

Truth definition
----------------
- Jet: true b-jet if ``HadronConeExclTruthLabelID == 5``.
- Tau: true hadronic tau according to the stored truth flag
  (``TRUTH_MODE_TAU = "label"``) or to a geometric match with the visible
  truth taus within ``DR_TRUTH_MATCH_TAU`` (``TRUTH_MODE_TAU = "geometric"``).

Analysis steps
--------------
A. Truth labelling of the selected jets and taus.

B. Pair-level classification. Each (jet, tau) pair with
   DeltaR < ``DR_THRESHOLD_KINEMATICS`` falls in one of four categories:
   true b-jet / fake tau (TF), fake b-jet / true tau (FT), both true (TT),
   both fake (FF). The full DeltaR(jet, tau) distribution is also produced
   per category, together with the event-level and pair-level fractions of
   each category in the overlap region.

C. Object-level kinematics. A jet (tau) is "in overlap" if at least one tau
   (jet) lies within ``DR_THRESHOLD_KINEMATICS``. Each object is counted
   once. pT, eta and phi are compared for overlapping vs isolated objects,
   and additionally split into true/fake populations. The pair categories
   are not used here, since one object can belong to pairs of different
   categories.

Object selection, binning and output paths are defined in ``params.py``.
"""

import sys
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

# params.py (and the obj_3_1.py utilities) live in the parent folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from params import (
    JET_SELECTION_MODE, JET_BTAG_BRANCH,
    JET_ETA_BRANCH, JET_PHI_BRANCH, JET_PT_BRANCH, JET_IS_ANALYSIS_BRANCH,
    JET_TRUTH_LABEL_BRANCH, JET_TRUTH_LABEL_B_VALUE,
    TAU_SELECTION_MODE, TAU_EFF_SCORE_BRANCH, TAU_SCORE_WP85_THRESHOLD,
    TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_PT_BRANCH, TAU_IS_ANALYSIS_BRANCH,
    TRUTH_MODE_TAU, TAU_TRUTH_MATCH_BRANCH,
    TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH, DR_TRUTH_MATCH_TAU,
    DR_THRESHOLD_KINEMATICS,
    CATEGORY_LABELS, CATEGORY_COLORS, CATEGORY_KEYS,
    OUTPUT_DIR_TRUTH,
    TRUTH_DR_THRESHOLDS, TRUTH_DR_HIST_MIN, TRUTH_DR_HIST_MAX,
    TRUTH_DR_HIST_BINSIZE, TRUTH_PT_HIST_MAX,
    NORMALIZE_TRUTH_HISTOGRAMS, OBJECT_KINEMATICS_VARIABLES,
)
from obj_3_1_geometric_overlap import (
    section, load_files, get_analysis_selection, delta_r, summarize,
    print_shoulder_table,
)


# ======================================================================
# PART A - RECO <-> TRUTH MATCHING
# ======================================================================

def match_reco_to_truth(reco_eta, reco_phi, truth_eta, truth_phi, dr_max):
    """
    Geometric matching between reconstructed and truth objects.

    For each reco object the minimum DeltaR with respect to the truth
    objects of the same event is computed; the object is matched if the
    minimum is below `dr_max`.

    Parameters
    ----------
    reco_eta, reco_phi : awkward.Array
        Jagged arrays [event][reco object] of pseudorapidity and azimuth
        of the reco objects.

    truth_eta, truth_phi : awkward.Array
        Jagged arrays [event][truth object] of pseudorapidity and azimuth
        of the truth objects.

    dr_max : float
        Maximum DeltaR for a reco object to be considered matched.

    Returns
    -------
    dr_min : awkward.Array
        Minimum DeltaR to a truth object for each reco object; `None`
        where the event has no truth object.

    is_matched : awkward.Array of bool
        True where `dr_min < dr_max`, False otherwise (including events
        with no truth object).
    """
    # First axis = reco objects, second axis = truth objects.
    reco_eta_ct, truth_eta_ct = ak.unzip(
        ak.cartesian([reco_eta, truth_eta], nested=True)
    )
    reco_phi_ct, truth_phi_ct = ak.unzip(
        ak.cartesian([reco_phi, truth_phi], nested=True)
    )

    dr_matrix = delta_r(reco_eta_ct, reco_phi_ct, truth_eta_ct, truth_phi_ct)

    dr_min = ak.min(dr_matrix, axis=-1)
    is_matched = ak.fill_none(dr_min < dr_max, False)

    return dr_min, is_matched


def label_jets_and_taus(a, jet_sel, tau_sel):
    """
    Assign a truth label (true/fake) to the selected jets and taus.

    Jets are true b-jets if their truth flavour label equals
    `JET_TRUTH_LABEL_B_VALUE`. Taus are labelled according to
    `TRUTH_MODE_TAU`: the stored hadronic-tau flag ("label") or a
    geometric match with the visible truth taus ("geometric").

    Parameters
    ----------
    a : awkward.Array
        Arrays of the tree branches read for the current file.

    jet_sel, tau_sel : awkward.Array of bool
        Jagged masks [event][object] selecting the jets and taus under
        study.

    Returns
    -------
    jet_is_true, tau_is_true : awkward.Array of bool
        Truth labels of the selected jets and taus.

    jet_dr_truth : None
        Placeholder: jets are always labelled from the truth flavour
        label, so no DeltaR to a truth object is available.

    tau_dr_truth : awkward.Array or None
        DeltaR to the closest visible truth tau in "geometric" mode;
        `None` in "label" mode.

    Raises
    ------
    RuntimeError
        If `TRUTH_MODE_TAU` is not "label" or "geometric".
    """
    jet_eta = a[JET_ETA_BRANCH][jet_sel]
    jet_phi = a[JET_PHI_BRANCH][jet_sel]

    tau_eta = a[TAU_ETA_BRANCH][tau_sel]
    tau_phi = a[TAU_PHI_BRANCH][tau_sel]

    flavour = a[JET_TRUTH_LABEL_BRANCH][jet_sel]
    jet_is_true = flavour == JET_TRUTH_LABEL_B_VALUE
    jet_dr_truth = None

    if TRUTH_MODE_TAU == "label":
        tau_is_true = a[TAU_TRUTH_MATCH_BRANCH][tau_sel] != 0
        tau_dr_truth = None

    elif TRUTH_MODE_TAU == "geometric":
        truth_tau_eta = a[TRUTH_TAU_ETA_BRANCH]
        truth_tau_phi = a[TRUTH_TAU_PHI_BRANCH]

        tau_dr_truth, tau_is_true = match_reco_to_truth(
            tau_eta, tau_phi, truth_tau_eta, truth_tau_phi,
            DR_TRUTH_MATCH_TAU,
        )

    else:
        raise RuntimeError(
            f"TRUTH_MODE_TAU='{TRUTH_MODE_TAU}' non gestito."
        )

    return jet_is_true, tau_is_true, jet_dr_truth, tau_dr_truth


# ======================================================================
# COMMON UTILITY: DeltaR MATRIX AND PAIR LABELS
# ======================================================================

def build_pair_geometry_and_labels(
    jet_eta, jet_phi, jet_label,
    tau_eta, tau_phi, tau_label,
):
    """
    Build, once per file, the DeltaR matrix of all (jet, tau) pairs and
    the truth labels broadcast to the same structure.

    The output is shared by the pair classification, the full DeltaR
    distribution, the event-level counts and the object-level overlap
    flags.

    Parameters
    ----------
    jet_eta, jet_phi : awkward.Array
        Jagged arrays [event][jet] of the selected jets.

    jet_label : awkward.Array of bool
        Truth label of each selected jet (True = true b-jet).

    tau_eta, tau_phi : awkward.Array
        Jagged arrays [event][tau] of the selected taus.

    tau_label : awkward.Array of bool
        Truth label of each selected tau (True = true hadronic tau).

    Returns
    -------
    pair_info : dict
        Arrays with structure [event][jet][tau]:

        - "dr": DeltaR(jet, tau);
        - "jet_label": truth label of the jet of each pair;
        - "tau_label": truth label of the tau of each pair.
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

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)

    return {
        "dr": dr_matrix,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }


# ======================================================================
# PART B - CLASSIFICATION OF THE OVERLAPPING PAIRS
# ======================================================================

def classify_overlapping_pairs(pair_info, dr_thr):
    """
    Count the overlapping pairs in each of the four truth categories.

    Only pairs with DeltaR < `dr_thr` are considered.

    Parameters
    ----------
    pair_info : dict
        Output of `build_pair_geometry_and_labels`.

    dr_thr : float
        DeltaR threshold defining the overlap.

    Returns
    -------
    categories : dict of int
        Number of overlapping pairs for each key of `CATEGORY_KEYS`.

    n_total : int
        Total number of overlapping pairs.
    """
    dr_matrix = pair_info["dr"]
    jet_label_ct = pair_info["jet_label"]
    tau_label_ct = pair_info["tau_label"]

    overlap_mask = dr_matrix < dr_thr

    jet_lab_flat = ak.to_numpy(ak.flatten(jet_label_ct, axis=None)).astype(bool)
    tau_lab_flat = ak.to_numpy(ak.flatten(tau_label_ct, axis=None)).astype(bool)
    overlap_flat = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)

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
    Split the full DeltaR(jet, tau) distribution into the four truth
    categories, without any cut on DeltaR.

    Parameters
    ----------
    pair_info : dict
        Output of `build_pair_geometry_and_labels`.

    Returns
    -------
    dr_by_category : dict of numpy.ndarray
        One-dimensional array of DeltaR values for each key of
        `CATEGORY_KEYS`.
    """
    dr_flat = ak.to_numpy(ak.flatten(pair_info["dr"], axis=None))
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

    return {cat: dr_flat[mask] for cat, mask in cat_masks.items()}


# ======================================================================
# PART C - OBJECT-LEVEL KINEMATICS
# ======================================================================

def classify_objects_kinematics(
    jet_pt, jet_eta, jet_phi, jet_label,
    tau_pt, tau_eta, tau_phi, tau_label,
    pair_info, dr_threshold,
):
    """
    Flag each selected jet and tau as overlapping or isolated and
    flatten their kinematics to one entry per object.

    A jet is in overlap if at least one selected tau lies within
    `dr_threshold`, and vice versa for taus. Each object is therefore
    counted once, regardless of the event multiplicity.

    Parameters
    ----------
    jet_pt, jet_eta, jet_phi : awkward.Array
        Jagged arrays [event][jet] of the selected jets.

    jet_label : awkward.Array of bool
        Truth label of each selected jet.

    tau_pt, tau_eta, tau_phi : awkward.Array
        Jagged arrays [event][tau] of the selected taus.

    tau_label : awkward.Array of bool
        Truth label of each selected tau.

    pair_info : dict
        Output of `build_pair_geometry_and_labels`; its "dr" matrix is
        used for the jet flags.

    dr_threshold : float
        DeltaR threshold defining the overlap.

    Returns
    -------
    object_data : dict
        ``object_data["jet"]`` and ``object_data["tau"]`` each hold
        one-dimensional numpy arrays with keys "pt", "eta", "phi",
        "is_true" (bool) and "in_overlap" (bool).

    Raises
    ------
    RuntimeError
        If the number of overlap flags differs from the number of
        selected objects.
    """
    jet_tau_dr = pair_info["dr"]
    jet_in_overlap = ak.any(jet_tau_dr < dr_threshold, axis=-1)

    # The tau matrix is rebuilt with taus on the outer axis so that taus in
    # events with no jets are not lost when reducing over the partner axis.
    tau_eta_ct, jet_eta_ct = ak.unzip(
        ak.cartesian([tau_eta, jet_eta], nested=True)
    )
    tau_phi_ct, jet_phi_ct = ak.unzip(
        ak.cartesian([tau_phi, jet_phi], nested=True)
    )

    tau_jet_dr = delta_r(tau_eta_ct, tau_phi_ct, jet_eta_ct, jet_phi_ct)
    tau_in_overlap = ak.any(tau_jet_dr < dr_threshold, axis=-1)

    def flat(x):
        return ak.to_numpy(ak.flatten(x, axis=None))

    jet_pt_flat = flat(jet_pt)
    jet_in_overlap_flat = flat(jet_in_overlap).astype(bool)

    tau_pt_flat = flat(tau_pt)
    tau_in_overlap_flat = flat(tau_in_overlap).astype(bool)

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
    """
    Concatenate the object-level data of several files.

    Parameters
    ----------
    parts_list : list of dict
        Outputs of `classify_objects_kinematics`, one per file.

    Returns
    -------
    merged : dict
        Same structure as a single `classify_objects_kinematics` output,
        with the arrays of all files concatenated (empty arrays if
        `parts_list` is empty).
    """
    variables = ("pt", "eta", "phi", "is_true", "in_overlap")

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
            merged[obj][var] = np.concatenate(arrays) if arrays else np.array([])

    return merged


# ======================================================================
# PLOT: FULL DeltaR DISTRIBUTION PER CATEGORY (PAIR-LEVEL)
# ======================================================================

def _step_hist_with_gaps(ax, values, bins, density, label, color,
                         linewidth=1.5):
    """
    Draw a histogram as a step line on an existing axis.

    Empty bins are kept at zero, so on a logarithmic y-axis they are not
    displayed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.

    values : numpy.ndarray
        Values to histogram.

    bins : numpy.ndarray
        Bin edges.

    density : bool
        If True, normalise the histogram to unit area.

    label : str
        Legend label; the number of entries is appended.

    color : str
        Line colour.

    linewidth : float, default=1.5
        Line width.
    """
    counts, edges = np.histogram(values, bins=bins, density=density)

    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)

    ax.plot(
        x,
        y,
        linewidth=linewidth,
        color=color,
        label=f"{label}  (n={values.size})",
    )


def save_dr_histogram_by_category(dr_by_category):
    """
    Plot the full DeltaR(jet, tau) distribution for the four truth
    categories and save it to `OUTPUT_DIR_TRUTH / "truth_categories"`.

    The plot is pair-level: it shows the shape of the small-DeltaR peak
    and of the combinatorial tail. The DeltaR thresholds in
    `TRUTH_DR_THRESHOLDS` are drawn as vertical lines.

    Parameters
    ----------
    dr_by_category : dict of numpy.ndarray
        Output of `flatten_dr_by_category`, merged over all files.
    """
    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip plot DeltaR per categoria."
        )
        return

    out_dir = OUTPUT_DIR_TRUTH / "truth_categories"
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = np.arange(
        TRUTH_DR_HIST_MIN,
        TRUTH_DR_HIST_MAX + TRUTH_DR_HIST_BINSIZE,
        TRUTH_DR_HIST_BINSIZE,
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
            density=NORMALIZE_TRUTH_HISTOGRAMS,
            label=CATEGORY_LABELS[cat],
            color=CATEGORY_COLORS[cat],
        )

    ax.set_yscale("log")
    ax.set_xlabel("$\\Delta R$(jet, tau)")
    ax.set_xlim(TRUTH_DR_HIST_MIN, TRUTH_DR_HIST_MAX)

    if NORMALIZE_TRUTH_HISTOGRAMS:
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

    for thr in TRUTH_DR_THRESHOLDS:
        ax.axvline(thr, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(
            thr + 0.005,
            0.95,
            fr"$\Delta R = {thr}$",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            verticalalignment="top",
            fontsize=8,
            color="black",
            alpha=0.8,
        )

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()

    suffix = (
        f"_normalized_cut{TRUTH_DR_HIST_MAX}"
        if NORMALIZE_TRUTH_HISTOGRAMS
        else f"_cut{TRUTH_DR_HIST_MAX}"
    )
    out_path = (
        out_dir / f"dr_jet_tau_by_category_{JET_SELECTION_MODE}{suffix}.png"
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot DeltaR per categoria salvato in: {out_path}")


# ======================================================================
# PLOT: OBJECT-LEVEL KINEMATICS
# ======================================================================

def _plot_hist_curves(ax, datasets, bins, density):
    """
    Draw several step histograms on a common axis.

    Histograms are normalised to unit area if `density` is True, otherwise
    they show raw counts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.

    datasets : list of tuple
        Entries ``(values, label, color)``, with `values` a numpy array
        and `color` a matplotlib colour or None for the default cycle.
        Empty datasets are skipped.

    bins : numpy.ndarray
        Bin edges shared by all datasets.

    density : bool
        If True, normalise each histogram to unit area.
    """
    for values, label, color in datasets:
        if values.size == 0:
            continue

        kwargs = {
            "bins": bins,
            "histtype": "step",
            "linewidth": 1.7,
            "density": density,
            "label": f"{label}  (n={values.size})",
        }

        if color is not None:
            kwargs["color"] = color

        ax.hist(values, **kwargs)


def _finish_object_plot(ax, xlabel, title, normalize):
    """
    Apply the common formatting of the object-level kinematic plots
    (log y-axis, labels, title, grid, legend).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format.

    xlabel, title : str
        x-axis label and plot title.

    normalize : bool
        If True the y-axis is labelled as a probability density,
        otherwise as a count.
    """
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densità di probabilità" if normalize else "Conteggio")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)


def save_object_kinematics_all_overlap_vs_isolated(object_data, dr_threshold):
    """
    Plot pT, eta and phi of all jets and all taus, comparing overlapping
    and isolated objects (6 plots), saved to
    `OUTPUT_DIR_TRUTH / "no_truth_categories"`.

    Each object appears exactly once.

    Parameters
    ----------
    object_data : dict
        Output of `merge_object_kinematics`.

    dr_threshold : float
        DeltaR threshold defining the overlap; used only in the plot
        titles.
    """
    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip dei plot object-level all overlap/non-overlap."
        )
        return

    out_dir = OUTPUT_DIR_TRUTH / "no_truth_categories"
    out_dir.mkdir(parents=True, exist_ok=True)

    for obj_name in ("jet", "tau"):
        for var_key, vmin, vmax, step, xlabel_suffix in OBJECT_KINEMATICS_VARIABLES:
            values = object_data[obj_name][var_key]
            overlap = object_data[obj_name]["in_overlap"]

            bins = np.arange(vmin, vmax + step, step)

            fig, ax = plt.subplots(figsize=(8, 5))

            _plot_hist_curves(
                ax,
                [
                    (values[overlap], "overlap", "tab:orange"),
                    (values[~overlap], "isolati", "tab:purple"),
                ],
                bins,
                NORMALIZE_TRUTH_HISTOGRAMS,
            )

            _finish_object_plot(
                ax,
                f"{obj_name} {xlabel_suffix}",
                f"{obj_name}: overlap vs isolati (event-level) "
                f"- $\\Delta R < {dr_threshold}$",
                normalize=NORMALIZE_TRUTH_HISTOGRAMS,
            )

            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_TRUTH_HISTOGRAMS else ""
            suffix_cut = f"_cut_{TRUTH_PT_HIST_MAX}" if var_key == "pt" else ""

            out_path = (
                out_dir
                / f"{obj_name}_{var_key}_all_overlap_vs_isolated_"
                f"{JET_SELECTION_MODE}{suffix}{suffix_cut}.png"
            )

            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[OK] Plot salvato in: {out_path}")


def save_object_kinematics_truth_split(object_data, dr_threshold):
    """
    Plot pT, eta and phi of jets and taus split by truth and overlap
    status (6 plots), saved to `OUTPUT_DIR_TRUTH / "truth_categories"`.

    Four populations are shown for each object: true + overlap,
    true + isolated, fake + overlap, fake + isolated. The split is
    object-level and does not use the pair categories of part B.

    Parameters
    ----------
    object_data : dict
        Output of `merge_object_kinematics`.

    dr_threshold : float
        DeltaR threshold defining the overlap; used only in the plot
        titles.
    """
    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip dei plot truth-split object-level."
        )
        return

    out_dir = OUTPUT_DIR_TRUTH / "truth_categories"
    out_dir.mkdir(parents=True, exist_ok=True)

    for obj_name in ("jet", "tau"):
        is_true = object_data[obj_name]["is_true"]
        in_overlap = object_data[obj_name]["in_overlap"]

        for var_key, vmin, vmax, step, xlabel_suffix in OBJECT_KINEMATICS_VARIABLES:
            values = object_data[obj_name][var_key]
            bins = np.arange(vmin, vmax + step, step)

            datasets = [
                (values[is_true & in_overlap], "vero + overlap", "tab:green"),
                (values[is_true & ~in_overlap], "vero + isolato", "tab:blue"),
                (values[~is_true & in_overlap], "fake + overlap", "tab:red"),
                (values[~is_true & ~in_overlap], "fake + isolato", "tab:gray"),
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            _plot_hist_curves(ax, datasets, bins, NORMALIZE_TRUTH_HISTOGRAMS)

            _finish_object_plot(
                ax,
                f"{obj_name} {xlabel_suffix}",
                f"{obj_name}: verita' × overlap/isolato (event-level)"
                f"- $\\Delta R < {dr_threshold}$",
                normalize=NORMALIZE_TRUTH_HISTOGRAMS,
            )

            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_TRUTH_HISTOGRAMS else ""
            suffix_cut = f"_cut_{TRUTH_PT_HIST_MAX}" if var_key == "pt" else ""

            out_path = (
                out_dir
                / f"{obj_name}_{var_key}_truth_split_"
                f"{JET_SELECTION_MODE}{suffix}{suffix_cut}.png"
            )

            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[OK] Plot salvato in: {out_path}")


# ======================================================================
# EVENT-LEVEL FRACTIONS OF THE PAIR CATEGORIES
# ======================================================================

def count_event_level_overlap_by_category(pair_info):
    """
    Count, for each truth category, the events containing at least one
    pair of that category.

    The four fractions need not sum to 100%: an event can contain pairs
    of different categories at the same time.

    Parameters
    ----------
    pair_info : dict
        Output of `build_pair_geometry_and_labels`.

    Returns
    -------
    event_overlaps : dict of int
        Events with at least one pair of the category having
        DeltaR < `DR_THRESHOLD_KINEMATICS`.

    event_totals : dict of int
        Events with at least one pair of the category at any DeltaR.
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
        has_overlap = ak.any(pair_mask & overlap_matrix, axis=-1)

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
    Print, for each truth category, the event-level and pair-level
    fraction of the overlap region.

    The event-level denominator is the number of events containing at
    least one pair of the category; the pair-level denominator is the
    number of pairs of the category.

    Parameters
    ----------
    dr_by_category : dict of numpy.ndarray
        Output of `flatten_dr_by_category`, merged over all files.

    event_overlap_counts : dict of int
        Events with at least one overlapping pair, per category.

    event_category_counts : dict of int
        Events with at least one pair, per category.

    dr_threshold : float
        DeltaR threshold defining the overlap.
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
# OBJECT-LEVEL SUMMARY
# ======================================================================

def print_object_level_summary(object_data, dr_threshold):
    """
    Print the number of overlapping and isolated jets and taus, in total
    and separately for true and fake objects.

    Parameters
    ----------
    object_data : dict
        Output of `merge_object_kinematics`.

    dr_threshold : float
        DeltaR threshold defining the overlap; used only in the header.
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

        for truth_name, truth_mask in (("veri", is_true), ("fake", ~is_true)):
            n_truth = int(np.count_nonzero(truth_mask))
            n_truth_overlap = int(np.count_nonzero(truth_mask & in_overlap))

            if n_truth > 0:
                frac = 100.0 * n_truth_overlap / n_truth
                print(
                    f"   {truth_name:24s}: "
                    f"{n_truth_overlap:8d} overlap / "
                    f"{n_truth:8d} totali "
                    f"({frac:6.2f}%)"
                )
            else:
                print(f"   {truth_name:24s}: nessun oggetto")


# ======================================================================
# MAIN
# ======================================================================

def main():
    """
    Run the truth-vs-reco overlap study on all available files.

    For each file the selected jets and taus are labelled, the pair
    matrix is built and the pair-level, event-level and object-level
    quantities are accumulated. After the loop the results are merged,
    printed and plotted.
    """
    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
        return

    branches = [
        TAU_ETA_BRANCH,
        TAU_PHI_BRANCH,
        TAU_PT_BRANCH,
        TAU_IS_ANALYSIS_BRANCH,
        JET_ETA_BRANCH,
        JET_PHI_BRANCH,
        JET_PT_BRANCH,
        JET_IS_ANALYSIS_BRANCH,
        JET_TRUTH_LABEL_BRANCH,
    ]

    if JET_SELECTION_MODE == "btag85":
        branches.append(JET_BTAG_BRANCH)

    if TAU_SELECTION_MODE == "score85":
        branches.append(TAU_EFF_SCORE_BRANCH)

    if TRUTH_MODE_TAU == "label":
        branches.append(TAU_TRUTH_MATCH_BRANCH)
    elif TRUTH_MODE_TAU == "geometric":
        branches += [TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH]
    else:
        raise RuntimeError(
            f"TRUTH_MODE_TAU='{TRUTH_MODE_TAU}' non gestito."
        )

    jet_dr_truth_parts = []
    tau_dr_truth_parts = []

    cat_totals = {cat: 0 for cat in CATEGORY_KEYS}
    n_overlap_totals = {thr: 0 for thr in TRUTH_DR_THRESHOLDS}

    dr_category_parts = []
    object_kinematics_parts = []

    event_overlap_counts = {cat: 0 for cat in CATEGORY_KEYS}
    event_category_counts = {cat: 0 for cat in CATEGORY_KEYS}

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]

        keys = set(tree.keys())
        missing = [b for b in branches if b not in keys]
        if missing:
            print(f"[WARNING] {item['file_name']}: branch mancanti: {missing}")
            print(
                "    Verificare i nomi dei branch in configurazione. "
                "File ignorato."
            )
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel_analysis = get_analysis_selection(
            tree, JET_IS_ANALYSIS_BRANCH, n_entries
        )
        tau_sel_analysis = get_analysis_selection(
            tree, TAU_IS_ANALYSIS_BRANCH, n_entries
        )

        if JET_SELECTION_MODE == "all":
            jet_sel = jet_sel_analysis
        else:
            jet_btag_sel = ak.fill_none(a[JET_BTAG_BRANCH], False) != 0
            jet_sel = jet_sel_analysis & jet_btag_sel

        if TAU_SELECTION_MODE == "all":
            tau_sel = tau_sel_analysis
        else:
            tau_score_sel = ak.fill_none(
                a[TAU_EFF_SCORE_BRANCH], -np.inf
            ) >= TAU_SCORE_WP85_THRESHOLD
            tau_sel = tau_sel_analysis & tau_score_sel

        jet_label, tau_label, jet_dr_truth, tau_dr_truth = label_jets_and_taus(
            a, jet_sel, tau_sel
        )

        if jet_dr_truth is not None:
            jet_dr_truth_parts.append(jet_dr_truth)
        if tau_dr_truth is not None:
            tau_dr_truth_parts.append(tau_dr_truth)

        jet_eta = a[JET_ETA_BRANCH][jet_sel]
        jet_phi = a[JET_PHI_BRANCH][jet_sel]
        jet_pt = a[JET_PT_BRANCH][jet_sel]

        tau_eta = a[TAU_ETA_BRANCH][tau_sel]
        tau_phi = a[TAU_PHI_BRANCH][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]

        pair_info = build_pair_geometry_and_labels(
            jet_eta, jet_phi, jet_label,
            tau_eta, tau_phi, tau_label,
        )

        for thr in TRUTH_DR_THRESHOLDS:
            cats, n_tot = classify_overlapping_pairs(pair_info, thr)

            n_overlap_totals[thr] += n_tot

            if thr == TRUTH_DR_THRESHOLDS[0]:
                for cat in CATEGORY_KEYS:
                    cat_totals[cat] += cats[cat]

        dr_category_parts.append(flatten_dr_by_category(pair_info))

        file_event_overlap, file_event_total = (
            count_event_level_overlap_by_category(pair_info)
        )

        for cat in CATEGORY_KEYS:
            event_overlap_counts[cat] += file_event_overlap[cat]
            event_category_counts[cat] += file_event_total[cat]

        object_kinematics_parts.append(
            classify_objects_kinematics(
                jet_pt, jet_eta, jet_phi, jet_label,
                tau_pt, tau_eta, tau_phi, tau_label,
                pair_info, DR_THRESHOLD_KINEMATICS,
            )
        )

    section("SOGLIA DI TRUTH-MATCHING (punto 3)")

    if jet_dr_truth_parts:
        jet_truth_summary = summarize(
            ak.concatenate(jet_dr_truth_parts), "DeltaR(reco jet, truth b)"
        )
        print_shoulder_table(jet_truth_summary, "DeltaR(reco jet, truth b)")

    if tau_dr_truth_parts:
        tau_truth_summary = summarize(
            ak.concatenate(tau_dr_truth_parts), "DeltaR(reco tau, truth tau)"
        )
        print_shoulder_table(tau_truth_summary, "DeltaR(reco tau, truth tau)")

    primary_overlap_threshold = DR_THRESHOLD_KINEMATICS

    section(
        "CLASSIFICAZIONE COPPIE OVERLAPPANTI "
        f"(soglia={primary_overlap_threshold}) - punto 5"
    )

    n_class_total = sum(cat_totals.values())

    for cat in CATEGORY_KEYS:
        value = cat_totals[cat]
        fraction = 100.0 * value / n_class_total if n_class_total > 0 else 0.0
        print(f"   {cat:30s}: {value:8d} ({fraction:6.2f}%)")

    print(f"   {'TOTALE':30s}: {n_class_total:8d}")

    section("CONTROLLO DI CONSISTENZA CON obj_3_1.py")

    for thr in TRUTH_DR_THRESHOLDS:
        print(
            f"   soglia {thr}: coppie overlappanti "
            f"(questo script) = {n_overlap_totals[thr]} "
            "-> confrontare con 'coppie (jet,tau) sovrapposte' "
            "stampato da obj_3_1.py per la stessa soglia "
            "(devono coincidere esattamente)"
        )

    dr_by_category = {cat: [] for cat in CATEGORY_KEYS}

    for part in dr_category_parts:
        for cat in CATEGORY_KEYS:
            dr_by_category[cat].append(part[cat])

    for cat in CATEGORY_KEYS:
        arrays = dr_by_category[cat]
        dr_by_category[cat] = np.concatenate(arrays) if arrays else np.array([])

    object_data = merge_object_kinematics(object_kinematics_parts)

    save_dr_histogram_by_category(dr_by_category)

    print_overlap_percentages(
        dr_by_category,
        event_overlap_counts,
        event_category_counts,
        DR_THRESHOLD_KINEMATICS,
    )

    print_object_level_summary(object_data, DR_THRESHOLD_KINEMATICS)

    save_object_kinematics_all_overlap_vs_isolated(
        object_data, DR_THRESHOLD_KINEMATICS
    )

    save_object_kinematics_truth_split(object_data, DR_THRESHOLD_KINEMATICS)


if __name__ == "__main__":
    main()