"""
Objective 3.1 - Event metadata of overlapping jet-tau pairs, split by
truth category.

Global goal
-----------
Check whether the overlap between RECO jets and taus
(DeltaR < ``DR_THRESHOLD_KINEMATICS``) depends on the pile-up
conditions of the event. For each overlapping pair the event-level
variables

- number of primary vertices (``nPrimaryVertices``),
- actual number of interactions per crossing,
- average number of interactions per crossing,

are histogrammed and plotted against the pair DeltaR, separately for
each truth category of the b-jet / hadronic-tau identification: (TF)
true b-jet and fake tau, (FT) fake b-jet and true tau, (TT) both true,
(FF) both fake. With ``AGGREGATE_CATEGORIES``, (TT) and (FF) are merged
into "other".

Truth labelling and plotting helpers come from
``truth_vs_reco_params.py``; selection utilities and DeltaR from
``obj_3_1.py``. All configuration is defined in ``params.py``.
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

from params import (
    JET_ETA_BRANCH, JET_PHI_BRANCH, JET_PT_BRANCH, JET_IS_ANALYSIS_BRANCH,
    TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_PT_BRANCH, TAU_IS_ANALYSIS_BRANCH,
    NPV_BRANCH, ACTUAL_MU_BRANCH, AVG_MU_BRANCH,
    JET_SELECTION_MODE, JET_BTAG_BRANCH,
    TAU_SELECTION_MODE, TAU_EFF_SCORE_BRANCH, TAU_SCORE_WP85_THRESHOLD,
    JET_TRUTH_LABEL_BRANCH,
    TRUTH_MODE_TAU, TAU_TRUTH_MATCH_BRANCH,
    TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH,
    AGGREGATE_CATEGORIES, NORMALIZE_PAIR_HISTOGRAMS,
    DR_THRESHOLD_KINEMATICS,
    CATEGORY_LABELS, CATEGORY_COLORS, CATEGORY_KEYS,
    CATEGORY_LABELS_AGG, CATEGORY_COLORS_AGG, CATEGORY_KEYS_AGG,
    OUTPUT_DIR_METADATA, VARIABLE_PLOT_CONFIG_METADATA, METADATA_VARIABLES,
)
from obj_3_1_geometric_overlap import section, load_files, get_analysis_selection, delta_r
from src.other_code.truth_vs_reco_params import (
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)


def build_pair_metadata_and_labels(
    jet_pt, jet_eta, jet_phi, jet_label,
    tau_pt, tau_eta, tau_phi, tau_label,
    n_pv, actual_mu, avg_mu,
):
    """
    Build, for every jet-tau combination of each event, the pair DeltaR,
    the event metadata and the truth labels.

    The jet-tau cartesian product (``nested=True``) gives arrays of
    shape ``[event][jet][tau]``; the per-event metadata are broadcast to
    the same shape.

    Parameters
    ----------
    jet_pt, jet_eta, jet_phi : awkward.Array
        Kinematics of the selected jets, shape ``[event][jet]``.

    jet_label : awkward.Array of bool
        True if the jet is a true b-jet.

    tau_pt, tau_eta, tau_phi : awkward.Array
        Kinematics of the selected taus, shape ``[event][tau]``.

    tau_label : awkward.Array of bool
        True if the tau is a true hadronic tau.

    n_pv, actual_mu, avg_mu : awkward.Array
        Per-event number of primary vertices, actual and average
        interactions per crossing, shape ``[event]``.

    Returns
    -------
    dict of str -> awkward.Array
        Arrays of shape ``[event][jet][tau]`` with keys ``pair_dr``,
        ``nPrimaryVertices``, ``actualInteractionsPerCrossing``,
        ``averageInteractionsPerCrossing``, ``jet_label``, ``tau_label``.
    """
    jet_pt_ct, tau_pt_ct = ak.unzip(ak.cartesian([jet_pt, tau_pt], nested=True))
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    n_pv_ct = ak.broadcast_arrays(n_pv, jet_pt_ct)[0]
    actual_mu_ct = ak.broadcast_arrays(actual_mu, jet_pt_ct)[0]
    avg_mu_ct = ak.broadcast_arrays(avg_mu, jet_pt_ct)[0]

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)

    return {
        "pair_dr": dr_matrix,
        "nPrimaryVertices": n_pv_ct,
        "actualInteractionsPerCrossing": actual_mu_ct,
        "averageInteractionsPerCrossing": avg_mu_ct,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }


def get_overlapping_pairs_metadata(pair_info, dr_thr):
    """
    Select the overlapping pairs and split them by truth category.

    A pair overlaps if ``pair_dr < dr_thr``. The categories follow
    ``AGGREGATE_CATEGORIES``: the four combinations (TF, FT, TT, FF) or
    TF, FT and "other" (TT and FF together).

    Parameters
    ----------
    pair_info : dict of str -> awkward.Array
        Output of `build_pair_metadata_and_labels`.

    dr_thr : float
        Maximum DeltaR of an overlapping pair.

    Returns
    -------
    dict of str -> dict of str -> numpy.ndarray
        ``result[category][variable]`` with ``variable`` in
        ``pair_dr``, ``nPrimaryVertices``,
        ``actualInteractionsPerCrossing``,
        ``averageInteractionsPerCrossing``: flat arrays of the values
        for the overlapping pairs of that category.
    """
    overlap_mask = pair_info["pair_dr"] < dr_thr

    def flat_overlap(key):
        """Flatten ``pair_info[key]`` and keep the overlapping pairs only."""
        flat_arr = ak.to_numpy(ak.flatten(pair_info[key], axis=None))
        flat_m = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat_arr[flat_m]

    jet_lab = flat_overlap("jet_label").astype(bool)
    tau_lab = flat_overlap("tau_label").astype(bool)

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
    variables = [
        "pair_dr", "nPrimaryVertices",
        "actualInteractionsPerCrossing", "averageInteractionsPerCrossing",
    ]

    for cat, cmask in cat_masks.items():
        for var in variables:
            res[cat][var] = flat_overlap(var)[cmask]

    return res


def merge_pair_metadata(parts_list):
    """
    Concatenate the per-file results of `get_overlapping_pairs_metadata`.

    Parameters
    ----------
    parts_list : list of dict
        One ``result[category][variable]`` dictionary per processed
        file.

    Returns
    -------
    dict of str -> dict of str -> numpy.ndarray
        Same structure as the input, with the arrays of all files
        concatenated (empty arrays if ``parts_list`` is empty).
    """
    variables = [
        "pair_dr", "nPrimaryVertices",
        "actualInteractionsPerCrossing", "averageInteractionsPerCrossing",
    ]
    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS

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


def save_metadata_plots(merged_data, dr_threshold):
    """
    Save one histogram per event-metadata variable, with one curve per
    truth category.

    Variables are those in ``METADATA_VARIABLES``; y scale and output
    folder come from ``VARIABLE_PLOT_CONFIG_METADATA`` (with an
    ``aggregated`` subfolder if ``AGGREGATE_CATEGORIES``). Nothing is
    done if matplotlib is unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_pair_metadata`.

    dr_threshold : float
        DeltaR threshold used to select the pairs; shown in the plot
        titles.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    for var_key, vmin, vmax, step, xlabel in METADATA_VARIABLES:

        config = VARIABLE_PLOT_CONFIG_METADATA.get(
            var_key, {"y_scale": "linear", "out_dir": OUTPUT_DIR_METADATA}
        )
        out_dir = Path(config["out_dir"])

        if AGGREGATE_CATEGORIES:
            out_dir = out_dir / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)

        y_scale = config["y_scale"]

        bins = np.arange(vmin, vmax + step, step)
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets = []
        for cat in cat_keys:
            values = merged_data[cat][var_key]
            datasets.append((values, labels[cat], colors[cat]))

        _plot_hist_curves(ax, datasets, bins)

        _finish_object_plot(
            ax,
            xlabel,
            f"Event Metadata in coppie (pair-level) - $\\Delta R < {dr_threshold}$",
            normalize=NORMALIZE_PAIR_HISTOGRAMS,
        )

        ax.set_yscale(y_scale)
        fig.tight_layout()

        suffix = "_normalized" if NORMALIZE_PAIR_HISTOGRAMS else ""
        suffix_agg = "_aggregated" if AGGREGATE_CATEGORIES else ""

        out_path = (
            out_dir
            / f"pair_metadata_{var_key}_"
            f"{JET_SELECTION_MODE}{suffix}{suffix_agg}.png"
        )

        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Plot saved to: {out_path}")


def save_scatter_plots_metadata(merged_data, dr_threshold):
    """
    Save scatter plots of each event-metadata variable versus the pair
    DeltaR, with one colour per truth category (legend entries include
    the number of pairs of each category).

    Files are written to ``OUTPUT_DIR_METADATA / "scatters"``. Nothing
    is done if matplotlib is unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_pair_metadata`.

    dr_threshold : float
        DeltaR threshold used to select the pairs; shown in the plot
        titles.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    out_dir = OUTPUT_DIR_METADATA / "scatters"
    out_dir.mkdir(parents=True, exist_ok=True)

    scatters_config = [
        ("nPrimaryVertices", "N. Primary Vertices"),
        ("actualInteractionsPerCrossing", r"Actual $\mu$"),
        ("averageInteractionsPerCrossing", r"Average $\mu$"),
    ]

    for var_key, ylabel in scatters_config:
        fig, ax = plt.subplots(figsize=(8, 6))

        for cat in cat_keys:
            dr_vals = merged_data[cat]["pair_dr"]
            meta_vals = merged_data[cat][var_key]
            count = len(dr_vals)

            ax.scatter(
                dr_vals,
                meta_vals,
                label=f"{labels[cat]} (N={count})",
                color=colors[cat],
                alpha=0.6,
                s=15,
                edgecolors="none",
            )

        ax.set_xlabel(r"$\Delta R$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Scatter: $\\Delta R$ vs {ylabel} ($\\Delta R < {dr_threshold}$)")
        ax.set_yscale("linear")
        ax.legend(title="Categorie", fontsize=9, loc="best")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"scatter_dr_vs_{var_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Scatter plot saved to: {out_path}")


def main():
    """
    Entry point of the event-metadata study.

    For each input file: read the branches, apply the analysis-level
    (and optional working point) selections, assign the truth labels,
    build the jet-tau pairs with the event metadata and keep the
    overlapping ones (DeltaR < ``DR_THRESHOLD_KINEMATICS``) by truth
    category. The per-file results are merged and used to save the
    histograms and the scatter plots. Files missing any required branch
    are skipped.

    Returns
    -------
    None
    """
    loaded = load_files()
    if not loaded:
        print("No input files available.")
        return

    branches = [
        TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_PT_BRANCH, TAU_IS_ANALYSIS_BRANCH,
        JET_ETA_BRANCH, JET_PHI_BRANCH, JET_PT_BRANCH, JET_IS_ANALYSIS_BRANCH,
        JET_TRUTH_LABEL_BRANCH,
        NPV_BRANCH, ACTUAL_MU_BRANCH, AVG_MU_BRANCH,
    ]

    if JET_SELECTION_MODE == "btag85":
        branches.append(JET_BTAG_BRANCH)

    if TAU_SELECTION_MODE == "score85":
        branches.append(TAU_EFF_SCORE_BRANCH)

    if TRUTH_MODE_TAU == "label":
        branches.append(TAU_TRUTH_MATCH_BRANCH)
    elif TRUTH_MODE_TAU == "geometric":
        branches.extend([TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH])

    pair_metadata_parts = []

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]

        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel_analysis = get_analysis_selection(tree, JET_IS_ANALYSIS_BRANCH, n_entries)
        tau_sel_analysis = get_analysis_selection(tree, TAU_IS_ANALYSIS_BRANCH, n_entries)

        if JET_SELECTION_MODE == "all":
            jet_sel = jet_sel_analysis
        else:
            jet_sel = jet_sel_analysis & (ak.fill_none(a[JET_BTAG_BRANCH], False) != 0)

        if TAU_SELECTION_MODE == "all":
            tau_sel = tau_sel_analysis
        else:
            tau_sel = tau_sel_analysis & (
                ak.fill_none(a[TAU_EFF_SCORE_BRANCH], -np.inf) >= TAU_SCORE_WP85_THRESHOLD
            )

        jet_label, tau_label, _, _ = label_jets_and_taus(a, jet_sel, tau_sel)

        jet_eta = a[JET_ETA_BRANCH][jet_sel]
        jet_phi = a[JET_PHI_BRANCH][jet_sel]
        jet_pt = a[JET_PT_BRANCH][jet_sel]

        tau_eta = a[TAU_ETA_BRANCH][tau_sel]
        tau_phi = a[TAU_PHI_BRANCH][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]

        n_pv = a[NPV_BRANCH]
        actual_mu = a[ACTUAL_MU_BRANCH]
        avg_mu = a[AVG_MU_BRANCH]

        pair_info = build_pair_metadata_and_labels(
            jet_pt, jet_eta, jet_phi, jet_label,
            tau_pt, tau_eta, tau_phi, tau_label,
            n_pv, actual_mu, avg_mu,
        )

        pair_metadata_parts.append(
            get_overlapping_pairs_metadata(pair_info, DR_THRESHOLD_KINEMATICS)
        )

    merged_pair_metadata = merge_pair_metadata(pair_metadata_parts)

    section("PAIR-LEVEL METADATA PLOTS")
    save_metadata_plots(merged_pair_metadata, DR_THRESHOLD_KINEMATICS)

    section("METADATA SCATTER PLOTS")
    save_scatter_plots_metadata(merged_pair_metadata, DR_THRESHOLD_KINEMATICS)


if __name__ == "__main__":
    main()