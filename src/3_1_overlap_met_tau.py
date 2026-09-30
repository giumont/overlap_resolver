"""
Objective 3.1 - Tau-MET variables of overlapping jet-tau pairs, split by
truth category.

Global goal
-----------
Extend the comparison of the kinematic characteristics of the
overlapping RECO (jet, tau) pairs (DeltaR < ``DR_THRESHOLD_KINEMATICS``)
with the physical correlation between the tau candidate and the missing
transverse energy (MET) of the event. Two variables are studied:

- the MET projected along the tau direction, MET_parallel;
- the transverse mass m_T of the tau-MET system.

Distributions are shown separately for the four truth categories of the
b-jet / hadronic-tau identification: (TF) true b-jet and fake tau, (FT)
fake b-jet and true tau, (TT) both true, (FF) both fake. This script
always uses the four standard categories (``AGGREGATE_CATEGORIES`` is
not applied).

The variable computations (`compute_dphi_met`, `compute_tau_met_proj`,
`compute_tau_mt`) are also imported by ``overlap_kinematics.py``.
Truth labelling and plotting helpers come from
``truth_vs_reco_params.py``. All configuration is defined in
``params.py``.
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
    JET_ETA_BRANCH, JET_PHI_BRANCH, JET_IS_ANALYSIS_BRANCH,
    TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_PT_BRANCH, TAU_IS_ANALYSIS_BRANCH,
    MET_BRANCH, MET_PHI_BRANCH,
    JET_SELECTION_MODE, JET_BTAG_BRANCH,
    TAU_SELECTION_MODE, TAU_SCORE_BRANCH, TAU_SCORE_WP85_THRESHOLD,
    JET_TRUTH_LABEL_BRANCH, TAU_TRUTH_MATCH_BRANCH,
    DR_THRESHOLD_KINEMATICS, NORMALIZE_PAIR_HISTOGRAMS,
    CATEGORY_LABELS, CATEGORY_COLORS, CATEGORY_KEYS,
    VARIABLE_PLOT_CONFIG_MET, TAU_MET_VARIABLES,
)
from obj_3_1 import section, load_files, get_analysis_selection, delta_r
from truth_vs_reco_params import (
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)


def compute_dphi_met(tau_phi, met_phi):
    """
    Azimuthal difference between the tau and the MET.

    Parameters
    ----------
    tau_phi : array-like of float
        Tau azimuth in radians.

    met_phi : array-like of float
        MET azimuth in radians (broadcastable against ``tau_phi``).

    Returns
    -------
    array-like of float
        ``tau_phi - met_phi`` wrapped into [-pi, pi].
    """
    return (tau_phi - met_phi + np.pi) % (2 * np.pi) - np.pi


def compute_tau_met_proj(tau_phi, met_val, met_phi):
    """
    MET projected along the tau direction, MET_parallel.

    Parameters
    ----------
    tau_phi : array-like of float
        Tau azimuth in radians.

    met_val : array-like of float
        MET magnitude (MeV).

    met_phi : array-like of float
        MET azimuth in radians.

    Returns
    -------
    array-like of float
        ``met_val * cos(Delta_phi(tau, MET))`` in MeV.
    """
    dphi_met = compute_dphi_met(tau_phi, met_phi)
    return met_val * np.cos(dphi_met)


def compute_tau_mt(tau_pt, tau_phi, met_val, met_phi):
    """
    Transverse mass of the tau-MET system.

    Parameters
    ----------
    tau_pt : array-like of float
        Tau transverse momentum (MeV).

    tau_phi : array-like of float
        Tau azimuth in radians.

    met_val : array-like of float
        MET magnitude (MeV).

    met_phi : array-like of float
        MET azimuth in radians.

    Returns
    -------
    array-like of float
        ``sqrt(2 * tau_pt * met_val * (1 - cos(Delta_phi)))`` in MeV,
        with the argument of the square root clipped at zero.
    """
    dphi_met = compute_dphi_met(tau_phi, met_phi)
    return np.sqrt(np.maximum(0, 2 * tau_pt * met_val * (1 - np.cos(dphi_met))))


def build_pair_met_kinematics(
    jet_eta, jet_phi, jet_label,
    tau_eta, tau_phi, tau_label,
    tau_met_proj, tau_mt,
):
    """
    Build, for every jet-tau combination of each event, the pair DeltaR,
    the tau-MET variables and the truth labels.

    The jet-tau cartesian product (``nested=True``) gives arrays of
    shape ``[event][jet][tau]``; the per-tau quantities are repeated
    along the jet axis.

    Parameters
    ----------
    jet_eta, jet_phi : awkward.Array
        Eta and azimuth of the selected jets, shape ``[event][jet]``.

    jet_label : awkward.Array of bool
        True if the jet is a true b-jet.

    tau_eta, tau_phi : awkward.Array
        Eta and azimuth of the selected taus, shape ``[event][tau]``.

    tau_label : awkward.Array of bool
        True if the tau is a true hadronic tau.

    tau_met_proj, tau_mt : awkward.Array
        Projected MET and transverse mass of each selected tau
        (see `compute_tau_met_proj`, `compute_tau_mt`).

    Returns
    -------
    dict of str -> awkward.Array
        Arrays of shape ``[event][jet][tau]`` with keys ``pair_dr``,
        ``tau_met_proj``, ``tau_mt``, ``jet_label``, ``tau_label``.
    """
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))

    _, tau_met_proj_ct = ak.unzip(ak.cartesian([jet_eta, tau_met_proj], nested=True))
    _, tau_mt_ct = ak.unzip(ak.cartesian([jet_eta, tau_mt], nested=True))

    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)

    return {
        "pair_dr": dr_matrix,
        "tau_met_proj": tau_met_proj_ct,
        "tau_mt": tau_mt_ct,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }


def get_overlapping_pairs_met(pair_info, dr_thr):
    """
    Select the overlapping pairs and split them by truth category.

    A pair overlaps if ``pair_dr < dr_thr``. The four standard
    categories (TF, FT, TT, FF) are always used.

    Parameters
    ----------
    pair_info : dict of str -> awkward.Array
        Output of `build_pair_met_kinematics`.

    dr_thr : float
        Maximum DeltaR of an overlapping pair.

    Returns
    -------
    dict of str -> dict of str -> numpy.ndarray
        ``result[category][variable]`` with ``variable`` in
        ``{"tau_met_proj", "tau_mt"}``: flat arrays of the values for
        the overlapping pairs of that category.
    """
    overlap_mask = pair_info["pair_dr"] < dr_thr

    def flat_overlap(key):
        """Flatten ``pair_info[key]`` and keep the overlapping pairs only."""
        flat_arr = ak.to_numpy(ak.flatten(pair_info[key], axis=None))
        flat_m = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat_arr[flat_m]

    jet_lab = flat_overlap("jet_label").astype(bool)
    tau_lab = flat_overlap("tau_label").astype(bool)

    cat_masks = {
        "a_jet_true_tau_fake": jet_lab & ~tau_lab,
        "b_jet_false_tau_true": ~jet_lab & tau_lab,
        "c_jet_true_tau_true": jet_lab & tau_lab,
        "d_jet_false_tau_false": ~jet_lab & ~tau_lab,
    }

    res = {cat: {} for cat in cat_masks.keys()}
    variables = ["tau_met_proj", "tau_mt"]

    for cat, cmask in cat_masks.items():
        for var in variables:
            res[cat][var] = flat_overlap(var)[cmask]

    return res


def merge_met_kinematics(parts_list):
    """
    Concatenate the per-file results of `get_overlapping_pairs_met`.

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
    variables = ["tau_met_proj", "tau_mt"]
    merged = {cat: {var: [] for var in variables} for cat in CATEGORY_KEYS}

    for part in parts_list:
        for cat in CATEGORY_KEYS:
            for var in variables:
                merged[cat][var].append(part[cat][var])

    for cat in CATEGORY_KEYS:
        for var in variables:
            arrays = merged[cat][var]
            merged[cat][var] = np.concatenate(arrays) if arrays else np.array([])

    return merged


def save_met_kinematics_plots(merged_data, dr_threshold):
    """
    Save one histogram per tau-MET variable, with one curve per truth
    category.

    Variables are those in ``TAU_MET_VARIABLES``; y scale and output
    folder come from ``VARIABLE_PLOT_CONFIG_MET``. Nothing is done if
    matplotlib is unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_met_kinematics`.

    dr_threshold : float
        DeltaR threshold used to select the pairs; shown in the plot
        titles.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    for var_key, vmin, vmax, step, xlabel in TAU_MET_VARIABLES:
        var_id = f"tau_{var_key}"
        config = VARIABLE_PLOT_CONFIG_MET[var_id]

        out_dir = Path(config["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        bins = np.arange(vmin, vmax + step, step)
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets = []
        for cat in CATEGORY_KEYS:
            values = merged_data[cat][var_id]
            datasets.append((values, CATEGORY_LABELS[cat], CATEGORY_COLORS[cat]))

        _plot_hist_curves(ax, datasets, bins)

        _finish_object_plot(
            ax,
            xlabel,
            f"Relazione $\\tau$-MET in coppie in overlap ($\\Delta R < {dr_threshold}$)",
            normalize=NORMALIZE_PAIR_HISTOGRAMS,
        )

        ax.set_yscale(config["y_scale"])
        fig.tight_layout()

        suffix = "_normalized" if NORMALIZE_PAIR_HISTOGRAMS else ""
        out_path = (
            out_dir
            / f"met_kinematics_{var_id}_jet{JET_SELECTION_MODE}_tau{TAU_SELECTION_MODE}{suffix}.png"
        )

        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Plot saved to: {out_path}")


def main():
    """
    Entry point of the tau-MET study.

    For each input file: read the branches, apply the analysis-level
    (and optional working point) selections, assign the truth labels,
    compute MET_parallel and m_T for each selected tau, build the
    jet-tau pairs and keep the overlapping ones by truth category. The
    per-file results are merged and plotted. Files missing any required
    branch are skipped. Unlike the other scripts, the b-tag score, tau
    score and truth branches are always required.

    Returns
    -------
    None
    """
    loaded = load_files()
    if not loaded:
        return

    branches = [
        TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_PT_BRANCH, TAU_IS_ANALYSIS_BRANCH,
        JET_ETA_BRANCH, JET_PHI_BRANCH, JET_IS_ANALYSIS_BRANCH,
        JET_BTAG_BRANCH, JET_TRUTH_LABEL_BRANCH,
        TAU_SCORE_BRANCH, TAU_TRUTH_MATCH_BRANCH,
        MET_BRANCH, MET_PHI_BRANCH,
    ]

    pair_met_parts = []

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
        elif JET_SELECTION_MODE == "btag85":
            jet_sel = jet_sel_analysis & (ak.fill_none(a[JET_BTAG_BRANCH], False) != 0)
        else:
            raise ValueError("JET_SELECTION_MODE non valido. Scegli 'all' o 'btag85'.")

        if TAU_SELECTION_MODE == "all":
            tau_sel = tau_sel_analysis
        elif TAU_SELECTION_MODE == "score85":
            tau_sel = tau_sel_analysis & (
                ak.fill_none(a[TAU_SCORE_BRANCH], -np.inf) >= TAU_SCORE_WP85_THRESHOLD
            )
        else:
            raise ValueError("TAU_SELECTION_MODE non valido. Scegli 'all' o 'score85'.")

        jet_label, tau_label, _, _ = label_jets_and_taus(a, jet_sel, tau_sel)

        jet_eta = a[JET_ETA_BRANCH][jet_sel]
        jet_phi = a[JET_PHI_BRANCH][jet_sel]

        tau_eta = a[TAU_ETA_BRANCH][tau_sel]
        tau_phi = a[TAU_PHI_BRANCH][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]

        met_val = a[MET_BRANCH]
        met_phi = a[MET_PHI_BRANCH]

        tau_met_proj = compute_tau_met_proj(tau_phi, met_val, met_phi)
        tau_mt = compute_tau_mt(tau_pt, tau_phi, met_val, met_phi)

        pair_info = build_pair_met_kinematics(
            jet_eta, jet_phi, jet_label,
            tau_eta, tau_phi, tau_label,
            tau_met_proj, tau_mt,
        )

        pair_met_parts.append(
            get_overlapping_pairs_met(pair_info, DR_THRESHOLD_KINEMATICS)
        )

    merged_met_kinematics = merge_met_kinematics(pair_met_parts)

    section("TAU-MET PHYSICAL CORRELATION PLOTS")
    save_met_kinematics_plots(merged_met_kinematics, DR_THRESHOLD_KINEMATICS)


if __name__ == "__main__":
    main()