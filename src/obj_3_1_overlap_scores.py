"""
Objective 3.1 - Offline identification scores of overlapping jet-tau
pairs, split by truth category.

Global goal
-----------
Compare the identification discriminants reconstructed by the offline
algorithms for the RECO (jet, tau) pairs that overlap geometrically
(DeltaR < ``DR_THRESHOLD_SCORES``): the continuous b-tagging quantile of
the jet (GN2v01) and the tau identification score selected by
``TAU_ID_SCORE_BRANCH``. Distributions are shown separately for the four
truth categories of the b-jet / hadronic-tau identification: (TF) true
b-jet and fake tau, (FT) fake b-jet and true tau, (TT) both true, (FF)
both fake.

The tau score plot is annotated with working points computed
dynamically: for each target efficiency in ``TARGET_EFFICIENCIES`` the
score threshold is set on the whole population of true hadronic taus
(not only the overlapping ones). These working points are the natural
starting point for Objective 3.2 (quality of the classification and
manual combination of the discriminants).

Only analysis-level jets and taus are used (no working point
selection). All configuration is defined in ``params.py``.
"""

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
    JET_SCORE_BRANCH, TAU_ID_SCORE_BRANCH,
    JET_TRUTH_LABEL_BRANCH, JET_TRUTH_LABEL_B_VALUE, TAU_TRUTH_MATCH_BRANCH,
    DR_THRESHOLD_SCORES, OUTPUT_DIR_SCORES,
    TARGET_EFFICIENCIES, WP_COLORS, JET_QUANTILES_MAP,
    CATEGORY_LABELS, CATEGORY_COLORS,
)
from obj_3_1_geometric_overlap import load_files, get_analysis_selection, delta_r, section


def compute_working_points(score_true, target_efficiencies, colors):
    """
    Compute the score thresholds corresponding to target efficiencies.

    The efficiency is defined on the whole population of true hadronic
    taus: the threshold for efficiency ``e`` is the (100 - e)-th
    percentile of their score, so that a fraction ``e`` of them lies
    above it.

    Parameters
    ----------
    score_true : numpy.ndarray
        Tau ID scores of the true hadronic taus.

    target_efficiencies : list of int or float
        Target efficiencies in percent.

    colors : list of str
        Plot colour of each working point (same length as
        ``target_efficiencies``).

    Returns
    -------
    list of tuple
        One ``(threshold, label, color)`` per working point, with
        ``label`` of the form ``"85%"``.
    """
    wp_list = []
    for target, color in zip(target_efficiencies, colors):
        thr = float(np.percentile(score_true, 100 - target))
        wp_list.append((thr, f"{target}%", color))
    return wp_list


def build_pair_scores(
    jet_eta, jet_phi, jet_pt, jet_score, jet_label,
    tau_eta, tau_phi, tau_pt, tau_score, tau_label,
):
    """
    Build all jet-tau pairs of each event, keep the overlapping ones
    (DeltaR < ``DR_THRESHOLD_SCORES``) and split them by truth category.

    The jet-tau cartesian product (``nested=True``) gives arrays of
    shape ``[event][jet][tau]``; jet quantities are repeated along the
    tau axis and vice versa.

    Parameters
    ----------
    jet_eta, jet_phi, jet_pt, jet_score : awkward.Array
        Kinematics and b-tag quantile of the selected jets, shape
        ``[event][jet]``.

    jet_label : awkward.Array of bool
        True if the jet is a true b-jet.

    tau_eta, tau_phi, tau_pt, tau_score : awkward.Array
        Kinematics and ID score of the selected taus, shape
        ``[event][tau]``.

    tau_label : awkward.Array of bool
        True if the tau is a true hadronic tau.

    Returns
    -------
    dict of str -> dict of str -> numpy.ndarray
        ``result[category][variable]`` with ``variable`` in
        ``jet_pt``, ``tau_pt``, ``jet_score``, ``tau_score``: flat
        arrays for the overlapping pairs of that category. Categories
        are the keys of ``CATEGORY_LABELS``.
    """
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    jet_pt_ct, tau_pt_ct = ak.unzip(ak.cartesian([jet_pt, tau_pt], nested=True))
    jet_score_ct, tau_score_ct = ak.unzip(ak.cartesian([jet_score, tau_score], nested=True))
    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)
    overlap_mask = dr_matrix < DR_THRESHOLD_SCORES

    def get_flat_overlap(arr):
        """Flatten ``arr`` and keep the overlapping pairs only."""
        flat_arr = ak.to_numpy(ak.flatten(arr, axis=None))
        flat_mask = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat_arr[flat_mask]

    j_pt = get_flat_overlap(jet_pt_ct)
    t_pt = get_flat_overlap(tau_pt_ct)
    j_score = get_flat_overlap(jet_score_ct)
    t_score = get_flat_overlap(tau_score_ct)
    j_lab = get_flat_overlap(jet_label_ct)
    t_lab = get_flat_overlap(tau_label_ct)

    cat_masks = {
        "a_jet_true_tau_fake": (j_lab == True) & (t_lab == False),
        "b_jet_false_tau_true": (j_lab == False) & (t_lab == True),
        "c_jet_true_tau_true": (j_lab == True) & (t_lab == True),
        "d_jet_false_tau_false": (j_lab == False) & (t_lab == False),
    }

    res = {cat: {} for cat in cat_masks.keys()}
    for cat, mask in cat_masks.items():
        res[cat]["jet_pt"] = j_pt[mask]
        res[cat]["tau_pt"] = t_pt[mask]
        res[cat]["jet_score"] = j_score[mask]
        res[cat]["tau_score"] = t_score[mask]

    return res


def merge_score_data(parts_list):
    """
    Concatenate the per-file results of `build_pair_scores`.

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
    variables = ["jet_pt", "tau_pt", "jet_score", "tau_score"]
    merged = {cat: {var: [] for var in variables} for cat in CATEGORY_LABELS.keys()}

    for part in parts_list:
        for cat in CATEGORY_LABELS.keys():
            for var in variables:
                merged[cat][var].append(part[cat][var])

    for cat in CATEGORY_LABELS.keys():
        for var in variables:
            merged[cat][var] = np.concatenate(merged[cat][var]) if merged[cat][var] else np.array([])

    return merged


def save_score_plots(merged_data, tau_wps):
    """
    Save the normalised distributions of the tau ID score and of the jet
    b-tag quantile, with one curve per truth category.

    The tau score plot also shows a vertical line for each working
    point. Files are written to ``OUTPUT_DIR_SCORES``. Nothing is done
    if matplotlib is unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_score_data`.

    tau_wps : list of tuple
        Output of `compute_working_points`.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    OUTPUT_DIR_SCORES.mkdir(parents=True, exist_ok=True)

    fig_tau, ax_tau = plt.subplots(figsize=(9, 6))
    tau_bins = np.linspace(0, 1.0, 100)

    for cat, label in CATEGORY_LABELS.items():
        data = merged_data[cat]["tau_score"]
        color = CATEGORY_COLORS[cat]
        if len(data) > 0:
            ax_tau.hist(
                data,
                bins=tau_bins,
                histtype="step",
                linewidth=1.5,
                density=True,
                label=f"{label} (N={len(data)})",
                color=color,
            )

    for threshold, eff, color in tau_wps:
        ax_tau.axvline(x=threshold, color=color, linestyle="--", linewidth=1.2, label=f"WP {eff} ({threshold:.3f})")

    ax_tau.set_xlabel(f"Tau ID Score\n({TAU_ID_SCORE_BRANCH})")
    ax_tau.set_ylabel("Densità normalizzata")
    ax_tau.set_title(f"Distribuzione Tau ID Score per categorie di verità ($\\Delta R < {DR_THRESHOLD_SCORES}$)")
    ax_tau.legend(fontsize=8, loc="upper right", ncol=2)
    ax_tau.grid(alpha=0.3)

    out_tau = OUTPUT_DIR_SCORES / f"tau_score_{TAU_ID_SCORE_BRANCH}_truth_categories_branch.png"
    fig_tau.tight_layout()
    fig_tau.savefig(out_tau, dpi=150)
    plt.close(fig_tau)
    print(f"[OK] Tau score plot saved to: {out_tau}")

    fig_jet, ax_jet = plt.subplots(figsize=(9, 6))
    jet_bins = np.arange(-1.5, 7.5, 1.0)

    for cat, label in CATEGORY_LABELS.items():
        data = merged_data[cat]["jet_score"]
        color = CATEGORY_COLORS[cat]
        if len(data) > 0:
            ax_jet.hist(
                data,
                bins=jet_bins,
                histtype="step",
                linewidth=1.5,
                density=True,
                label=f"{label} (N={len(data)})",
                color=color,
            )

    ax_jet.set_xticks([-1, 1, 2, 3, 4, 5, 6])
    ax_jet.set_xticklabels([JET_QUANTILES_MAP[k] for k in [-1, 1, 2, 3, 4, 5, 6]])
    ax_jet.set_xlabel(f"Jet b-tag Target Efficiency Quantile\n({JET_SCORE_BRANCH})")
    ax_jet.set_ylabel("Densità normalizzata")
    ax_jet.set_title(f"Distribuzione b-tag quantiles per categorie di verità ($\\Delta R < {DR_THRESHOLD_SCORES}$)")
    ax_jet.legend(fontsize=9, loc="upper right")
    ax_jet.grid(alpha=0.3, axis="y")

    out_jet = OUTPUT_DIR_SCORES / "jet_quantile_truth_categories.png"
    fig_jet.tight_layout()
    fig_jet.savefig(out_jet, dpi=150)
    plt.close(fig_jet)
    print(f"[OK] Jet score plot saved to: {out_jet}")


def save_score_scatters(merged_data):
    """
    Save scatter plots of the ID scores versus the object pT: jet b-tag
    quantile vs jet pT and tau ID score vs tau pT.

    Each figure is a 2x2 grid with one panel per truth category. The
    jet quantile is discrete, so a random vertical jitter (unseeded) is
    added for display only, and the marker transparency adapts to the
    number of points. Files are written to
    ``OUTPUT_DIR_SCORES / "scatters"``. Nothing is done if matplotlib is
    unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_score_data`.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    out_dir = OUTPUT_DIR_SCORES / "scatters"
    out_dir.mkdir(parents=True, exist_ok=True)

    scatters_config = [
        ("jet_pt", "jet_score", "pT Jet [MeV]", f"Jet b-tag Target Efficiency Quantile\n({JET_SCORE_BRANCH})"),
        ("tau_pt", "tau_score", "pT Tau [MeV]", f"Tau ID Score\n({TAU_ID_SCORE_BRANCH})"),
    ]

    cat_keys = list(CATEGORY_LABELS.keys())

    for pt_key, score_key, xlabel, ylabel in scatters_config:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        is_jet_score = (score_key == "jet_score")

        for idx, cat in enumerate(cat_keys):
            ax = axes_flat[idx]
            pt_vals = merged_data[cat][pt_key]
            score_vals = merged_data[cat][score_key]
            label = CATEGORY_LABELS[cat]
            color = CATEGORY_COLORS[cat]
            count = len(pt_vals)

            if is_jet_score and count > 0:
                jitter = np.random.uniform(-0.18, 0.18, size=count)
                y_plot = score_vals + jitter
            else:
                y_plot = score_vals

            alpha_val = max(0.05, min(0.5, 1000.0 / max(count, 1)))

            ax.scatter(
                pt_vals,
                y_plot,
                color=color,
                alpha=alpha_val,
                s=8,
                edgecolors="none",
            )

            ax.set_xscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_title(f"{label}\n(N = {count:,})", fontsize=10, fontweight="bold")

            if is_jet_score:
                ax.set_yticks([-1, 1, 2, 3, 4, 5, 6])
                ax.set_yticklabels([JET_QUANTILES_MAP[k] for k in [-1, 1, 2, 3, 4, 5, 6]])
                ax.set_ylim(-1.6, 6.6)
            else:
                ax.set_ylim(-0.05, 1.05)

        for ax in axes[1, :]:
            ax.set_xlabel(xlabel, fontsize=11)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=11)

        fig.suptitle(
            f"Scatter: {ylabel.splitlines()[0]} vs {xlabel} ($\\Delta R < {DR_THRESHOLD_SCORES}$)",
            fontsize=13, y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if score_key == "tau_score":
            filename = f"grid_scatter_{score_key}_{TAU_ID_SCORE_BRANCH}_vs_pt.png"
        else:
            filename = f"grid_scatter_{score_key}_vs_pt.png"
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"[OK] Grid scatter plot saved to: {out_path}")


def main():
    """
    Entry point of the identification-score study.

    For each input file: read the branches, apply the analysis-level
    selections, label jets (true b-jet) and taus (true hadronic tau),
    collect the scores of the true taus for the working points, and
    build the overlapping pairs by truth category. The working points
    are then computed on all true taus, the per-file results are merged
    and the distributions and scatter plots are saved. Files missing
    any required branch are skipped.

    Returns
    -------
    None
    """
    loaded = load_files()
    if not loaded:
        return

    branches = [
        TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_PT_BRANCH, TAU_IS_ANALYSIS_BRANCH,
        TAU_ID_SCORE_BRANCH, TAU_TRUTH_MATCH_BRANCH,
        JET_ETA_BRANCH, JET_PHI_BRANCH, JET_PT_BRANCH, JET_IS_ANALYSIS_BRANCH,
        JET_SCORE_BRANCH, JET_TRUTH_LABEL_BRANCH,
    ]

    parts = []
    global_true_tau_scores = []

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]

        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel = get_analysis_selection(tree, JET_IS_ANALYSIS_BRANCH, n_entries)
        tau_sel = get_analysis_selection(tree, TAU_IS_ANALYSIS_BRANCH, n_entries)

        jet_eta = a[JET_ETA_BRANCH][jet_sel]
        jet_phi = a[JET_PHI_BRANCH][jet_sel]
        jet_score = a[JET_SCORE_BRANCH][jet_sel]
        jet_label = a[JET_TRUTH_LABEL_BRANCH][jet_sel] == JET_TRUTH_LABEL_B_VALUE

        tau_eta = a[TAU_ETA_BRANCH][tau_sel]
        tau_phi = a[TAU_PHI_BRANCH][tau_sel]
        tau_score = a[TAU_ID_SCORE_BRANCH][tau_sel]
        tau_label = a[TAU_TRUTH_MATCH_BRANCH][tau_sel] == 1

        # Scores of all true taus (not only overlapping ones) for the WPs.
        tau_score_flat = ak.to_numpy(ak.flatten(tau_score, axis=None))
        tau_label_flat = ak.to_numpy(ak.flatten(tau_label, axis=None))
        valid_mask = np.isfinite(tau_score_flat)
        global_true_tau_scores.append(tau_score_flat[valid_mask & tau_label_flat])

        jet_pt = a[JET_PT_BRANCH][jet_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]

        parts.append(build_pair_scores(
            jet_eta, jet_phi, jet_pt, jet_score, jet_label,
            tau_eta, tau_phi, tau_pt, tau_score, tau_label,
        ))

    all_true_taus = np.concatenate(global_true_tau_scores) if global_true_tau_scores else np.array([])
    if len(all_true_taus) == 0:
        print("Error: no true hadronic tau found in the processed files, cannot compute the working points.")
        return

    tau_wps = compute_working_points(all_true_taus, TARGET_EFFICIENCIES, WP_COLORS)

    merged = merge_score_data(parts)

    section("PAIR-LEVEL SCORE PLOTS")
    print(f"Computed {len(tau_wps)} dynamic thresholds for branch {TAU_ID_SCORE_BRANCH}.")
    save_score_plots(merged, tau_wps)

    section("SCATTER PLOTS (SCORE VS PT)")
    save_score_scatters(merged)


if __name__ == "__main__":
    main()