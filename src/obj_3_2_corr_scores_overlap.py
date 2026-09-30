"""
Correlation between b-tag quantile and tau ID score in overlapping pairs.

Role in the project: for each (analysis-level reco jet, reco tau) pair with
DeltaR below `params.DR_THRESHOLD_CORR`, this script relates the GN2v01
continuous b-tag quantile of the jet (`params.JET_SCORE_BRANCH`) to the tau
ID score (`params.TAU_ID_SCORE_BRANCH`). It quantifies the correlation
between the two identification discriminants in the overlap region
(objective 3.1) and provides the input for their manual combination
(objective 3.2); results are stored under ``output/obj_3_2``.

Pairs are split in the four truth categories (b-jet true/fake x hadronic
tau true/fake). For each category the script draws, for every b-tag
quantile bin, the distribution of the tau score (violin plot), with the
tau working points overlaid as horizontal lines, and reports:

- the correlation ratio (eta) between the quantile (treated as a
  categorical variable, so that the N/A bin -1 is handled correctly) and
  the tau score;
- the Spearman rank correlation, restricted to quantile bins > 0.

The tau working points are the score thresholds giving the target signal
efficiencies `params.TARGET_EFFICIENCIES` on all true hadronic taus at
analysis level (not only those in overlap).

Configuration is in ``params.py``; utilities come from ``obj_3_1.py``.
"""

import numpy as np
import awkward as ak
from scipy.stats import spearmanr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

import params
from obj_3_1_geometric_overlap import section, load_files, get_analysis_selection, delta_r


def compute_correlation_ratio(categories, measurements):
    """
    Correlation ratio (eta) between a categorical and a continuous variable.

    eta = sqrt(SS_between / SS_total), where SS_between is the weighted
    dispersion of the per-category means around the global mean. Unlike a
    rank correlation it does not assume an order of the categories, so
    non-ordinal values such as -1 are handled correctly.

    Parameters
    ----------
    categories : array-like
        Category of each measurement (here the b-tag quantile bin).

    measurements : array-like
        Continuous values (here the tau score), same length as
        `categories`.

    Returns
    -------
    float
        eta in [0, 1]; 0.0 if `measurements` has zero variance.
    """
    cats = np.array(categories)
    meas = np.array(measurements)

    total_mean = np.mean(meas)
    ss_total = np.sum((meas - total_mean) ** 2)

    if ss_total == 0:
        return 0.0

    ss_between = 0.0
    for cat_val in np.unique(cats):
        mask = (cats == cat_val)
        cat_mean = np.mean(meas[mask])
        n_cat = np.sum(mask)
        ss_between += n_cat * (cat_mean - total_mean) ** 2

    return np.sqrt(ss_between / ss_total)


def compute_working_points(score_true, target_efficiencies, colors):
    """
    Tau score thresholds corresponding to target signal efficiencies.

    The threshold for efficiency `t` (%) is the ``100 - t`` percentile of
    the score of true hadronic taus, so that a fraction `t` of them has
    score above it (score higher = more signal-like).

    Parameters
    ----------
    score_true : numpy.ndarray
        Scores of all true hadronic taus.

    target_efficiencies : list of float
        Target efficiencies in percent.

    colors : list of str
        Plot color of each working point, same length as
        `target_efficiencies`.

    Returns
    -------
    list of tuple
        ``(threshold, label, color)`` for each target, with
        ``label = "<target>%"``.
    """
    wp_list = []
    for target, color in zip(target_efficiencies, colors):
        thr = float(np.percentile(score_true, 100 - target))
        wp_list.append((thr, f"{target}%", color))
    return wp_list


def extract_overlapping_scores(
    jet_eta, jet_phi, jet_score,
    tau_eta, tau_phi, tau_score,
    dr_threshold,
    jet_truth=None, tau_truth=None
):
    """
    Extract the scores of the (jet, tau) pairs with DeltaR < `dr_threshold`.

    All jet-tau combinations of each event are built; pairs whose jet
    score is not finite or < `params.CORR_JET_SCORE_MIN`, or whose tau
    score is not finite or <= `params.CORR_TAU_SCORE_MIN`, are discarded.

    Parameters
    ----------
    jet_eta, jet_phi, jet_score : ak.Array
        Jagged per-event arrays of the selected jets.

    tau_eta, tau_phi, tau_score : ak.Array
        Jagged per-event arrays of the selected taus.

    dr_threshold : float
        Maximum DeltaR of an overlapping pair.

    jet_truth, tau_truth : ak.Array or None, default=None
        Jagged truth labels of jets and taus. Truth columns are returned
        only if both are given.

    Returns
    -------
    dict of numpy.ndarray
        ``jet_score`` and ``tau_score`` of the valid overlapping pairs,
        plus ``jet_truth`` and ``tau_truth`` when truth is provided.
    """
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))
    jet_score_ct, tau_score_ct = ak.unzip(ak.cartesian([jet_score, tau_score], nested=True))

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)
    overlap_mask = dr_matrix < dr_threshold

    def apply_mask(arr):
        flat = ak.to_numpy(ak.flatten(arr, axis=None))
        mask_flat = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)
        return flat[mask_flat]

    j_scores_overlap = apply_mask(jet_score_ct)
    t_scores_overlap = apply_mask(tau_score_ct)

    valid_mask = (
        np.isfinite(j_scores_overlap) &
        np.isfinite(t_scores_overlap) &
        (j_scores_overlap >= params.CORR_JET_SCORE_MIN) &
        (t_scores_overlap > params.CORR_TAU_SCORE_MIN)
    )

    res = {
        "jet_score": j_scores_overlap[valid_mask],
        "tau_score": t_scores_overlap[valid_mask]
    }

    if jet_truth is not None and tau_truth is not None:
        j_truth_ct, t_truth_ct = ak.unzip(ak.cartesian([jet_truth, tau_truth], nested=True))
        res["jet_truth"] = apply_mask(j_truth_ct)[valid_mask]
        res["tau_truth"] = apply_mask(t_truth_ct)[valid_mask]

    return res


def save_truth_type_violin_plot(jet_scores, tau_scores, jet_truth, tau_truth, score_name, tau_wps):
    """
    Plot the tau score vs b-tag quantile bin, one panel per truth category.

    Each panel is a violin plot of the tau score in every quantile bin
    present in the category, with the tau working points as horizontal
    lines and a box reporting the number of pairs, the correlation ratio
    and the Spearman coefficient (bins > 0 only). The figure is saved in
    `params.OUTPUT_DIR_CORR_SCORES`. Nothing is done if matplotlib is
    unavailable.

    Parameters
    ----------
    jet_scores, tau_scores : numpy.ndarray
        b-tag quantile and tau score of each overlapping pair.

    jet_truth, tau_truth : numpy.ndarray
        Truth labels of each pair. A jet is a true b-jet if its label
        equals `params.JET_TRUTH_LABEL_B_VALUE`, a tau is a true hadronic
        tau if its label equals `params.TAU_TRUTH_TRUE_VALUE`.

    score_name : str
        Name of the tau score, used in labels and in the file name.

    tau_wps : list of tuple
        ``(threshold, label, color)`` as returned by
        `compute_working_points`.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    params.OUTPUT_DIR_CORR_SCORES.mkdir(parents=True, exist_ok=True)

    jet_is_b = jet_truth == params.JET_TRUTH_LABEL_B_VALUE
    tau_is_true = tau_truth == params.TAU_TRUTH_TRUE_VALUE

    categories = {
        "(a) Jet Vero / Tau Vero": jet_is_b & tau_is_true,
        "(b) Jet Vero / Tau Fake": jet_is_b & ~tau_is_true,
        "(c) Jet Fake / Tau Vero": ~jet_is_b & tau_is_true,
        "(d) Jet Fake / Tau Fake": ~jet_is_b & ~tau_is_true,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (title, mask) in enumerate(categories.items()):
        ax = axes[idx]
        j_s = jet_scores[mask]
        t_s = tau_scores[mask]
        n_pairs = len(j_s)

        eta_ratio = np.nan
        spearman_rho = np.nan

        if n_pairs > 1:
            eta_ratio = compute_correlation_ratio(j_s, t_s)
            ordinal_mask = j_s > 0
            if np.sum(ordinal_mask) > 1:
                spearman_rho, _ = spearmanr(j_s[ordinal_mask], t_s[ordinal_mask])

        unique_quantiles = np.sort(np.unique(j_s))
        present_quantiles = [q for q in unique_quantiles if q in params.JET_QUANTILES_MAP]

        if len(present_quantiles) > 0:
            data_by_quantile = [t_s[j_s == q] for q in present_quantiles]
            parts = ax.violinplot(data_by_quantile, positions=range(len(present_quantiles)), showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('#1f77b4')
                pc.set_alpha(0.6)
            ax.set_xticks(range(len(present_quantiles)))
            ax.set_xticklabels([params.JET_QUANTILES_MAP[q] for q in present_quantiles])

        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(f"Jet b-tag Target Efficiency / Quantile Bin ({params.JET_SCORE_BRANCH})")
        ax.set_ylabel(f"Tau Score ({score_name})")
        ax.set_title(title)

        for threshold, eff, color in tau_wps:
            lbl = f"WP {eff} ({threshold:.3f})" if idx == 0 else None
            ax.axhline(y=threshold, color=color, linestyle="--", linewidth=1.2, alpha=0.8, zorder=2, label=lbl)

        if idx == 0:
            ax.legend(fontsize=8, loc='upper right', title="Soglie Tau ID", ncol=2)

        textstr = (
            f"Coppie ($\\Delta R < {params.DR_THRESHOLD_CORR}$): {n_pairs}\n"
            f"Corr. Ratio ($\\eta$): {eta_ratio:.3f}\n"
            f"Spearman $\\rho_s$ (bin>0): {spearman_rho:+.3f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    out_path = params.OUTPUT_DIR_CORR_SCORES / (
        f"correlation_{score_name}_vs_btag_dr{params.DR_THRESHOLD_CORR}_truth_types.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Plot salvato in: {out_path}")


def main():
    """
    Run the correlation study over all input files.

    For each file: select analysis-level jets and taus, collect the scores
    of all true hadronic taus (for the working points) and of the
    overlapping pairs. Then compute the tau working points, print a
    summary and save the violin plot. Files lacking any required branch
    are skipped.

    Returns
    -------
    None
    """
    loaded = load_files()
    if not loaded:
        return

    branches = [
        params.TAU_ETA_BRANCH, params.TAU_PHI_BRANCH,
        params.TAU_IS_ANALYSIS_BRANCH, params.TAU_ID_SCORE_BRANCH,
        params.JET_ETA_BRANCH, params.JET_PHI_BRANCH,
        params.JET_IS_ANALYSIS_BRANCH, params.JET_SCORE_BRANCH,
        params.JET_TRUTH_LABEL_BRANCH, params.TAU_TRUTH_MATCH_BRANCH,
    ]

    all_jet_scores, all_tau_scores = [], []
    all_jet_truths, all_tau_truths = [], []
    global_true_tau_scores = []

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]
        if any(b not in tree.keys() for b in branches):
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel = get_analysis_selection(tree, params.JET_IS_ANALYSIS_BRANCH, n_entries)
        tau_sel = get_analysis_selection(tree, params.TAU_IS_ANALYSIS_BRANCH, n_entries)

        tau_score_all = a[params.TAU_ID_SCORE_BRANCH][tau_sel]
        tau_label_all = a[params.TAU_TRUTH_MATCH_BRANCH][tau_sel] == params.TAU_TRUTH_TRUE_VALUE
        tau_score_flat = ak.to_numpy(ak.flatten(tau_score_all, axis=None))
        tau_label_flat = ak.to_numpy(ak.flatten(tau_label_all, axis=None))
        valid_mask_wp = np.isfinite(tau_score_flat)
        global_true_tau_scores.append(tau_score_flat[valid_mask_wp & tau_label_flat])

        extracted = extract_overlapping_scores(
            a[params.JET_ETA_BRANCH][jet_sel], a[params.JET_PHI_BRANCH][jet_sel],
            a[params.JET_SCORE_BRANCH][jet_sel],
            a[params.TAU_ETA_BRANCH][tau_sel], a[params.TAU_PHI_BRANCH][tau_sel],
            a[params.TAU_ID_SCORE_BRANCH][tau_sel],
            params.DR_THRESHOLD_CORR,
            jet_truth=a[params.JET_TRUTH_LABEL_BRANCH][jet_sel],
            tau_truth=a[params.TAU_TRUTH_MATCH_BRANCH][tau_sel],
        )

        all_jet_scores.append(extracted["jet_score"])
        all_tau_scores.append(extracted["tau_score"])
        all_jet_truths.append(extracted["jet_truth"])
        all_tau_truths.append(extracted["tau_truth"])

    all_true_taus = np.concatenate(global_true_tau_scores) if global_true_tau_scores else np.array([])
    if len(all_true_taus) == 0:
        print("Errore: Nessun vero tau adronico trovato nei file processati per calcolare i WP.")
        return

    tau_wps = compute_working_points(all_true_taus, params.TARGET_EFFICIENCIES, params.WP_COLORS)
    print(f"Calcolate {len(tau_wps)} soglie dinamiche per il branch {params.TAU_ID_SCORE_BRANCH}.")

    j_s = np.concatenate(all_jet_scores) if all_jet_scores else np.array([])
    t_s = np.concatenate(all_tau_scores) if all_tau_scores else np.array([])
    j_t = np.concatenate(all_jet_truths) if all_jet_truths else np.array([])
    t_t = np.concatenate(all_tau_truths) if all_tau_truths else np.array([])

    section(f"ANALISI COMPARATIVA: {params.TAU_ID_SCORE_BRANCH}")
    print(f"Coppie estratte in overlap (dR < {params.DR_THRESHOLD_CORR}): {len(j_s)}")

    if len(j_s) > 0:
        save_truth_type_violin_plot(
            j_s, t_s, j_t, t_t, score_name=params.TAU_ID_SCORE_BRANCH, tau_wps=tau_wps
        )


if __name__ == "__main__":
    main()