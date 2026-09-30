"""
Objective 3.1 - Pair-level kinematics of overlapping jet-tau pairs,
"no-repetitions" variant (global greedy matching).

Global goal
-----------
Same goal as ``overlap_kinematics.py``: compare the kinematic
characteristics of the overlapping RECO (jet, tau) pairs, split by the
four truth categories (TF, FT, TT, FF) of the b-jet / hadronic-tau
identification. The only difference is the criterion used to build the
pairs.

In ``overlap_kinematics.py`` a pair enters the analysis if and only if
DeltaR(jet, tau) < ``DR_THRESHOLD_KINEMATICS``, so the same jet can
appear in several pairs (with different taus) and vice versa: an event
with 1 jet and 3 taus all within threshold gives 3 pairs sharing the
jet. Here every jet and every tau appears in at most ONE pair per
event, using a global greedy matching (nearest neighbour without
replacement), for each event:

1. compute the DeltaR(jet_i, tau_j) matrix;
2. find the pair (i, j) with the absolute minimum DeltaR over the whole
   remaining matrix;
3. if that DeltaR is below the threshold, store the pair and remove row
   i and column j (jet_i and tau_j are no longer available);
4. repeat until the minimum residual DeltaR is not below the threshold
   or jets or taus are exhausted.

A sequential "tau by tau" algorithm (each tau takes the closest free
jet, in branch order) would be order-dependent: a jet contested by two
taus goes to the first processed one even if it is closer to the other.
Picking at each step the absolute minimum DeltaR among the available
pairs makes the result independent of the storage order of jets and
taus.

Selections, truth categories, plotted variables and plot structure are
those of ``overlap_kinematics.py``; outputs go to a separate folder
(``OUTPUT_DIR_NO_REPS``) so that they do not overwrite the other
variant. Truth labelling and plotting helpers come from
``truth_vs_reco_params.py``; the MET-based tau variables from
``overlap_met_tau.py``. All configuration is defined in ``params.py``.
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
    JET_ETA_BRANCH, JET_PHI_BRANCH, JET_PT_BRANCH, JET_MASS_BRANCH,
    JET_N_MUONS_BRANCH, JET_IS_ANALYSIS_BRANCH,
    TAU_ETA_BRANCH, TAU_PHI_BRANCH, TAU_PT_BRANCH, TAU_NPRONG_BRANCH,
    TAU_DECAYMODE_BRANCH, TAU_CHARGE_BRANCH, TAU_IS_ANALYSIS_BRANCH,
    JET_SELECTION_MODE, JET_BTAG_BRANCH,
    TAU_SELECTION_MODE, TAU_SCORE_BRANCH, TAU_SCORE_WP85_THRESHOLD,
    JET_TRUTH_LABEL_BRANCH,
    TRUTH_MODE_TAU, TAU_TRUTH_MATCH_BRANCH,
    TRUTH_TAU_ETA_BRANCH, TRUTH_TAU_PHI_BRANCH,
    AGGREGATE_CATEGORIES, NORMALIZE_PAIR_HISTOGRAMS,
    COMPUTE_PT_RATIO, COMPUTE_MET_PROJ, COMPUTE_MT,
    DR_THRESHOLD_KINEMATICS, PT_HIST_MAX_NO_REPS, M_HIST_MAX,
    CATEGORY_LABELS, CATEGORY_COLORS, CATEGORY_KEYS,
    CATEGORY_LABELS_AGG, CATEGORY_COLORS_AGG, CATEGORY_KEYS_AGG,
    OUTPUT_DIR_NO_REPS, VARIABLE_PLOT_CONFIG_NO_REPS,
    JET_VARIABLES_NO_REPS, TAU_VARIABLES_NO_REPS, PAIR_VARIABLES_NO_REPS,
)
from obj_3_1 import section, load_files, get_analysis_selection, delta_r
from truth_vs_reco_params import (
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)
from overlap_met_tau import compute_tau_met_proj, compute_tau_mt


def greedy_match_no_reps_indices(jet_eta, jet_phi, tau_eta, tau_phi, dr_thr):
    """
    Match jets and taus one-to-one, event by event, with a global greedy
    algorithm (nearest neighbour without replacement).

    At each step the still-available (jet, tau) pair with the absolute
    minimum DeltaR in the event is taken, if its DeltaR is below
    ``dr_thr``, and both objects are removed from the pool. The loop
    stops when the minimum residual DeltaR is not below the threshold or
    jets or taus run out, so an event gives at most ``min(n_jet, n_tau)``
    pairs and each object appears in at most one pair.

    The loop is intrinsically iterative per event, so the inputs are
    converted to Python lists and each event uses NumPy on its own
    (small) DeltaR matrix.

    Parameters
    ----------
    jet_eta, jet_phi : awkward.Array
        Eta and azimuth of the selected jets, shape ``[event][jet]``.

    tau_eta, tau_phi : awkward.Array
        Eta and azimuth of the selected taus, shape ``[event][tau]``.

    dr_thr : float
        DeltaR threshold: pairs are formed only below it.

    Returns
    -------
    matched_jet_idx, matched_tau_idx : awkward.Array
        Jagged arrays, one list per event, with the LOCAL indices (into
        the selected jet and tau collections given as input) of the
        matched objects. The i-th pair of an event is
        ``(matched_jet_idx[evt][i], matched_tau_idx[evt][i])``.
    """
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
    Diagnostic check that, in every event, the matched jet indices are
    all distinct, and likewise the tau indices (a structural guarantee
    of the greedy matching). Prints the outcome and the total number of
    pairs.

    Parameters
    ----------
    matched_jet_idx, matched_tau_idx : awkward.Array
        Output of `greedy_match_no_reps_indices`.

    label : str, default=""
        Text (e.g. the file name) identifying the check in the printout.

    Returns
    -------
    None
    """
    def has_no_duplicates_per_event(idx_array):
        """True if no event contains a repeated index."""
        as_list = ak.to_list(idx_array)
        for ev_idx in as_list:
            if len(ev_idx) != len(set(ev_idx)):
                return False
        return True

    ok_jet = has_no_duplicates_per_event(matched_jet_idx)
    ok_tau = has_no_duplicates_per_event(matched_tau_idx)

    n_pairs = int(ak.sum(ak.num(matched_jet_idx)))

    status_jet = "OK" if ok_jet else "FAILED"
    status_tau = "OK" if ok_tau else "FAILED"

    print(
        f"   [{label}] no-repetition check: "
        f"jet={status_jet}  tau={status_tau}  (total pairs: {n_pairs})"
    )


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
    Build jet, tau and pair variables for the pairs found by the greedy
    matching.

    Unlike `build_pair_kinematics_and_labels` in ``overlap_kinematics.py``,
    no jet x tau cartesian product is built: the pairs are already
    identified by the matching, so the matched objects are selected with
    awkward jagged indexing. Every pair has DeltaR below the threshold by
    construction, so no overlap mask is needed.

    Parameters
    ----------
    jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons : awkward.Array
        Kinematics (pT and mass in MeV) and number of soft muons of the
        selected jets, shape ``[event][jet]``.

    jet_label : awkward.Array of bool
        True if the jet is a true b-jet.

    tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge : awkward.Array
        Kinematics (pT in MeV), number of prongs, decay mode and charge
        of the selected taus, shape ``[event][tau]``.

    tau_label : awkward.Array of bool
        True if the tau is a true hadronic tau.

    matched_jet_idx, matched_tau_idx : awkward.Array
        Output of `greedy_match_no_reps_indices`.

    met, met_phi : awkward.Array or None, default=None
        Per-event missing transverse energy and its azimuth. The
        MET-based features are computed only if both are given; the
        current `main` does not pass them.

    compute_pt_ratio : bool, default=COMPUTE_PT_RATIO
        Add ``pair_pt_ratio`` = jet pT / tau pT.

    compute_met_proj : bool, default=COMPUTE_MET_PROJ
        Add ``tau_met_proj``. Requires ``met`` and ``met_phi``.

    compute_mt : bool, default=COMPUTE_MT
        Add ``tau_mt``. Requires ``met`` and ``met_phi``.

    Returns
    -------
    dict of str -> awkward.Array
        Arrays of shape ``[event][pair]`` with keys ``pair_dr``,
        ``pair_deta``, ``pair_dphi``, ``jet_*`` (pt, eta, phi, mass,
        n_muons), ``tau_*`` (pt, eta, phi, nProng, decayMode, charge),
        ``jet_label``, ``tau_label`` and, when requested,
        ``pair_pt_ratio``, ``tau_met_proj``, ``tau_mt``.
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
    Split the matched pairs by truth category.

    Counterpart of `get_overlapping_pairs_kinematics` in
    ``overlap_kinematics.py`` without the DeltaR mask, which is already
    embedded in the pair construction. The categories follow
    ``AGGREGATE_CATEGORIES``: the four combinations (TF, FT, TT, FF) or
    TF, FT and "other" (TT and FF together).

    Parameters
    ----------
    features : dict of str -> awkward.Array
        Output of `build_matched_pair_features`.

    Returns
    -------
    dict of str -> dict of str -> numpy.ndarray
        ``result[category][variable]`` is the flat array of the values
        of ``variable`` for the pairs of that category. The truth labels
        themselves are not included as variables.
    """
    def flat(key):
        """Flatten ``features[key]`` to a 1D numpy array."""
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
    """
    Concatenate the per-file results of `categorize_matched_pairs`.

    Parameters
    ----------
    parts_list : list of dict
        One ``result[category][variable]`` dictionary per processed
        file. The variables are taken from the first element.

    Returns
    -------
    dict of str -> dict of str -> numpy.ndarray
        Same structure as the input, with the arrays of all files
        concatenated. Empty dict if ``parts_list`` is empty.
    """
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


def save_pair_kinematics_plots(merged_data, dr_threshold):
    """
    Save one histogram per variable, with one curve per truth category.

    Jet, tau and pair variables listed in ``JET_VARIABLES_NO_REPS``,
    ``TAU_VARIABLES_NO_REPS`` and ``PAIR_VARIABLES_NO_REPS`` are plotted,
    with the y scale and the output folder taken from
    ``VARIABLE_PLOT_CONFIG_NO_REPS``. Files are written to a subfolder
    ``DR_max_<DR_THRESHOLD_KINEMATICS>`` (plus ``aggregated`` if
    ``AGGREGATE_CATEGORIES``). Nothing is done if matplotlib is
    unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_pair_kinematics`.

    dr_threshold : float
        DeltaR threshold used to build the pairs; shown in the plot
        titles only (folder names use ``DR_THRESHOLD_KINEMATICS``).

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    for obj_name, obj_title in (("jet", "jet"), ("tau", "tau"), ("pair", "coppia")):

        if obj_name == "jet":
            var_list = JET_VARIABLES_NO_REPS
        elif obj_name == "tau":
            var_list = TAU_VARIABLES_NO_REPS
        else:
            var_list = PAIR_VARIABLES_NO_REPS

        for var_key, vmin, vmax, step, xlabel in var_list:

            var_id = f"{obj_name}_{var_key}"
            config = VARIABLE_PLOT_CONFIG_NO_REPS.get(
                var_id,
                {"y_scale": "linear", "out_dir": OUTPUT_DIR_NO_REPS / "pair_kinematics_categories_no_reps"},
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
                normalize=NORMALIZE_PAIR_HISTOGRAMS,
            )

            ax.set_yscale(y_scale)
            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_PAIR_HISTOGRAMS else ""
            if var_key == "pt":
                suffix_cut = f"_cut_{PT_HIST_MAX_NO_REPS}"
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

            print(f"[OK] Plot saved to: {out_path}")


def save_scatter_plots(merged_data, dr_threshold):
    """
    Save scatter plots of DeltaR versus jet pT and versus tau pT, with
    one colour per truth category (legend entries include the number of
    pairs of each category).

    Files are written to ``OUTPUT_DIR_NO_REPS / "scatters"``. Nothing is
    done if matplotlib is unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_pair_kinematics`.

    dr_threshold : float
        Unused; the plot titles use ``DR_THRESHOLD_KINEMATICS``.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        return

    cat_keys = CATEGORY_KEYS_AGG if AGGREGATE_CATEGORIES else CATEGORY_KEYS
    labels = CATEGORY_LABELS_AGG if AGGREGATE_CATEGORIES else CATEGORY_LABELS
    colors = CATEGORY_COLORS_AGG if AGGREGATE_CATEGORIES else CATEGORY_COLORS

    out_dir = OUTPUT_DIR_NO_REPS / "scatters"
    out_dir.mkdir(parents=True, exist_ok=True)

    scatters_config = [
        ("jet_pt", "pT Jet [MeV]"),
        ("tau_pt", "pT Tau [MeV]"),
    ]

    for var_key, ylabel in scatters_config:
        fig, ax = plt.subplots(figsize=(8, 6))

        for cat in cat_keys:
            dr_vals = merged_data[cat]["pair_dr"]
            pt_vals = merged_data[cat][var_key]
            count = len(dr_vals)

            ax.scatter(
                dr_vals,
                pt_vals,
                label=f"{labels[cat]} (N={count})",
                color=colors[cat],
                alpha=0.6,
                s=15,
                edgecolors="none",
            )

        ax.set_xlabel(r"$\Delta R$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Scatter (no-rep): $\\Delta R$ vs {ylabel} ($\\Delta R < {DR_THRESHOLD_KINEMATICS}$)")
        ax.set_yscale("log")
        ax.legend(title="Categorie", fontsize=9, loc="best")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"scatter_no_reps_dr_vs_{var_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Scatter plot saved to: {out_path}")


def main():
    """
    Entry point of the no-repetitions pair-level kinematics study.

    For each input file: read the branches, apply the analysis-level
    (and optional working point) selections, assign the truth labels,
    build the one-to-one jet-tau pairs with the global greedy matching
    (DeltaR < ``DR_THRESHOLD_KINEMATICS``), check that no object is
    repeated, and split the pairs by truth category. The per-file
    results are merged and used to save the histograms and the scatter
    plots. Files missing any required branch are skipped.

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
        TAU_NPRONG_BRANCH, TAU_DECAYMODE_BRANCH, TAU_CHARGE_BRANCH,
        JET_ETA_BRANCH, JET_PHI_BRANCH, JET_PT_BRANCH,
        JET_MASS_BRANCH, JET_N_MUONS_BRANCH, JET_IS_ANALYSIS_BRANCH,
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

    section("GREEDY MATCHING WITHOUT REPETITIONS")

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
                ak.fill_none(a[TAU_SCORE_BRANCH], -np.inf) >= TAU_SCORE_WP85_THRESHOLD
            )

        jet_label, tau_label, _, _ = label_jets_and_taus(a, jet_sel, tau_sel)

        jet_eta = a[JET_ETA_BRANCH][jet_sel]
        jet_phi = a[JET_PHI_BRANCH][jet_sel]
        jet_pt = a[JET_PT_BRANCH][jet_sel]
        jet_mass = a[JET_MASS_BRANCH][jet_sel]
        jet_n_muons = a[JET_N_MUONS_BRANCH][jet_sel]

        tau_eta = a[TAU_ETA_BRANCH][tau_sel]
        tau_phi = a[TAU_PHI_BRANCH][tau_sel]
        tau_pt = a[TAU_PT_BRANCH][tau_sel]
        tau_nProng = a[TAU_NPRONG_BRANCH][tau_sel]
        tau_decayMode = a[TAU_DECAYMODE_BRANCH][tau_sel]
        tau_charge = a[TAU_CHARGE_BRANCH][tau_sel]

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

    section("PAIR-LEVEL KINEMATICS PLOTS (NO-REPS)")
    save_pair_kinematics_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)

    section("SCATTER PLOTS (NO-REPS)")
    save_scatter_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)


if __name__ == "__main__":
    main()