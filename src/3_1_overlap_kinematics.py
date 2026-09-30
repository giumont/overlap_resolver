"""
Objective 3.1 - Pair-level kinematics of overlapping jet-tau pairs,
split by truth category.

Global goal
-----------
Complete the overlap study of ``3.1_geometric_overlap`` by comparing the kinematic
characteristics of the RECO (jet, tau) pairs that geometrically overlap
(DeltaR < ``DR_THRESHOLD_KINEMATICS``), separately for each combination
of truth information:

- (TF) true b-jet, fake hadronic tau
- (FT) fake b-jet, true hadronic tau
- (TT) true b-jet, true hadronic tau
- (FF) both fake

Optionally (``AGGREGATE_CATEGORIES``) (TT) and (FF) are merged into
"other".

For every jet-tau combination of an event the script builds jet
variables (pT, mass, number of soft muons, eta, phi), tau variables (pT,
nProng, decayMode, charge, eta, phi) and pair variables (pT ratio,
DeltaR, Delta eta, Delta phi). Pairs are then selected by DeltaR,
assigned to a truth category and merged over all files. The output is a
set of normalised histograms per variable (one curve per category) and
scatter plots of DeltaR versus jet/tau pT.

Truth labelling (`label_jets_and_taus`), matching (`match_reco_to_truth`),
plotting helpers (`_plot_hist_curves`, `_finish_object_plot`) and the
MET-based tau variables (`compute_tau_met_proj`, `compute_tau_mt`) are
imported from ``truth_vs_reco_params.py`` and ``overlap_met_tau.py``.
All configuration is defined in ``params.py``.
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
    OUTPUT_DIR_PAIR_KIN,
    AGGREGATE_CATEGORIES, NORMALIZE_PAIR_HISTOGRAMS,
    COMPUTE_PT_RATIO, COMPUTE_MET_PROJ, COMPUTE_MT,
    DR_THRESHOLD_KINEMATICS, PT_HIST_MAX, M_HIST_MAX,
    CATEGORY_LABELS, CATEGORY_COLORS, CATEGORY_KEYS,
    CATEGORY_LABELS_AGG, CATEGORY_COLORS_AGG, CATEGORY_KEYS_AGG,
    VARIABLE_PLOT_CONFIG, JET_VARIABLES, TAU_VARIABLES, PAIR_VARIABLES,
)
from obj_3_1 import section, load_files, get_analysis_selection, delta_r
from truth_vs_reco_params import (
    match_reco_to_truth,
    label_jets_and_taus,
    _plot_hist_curves,
    _finish_object_plot,
)
from overlap_met_tau import compute_tau_met_proj, compute_tau_mt


def build_pair_kinematics_and_labels(
    jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
    tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
    met=None, met_phi=None,
    compute_pt_ratio=COMPUTE_PT_RATIO,
    compute_met_proj=COMPUTE_MET_PROJ,
    compute_mt=COMPUTE_MT,
):
    """
    Build jet, tau and pair variables for every jet-tau combination of
    each event, together with the truth labels of the two objects.

    All jet and tau inputs are jagged arrays of shape
    ``[event][object]`` for the selected objects. The jet-tau
    cartesian product (``nested=True``) gives arrays of shape
    ``[event][jet][tau]``: jet quantities are repeated along the tau
    axis and tau quantities along the jet axis.

    Parameters
    ----------
    jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons : awkward.Array
        Jet kinematics (pT and mass in MeV) and number of soft muons.

    jet_label : awkward.Array of bool
        True if the jet is a true b-jet.

    tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge : awkward.Array
        Tau kinematics (pT in MeV), number of prongs, decay mode and
        charge.

    tau_label : awkward.Array of bool
        True if the tau is a true hadronic tau.

    met, met_phi : awkward.Array or None, default=None
        Per-event missing transverse energy and its azimuth. The
        MET-based features are computed only if both are given; the
        current `main` does not pass them.

    compute_pt_ratio : bool, default=COMPUTE_PT_RATIO
        Add ``pair_pt_ratio`` = jet pT / tau pT.

    compute_met_proj : bool, default=COMPUTE_MET_PROJ
        Add ``tau_met_proj`` (projection of the MET along the tau
        direction, via `compute_tau_met_proj`). Requires ``met`` and
        ``met_phi``.

    compute_mt : bool, default=COMPUTE_MT
        Add ``tau_mt`` (transverse mass of the tau-MET system, via
        `compute_tau_mt`). Requires ``met`` and ``met_phi``.

    Returns
    -------
    dict of str -> awkward.Array
        Arrays of shape ``[event][jet][tau]`` with keys ``pair_dr``,
        ``pair_deta``, ``pair_dphi``, ``jet_*`` (pt, eta, phi, mass,
        n_muons), ``tau_*`` (pt, eta, phi, nProng, decayMode, charge),
        ``jet_label``, ``tau_label`` and, when requested,
        ``pair_pt_ratio``, ``tau_met_proj``, ``tau_mt``.
    """
    jet_pt_ct, tau_pt_ct = ak.unzip(ak.cartesian([jet_pt, tau_pt], nested=True))
    jet_eta_ct, tau_eta_ct = ak.unzip(ak.cartesian([jet_eta, tau_eta], nested=True))
    jet_phi_ct, tau_phi_ct = ak.unzip(ak.cartesian([jet_phi, tau_phi], nested=True))

    jet_mass_ct, _ = ak.unzip(ak.cartesian([jet_mass, tau_pt], nested=True))
    jet_n_muons_ct, _ = ak.unzip(ak.cartesian([jet_n_muons, tau_pt], nested=True))

    _, tau_nProng_ct = ak.unzip(ak.cartesian([jet_pt, tau_nProng], nested=True))
    _, tau_decayMode_ct = ak.unzip(ak.cartesian([jet_pt, tau_decayMode], nested=True))
    _, tau_charge_ct = ak.unzip(ak.cartesian([jet_pt, tau_charge], nested=True))

    jet_label_ct, tau_label_ct = ak.unzip(ak.cartesian([jet_label, tau_label], nested=True))

    dr_matrix = delta_r(jet_eta_ct, jet_phi_ct, tau_eta_ct, tau_phi_ct)
    deta_matrix = jet_eta_ct - tau_eta_ct
    dphi_matrix = (jet_phi_ct - tau_phi_ct + np.pi) % (2 * np.pi) - np.pi

    features = {
        "pair_dr": dr_matrix,
        "pair_deta": deta_matrix,
        "pair_dphi": dphi_matrix,
        "jet_pt": jet_pt_ct,
        "jet_eta": jet_eta_ct,
        "jet_phi": jet_phi_ct,
        "jet_mass": jet_mass_ct,
        "jet_n_muons": jet_n_muons_ct,
        "tau_pt": tau_pt_ct,
        "tau_eta": tau_eta_ct,
        "tau_phi": tau_phi_ct,
        "tau_nProng": tau_nProng_ct,
        "tau_decayMode": tau_decayMode_ct,
        "tau_charge": tau_charge_ct,
        "jet_label": jet_label_ct,
        "tau_label": tau_label_ct,
    }

    if compute_pt_ratio:
        features["pair_pt_ratio"] = jet_pt_ct / tau_pt_ct

    if compute_met_proj and met is not None and met_phi is not None:
        met_bcast = ak.broadcast_arrays(met, tau_pt_ct)[0]
        met_phi_bcast = ak.broadcast_arrays(met_phi, tau_phi_ct)[0]
        features["tau_met_proj"] = compute_tau_met_proj(tau_phi_ct, met_bcast, met_phi_bcast)

    if compute_mt and met is not None and met_phi is not None:
        met_bcast = ak.broadcast_arrays(met, tau_pt_ct)[0]
        met_phi_bcast = ak.broadcast_arrays(met_phi, tau_phi_ct)[0]
        features["tau_mt"] = compute_tau_mt(tau_pt_ct, tau_phi_ct, met_bcast, met_phi_bcast)

    return features


def get_overlapping_pairs_kinematics(pair_info, dr_thr):
    """
    Select the overlapping pairs and split them by truth category.

    A pair overlaps if ``pair_dr < dr_thr``. The categories follow
    ``AGGREGATE_CATEGORIES``: the four combinations (TF, FT, TT, FF) or
    TF, FT and "other" (TT and FF together).

    Parameters
    ----------
    pair_info : dict of str -> awkward.Array
        Output of `build_pair_kinematics_and_labels`.

    dr_thr : float
        Maximum DeltaR of an overlapping pair.

    Returns
    -------
    dict of str -> dict of str -> numpy.ndarray
        ``result[category][variable]`` is the flat array of the values
        of ``variable`` for the overlapping pairs of that category. The
        truth labels themselves are not included as variables.
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

    variables = [k for k in pair_info.keys() if k not in ["jet_label", "tau_label"]]

    for cat, cmask in cat_masks.items():
        for var in variables:
            res[cat][var] = flat_overlap(var)[cmask]

    return res


def merge_pair_kinematics(parts_list):
    """
    Concatenate the per-file results of `get_overlapping_pairs_kinematics`.

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

    Jet, tau and pair variables listed in ``JET_VARIABLES``,
    ``TAU_VARIABLES`` and ``PAIR_VARIABLES`` are plotted, with the y
    scale and the output folder taken from ``VARIABLE_PLOT_CONFIG``.
    Files are written to a subfolder ``DR_max_<DR_THRESHOLD_KINEMATICS>``
    (plus ``aggregated`` if ``AGGREGATE_CATEGORIES``). Nothing is done
    if matplotlib is unavailable.

    Parameters
    ----------
    merged_data : dict of str -> dict of str -> numpy.ndarray
        Output of `merge_pair_kinematics`.

    dr_threshold : float
        DeltaR threshold used to select the pairs; shown in the plot
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
            var_list = JET_VARIABLES
        elif obj_name == "tau":
            var_list = TAU_VARIABLES
        else:
            var_list = PAIR_VARIABLES

        for var_key, vmin, vmax, step, xlabel in var_list:

            var_id = f"{obj_name}_{var_key}"
            config = VARIABLE_PLOT_CONFIG.get(
                var_id,
                {"y_scale": "linear", "out_dir": OUTPUT_DIR_PAIR_KIN / "pair_kinematics_categories"},
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
                f"Cinematica {obj_title} in coppie (pair-level) - $\\Delta R < {dr_threshold}$",
                normalize=NORMALIZE_PAIR_HISTOGRAMS,
            )

            ax.set_yscale(y_scale)
            fig.tight_layout()

            suffix = "_normalized" if NORMALIZE_PAIR_HISTOGRAMS else ""
            if var_key == "pt":
                suffix_cut = f"_cut_{PT_HIST_MAX}"
            elif var_key == "mass":
                suffix_cut = f"_cut_{M_HIST_MAX}"
            else:
                suffix_cut = ""
            suffix_agg = "_aggregated" if AGGREGATE_CATEGORIES else ""

            out_path = (
                plot_dir
                / f"pair_kinematics_{var_id}_"
                f"{JET_SELECTION_MODE}{suffix}{suffix_cut}{suffix_agg}.png"
            )

            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[OK] Plot salvato in: {out_path}")


def save_scatter_plots(merged_data, dr_threshold):
    """
    Save scatter plots of DeltaR versus jet pT and versus tau pT, with
    one colour per truth category (legend entries include the number of
    pairs of each category).

    Files are written to ``OUTPUT_DIR_PAIR_KIN / "scatters"``. Nothing
    is done if matplotlib is unavailable.

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

    out_dir = OUTPUT_DIR_PAIR_KIN / "scatters"
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
        ax.set_title(f"Scatter: $\\Delta R$ vs {ylabel} ($\\Delta R < {DR_THRESHOLD_KINEMATICS}$)")
        ax.set_yscale("log")
        ax.legend(title="Categorie", fontsize=9, loc="best")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"scatter_dr_vs_{var_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Scatter plot salvato in: {out_path}")


def main():
    """
    Entry point of the pair-level kinematics study.

    For each input file: read the branches, apply the analysis-level
    (and optional working point) selections, assign the truth labels to
    jets and taus, build all jet-tau pairs, keep those with
    DeltaR < ``DR_THRESHOLD_KINEMATICS`` split by truth category. The
    per-file results are merged and used to save the histograms and the
    scatter plots. Files missing any required branch are skipped.

    Returns
    -------
    None
    """
    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
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

        pair_info = build_pair_kinematics_and_labels(
            jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
            tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
        )

        pair_kinematics_parts.append(
            get_overlapping_pairs_kinematics(pair_info, DR_THRESHOLD_KINEMATICS)
        )

    merged_pair_kinematics = merge_pair_kinematics(pair_kinematics_parts)

    section("PLOT CINEMATICA PAIR-LEVEL")
    save_pair_kinematics_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)

    section("SCATTER PLOTS")
    save_scatter_plots(merged_pair_kinematics, DR_THRESHOLD_KINEMATICS)


if __name__ == "__main__":
    main()