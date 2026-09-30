"""
Empirical efficiency calibration of the continuous b-tagging SF branch.

Objective 3.2: on the signal sample, quantify the quality of the b-jet
classification. This script analyses the continuous branch
``recojet_antikt4PFlow_ftag_effSF_GN2v01_Continuous___NOSYS`` of the analysis
jets, using the truth flavour label to separate true b-jets from the rest.

Analysis steps
--------------
1. Distribution of the continuous score (range, percentiles, fraction of
   true b-jets).
2. b-jet efficiency and background efficiency measured on the dataset as a
   function of a threshold scanned over the score range.
3. Empirical threshold, background efficiency and rejection for each target
   b-jet efficiency in ``TARGET_EFFICIENCIES``.
4. ROC curve and AUC through ``plot_roc_curve`` of the external
   ``flavour_tagging`` repository.

The direction of the cut is set by ``BJET_SF_HIGHER_IS_BETTER``. Branch
names, scan settings and output paths are defined in ``params.py``; the
utilities of ``obj_3_1.py`` are reused for file loading and selection.
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
    JET_IS_ANALYSIS_BRANCH, JET_TRUTH_LABEL_BRANCH, JET_TRUTH_LABEL_B_VALUE,
    JET_SF_CONTINUOUS_BRANCH, OUTPUT_DIR_BJET_SF,
    BJET_SF_N_SCAN_POINTS, BJET_SF_HIGHER_IS_BETTER,
    TARGET_EFFICIENCIES, FLAVOUR_TAGGING_DIR,
)
from obj_3_1 import section, load_files, get_analysis_selection

sys.path.append(str(FLAVOUR_TAGGING_DIR))
from src.graphics import plot_roc_curve


# ======================================================================
# UTILITY
# ======================================================================

def flatten_clean(values):
    """
    Flatten a jagged array to a one-dimensional numpy array, dropping
    missing and non-finite values.

    Parameters
    ----------
    values : awkward.Array
        Array of any nesting depth.

    Returns
    -------
    flat : numpy.ndarray
        Flattened values without `None`, NaN or infinities.
    """
    flat = ak.flatten(values, axis=None)
    flat = ak.drop_none(flat)
    flat = ak.to_numpy(flat)
    if flat.size > 0 and np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]
    return flat


# ======================================================================
# DISTRIBUTION OF THE CONTINUOUS SCORE
# ======================================================================

def print_continuous_properties(n_files, counts_per_event, score_all, is_true_all):
    """
    Print the basic properties of the continuous score of the analysis
    jets: range, percentiles and fraction of true b-jets.

    Parameters
    ----------
    n_files : int
        Number of files processed.

    counts_per_event : numpy.ndarray
        Number of selected jets per event. Currently unused.

    score_all : numpy.ndarray
        Continuous score of all selected jets, without missing values.

    is_true_all : numpy.ndarray of bool
        True where the jet is a true b-jet; same length as `score_all`.
    """
    section(f"PROPRIETA' DI {JET_SF_CONTINUOUS_BRANCH} (jet analysis-level)")

    print(f"   n. file processati: {n_files}")
    print(f"   shape dopo flatten: {score_all.shape}")

    if score_all.size == 0:
        print("   nessun valore disponibile.")
        return

    print("\n   intervallo osservato:")
    print(f"      min = {np.min(score_all):.6f}")
    print(f"      max = {np.max(score_all):.6f}")

    print("\n   percentili sull'intera collezione:")
    print(f"      mean   = {np.mean(score_all):.6f}")
    print(f"      median = {np.median(score_all):.6f}")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"      p{p:02d}    = {np.percentile(score_all, p):.6f}")

    n_true = int(np.sum(is_true_all))
    n_fake = int(np.sum(~is_true_all))
    print("\n   composizione per truth-matching:")
    print(f"      vero jet-b: {n_true:8d}  ({100.0 * n_true / score_all.size:.3f}%)")
    print(f"      fake      : {n_fake:8d}  ({100.0 * n_fake / score_all.size:.3f}%)")


# ======================================================================
# EMPIRICAL EFFICIENCY ON THE CONTINUOUS AXIS
# ======================================================================

def compute_efficiency_curve(score_true, score_fake, thresholds):
    """
    Efficiency of the score cut on true b-jets and on fake jets as a
    function of the threshold.

    The cut is ``score >= threshold`` if `BJET_SF_HIGHER_IS_BETTER` is
    True, ``score <= threshold`` otherwise.

    Parameters
    ----------
    score_true : numpy.ndarray
        Scores of the true b-jets.

    score_fake : numpy.ndarray
        Scores of the fake (non-b) jets.

    thresholds : numpy.ndarray
        Thresholds to scan.

    Returns
    -------
    eff_true, eff_fake : numpy.ndarray
        Fraction of true and fake jets passing the cut, one value per
        threshold.
    """
    n_true = max(score_true.size, 1)
    n_fake = max(score_fake.size, 1)

    if BJET_SF_HIGHER_IS_BETTER:
        eff_true = np.array([np.sum(score_true >= thr) / n_true for thr in thresholds])
        eff_fake = np.array([np.sum(score_fake >= thr) / n_fake for thr in thresholds])
    else:
        eff_true = np.array([np.sum(score_true <= thr) / n_true for thr in thresholds])
        eff_fake = np.array([np.sum(score_fake <= thr) / n_fake for thr in thresholds])

    return eff_true, eff_fake


def save_calibration_plot(thresholds, eff_true, eff_fake):
    """
    Plot the b-jet and background efficiencies as a function of the
    threshold and save it to
    `OUTPUT_DIR_BJET_SF / "continuous_score_efficiency.png"`.

    Parameters
    ----------
    thresholds : numpy.ndarray
        Scanned thresholds (x-axis).

    eff_true, eff_fake : numpy.ndarray
        Output of `compute_efficiency_curve`.
    """
    if not HAS_MPL:
        return

    OUTPUT_DIR_BJET_SF.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(thresholds, eff_true, "-", color="tab:blue", linewidth=2,
            label=r"$\epsilon_b$ REALE (veri b)")
    ax.plot(thresholds, eff_fake, "--", color="tab:gray", linewidth=2,
            label=r"$\epsilon_{bkg}$ REALE (fake)")

    ax.set_xlabel(f"Valore di {JET_SF_CONTINUOUS_BRANCH}")
    ax.set_ylabel("Efficienza")
    ax.set_title("Efficienza reale in funzione della soglia continua")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out_path = OUTPUT_DIR_BJET_SF / "continuous_score_efficiency.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Plot salvato in: {out_path}")


def print_empirical_working_points(score_true, score_fake, thresholds, eff_true, eff_fake):
    """
    Print, for each target b-jet efficiency in `TARGET_EFFICIENCIES`, the
    scanned threshold whose measured efficiency is closest to the
    target, with the corresponding background efficiency and rejection.

    Parameters
    ----------
    score_true, score_fake : numpy.ndarray
        Scores of the true b-jets and of the fake jets. Currently unused;
        the table is built from the efficiency curves.

    thresholds : numpy.ndarray
        Scanned thresholds.

    eff_true, eff_fake : numpy.ndarray
        Output of `compute_efficiency_curve`.
    """
    section("SOGLIE EMPIRICHE RICALIBRATE SUL BRANCH CONTINUO")
    print(f"   {'Target Eff.':>12} | {'Soglia stimata':>15} | {'Eff. Bkg':>12} | {'Rejection':>12}")
    print("   " + "-" * 58)

    for target in TARGET_EFFICIENCIES:
        target_frac = target / 100.0
        idx = np.argmin(np.abs(eff_true - target_frac))
        thr = thresholds[idx]
        f_eff = eff_fake[idx]
        rej = 1.0 / f_eff if f_eff > 0 else np.inf

        print(f"   {target:>11d}% | {thr:>15.6f} | {f_eff*100:>11.2f}% | {rej:>12.2f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    """
    Run the continuous-score analysis on all available files.

    Scores and truth labels of the analysis jets are collected from each
    file that contains the continuous branch, then the score distribution,
    the efficiency scan, the empirical working points and the ROC curve
    are produced.
    """
    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
        return

    branches = [
        JET_SF_CONTINUOUS_BRANCH,
        JET_IS_ANALYSIS_BRANCH,
        JET_TRUTH_LABEL_BRANCH,
    ]
    score_parts, is_true_parts, counts_parts = [], [], []

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]

        if JET_SF_CONTINUOUS_BRANCH not in tree.keys():
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")
        jet_sel = get_analysis_selection(tree, JET_IS_ANALYSIS_BRANCH, n_entries)

        score_sel = a[JET_SF_CONTINUOUS_BRANCH][jet_sel]
        truth_sel = (
            ak.fill_none(a[JET_TRUTH_LABEL_BRANCH][jet_sel], -1)
            == JET_TRUTH_LABEL_B_VALUE
        )

        counts_parts.append(ak.num(score_sel, axis=1))
        score_parts.append(score_sel)
        is_true_parts.append(truth_sel)

    if not score_parts:
        print("Branch continuo non trovato nei file.")
        return

    score_all_ak = ak.concatenate(score_parts)
    is_true_all_ak = ak.concatenate(is_true_parts)
    counts_per_event = ak.to_numpy(ak.concatenate(counts_parts))

    # Truth flags are aligned to the finite scores, matching flatten_clean.
    score_all = flatten_clean(score_all_ak)
    is_true_flat = ak.to_numpy(ak.flatten(is_true_all_ak, axis=None))
    score_all_raw = ak.to_numpy(ak.flatten(score_all_ak, axis=None))
    is_true_all = is_true_flat[np.isfinite(score_all_raw)].astype(bool)

    print_continuous_properties(
        len(score_parts), counts_per_event, score_all, is_true_all
    )

    score_true = score_all[is_true_all]
    score_fake = score_all[~is_true_all]

    if score_true.size > 0:
        thresholds = np.linspace(
            np.min(score_all), np.max(score_all), BJET_SF_N_SCAN_POINTS
        )

        eff_true, eff_fake = compute_efficiency_curve(
            score_true, score_fake, thresholds
        )
        save_calibration_plot(thresholds, eff_true, eff_fake)
        print_empirical_working_points(
            score_true, score_fake, thresholds, eff_true, eff_fake
        )

        section("SUITE DI VALUTAZIONE ESTERNA (graphics.py)")

        y_score_for_roc = score_all if BJET_SF_HIGHER_IS_BETTER else -score_all

        _, _, _, auc = plot_roc_curve(
            y_true=is_true_all,
            y_score=y_score_for_roc,
            save_dir=OUTPUT_DIR_BJET_SF,
            filename="continuous_roc_curve",
        )
        print(f"   AUC: {auc:.4f}")


if __name__ == "__main__":
    main()