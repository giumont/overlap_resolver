"""
Classification quality of a tau ID score (hadronic tau vs fake).

Role in the project: objective 3.2, first step "quantify the quality of
the classification separately for b-jets and tau-jets". This script
covers the tau side for the score `params.TAU_SCORE_ANALYSIS_BRANCH`, and
is the counterpart of ``bjet_quantile_analysis.py``. It is meant to run in
the same folder as ``obj_3_1.py``, whose utilities (``section``,
``load_files``, ``get_analysis_selection``, ``summarize``) it reuses.

Conventions:

- Collection: analysis-level taus (`params.TAU_IS_ANALYSIS_BRANCH` != 0).
- True hadronic tau (signal): `params.TAU_TRUTH_MATCH_BRANCH` != 0, the
  same "label" convention used elsewhere in the pipeline. "Fake" is every
  other analysis-level tau; it is only a reference population (it may
  contain real taus not truth-matched at visible level, or jets).
- Score: higher = more signal-like. Signal efficiency at threshold thr is
  P(score >= thr | true hadronic tau). The direction must be checked on
  the plots; if the two populations turn out inverted, the cut direction
  has to be reversed before using the working points.
- Working points: thresholds giving the signal efficiencies
  `params.TARGET_EFFICIENCIES`, the same family of targets used for the
  b-tagger (same definition, not the same numerical thresholds, which are
  specific to each discriminant).

Structure:

- Part A: properties of the score, overall and per subsample.
- Part B: signal/fake efficiency vs threshold scan, with plot.
- Part C: ROC curve, working points (``compute_working_points`` from
  ``flavour_tagging/src/graphics.py``) and evaluation plots and metrics at
  the 85% working point.

Configuration is in ``params.py``.
"""

import sys

import numpy as np
import awkward as ak

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

import params
from obj_3_1_geometric_overlap import section, load_files, get_analysis_selection, summarize

from flavour_tag_ml.graphics import (
    plot_roc_curve,
    plot_background_rejection,
    plot_score_distribution,
    plot_efficiency_vs_threshold,
    plot_confusion_matrix,
    compute_classification_metrics,
    compute_working_points,
)


def flatten_clean(values):
    """
    Flatten an array to 1D numpy, dropping None and non-finite values.

    Parameters
    ----------
    values : ak.Array
        Jagged or regular array.

    Returns
    -------
    numpy.ndarray
        1D array; for floating dtypes NaN and inf are removed.
    """
    flat = ak.flatten(values, axis=None)
    flat = ak.drop_none(flat)
    flat = ak.to_numpy(flat)

    if flat.size == 0:
        return flat

    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]

    return flat


def print_score_properties(n_taus_raw, counts_per_event, score_all, is_true_all):
    """
    Print shape, range and percentiles of the score.

    Statistics are printed for the whole collection and, via
    ``obj_3_1.summarize``, separately for true hadronic taus and fakes.

    Parameters
    ----------
    n_taus_raw : int
        Number of input files used (printed as "eventi processati").

    counts_per_event : numpy.ndarray
        Number of analysis-level taus in each event.

    score_all : numpy.ndarray
        Finite scores of all analysis-level taus.

    is_true_all : numpy.ndarray of bool
        True hadronic tau flag, aligned with `score_all`.

    Returns
    -------
    None
    """
    section(f"PROPRIETA' DI {params.TAU_SCORE_ANALYSIS_BRANCH} (tau analysis-level)")

    print(f"   branch (ROOT): std::vector<float>, per evento")
    print(f"   n. eventi processati: {n_taus_raw}")
    print(
        f"   n. tau per evento (analysis-level) - "
        f"min={np.min(counts_per_event) if counts_per_event.size else 0}  "
        f"max={np.max(counts_per_event) if counts_per_event.size else 0}  "
        f"mean={np.mean(counts_per_event) if counts_per_event.size else 0:.4f}"
    )
    print(f"   shape dopo flatten (n. tau analysis-level totali): {score_all.shape}")

    if score_all.size == 0:
        print("   nessun valore disponibile: interrompo la stampa delle proprieta'.")
        return

    print(f"   dtype: {score_all.dtype}")
    print(f"\n   intervallo osservato:")
    print(f"      min = {np.min(score_all):.6f}")
    print(f"      max = {np.max(score_all):.6f}")

    print(f"\n   valori (percentili sull'intera collezione):")
    print(f"      mean   = {np.mean(score_all):.6f}")
    print(f"      median = {np.median(score_all):.6f}")
    print(f"      std    = {np.std(score_all):.6f}")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"      p{p:02d}    = {np.percentile(score_all, p):.6f}")

    n_true = int(np.sum(is_true_all))
    n_fake = int(np.sum(~is_true_all))
    print(
        f"\n   composizione per truth-matching "
        f"({params.TAU_TRUTH_MATCH_BRANCH} != 0 = vero tau adronico):"
    )
    print(f"      vero tau adronico: {n_true:8d}  ({100.0 * n_true / score_all.size:.3f}%)")
    print(f"      fake             : {n_fake:8d}  ({100.0 * n_fake / score_all.size:.3f}%)")

    section("PROPRIETA' PER SOTTOCAMPIONE (vero tau adronico vs fake)")
    summarize(ak.Array(score_all[is_true_all]), "score - vero tau adronico")
    summarize(ak.Array(score_all[~is_true_all]), "score - fake")


def compute_efficiency_curve(score_true, score_fake, thresholds):
    """
    Efficiency of the cut ``score >= thr`` on signal and fake taus.

    Parameters
    ----------
    score_true, score_fake : numpy.ndarray
        Scores of true hadronic taus and of fakes.

    thresholds : numpy.ndarray
        Thresholds to scan.

    Returns
    -------
    eff_true, eff_fake : numpy.ndarray
        ``P(score >= thr | true tau)`` and ``P(score >= thr | fake)`` for
        each threshold. An empty population is treated as having size 1
        (efficiency 0) to avoid division by zero.
    """
    n_true = max(score_true.size, 1)
    n_fake = max(score_fake.size, 1)

    eff_true = np.array(
        [np.sum(score_true >= thr) / n_true for thr in thresholds]
    )
    eff_fake = np.array(
        [np.sum(score_fake >= thr) / n_fake for thr in thresholds]
    )

    return eff_true, eff_fake


def save_efficiency_plot(thresholds, eff_true, eff_fake, wp_table):
    """
    Plot signal and fake efficiency vs score threshold, with working points.

    The figure is saved in `params.OUTPUT_DIR_TAU_TAGGER`. Nothing is done
    if matplotlib is unavailable.

    Parameters
    ----------
    thresholds : numpy.ndarray
        Scanned thresholds (x axis).

    eff_true, eff_fake : numpy.ndarray
        Efficiencies from `compute_efficiency_curve`.

    wp_table : list of dict
        Working points from ``compute_working_points``; uses the keys
        ``threshold`` and ``target_efficiency`` (fraction or percent).
        Each one is drawn as a vertical line labelled with its target.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        print("\n[WARNING] matplotlib non disponibile: skip del plot di efficienza.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        thresholds, eff_true,
        color="tab:blue", linewidth=1.8,
        label=r"$\epsilon_{\tau}$, (tau_truth_IsHadronicTau == 1)",
    )
    ax.plot(
        thresholds, eff_fake,
        color="tab:gray", linewidth=1.2, linestyle="--",
        label=r"$\epsilon_{bkg}$, (tau_truth_IsHadronicTau == 0)",
    )

    for wp in wp_table:
        thr_wp = wp["threshold"]
        target_pct = wp["target_efficiency"] * 100 if wp["target_efficiency"] <= 1.0 else wp["target_efficiency"]
        ax.axvline(thr_wp, color="tab:red", linestyle=":", linewidth=1.0)
        ax.text(
            thr_wp, 1.02, f"{target_pct:.0f}%",
            rotation=0, ha="center", va="bottom",
            fontsize=8, color="tab:red",
        )

    ax.set_xlabel(f"soglia su {params.TAU_SCORE_ANALYSIS_BRANCH}")
    ax.set_ylabel("efficienza")
    ax.set_ylim(0.0, 1.08)
    ax.set_title(
        "Efficienza di riconoscimento del tau al variare dello score\n"
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = params.OUTPUT_DIR_TAU_TAGGER / (
        f"tau_gn_score{params.TAU_SCORE_ANALYSIS_BRANCH}_efficiency_vs_threshold.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot salvato in: {out_path}")


def print_working_points(wp_table):
    """
    Print the working-point table.

    Parameters
    ----------
    wp_table : list of dict
        Output of ``compute_working_points``; uses the keys
        ``target_efficiency`` (fraction or percent), ``threshold``,
        ``achieved_efficiency``, ``background_efficiency`` and
        ``background_rejection``.

    Returns
    -------
    None
    """
    section(
        "SOGLIE OPERATIVE COMPARABILI AL B-TAGGING (punto 3) - "
        f"famiglia di working point {params.TARGET_EFFICIENCIES}% "
        "(stessa logica FixedCutBEff)"
    )

    print(
        f"   {'target eff.':>12} {'soglia score':>14} "
        f"{'eff. misurata':>14} {'bkg eff.':>16} {'bkg rejection':>15}"
    )

    for wp in wp_table:
        target_pct = wp["target_efficiency"] * 100 if wp["target_efficiency"] <= 1.0 else wp["target_efficiency"]
        rej = wp["background_rejection"]
        rej_str = f"{rej:.2f}" if np.isfinite(rej) else "inf"
        print(
            f"   {target_pct:>11.0f}% {wp['threshold']:>14.6f} "
            f"{100.0 * wp['achieved_efficiency']:>13.3f}% "
            f"{100.0 * wp['background_efficiency']:>15.3f}% {rej_str:>15}"
        )


def main():
    """
    Run the tau score analysis over all input files.

    Reads score and truth of analysis-level taus (files lacking a required
    branch are skipped), prints the score properties (Part A), scans the
    efficiency vs threshold and saves its plot (Part B), then computes the
    ROC curve, the working points and the evaluation plots and metrics at
    the 85% working point (Part C). Outputs go to
    `params.OUTPUT_DIR_TAU_TAGGER`.

    Returns
    -------
    None
    """
    params.OUTPUT_DIR_TAU_TAGGER.mkdir(parents=True, exist_ok=True)

    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
        return

    branches = [
        params.TAU_SCORE_ANALYSIS_BRANCH,
        params.TAU_IS_ANALYSIS_BRANCH,
        params.TAU_TRUTH_MATCH_BRANCH,
    ]

    score_parts = []
    is_true_parts = []
    counts_per_event_parts = []
    n_files_used = 0

    for item in loaded:
        tree, n_entries, filename = item["tree"], item["n_entries"], item["file_name"]

        keys = set(tree.keys())
        missing = [b for b in branches if b not in keys]
        if missing:
            print(f"[WARNING] {filename}: branch mancanti: {missing}")
            print("    Verificare i PLACEHOLDER in CONFIGURAZIONE. File ignorato.")
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        tau_sel = get_analysis_selection(
            tree, params.TAU_IS_ANALYSIS_BRANCH, n_entries,
        )

        score_sel = a[params.TAU_SCORE_ANALYSIS_BRANCH][tau_sel]
        truth_sel = ak.fill_none(a[params.TAU_TRUTH_MATCH_BRANCH][tau_sel], 0) != 0

        counts_per_event_parts.append(ak.num(score_sel, axis=1))
        score_parts.append(score_sel)
        is_true_parts.append(truth_sel)

        n_files_used += 1

    if not score_parts:
        print("Nessun file con i branch necessari. Interrompo.")
        return

    score_all_ak = ak.concatenate(score_parts)
    is_true_all_ak = ak.concatenate(is_true_parts)
    counts_per_event = ak.to_numpy(ak.concatenate(counts_per_event_parts))

    score_all = flatten_clean(score_all_ak)
    # Same finiteness mask as flatten_clean, to keep labels aligned with scores.
    is_true_flat = ak.to_numpy(ak.flatten(is_true_all_ak, axis=None))
    score_all_raw = ak.to_numpy(ak.flatten(score_all_ak, axis=None))
    finite_mask = np.isfinite(score_all_raw)
    is_true_all = is_true_flat[finite_mask].astype(bool)

    print_score_properties(
        n_files_used, counts_per_event, score_all, is_true_all,
    )

    score_true = score_all[is_true_all]
    score_fake = score_all[~is_true_all]

    if score_true.size == 0:
        print(
            "\n[WARNING] nessun vero tau adronico trovato: impossibile "
            "calcolare l'efficienza (punto 2) e le soglie (punto 3)."
        )
        return

    thresholds = np.linspace(
        np.min(score_all), np.max(score_all), params.N_SCAN_POINTS,
    )
    eff_true, eff_fake = compute_efficiency_curve(
        score_true, score_fake, thresholds,
    )

    section("CURVA DI EFFICIENZA (punto 2)")
    print(
        f"   soglia scansionata fra {thresholds[0]:.6f} e "
        f"{thresholds[-1]:.6f} ({params.N_SCAN_POINTS} punti)"
    )
    for target in params.TARGET_EFFICIENCIES:
        idx = int(np.argmin(np.abs(eff_true - target / 100.0)))
        print(
            f"   eff. riconoscimento tau ~= {target}%  "
            f"-> soglia score ~= {thresholds[idx]:.6f}  "
            f"(eff. misurata alla soglia scansionata = "
            f"{100.0 * eff_true[idx]:.3f}%)"
        )

    target_effs_frac = [t / 100.0 for t in params.TARGET_EFFICIENCIES]

    fpr, tpr, thresholds_roc, auc = plot_roc_curve(
        y_true=is_true_all,
        y_score=score_all,
        save_dir=params.OUTPUT_DIR_TAU_TAGGER,
        filename="tau_roc_curve"
    )

    wp_table = compute_working_points(
        fpr, tpr, thresholds_roc,
        target_effs=target_effs_frac
    )

    print_working_points(wp_table)
    save_efficiency_plot(thresholds, eff_true, eff_fake, wp_table)

    section("SUITE DI VALUTAZIONE (graphics.py)")
    print(f"   AUC calcolata: {auc:.4f}")

    plot_background_rejection(
        tpr, fpr,
        save_dir=params.OUTPUT_DIR_TAU_TAGGER,
        filename="tau_background_rejection",
        eff_range=(0.7, 1)  # efficiency range available for the b-tagger
    )

    plot_score_distribution(
        y_true=is_true_all,
        y_score=score_all,
        working_points=wp_table,
        save_dir=params.OUTPUT_DIR_TAU_TAGGER,
        filename="tau_score_distribution"
    )

    plot_efficiency_vs_threshold(
        y_true=is_true_all,
        y_score=score_all,
        save_dir=params.OUTPUT_DIR_TAU_TAGGER,
        filename="tau_eff_vs_threshold"
    )

    thr_85 = next((wp["threshold"] for wp in wp_table if abs(wp["target_efficiency"] - 0.85) < 1e-4), 0.5)

    plot_confusion_matrix(
        y_true=is_true_all,
        y_score=score_all,
        threshold=thr_85,
        save_dir=params.OUTPUT_DIR_TAU_TAGGER,
        filename="tau_confusion_matrix_85"
    )

    metrics_85 = compute_classification_metrics(
        is_true_all, score_all, threshold=thr_85
    )

    print(f"\n   Metriche a efficienza 85% (soglia = {thr_85:.6f}):")
    for key, value in metrics_85.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.4f}")
        else:
            print(f"      {key}: {value}")

    section("FINE")


if __name__ == "__main__":
    main()