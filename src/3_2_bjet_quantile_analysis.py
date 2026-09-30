"""
Classification quality and calibration of the GN2v01 continuous b-tag
quantile (b-jet vs non-b jet).

Role in the project: objective 3.2, first step "quantify the quality of
the classification separately for b-jets and tau-jets". This script covers
the b-jet side for `params.JET_SCORE_BRANCH`, and is the counterpart of
``tau_score_analysis.py``. It is meant to run in the same folder as
``obj_3_1.py``, whose utilities (``section``, ``load_files``,
``get_analysis_selection``, ``summarize``) it reuses.

The branch is not a raw score: it is a discrete quantile assigned by ATLAS
from a reference calibration. Bin b in {1, ..., 6} corresponds to a nominal
b-jet efficiency of `params.JET_QUANTILE_BIN_NOMINAL_EFF` [b] (97, 90, 85,
80, 75, 70 %) for the cut ``quantile >= b``; -1 means not available. A
higher bin is therefore a tighter, more b-like selection. The question
addressed is whether these nominal efficiencies hold on this dataset.

Conventions:

- Collection: analysis-level jets (`params.JET_IS_ANALYSIS_BRANCH` != 0).
- True b-jet (signal): `params.JET_TRUTH_LABEL_BRANCH` ==
  `params.JET_TRUTH_LABEL_B_VALUE`. "Fake" is every other analysis-level
  jet, used only as reference population.
- Efficiency at bin b: P(quantile >= b | true b-jet).

Structure:

- Part A: properties of the quantile, overall and per subsample, and
  real efficiency and fake acceptance per bin.
- Part B: real vs nominal efficiency per bin, with plot.
- Part C: independent check of the ready-made working point
  `params.JET_BTAG_BRANCH` (FixedCutBEff_85, nominal efficiency
  `params.BTAG_WP85_NOMINAL_EFF` %).
- Part D: ROC curve, working points (``compute_working_points`` from
  ``flavour_tagging/src/graphics.py``) for `params.TARGET_EFFICIENCIES`,
  and evaluation plots and metrics at the 85% working point.

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
from obj_3_1 import section, load_files, get_analysis_selection, summarize

sys.path.append(str(params.FLAVOUR_TAGGING_DIR))
from src.graphics import (
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


def print_quantile_properties(n_files, counts_per_event, quant_all, is_true_all):
    """
    Print shape, range and percentiles of the quantile.

    Statistics are printed for the whole collection and, via
    ``obj_3_1.summarize``, separately for true b-jets and fakes.

    Parameters
    ----------
    n_files : int
        Number of input files used.

    counts_per_event : numpy.ndarray
        Number of analysis-level jets in each event.

    quant_all : numpy.ndarray
        Finite quantile values of all analysis-level jets.

    is_true_all : numpy.ndarray of bool
        True b-jet flag, aligned with `quant_all`.

    Returns
    -------
    None
    """
    section(f"PROPRIETA' DI {params.JET_SCORE_BRANCH} (jet analysis-level)")

    print(f"   branch (ROOT): std::vector<float>, per evento")
    print(f"   n. file processati: {n_files}")
    print(
        f"   n. jet per evento (analysis-level) - "
        f"min={np.min(counts_per_event) if counts_per_event.size else 0}  "
        f"max={np.max(counts_per_event) if counts_per_event.size else 0}  "
        f"mean={np.mean(counts_per_event) if counts_per_event.size else 0:.4f}"
    )
    print(f"   shape dopo flatten (n. jet analysis-level totali): {quant_all.shape}")

    if quant_all.size == 0:
        print("   nessun valore disponibile: interrompo la stampa delle proprieta'.")
        return

    print(f"   dtype: {quant_all.dtype}")
    print(f"\n   intervallo osservato:")
    print(f"      min = {np.min(quant_all):.6f}")
    print(f"      max = {np.max(quant_all):.6f}")

    print(f"\n   valori (percentili sull'intera collezione):")
    print(f"      mean   = {np.mean(quant_all):.6f}")
    print(f"      median = {np.median(quant_all):.6f}")
    print(f"      std    = {np.std(quant_all):.6f}")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"      p{p:02d}    = {np.percentile(quant_all, p):.6f}")

    n_true = int(np.sum(is_true_all))
    n_fake = int(np.sum(~is_true_all))
    print(
        f"\n   composizione per truth-matching "
        f"({params.JET_TRUTH_LABEL_BRANCH} == {params.JET_TRUTH_LABEL_B_VALUE} = vero jet-b):"
    )
    print(f"      vero jet-b: {n_true:8d}  ({100.0 * n_true / quant_all.size:.3f}%)")
    print(f"      fake      : {n_fake:8d}  ({100.0 * n_fake / quant_all.size:.3f}%)")

    section("PROPRIETA' PER SOTTOCAMPIONE (vero jet-b vs fake)")
    summarize(ak.Array(quant_all[is_true_all]), "quantile - vero jet-b")
    summarize(ak.Array(quant_all[~is_true_all]), "quantile - fake")


def compute_efficiency_curve(quant_true, quant_fake, thresholds):
    """
    Efficiency of the cut ``quantile >= thr`` on true b-jets and fakes.

    Parameters
    ----------
    quant_true, quant_fake : numpy.ndarray
        Quantile values of true b-jets and of fakes.

    thresholds : numpy.ndarray
        Thresholds (here the discrete bins).

    Returns
    -------
    eff_true, eff_fake : numpy.ndarray
        ``P(quantile >= thr | true b)`` and ``P(quantile >= thr | fake)``
        for each threshold. An empty population is treated as having size
        1 (efficiency 0) to avoid division by zero.
    """
    n_true = max(quant_true.size, 1)
    n_fake = max(quant_fake.size, 1)

    eff_true = np.array(
        [np.sum(quant_true >= thr) / n_true for thr in thresholds]
    )
    eff_fake = np.array(
        [np.sum(quant_fake >= thr) / n_fake for thr in thresholds]
    )

    return eff_true, eff_fake


def save_calibration_plot(thresholds, eff_true, eff_fake):
    """
    Plot real vs nominal efficiency for each discrete quantile bin.

    Shows the measured b-jet efficiency, the nominal efficiency from
    `params.JET_QUANTILE_BIN_NOMINAL_EFF` and the measured fake
    acceptance. The figure is saved as ``jet_gn2_quantile_calibration.png``
    in `params.OUTPUT_DIR_BJET_TAGGER`. Nothing is done if matplotlib is
    unavailable.

    Parameters
    ----------
    thresholds : numpy.ndarray of int
        Quantile bins (x axis); all must be keys of
        `params.JET_QUANTILE_BIN_NOMINAL_EFF`.

    eff_true, eff_fake : numpy.ndarray
        Measured efficiencies from `compute_efficiency_curve`.

    Returns
    -------
    None
    """
    if not HAS_MPL:
        print(
            "\n[WARNING] matplotlib non disponibile: "
            "skip del plot di calibrazione."
        )
        return

    params.OUTPUT_DIR_BJET_TAGGER.mkdir(parents=True, exist_ok=True)

    eff_nominal = np.array(
        [params.JET_QUANTILE_BIN_NOMINAL_EFF[int(b)] / 100.0 for b in thresholds]
    )
    nominal_levels = sorted(params.JET_QUANTILE_BIN_NOMINAL_EFF.values())

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for i, eff_val in enumerate(nominal_levels):
        ax.axhline(
            y=eff_val / 100.0,
            color="red",
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
            label="Livelli nominali WP" if i == 0 else "",
        )

    ax.plot(
        thresholds, eff_true, "o-",
        color="tab:blue", linewidth=1.8,
        label=r"$\epsilon_b$ REALE (misurata, HadronConeExclTruthLabelID==5)",
    )
    ax.plot(
        thresholds, eff_nominal, "s--",
        color="black", linewidth=1.2,
        label=r"$\epsilon_b$ NOMINALE attesa",
    )
    ax.plot(
        thresholds, eff_fake, "^:",
        color="tab:gray", linewidth=1.2,
        label=r"$\epsilon_{bkg}$ REALE (truth label $\neq$ 5)",
    )

    ax.set_xlabel(f"Bin discreto su {params.JET_SCORE_BRANCH}")
    ax.set_ylabel("Efficienza / Frazione")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(thresholds)

    y_ticks = sorted(set([0.0, 0.2, 0.4, 0.6, 1.0] + [v / 100.0 for v in nominal_levels]))
    ax.set_yticks(y_ticks)

    ax.set_title(
        "GN2v01: efficienza di b-tag reale vs nominale per bin discreto\n"
    )
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = params.OUTPUT_DIR_BJET_TAGGER / "jet_gn2_quantile_calibration.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Plot salvato in: {out_path}")


def print_nominal_vs_real(quant_true):
    """
    Print real vs nominal b-jet efficiency for each quantile bin.

    For each bin b (from the tightest to the loosest) prints the nominal
    efficiency from `params.JET_QUANTILE_BIN_NOMINAL_EFF`, the measured
    ``P(quantile >= b | true b)`` and their difference in percentage
    points.

    Parameters
    ----------
    quant_true : numpy.ndarray
        Quantile values of the true b-jets.

    Returns
    -------
    None
    """
    section("EFFICIENZA REALE VS NOMINALE PER BIN DISCRETO (punto 2)")

    n_true_tot = quant_true.size
    print(f"   {'Bin (>=)':>10} | {'Eff. Nominale':>15} | {'Eff. Reale':>12} | {'Scostamento (pp)':>18}")
    print("   " + "-" * 65)

    for b in sorted(params.JET_QUANTILE_BIN_NOMINAL_EFF, reverse=True):
        eff_real = 100.0 * np.sum(quant_true >= b) / n_true_tot
        eff_nom = params.JET_QUANTILE_BIN_NOMINAL_EFF[b]
        delta = eff_real - eff_nom
        print(f"   {b:>10d} | {eff_nom:>14.1f}% | {eff_real:>11.3f}% | {delta:>+17.3f}")


def check_fixed_wp85(a_parts, jet_sel_parts, is_true_ak_parts):
    """
    Measure the real efficiency of the ready-made FixedCutBEff_85 working point.

    Aggregates all files and computes the fraction of true b-jets with
    `params.JET_BTAG_BRANCH` != 0, to be compared with the nominal
    `params.BTAG_WP85_NOMINAL_EFF` %. Files lacking the branch are
    ignored; the check is skipped if no file has it, if the arrays are
    empty or misaligned, or if there are no true b-jets.

    Parameters
    ----------
    a_parts : list of ak.Array
        Branch arrays of each file.

    jet_sel_parts : list of ak.Array
        Analysis-level jet selection of each file.

    is_true_ak_parts : list of ak.Array
        Jagged true b-jet flags of the selected jets of each file.

    Returns
    -------
    None
    """
    section(
        "CONTROLLO INDIPENDENTE: efficienza reale del WP gia' pronto "
        f"{params.JET_BTAG_BRANCH}"
    )

    wp_pass_all = []
    is_true_all = []

    for a, jet_sel, truth_sel in zip(a_parts, jet_sel_parts, is_true_ak_parts):
        if params.JET_BTAG_BRANCH not in a.fields:
            continue
        wp_pass_sel = ak.fill_none(a[params.JET_BTAG_BRANCH][jet_sel], False) != 0
        wp_pass_all.append(flatten_clean(ak.values_astype(wp_pass_sel, np.float64)) != 0)
        is_true_all.append(flatten_clean(ak.values_astype(truth_sel, np.float64)) != 0)

    if not wp_pass_all:
        print(f"   [WARNING] branch {params.JET_BTAG_BRANCH} non trovato in nessun file. Skip.")
        return

    wp_pass_flat = np.concatenate(wp_pass_all)
    is_true_flat = np.concatenate(is_true_all)

    if is_true_flat.size == 0 or wp_pass_flat.size != is_true_flat.size:
        print("   [WARNING] array non allineati o vuoti. Skip del controllo.")
        return

    n_true = int(np.sum(is_true_flat))
    if n_true == 0:
        print("   [WARNING] nessun vero jet-b trovato. Skip del controllo.")
        return

    eff_real = float(np.sum(wp_pass_flat & is_true_flat)) / n_true

    print(
        f"   efficienza nominale WP FixedCutBEff_85: "
        f"{params.BTAG_WP85_NOMINAL_EFF}%"
    )
    print(f"   efficienza reale misurata su questo dataset: {100.0 * eff_real:.3f}%")
    print(
        f"   scostamento: {100.0 * eff_real - params.BTAG_WP85_NOMINAL_EFF:+.3f} pp"
    )


def print_recalibrated_working_points(wp_table):
    """
    Print the working-point table obtained on this dataset.

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
        "SOGLIE RICALIBRATE SUL QUANTILE (punto 4) - famiglia di working "
        f"point {params.TARGET_EFFICIENCIES}%"
    )

    print(
        f"   {'target eff.':>12} {'soglia ricalibrata':>20} "
        f"{'eff. misurata':>14} {'bkg eff.':>15} {'bkg rejection':>15}"
    )

    for wp in wp_table:
        target_pct = wp["target_efficiency"] * 100 if wp["target_efficiency"] <= 1.0 else wp["target_efficiency"]
        rej = wp["background_rejection"]
        rej_str = f"{rej:.2f}" if np.isfinite(rej) else "inf"
        print(
            f"   {target_pct:>11.0f}% {wp['threshold']:>20.1f} "
            f"{100.0 * wp['achieved_efficiency']:>13.3f}% "
            f"{100.0 * wp['background_efficiency']:>14.3f}% {rej_str:>15}"
        )

    print(
        "\n   NOTA: La 'soglia ricalibrata' identifica il bin discreto ottimale "
        "sui b veri di QUESTO dataset. Se si discosta dalle efficienze attese, "
        "il tagger necessita di ricalibrazione dell'asse."
    )


def main():
    """
    Run the b-tag quantile analysis over all input files.

    Reads quantile, truth and FixedCutBEff_85 flag of analysis-level jets
    (files lacking a required branch are skipped), then runs Parts A to D
    described in the module docstring. Outputs go to
    `params.OUTPUT_DIR_BJET_TAGGER`.

    Returns
    -------
    None
    """
    loaded = load_files()
    if not loaded:
        print("Nessun file disponibile.")
        return

    branches = [
        params.JET_SCORE_BRANCH,
        params.JET_IS_ANALYSIS_BRANCH,
        params.JET_TRUTH_LABEL_BRANCH,
        params.JET_BTAG_BRANCH,
    ]

    quant_parts = []
    is_true_parts = []
    counts_per_event_parts = []
    n_files_used = 0

    # Kept for Part C, which needs the raw arrays and selections per file.
    a_parts_for_wp_check = []
    jet_sel_parts_for_wp_check = []
    is_true_ak_parts_for_wp_check = []

    for item in loaded:
        tree, n_entries, filename = item["tree"], item["n_entries"], item["file_name"]

        keys = set(tree.keys())
        missing = [b for b in branches if b not in keys]
        if missing:
            print(f"[WARNING] {filename}: branch mancanti: {missing}")
            print("    Verificare i PLACEHOLDER in CONFIGURAZIONE. File ignorato.")
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel = get_analysis_selection(
            tree, params.JET_IS_ANALYSIS_BRANCH, n_entries,
        )

        quant_sel = a[params.JET_SCORE_BRANCH][jet_sel]
        truth_sel = ak.fill_none(
            a[params.JET_TRUTH_LABEL_BRANCH][jet_sel], -1,
        ) == params.JET_TRUTH_LABEL_B_VALUE

        counts_per_event_parts.append(ak.num(quant_sel, axis=1))
        quant_parts.append(quant_sel)
        is_true_parts.append(truth_sel)

        a_parts_for_wp_check.append(a)
        jet_sel_parts_for_wp_check.append(jet_sel)
        is_true_ak_parts_for_wp_check.append(truth_sel)

        n_files_used += 1

    if not quant_parts:
        print("Nessun file con i branch necessari. Interrompo.")
        return

    quant_all_ak = ak.concatenate(quant_parts)
    is_true_all_ak = ak.concatenate(is_true_parts)
    counts_per_event = ak.to_numpy(ak.concatenate(counts_per_event_parts))

    quant_all = flatten_clean(quant_all_ak)
    # Same finiteness mask as flatten_clean, to keep labels aligned with quantiles.
    is_true_flat = ak.to_numpy(ak.flatten(is_true_all_ak, axis=None))
    quant_all_raw = ak.to_numpy(ak.flatten(quant_all_ak, axis=None))
    finite_mask = np.isfinite(quant_all_raw)
    is_true_all = is_true_flat[finite_mask].astype(bool)

    print_quantile_properties(
        n_files_used, counts_per_event, quant_all, is_true_all,
    )

    quant_true = quant_all[is_true_all]
    quant_fake = quant_all[~is_true_all]

    if quant_true.size == 0:
        print(
            "\n[WARNING] nessun vero jet-b trovato: impossibile "
            "calcolare l'efficienza (punto 2) e le soglie (punto 4)."
        )
        return

    bins = sorted(params.JET_QUANTILE_BIN_NOMINAL_EFF)

    print("\n=== ANALISI DISCRETA PER BIN INTERO ===")
    print(f"{'Bin (>=)':>10} | {'True b passati':>15} | {'Eff. True b':>12} | {'Fake passati':>15} | {'Fake Acc.':>12}")
    print("-" * 75)

    n_true_tot = quant_true.size
    n_fake_tot = quant_fake.size

    for b in bins:
        n_t_pass = np.sum(quant_true >= b)
        n_f_pass = np.sum(quant_fake >= b)
        eff_t = n_t_pass / n_true_tot if n_true_tot else 0
        eff_f = n_f_pass / n_fake_tot if n_fake_tot else 0
        print(f"{b:>10d} | {n_t_pass:>15d} | {100.0*eff_t:>11.2f}% | {n_f_pass:>15d} | {100.0*eff_f:>11.2f}%")

    thresholds = np.array(bins, dtype=int)

    eff_true, eff_fake = compute_efficiency_curve(
        quant_true, quant_fake, thresholds,
    )

    print_nominal_vs_real(quant_true=quant_true)
    save_calibration_plot(thresholds, eff_true, eff_fake)

    check_fixed_wp85(
        a_parts_for_wp_check, jet_sel_parts_for_wp_check, is_true_ak_parts_for_wp_check,
    )

    target_effs_frac = [t / 100.0 for t in params.TARGET_EFFICIENCIES]

    fpr, tpr, thresholds_roc, auc = plot_roc_curve(
        y_true=is_true_all,
        y_score=quant_all,
        save_dir=params.OUTPUT_DIR_BJET_TAGGER,
        filename="bjet_gn2_roc_curve"
    )

    wp_table = compute_working_points(
        fpr, tpr, thresholds_roc,
        target_effs=target_effs_frac
    )

    print_recalibrated_working_points(wp_table)

    section("SUITE DI VALUTAZIONE (graphics.py)")
    print(f"   AUC calcolata: {auc:.4f}")

    plot_background_rejection(
        tpr, fpr,
        save_dir=params.OUTPUT_DIR_BJET_TAGGER,
        filename="bjet_gn2_background_rejection"
    )

    plot_score_distribution(
        y_true=is_true_all,
        y_score=quant_all,
        working_points=wp_table,
        save_dir=params.OUTPUT_DIR_BJET_TAGGER,
        filename="bjet_gn2_score_distribution"
    )

    plot_efficiency_vs_threshold(
        y_true=is_true_all,
        y_score=quant_all,
        save_dir=params.OUTPUT_DIR_BJET_TAGGER,
        filename="bjet_gn2_eff_vs_threshold"
    )

    thr_85 = next((wp["threshold"] for wp in wp_table if abs(wp["target_efficiency"] - 0.85) < 1e-4), 3.0)

    plot_confusion_matrix(
        y_true=is_true_all,
        y_score=quant_all,
        threshold=thr_85,
        save_dir=params.OUTPUT_DIR_BJET_TAGGER,
        filename="bjet_gn2_confusion_matrix_85"
    )

    metrics_85 = compute_classification_metrics(
        is_true_all, quant_all, threshold=thr_85
    )

    print(f"\n   Metriche a efficienza 85% (soglia bin >= {thr_85:.0f}):")
    for key, value in metrics_85.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.4f}")
        else:
            print(f"      {key}: {value}")

    section("FINE")


if __name__ == "__main__":
    main()