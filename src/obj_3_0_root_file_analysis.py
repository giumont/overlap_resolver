"""
Preliminary exploration of a ntuple
(``AnalysisMiniTree``).

Role in the project: first step before objectives 3.1 and 3.2. It lists
the branches of the tree grouped by object family (reco jets, reco taus,
truth objects, bbtt analysis-level variables, ...), and prints summary
statistics and saves a distribution plot for the key branches (kinematics,
b-tagging and tau ID discriminants, truth labels) used in the overlap and
classification studies. It works on both the signal (GGF) and the ttbar
background files.

Requirements: uproot, awkward, numpy, matplotlib.

Usage::

    python3 root_file_analysis.py HH_bbtt/output_GGF_mc23a_bypass_noOR_000001.root [n_entries_cap]
    python3 root_file_analysis.py fondo_tt/output_TTBAR_mc23a_bypass_noOR_000001.root [n_entries_cap]

All configuration (tree name, branch groups, key branches, output
directory, number of bins) lives in ``params.py``.
"""

import os
import re
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

import params


def classify_branch(name):
    """
    Assign a branch to its family based on the name prefix.

    Parameters
    ----------
    name : str
        Branch name.

    Returns
    -------
    str
        Description of the first group in `params.BRANCH_PREFIX_GROUPS`
        whose prefix matches `name`, or `params.BRANCH_GROUP_DEFAULT` if
        none matches.
    """
    for prefix, label in params.BRANCH_PREFIX_GROUPS:
        if name.startswith(prefix):
            return label
    return params.BRANCH_GROUP_DEFAULT


def describe_structure(tree):
    """
    Print the general structure of the tree.

    Reports the number of entries and branches, then lists every branch
    with its type, grouped by family (see `classify_branch`) and with the
    largest families first.

    Parameters
    ----------
    tree : uproot.TTree
        Tree to describe.

    Returns
    -------
    None
    """
    print("=" * 80)
    print(f"TTree: {tree.name}")
    print(f"Numero di eventi (entries): {tree.num_entries}")
    print(f"Numero di branch: {len(tree.keys())}")
    print("=" * 80)

    groups = {}
    for name in tree.keys():
        groups.setdefault(classify_branch(name), []).append(name)

    print("\n--- Raggruppamento branch per famiglia ---")
    for label, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n[{label}]  ({len(names)} branch)")
        for n in sorted(names):
            print(f"    {n:65s} {tree[n].typename}")


def basic_stats_for_branch(tree, branch_name, n_entries_cap=None):
    """
    Print and return summary statistics of one branch.

    Works for both scalar (one value per event) and jagged (one list per
    event) branches; jagged branches are flattened over all objects.
    NaN values are excluded from mean, std, min and max.

    Parameters
    ----------
    tree : uproot.TTree
        Tree containing the branch.

    branch_name : str
        Name of the branch to summarise.

    n_entries_cap : int or None, default=None
        Maximum number of events read; None reads all of them.

    Returns
    -------
    dict or None
        None if the branch is empty or non-numeric. Otherwise a dict with
        keys ``array`` (flattened numpy array, NaNs included),
        ``n_values``, ``n_objects_per_event_avg`` (1.0 for scalar
        branches), ``mean``, ``std``, ``min``, ``max`` and ``nan_frac``.
    """
    array = tree[branch_name].array(entry_stop=n_entries_cap, library="ak")

    is_jagged = array.ndim > 1
    flat = ak.flatten(array, axis=None) if is_jagged else array
    flat_np = ak.to_numpy(flat) if len(flat) > 0 else np.array([])

    if flat_np.size == 0:
        print(f"{branch_name:65s} nessun valore disponibile")
        return None

    if not np.issubdtype(flat_np.dtype, np.number):
        print(f"{branch_name:65s} tipo non numerico ({flat_np.dtype}), skip stat")
        return None

    is_float = np.issubdtype(flat_np.dtype, np.floating)
    nan_frac = np.isnan(flat_np).mean() if is_float else 0.0
    valid = flat_np[~np.isnan(flat_np)] if is_float else flat_np

    stats = {
        "n_values": flat_np.size,
        "n_objects_per_event_avg": (flat_np.size / len(array)) if is_jagged else 1.0,
        "mean": float(np.mean(valid)) if valid.size else float("nan"),
        "std": float(np.std(valid)) if valid.size else float("nan"),
        "min": float(np.min(valid)) if valid.size else float("nan"),
        "max": float(np.max(valid)) if valid.size else float("nan"),
        "nan_frac": float(nan_frac),
    }

    print(
        f"{branch_name:45s} n={stats['n_values']:>9d}  "
        f"mean={stats['mean']:>10.4g}  std={stats['std']:>10.4g}  "
        f"min={stats['min']:>10.4g}  max={stats['max']:>10.4g}  "
        f"nan_frac={stats['nan_frac']:.3f}"
    )
    return {"array": flat_np, **stats}


def plot_distribution(values, branch_name, out_dir=".", bins=params.EXPLORATION_N_BINS, range_=None):
    """
    Save a histogram of a branch to ``<out_dir>/dist_<branch>.png``.

    Parameters
    ----------
    values : array-like
        Flattened values to histogram.

    branch_name : str
        Branch name, used as x-axis label, title and (with non-alphanumeric
        characters replaced by ``_``) in the file name.

    out_dir : str, default="."
        Output directory; must already exist.

    bins : int, default=`params.EXPLORATION_N_BINS`
        Number of histogram bins.

    range_ : tuple of float or None, default=None
        ``(min, max)`` histogram range; None uses the data range.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, range=range_, histtype="stepfilled", alpha=0.7)
    ax.set_xlabel(branch_name)
    ax.set_ylabel("Eventi / oggetti")
    ax.set_title(f"Distribuzione: {branch_name}")
    fig.tight_layout()
    out_path = f"{out_dir}/dist_{re.sub('[^A-Za-z0-9_]', '_', branch_name)}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  -> istogramma salvato in {out_path}")


def main(root_path, out_dir=params.OUTPUT_DIR_ROOT_EXPLORATION, n_entries_cap=params.N_ENTRIES_CAP):
    """
    Explore one ntuple: structure, then statistics and plots of key branches.

    Prints the branch structure (`describe_structure`), then for each
    branch in `params.EXPLORATION_KEY_BRANCHES` prints summary statistics
    and saves a distribution plot. Branches missing from the tree are
    reported and skipped.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Path of the ``.root`` file to open.

    out_dir : str or pathlib.Path, default=`params.OUTPUT_DIR_ROOT_EXPLORATION`
        Directory where the plots are saved (created if missing).

    n_entries_cap : int or None, default=`params.N_ENTRIES_CAP`
        Maximum number of events read per branch; None reads all of them.

    Returns
    -------
    None
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with uproot.open(root_path) as f:
        tree = f[params.TREE_NAME]

        describe_structure(tree)

        print("\n" + "=" * 80)
        print("STATISTICHE DI BASE (branch chiave)")
        print("=" * 80)

        for branch_name in params.EXPLORATION_KEY_BRANCHES:
            if branch_name not in tree.keys():
                print(f"{branch_name:45s} NON TROVATO nel tree, skip")
                continue
            result = basic_stats_for_branch(tree, branch_name, n_entries_cap=n_entries_cap)
            if result is not None and result["n_values"] > 0:
                plot_distribution(result["array"], branch_name, out_dir=out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python root_file_analysis.py <path_al_file.root> [n_entries_cap]")
        sys.exit(1)

    cap = int(sys.argv[2]) if len(sys.argv) > 2 else params.N_ENTRIES_CAP
    main(sys.argv[1], n_entries_cap=cap)