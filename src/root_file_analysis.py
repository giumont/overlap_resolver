"""
Esplorazione preliminare di un ntuple ATLAS HH->bbtautau (AnalysisMiniTree).

Requisiti: uproot, awkward, numpy, matplotlib
    pip install uproot awkward numpy matplotlib

Uso:
    python3 root_file_analysis.py HH_bbtt/output_GGF_mc23a_bypass_noOR_000001.root
Oppure 
    python3 root_file_analysis.py fondo_tt/output_TTBAR_mc23a_bypass_noOR_000001.root 
"""

import sys
import re
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt

TREE_NAME = "AnalysisMiniTree"

# Famiglie di branch, per raggruppare l'output in modo leggibile
PREFIX_GROUPS = [
    ("bbtt_", "Livello analisi (oggetti/coppie gia' selezionate dall'algoritmo bbtt)"),
    ("recojet_antikt4PFlow_", "Jet ricostruiti (reco, anti-kt R=0.4 PFlow)"),
    ("tau_", "Tau adronici ricostruiti (reco)"),
    ("el_", "Elettroni ricostruiti (reco)"),
    ("mu_", "Muoni ricostruiti (reco)"),
    ("truthjet_antikt4_", "Jet di verita' (truth)"),
    ("truthtau_", "Tau di verita' (truth)"),
    ("truthelectron_", "Elettroni di verita' (truth)"),
    ("truthmuon_", "Muoni di verita' (truth)"),
    ("truthmet_", "MET di verita' (truth)"),
    ("truth_", "Verita' a livello di evento/generatore (Higgs, HH, PDF info)"),
    ("trigPassed_", "Decisioni di trigger HLT specifiche"),
]


def classify_branch(name):
    for prefix, label in PREFIX_GROUPS:
        if name.startswith(prefix):
            return label
    return "Metadati di evento / varie"


def describe_structure(tree):
    """Stampa struttura generale del tree: n. entries, n. branch, tipi, raggruppamento."""
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
            typename = tree[n].typename
            print(f"    {n:65s} {typename}")


def basic_stats_for_branch(tree, branch_name, n_entries_cap=None):
    """Statistiche di base per un branch, gestendo sia scalari sia jagged array."""
    array = tree[branch_name].array(entry_stop=n_entries_cap, library="ak")

    # jagged (una lista per evento) -> appiattisco per le statistiche globali
    is_jagged = array.ndim > 1
    flat = ak.flatten(array, axis=None) if is_jagged else array

    flat_np = ak.to_numpy(flat) if len(flat) > 0 else np.array([])

    if flat_np.size == 0:
        print(f"{branch_name:65s} nessun valore disponibile")
        return None

    # solo per branch numerici
    if not np.issubdtype(flat_np.dtype, np.number):
        print(f"{branch_name:65s} tipo non numerico ({flat_np.dtype}), skip stat")
        return None

    nan_frac = np.isnan(flat_np).mean() if np.issubdtype(flat_np.dtype, np.floating) else 0.0
    valid = flat_np[~np.isnan(flat_np)] if np.issubdtype(flat_np.dtype, np.floating) else flat_np

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


def plot_distribution(values, branch_name, out_dir=".", bins=60, range_=None):
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


def main(root_path, out_dir="./output/root_files_analysis", n_entries_cap=None):
    import os
    os.makedirs(out_dir, exist_ok=True)

    with uproot.open(root_path) as f:
        tree = f[TREE_NAME]

        describe_structure(tree)

        # --- Statistiche di base + istogrammi per un set di variabili chiave ---
        # (kinematica reco/truth principale + discriminanti di ID rilevanti per l'obiettivo 3.1)
        key_branches = [
            # cinematica jet reco
            "recojet_antikt4PFlow_pt___NOSYS",
            "recojet_antikt4PFlow_eta",
            "recojet_antikt4PFlow_phi",
            # b-tagging
            "recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous",
            "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
            # cinematica tau reco
            "tau_pt___NOSYS",
            "tau_eta",
            "tau_phi",
            # discriminanti tau ID
            "tau_RNNJetScoreSigTrans",
            "tau_GNTauScoreSigTrans_v0prune",
            # variabili di evento gia' selezionate (bbtt)
            "bbtt_HH_m___NOSYS",
            "bbtt_H_bb_m___NOSYS",
            "bbtt_mmc_m___NOSYS",
        ]

        print("\n" + "=" * 80)
        print("STATISTICHE DI BASE (branch chiave)")
        print("=" * 80)

        for branch_name in key_branches:
            if branch_name not in tree.keys():
                print(f"{branch_name:45s} NON TROVATO nel tree, skip")
                continue
            result = basic_stats_for_branch(tree, branch_name, n_entries_cap=n_entries_cap)
            if result is not None and result["n_values"] > 0:
                plot_distribution(result["array"], branch_name, out_dir=out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python explore_root_file.py <path_al_file.root> [n_entries_cap]")
        sys.exit(1)

    root_path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(root_path, n_entries_cap=cap)