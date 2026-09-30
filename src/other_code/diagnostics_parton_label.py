"""
Diagnostic of the truth information of jets labelled as taus (label 15).

Role in the project: the truth categories used in objectives 3.1 and 3.2
(true/fake b-jet x true/fake hadronic tau) rely on the truth labels of the
signal sample. This script checks whether the reco jets with
``HadronConeExclTruthLabelID == 15`` (jets that are the hadronic shower of
a tau, i.e. physical jet-tau overlap) carry consistent truth information.
Two subpopulations are compared:

- COHERENT: label == 15, ``parentHiggsParentsMask != 0`` and equal to the
  mask of the true hadronic taus of the same event. In the previous
  diagnostic (``diagnostics_tau_vs_taujet.py``, STEP 1) these are ~99.8% of
  the label-15 jets with non-zero mask, with a median DeltaR of ~0.009 to
  the true tau: the same hadronic shower reconstructed twice.
- MASK==0: label == 15 but ``parentHiggsParentsMask == 0`` (not a child of
  any Higgs), despite a minimum DeltaR to the true tau similar to the
  coherent one (median ~0.014 in the previous test, without filters).

The "incoherent" jets of the previous diagnostic (mask != 0 but different
from the tau one) have mask != 0, so they belong to neither population.

For each population the script prints the distribution of
``recojet_antikt4PFlow_PartonTruthLabelID`` (a large fraction of -1 for
MASK==0 would point to truth information not propagated upstream; well
defined parton labels would point to a different issue) and the minimum
DeltaR to the nearest true tau of the same event. Finally it looks at the
parton label of the MASK==0 jets with a very small DeltaR
(`params.DIAG_DR_VERY_CLOSE`).

The analysis runs on the signal sample only (``params.SAMPLES["signal"]``),
where the Higgs genealogy exists. Paths come from ``params.py`` and are
resolved relative to its folder, since this file lives in ``other_code/``.

Usage::

    python diagnostics_parton_label.py [path/to/file.root] [n_entries_cap]

Without a path, the first signal file of ``params.py`` is used.

Requirements: uproot, awkward, numpy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import awkward as ak
import numpy as np
import uproot

import params


def default_root_path():
    """
    Path of the first signal file defined in ``params.py``.

    Returns
    -------
    pathlib.Path
        ``PROJECT_DIR / root_dir / <prefix>01<suffix>`` of the signal
        sample.
    """
    signal = params.SAMPLES["signal"]
    return params.PROJECT_DIR / signal["root_dir"] / f"{signal['file_prefix']}01{params.FILE_SUFFIX}"


def delta_phi(phi1, phi2):
    """
    Azimuthal difference wrapped to [-pi, pi).

    Parameters
    ----------
    phi1, phi2 : array-like
        Azimuthal angles in radians.

    Returns
    -------
    array-like
        ``phi1 - phi2`` wrapped to [-pi, pi).
    """
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    """
    Angular distance DeltaR = sqrt(deta^2 + dphi^2).

    Parameters
    ----------
    eta1, phi1, eta2, phi2 : array-like
        Pseudorapidity and azimuth of the two objects.

    Returns
    -------
    array-like
        DeltaR, with the same structure as the inputs.
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta ** 2 + dphi ** 2)


def section(title):
    """
    Print a title between two separator lines.

    Parameters
    ----------
    title : str
        Text to print.

    Returns
    -------
    None
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_dr(values, label):
    """
    Print n, mean, median, p10 and p90 of a DeltaR array.

    Parameters
    ----------
    values : ak.Array
        DeltaR values (any nesting, flattened).

    label : str
        Label printed before the statistics.

    Returns
    -------
    None
    """
    flat = ak.to_numpy(ak.flatten(values, axis=None))
    if flat.size == 0:
        print(f"   {label}: nessun valore disponibile")
        return
    print(f"   {label}: n={flat.size:>7d}  mean={np.mean(flat):.4f}  "
          f"median={np.median(flat):.4f}  p10={np.percentile(flat,10):.4f}  "
          f"p90={np.percentile(flat,90):.4f}")


def value_counts_flat(values, top=15):
    """
    Print the most frequent distinct values of an array.

    Parameters
    ----------
    values : ak.Array
        Values to count (any nesting, flattened).

    top : int, default=15
        Number of values printed, by decreasing count; the number of
        remaining distinct values is reported.

    Returns
    -------
    None
    """
    flat = ak.to_numpy(ak.flatten(values, axis=None))
    if flat.size == 0:
        print("      nessun valore disponibile")
        return
    vals, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    for v, c in list(zip(vals[order], counts[order]))[:top]:
        print(f"      {v!r:>8}  count={c:>8d}  ({100 * c / flat.size:5.2f}%)")
    if len(vals) > top:
        print(f"      ... e altri {len(vals) - top} valori distinti")


def main(root_path, n_entries_cap=None):
    """
    Compare the COHERENT and MASK==0 populations of label-15 jets.

    Events are restricted to those where all true hadronic taus with
    non-zero mask share the same mask (the reference mask). Prints the
    parton-label distribution and the fraction of label -1 of both
    populations, their minimum DeltaR to the true taus, and the parton
    label of the MASK==0 jets with DeltaR < `params.DIAG_DR_VERY_CLOSE`.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Path of the ``.root`` file.

    n_entries_cap : int or None, default=None
        Maximum number of events read; None reads all of them.

    Returns
    -------
    None
    """
    branches = [
        params.TAU_ETA_BRANCH, params.TAU_PHI_BRANCH, params.TAU_TRUTH_MATCH_BRANCH,
        params.TAU_PARENT_HIGGS_MASK_BRANCH, params.TAU_IS_ANALYSIS_BRANCH,
        params.JET_ETA_BRANCH, params.JET_PHI_BRANCH,
        params.JET_TRUTH_LABEL_BRANCH,
        params.JET_PARENT_HIGGS_MASK_BRANCH,
        params.JET_PARTON_LABEL_BRANCH,
        params.JET_IS_ANALYSIS_BRANCH,
    ]

    with uproot.open(root_path) as f:
        tree = f[params.TREE_NAME]
        a = tree.arrays(branches, entry_stop=n_entries_cap, library="ak")

    tau_sel = a[params.TAU_IS_ANALYSIS_BRANCH] == 1
    jet_sel = a[params.JET_IS_ANALYSIS_BRANCH] == 1

    tau_eta = a[params.TAU_ETA_BRANCH][tau_sel]
    tau_phi = a[params.TAU_PHI_BRANCH][tau_sel]
    tau_ishad = a[params.TAU_TRUTH_MATCH_BRANCH][tau_sel]
    tau_mask = a[params.TAU_PARENT_HIGGS_MASK_BRANCH][tau_sel]

    jet_eta = a[params.JET_ETA_BRANCH][jet_sel]
    jet_phi = a[params.JET_PHI_BRANCH][jet_sel]
    jet_label = a[params.JET_TRUTH_LABEL_BRANCH][jet_sel]
    jet_mask = a[params.JET_PARENT_HIGGS_MASK_BRANCH][jet_sel]
    jet_parton = a[params.JET_PARTON_LABEL_BRANCH][jet_sel]

    real_tau = tau_ishad == 1
    real_tau_mask = tau_mask[real_tau]
    nonzero_real_tau_mask = real_tau_mask[real_tau_mask != 0]

    has_ref = ak.num(nonzero_real_tau_mask, axis=1) > 0
    min_mask = ak.min(nonzero_real_tau_mask, axis=1)
    max_mask = ak.max(nonzero_real_tau_mask, axis=1)
    consistent_tau_evt = ak.fill_none(min_mask == max_mask, True)

    valid_evt = has_ref & consistent_tau_evt
    ref_mask = min_mask

    real_tau_eta_v = tau_eta[real_tau][valid_evt]
    real_tau_phi_v = tau_phi[real_tau][valid_evt]
    ref_mask_v = ref_mask[valid_evt]

    jet15 = jet_label == params.JET_TRUTH_LABEL_TAU_VALUE

    jet15_nonzero = jet15 & (jet_mask != 0)

    jet_coh_mask = jet_mask[jet15_nonzero][valid_evt]
    jet_coh_eta = jet_eta[jet15_nonzero][valid_evt]
    jet_coh_phi = jet_phi[jet15_nonzero][valid_evt]
    jet_coh_parton = jet_parton[jet15_nonzero][valid_evt]

    match_ok = jet_coh_mask == ref_mask_v

    coh_eta = jet_coh_eta[match_ok]
    coh_phi = jet_coh_phi[match_ok]
    coh_parton = jet_coh_parton[match_ok]

    jet15_mask0 = jet15 & (jet_mask == 0)

    mask0_eta = jet_eta[jet15_mask0][valid_evt]
    mask0_phi = jet_phi[jet15_mask0][valid_evt]
    mask0_parton = jet_parton[jet15_mask0][valid_evt]

    section("PartonTruthLabelID - popolazione COERENTE (label==15, mask!=0, concorde col tau)")
    value_counts_flat(coh_parton)

    section("PartonTruthLabelID - popolazione MASK==0 (label==15, mask==0)")
    value_counts_flat(mask0_parton)

    n_coh = int(ak.sum(ak.num(coh_parton, axis=1)))
    n_coh_undef = int(ak.sum(ak.sum(coh_parton == -1, axis=1)))
    n_m0 = int(ak.sum(ak.num(mask0_parton, axis=1)))
    n_m0_undef = int(ak.sum(ak.sum(mask0_parton == -1, axis=1)))

    print(f"\n   COERENTE : {n_coh_undef}/{n_coh} ({100*n_coh_undef/max(n_coh,1):.2f}%) con PartonTruthLabelID==-1")
    print(f"   MASK==0  : {n_m0_undef}/{n_m0} ({100*n_m0_undef/max(n_m0,1):.2f}%) con PartonTruthLabelID==-1")
    print("\n   Interpretazione:")
    print("   - se MASK==0 ha una frazione di -1 molto piu' alta della COERENTE: e' un indizio")
    print("     che l'informazione di verita' (parton E genealogia) e' incompleta/indefinita per")
    print("     questi jet, a sostegno di un problema di propagazione a monte (ipotesi 1).")
    print("   - se MASK==0 mostra invece parton-label ben definiti (in particolare 5=b, non -1):")
    print("     significherebbe che sono jet con un flavour parton chiaro ma genealogia Higgs")
    print("     non tracciata, un pattern diverso da investigare ulteriormente.")

    section("DeltaR(jet, tau vero piu' vicino) - confronto diretto fra le due popolazioni")

    jet_eta_c1, tau_eta_c1 = ak.unzip(ak.cartesian([coh_eta, real_tau_eta_v], nested=True))
    jet_phi_c1, tau_phi_c1 = ak.unzip(ak.cartesian([coh_phi, real_tau_phi_v], nested=True))
    dr_coh = ak.min(delta_r(jet_eta_c1, jet_phi_c1, tau_eta_c1, tau_phi_c1), axis=-1)

    jet_eta_c2, tau_eta_c2 = ak.unzip(ak.cartesian([mask0_eta, real_tau_eta_v], nested=True))
    jet_phi_c2, tau_phi_c2 = ak.unzip(ak.cartesian([mask0_phi, real_tau_phi_v], nested=True))
    dr_m0 = ak.min(delta_r(jet_eta_c2, jet_phi_c2, tau_eta_c2, tau_phi_c2), axis=-1)

    summarize_dr(dr_coh, "DeltaR min - popolazione COERENTE")
    summarize_dr(dr_m0, "DeltaR min - popolazione MASK==0")

    dr_close = params.DIAG_DR_VERY_CLOSE
    section(f"Incrocio: jet MASK==0 con DeltaR molto piccolo (<{dr_close}) - PartonTruthLabelID")

    dr_m0_flat = ak.flatten(dr_m0, axis=None)
    parton_m0_flat = ak.flatten(mask0_parton, axis=None)
    very_close = ak.to_numpy(dr_m0_flat) < dr_close
    n_very_close = int(np.sum(very_close))
    print(f"   jet MASK==0 con DeltaR<{dr_close} al tau vero: {n_very_close} su {len(dr_m0_flat)} "
          f"({100*n_very_close/max(len(dr_m0_flat),1):.2f}%)")
    if n_very_close > 0:
        parton_close = ak.to_numpy(parton_m0_flat)[very_close]
        vals, counts = np.unique(parton_close, return_counts=True)
        order = np.argsort(-counts)
        for v, c in list(zip(vals[order], counts[order]))[:15]:
            print(f"      PartonTruthLabelID={v!r:>6}  count={c:>7d}  ({100*c/n_very_close:5.2f}%)")
        print("\n   Se qui domina PartonTruthLabelID==-1 nonostante il DeltaR piccolissimo:")
        print("   e' la conferma piu' diretta che per questi jet manca a monte l'informazione")
        print("   di verita' (sia parton che Higgs-genealogia), pur essendo geometricamente")
        print("   indistinguibili dal vero overlap b/tau fisico gia' isolato nella popolazione")
        print("   coerente -> probabile artefatto di propagazione della truth-info, non fisica.")


if __name__ == "__main__":
    root_path = sys.argv[1] if len(sys.argv) > 1 else default_root_path()
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(root_path, n_entries_cap=cap)