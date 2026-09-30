"""
Confronto tra due sottopopolazioni di jet con HadronConeExclTruthLabelID==15:

  Popolazione COERENTE : label==15, mask!=0, mask concorde con quella dei
                          tau veri dello stesso evento (STEP 1 dello script
                          precedente: ~99.8% di questi casi, DeltaR mediano
                          ~0.009 -> overlap fisico genuino, stesso sciame
                          adronico ricostruito due volte).

  Popolazione MASK==0  : label==15, ma parentHiggsParentsMask==0 (quindi
                          "non figlio di nessun H" secondo quella variabile),
                          pur avendo un DeltaR minimo al tau vero molto
                          simile alla popolazione coerente (mediana ~0.014
                          nel test precedente, senza filtro).

  NOTA: i 9 casi "incoerenti" (mask!=0 ma diversa da quella del tau) trovati
  nello script precedente hanno per definizione mask!=0, quindi sono gia'
  esclusi dalla popolazione MASK==0 qui studiata senza bisogno di un filtro
  aggiuntivo; non vengono inclusi in nessuna delle due popolazioni confrontate.

Per ciascuna delle due popolazioni si stampano:
  - la distribuzione di recojet_antikt4PFlow_PartonTruthLabelID
    (per capire se il mask==0 e' associato a un parton-label indefinito,
    a sostegno di un problema di propagazione della genealogia a monte,
    oppure a un parton-label ben definito, che aprirebbe un'altra pista)
  - la distribuzione di DeltaR minimo al tau vero piu' vicino nello stesso
    evento, come termine di paragone diretto tra le due popolazioni

Requisiti: uproot, awkward, numpy
    pip install uproot awkward numpy

Uso:
    python check_label15_mask0_partonlabel.py /path/to/file.root [n_entries_cap]
"""

import sys
import numpy as np
import awkward as ak
import uproot

TREE_NAME = "AnalysisMiniTree"


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta ** 2 + dphi ** 2)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_dr(values, label):
    flat = ak.to_numpy(ak.flatten(values, axis=None))
    if flat.size == 0:
        print(f"   {label}: nessun valore disponibile")
        return
    print(f"   {label}: n={flat.size:>7d}  mean={np.mean(flat):.4f}  "
          f"median={np.median(flat):.4f}  p10={np.percentile(flat,10):.4f}  "
          f"p90={np.percentile(flat,90):.4f}")


def value_counts_flat(values, top=15):
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
    branches = [
        "tau_eta", "tau_phi", "tau_truth_IsHadronicTau",
        "tau_parentHiggsParentsMask", "tau_isAnalysisTau___NOSYS",
        "recojet_antikt4PFlow_eta", "recojet_antikt4PFlow_phi",
        "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
        "recojet_antikt4PFlow_parentHiggsParentsMask",
        "recojet_antikt4PFlow_PartonTruthLabelID",
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
    ]

    with uproot.open(root_path) as f:
        tree = f[TREE_NAME]
        a = tree.arrays(branches, entry_stop=n_entries_cap, library="ak")

    # --- selezione analysis-level ---
    tau_sel = a["tau_isAnalysisTau___NOSYS"] == 1
    jet_sel = a["recojet_antikt4PFlow_isAnalysisJet___NOSYS"] == 1

    tau_eta = a["tau_eta"][tau_sel]
    tau_phi = a["tau_phi"][tau_sel]
    tau_ishad = a["tau_truth_IsHadronicTau"][tau_sel]
    tau_mask = a["tau_parentHiggsParentsMask"][tau_sel]

    jet_eta = a["recojet_antikt4PFlow_eta"][jet_sel]
    jet_phi = a["recojet_antikt4PFlow_phi"][jet_sel]
    jet_label = a["recojet_antikt4PFlow_HadronConeExclTruthLabelID"][jet_sel]
    jet_mask = a["recojet_antikt4PFlow_parentHiggsParentsMask"][jet_sel]
    jet_parton = a["recojet_antikt4PFlow_PartonTruthLabelID"][jet_sel]

    # --- STEP 0 (come nello script precedente): mask di riferimento dei tau veri per evento ---
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

    jet15 = jet_label == 15

    # ------------------------------------------------------------------
    # Popolazione COERENTE: label==15, mask!=0, concorde col tau (STEP 1)
    # ------------------------------------------------------------------
    jet15_nonzero = jet15 & (jet_mask != 0)

    jet_coh_mask = jet_mask[jet15_nonzero][valid_evt]
    jet_coh_eta = jet_eta[jet15_nonzero][valid_evt]
    jet_coh_phi = jet_phi[jet15_nonzero][valid_evt]
    jet_coh_parton = jet_parton[jet15_nonzero][valid_evt]

    match_ok = jet_coh_mask == ref_mask_v  # True = coerente

    coh_eta = jet_coh_eta[match_ok]
    coh_phi = jet_coh_phi[match_ok]
    coh_parton = jet_coh_parton[match_ok]

    # ------------------------------------------------------------------
    # Popolazione MASK==0: label==15, mask==0 (i 9 casi incoerenti hanno
    # mask!=0 e sono quindi gia' esclusi da questa selezione)
    # ------------------------------------------------------------------
    jet15_mask0 = jet15 & (jet_mask == 0)

    mask0_eta = jet_eta[jet15_mask0][valid_evt]
    mask0_phi = jet_phi[jet15_mask0][valid_evt]
    mask0_parton = jet_parton[jet15_mask0][valid_evt]

    # ------------------------------------------------------------------
    # PartonTruthLabelID: distribuzione a confronto
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # DeltaR minimo al tau vero piu' vicino, a confronto diretto
    # ------------------------------------------------------------------
    section("DeltaR(jet, tau vero piu' vicino) - confronto diretto fra le due popolazioni")

    jet_eta_c1, tau_eta_c1 = ak.unzip(ak.cartesian([coh_eta, real_tau_eta_v], nested=True))
    jet_phi_c1, tau_phi_c1 = ak.unzip(ak.cartesian([coh_phi, real_tau_phi_v], nested=True))
    dr_coh = ak.min(delta_r(jet_eta_c1, jet_phi_c1, tau_eta_c1, tau_phi_c1), axis=-1)

    jet_eta_c2, tau_eta_c2 = ak.unzip(ak.cartesian([mask0_eta, real_tau_eta_v], nested=True))
    jet_phi_c2, tau_phi_c2 = ak.unzip(ak.cartesian([mask0_phi, real_tau_phi_v], nested=True))
    dr_m0 = ak.min(delta_r(jet_eta_c2, jet_phi_c2, tau_eta_c2, tau_phi_c2), axis=-1)

    summarize_dr(dr_coh, "DeltaR min - popolazione COERENTE")
    summarize_dr(dr_m0, "DeltaR min - popolazione MASK==0")

    # ------------------------------------------------------------------
    # Incrocio finale: fra i jet MASK==0 geometricamente vicinissimi al tau
    # (stesso ordine di grandezza della popolazione coerente), che aspetto
    # ha il parton-label? Isola il sottocaso piu' sospetto.
    # ------------------------------------------------------------------
    section("Incrocio: jet MASK==0 con DeltaR molto piccolo (<0.05) - PartonTruthLabelID")

    dr_m0_flat = ak.flatten(dr_m0, axis=None)
    parton_m0_flat = ak.flatten(mask0_parton, axis=None)
    very_close = ak.to_numpy(dr_m0_flat) < 0.05
    n_very_close = int(np.sum(very_close))
    print(f"   jet MASK==0 con DeltaR<0.05 al tau vero: {n_very_close} su {len(dr_m0_flat)} "
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
    if len(sys.argv) < 2:
        print("Uso: python check_label15_mask0_partonlabel.py <path_al_file.root> [n_entries_cap]")
        sys.exit(1)
    root_path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(root_path, n_entries_cap=cap)