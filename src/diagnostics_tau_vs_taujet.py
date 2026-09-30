from pathlib import Path

import numpy as np
import awkward as ak
import uproot


# ======================================================================
# CONFIGURAZIONE
# ======================================================================

ROOT_DIR = Path("HH_bbtt")

FILE_PREFIX = "output_GGF_mc23a_bypass_noOR_0000"
FILE_SUFFIX = ".root"

TREE_NAME = "AnalysisMiniTree"

# None = tutti gli eventi di ogni file
# Esempio: 10000 -> usa i primi 10000 eventi di CIASCUN file
N_ENTRIES_CAP = None


# ======================================================================
# UTILITY
# ======================================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def file_path(index):
    """
    index = 1 ... 14

    Esempio:
        index=1  -> output_GGF_mc23a_bypass_noOR_000001.root
        index=14 -> output_GGF_mc23a_bypass_noOR_000014.root
    """
    return (
        ROOT_DIR
        / f"{FILE_PREFIX}{index:02d}{FILE_SUFFIX}"
    )


def load_files():
    """
    Cerca i 14 file ROOT attesi e restituisce una lista di dict:

        {
            "index":     indice del file,
            "path":      Path,
            "file_name": nome del file,
            "root_file": handle uproot,
            "tree":      AnalysisMiniTree,
            "n_entries": numero di eventi effettivamente usati
        }

    I file mancanti o non validi vengono segnalati e ignorati.
    """

    loaded = []

    section("FILE INPUT")

    for i in range(1, 15):

        path = file_path(i)

        if not path.exists():
            print(f"[MANCANTE] {path}")
            continue

        try:
            root_file = uproot.open(path)

            if TREE_NAME not in root_file:
                print(
                    f"[ERRORE] {path}: "
                    f"tree '{TREE_NAME}' non trovato"
                )
                continue

            tree = root_file[TREE_NAME]

            n_entries = tree.num_entries

            if N_ENTRIES_CAP is not None:
                n_entries_used = min(
                    n_entries,
                    N_ENTRIES_CAP,
                )
            else:
                n_entries_used = n_entries

            print(
                f"[OK] {path}   "
                f"entries={n_entries}   "
                f"usati={n_entries_used}"
            )

            loaded.append(
                {
                    "index": i,
                    "path": path,
                    "file_name": path.name,
                    "root_file": root_file,
                    "tree": tree,
                    "n_entries": n_entries_used,
                }
            )

        except Exception as exc:
            print(
                f"[ERRORE] {path}: {exc}"
            )

    print(
        f"\nFile caricati correttamente: "
        f"{len(loaded)}/14"
    )

    return loaded


def get_analysis_selection(
    tree,
    selection_branch,
    n_entries,
):
    """
    Legge un branch isAnalysis... e lo converte esplicitamente
    in una maschera booleana.

    None -> False
    0    -> False
    != 0 -> True
    """

    selection = tree[selection_branch].array(
        entry_stop=n_entries,
        library="ak",
    )

    selection = ak.fill_none(
        selection,
        0,
    )

    return selection != 0


def clean_bool_branch(arr):
    """
    Converte robustamente un branch booleano/int8/uint8
    in bool awkward.

    None -> False
    """

    arr = ak.fill_none(
        arr,
        0,
    )

    return arr != 0


def summarize(values, label):
    """
    values può essere:
      - awkward jagged array
      - awkward regular array
      - numpy array

    Flatten, rimozione dei None e dei NaN, quindi:

      n
      mean
      median
      p10
      p90
    """

    flat = ak.flatten(
        values,
        axis=None,
    )

    flat = ak.drop_none(flat)

    flat = ak.to_numpy(flat)

    if flat.size == 0:
        print(
            f"   {label}: nessun valore disponibile"
        )
        return

    if np.issubdtype(
        flat.dtype,
        np.floating,
    ):
        flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        print(
            f"   {label}: nessun valore finito disponibile"
        )
        return

    print(
        f"   {label}: "
        f"n={flat.size:>10d}  "
        f"mean={np.mean(flat):.4f}  "
        f"median={np.median(flat):.4f}  "
        f"p10={np.percentile(flat, 10):.4f}  "
        f"p90={np.percentile(flat, 90):.4f}"
    )


def delta_phi(phi1, phi2):
    """
    Delta phi riportato nell'intervallo [-pi, pi).
    """

    dphi = phi1 - phi2

    return (
        (dphi + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )


def delta_r(
    eta1,
    phi1,
    eta2,
    phi2,
):
    """
    DeltaR = sqrt((Delta eta)^2 + (Delta phi)^2)
    """

    deta = eta1 - eta2

    dphi = delta_phi(
        phi1,
        phi2,
    )

    return np.sqrt(
        deta**2 + dphi**2
    )


# ======================================================================
# STEP 0 + STEP 1 + STEP 2
# ======================================================================

def analyze_label15_higgs_consistency(
    loaded,
):
    """
    Diagnostica aggregata su tutti i file.

    STEP 0
    ------
    Verifica che, all'interno dello stesso evento, tutti i tau
    adronici veri con parentHiggsParentsMask != 0 abbiano la
    stessa mask.

    STEP 1
    ------
    DEFINIZIONE ESPLICITA DEL SUBSET:

        HadronConeExclTruthLabelID == 15
        AND
        parentHiggsParentsMask != 0

    Solo questo subset viene confrontato con la mask di riferimento
    dei tau adronici veri dello stesso evento.

    STEP 2
    ------
    Usa ESATTAMENTE lo stesso subset di jet dello STEP 1:

        label == 15
        AND
        mask != 0

    Per questi jet viene calcolato il DeltaR minimo rispetto ai tau
    adronici veri dello stesso evento, separando:

        - jet coerenti
        - jet incoerenti

    TUTTE le quantità vengono aggregate sui file caricati.
    """

    # ------------------------------------------------------------------
    # BRANCHES
    # ------------------------------------------------------------------

    branches = [
        "tau_eta",
        "tau_phi",
        "tau_truth_IsHadronicTau",
        "tau_parentHiggsParentsMask",
        "tau_isAnalysisTau___NOSYS",

        "recojet_antikt4PFlow_eta",
        "recojet_antikt4PFlow_phi",
        "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
        "recojet_antikt4PFlow_parentHiggsParentsMask",
        "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
    ]

    # ------------------------------------------------------------------
    # ACCUMULATORI GLOBALI
    # ------------------------------------------------------------------

    # STEP 0
    aggregate_events = 0
    aggregate_events_with_ref = 0
    aggregate_events_inconsistent_tau = 0
    aggregate_events_valid = 0

    # STEP 1
    aggregate_jet15_total = 0
    aggregate_jet15_mask0 = 0
    aggregate_jet15_nonzero_total = 0
    aggregate_compared = 0
    aggregate_consistent = 0
    aggregate_inconsistent = 0

    # STEP 2
    dr_consistent_parts = []
    dr_inconsistent_parts = []

    # ------------------------------------------------------------------
    # LOOP SU TUTTI I FILE
    # ------------------------------------------------------------------

    for item in loaded:

        filename = item["file_name"]
        tree = item["tree"]
        n_entries = item["n_entries"]

        print()
        print("-" * 100)
        print(
            f"Analisi file: {filename}"
        )
        print("-" * 100)

        keys = set(tree.keys())

        missing = [
            branch
            for branch in branches
            if branch not in keys
        ]

        if missing:

            print(
                f"[WARNING] {filename}: "
                "branch mancanti:"
            )

            for branch in missing:
                print(
                    f"    - {branch}"
                )

            print(
                "    File ignorato."
            )

            continue

        # ==============================================================
        # LETTURA
        # ==============================================================

        a = tree.arrays(
            branches,
            entry_stop=n_entries,
            library="ak",
        )

        n_events = len(
            a["tau_eta"]
        )

        # ==============================================================
        # SELEZIONE ANALYSIS-LEVEL
        # ==============================================================

        tau_sel = get_analysis_selection(
            tree,
            "tau_isAnalysisTau___NOSYS",
            n_entries,
        )

        jet_sel = get_analysis_selection(
            tree,
            "recojet_antikt4PFlow_isAnalysisJet___NOSYS",
            n_entries,
        )

        tau_eta = (
            a["tau_eta"][tau_sel]
        )

        tau_phi = (
            a["tau_phi"][tau_sel]
        )

        tau_ishad = clean_bool_branch(
            a["tau_truth_IsHadronicTau"][tau_sel]
        )

        tau_mask = ak.fill_none(
            a["tau_parentHiggsParentsMask"][tau_sel],
            0,
        )

        jet_eta = (
            a["recojet_antikt4PFlow_eta"][jet_sel]
        )

        jet_phi = (
            a["recojet_antikt4PFlow_phi"][jet_sel]
        )

        jet_label = ak.fill_none(
            a[
                "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
            ][jet_sel],
            -999999,
        )

        jet_mask = ak.fill_none(
            a[
                "recojet_antikt4PFlow_parentHiggsParentsMask"
            ][jet_sel],
            0,
        )

        # ==============================================================
        # STEP 0
        # ==============================================================

        real_tau = (
            tau_ishad
        )

        real_tau_mask = tau_mask[
            real_tau
        ]

        # Solo tau veri con mask != 0
        nonzero_real_tau_mask = (
            real_tau_mask[
                real_tau_mask != 0
            ]
        )

        # Numero di tau veri con mask != 0 per evento
        n_real_tau_nz = ak.num(
            nonzero_real_tau_mask,
            axis=1,
        )

        has_ref = (
            n_real_tau_nz > 0
        )

        # min/max all'interno dell'evento
        min_mask = ak.min(
            nonzero_real_tau_mask,
            axis=1,
        )

        max_mask = ak.max(
            nonzero_real_tau_mask,
            axis=1,
        )

        consistent_tau_evt = ak.fill_none(
            min_mask == max_mask,
            True,
        )

        # Evento utilizzabile negli step successivi:
        # almeno un tau vero con mask!=0
        # e mask uniforme tra questi tau
        valid_evt = (
            has_ref
            & consistent_tau_evt
        )

        ref_mask = min_mask

        n_evt_with_ref = int(
            ak.sum(has_ref)
        )

        n_evt_inconsistent = int(
            ak.sum(
                has_ref
                & ~consistent_tau_evt
            )
        )

        n_evt_valid = int(
            ak.sum(valid_evt)
        )

        aggregate_events += n_events
        aggregate_events_with_ref += (
            n_evt_with_ref
        )
        aggregate_events_inconsistent_tau += (
            n_evt_inconsistent
        )
        aggregate_events_valid += (
            n_evt_valid
        )

        # ==============================================================
        # STEP 1
        # ==============================================================

        jet15 = (
            jet_label == 15
        )

        # --------------------------------------------------------------
        # SUBSET UNICO USATO DA STEP 1 E STEP 2
        #
        # Questo è il punto centrale:
        #
        #     label == 15
        #     AND
        #     mask != 0
        #
        # Da qui in poi STEP 1 e STEP 2 lavorano esclusivamente
        # su questa popolazione.
        # --------------------------------------------------------------

        jet15_nonzero = (
            jet15
            & (jet_mask != 0)
        )

        jet15_mask0 = (
            jet15
            & (jet_mask == 0)
        )

        # --------------------------------------------------------------
        # CORREZIONE DEL BUG:
        #
        # NON usare:
        #
        #     ak.num(jet15, axis=1)
        #
        # perché ak.num conta gli elementi della lista, non i True.
        #
        # Per contare i jet che soddisfano una maschera booleana
        # bisogna usare ak.sum(mask).
        # --------------------------------------------------------------

        n_jet15_total = int(
            ak.sum(jet15)
        )

        n_jet15_mask0 = int(
            ak.sum(jet15_mask0)
        )

        n_jet15_nonzero = int(
            ak.sum(jet15_nonzero)
        )

        aggregate_jet15_total += (
            n_jet15_total
        )

        aggregate_jet15_mask0 += (
            n_jet15_mask0
        )

        aggregate_jet15_nonzero_total += (
            n_jet15_nonzero
        )

        # --------------------------------------------------------------
        # Restrizione agli eventi validi
        # --------------------------------------------------------------

        valid_ref_mask = (
            ref_mask[valid_evt]
        )

        # Questo è IL subset dello STEP 1
        subset_mask_v = (
            jet15_nonzero[valid_evt]
        )

        jet15_mask_v = (
            jet_mask[valid_evt][subset_mask_v]
        )

        # --------------------------------------------------------------
        # Broadcast esplicito della mask di riferimento
        #
        # valid_ref_mask:
        #   [evento]
        #
        # jet15_mask_v:
        #   [evento][jet]
        # --------------------------------------------------------------

        ref_mask_broadcast, _ = ak.broadcast_arrays(
            valid_ref_mask,
            jet15_mask_v,
        )

        match_ok = (
            jet15_mask_v
            == ref_mask_broadcast
        )

        n_compared = int(
            ak.sum(
                ak.num(
                    jet15_mask_v,
                    axis=1,
                )
            )
        )

        n_consistent = int(
            ak.sum(match_ok)
        )

        n_inconsistent = (
            n_compared
            - n_consistent
        )

        aggregate_compared += (
            n_compared
        )

        aggregate_consistent += (
            n_consistent
        )

        aggregate_inconsistent += (
            n_inconsistent
        )

        # ==============================================================
        # STEP 2
        # ==============================================================

        # --------------------------------------------------------------
        # IMPORTANTE:
        #
        # STEP 2 usa ESATTAMENTE:
        #
        #       subset_mask_v
        #
        # cioè gli stessi jet già selezionati nello STEP 1:
        #
        #       label == 15
        #       AND
        #       mask != 0
        #
        # I jet con mask==0 NON entrano nel calcolo del DeltaR.
        # --------------------------------------------------------------

        real_tau_eta = (
            tau_eta[real_tau]
        )

        real_tau_phi = (
            tau_phi[real_tau]
        )

        real_tau_eta_v = (
            real_tau_eta[valid_evt]
        )

        real_tau_phi_v = (
            real_tau_phi[valid_evt]
        )

        # Applico ESATTAMENTE lo stesso subset dello STEP 1
        jet_eta_v = (
            jet_eta[valid_evt][subset_mask_v]
        )

        jet_phi_v = (
            jet_phi[valid_evt][subset_mask_v]
        )

        # --------------------------------------------------------------
        # Controllo interno:
        #
        # il numero di jet usati per Step 2 deve coincidere
        # con quello confrontato nello Step 1.
        # --------------------------------------------------------------

        n_step2_jets = int(
            ak.sum(
                ak.num(
                    jet_eta_v,
                    axis=1,
                )
            )
        )

        if n_step2_jets != n_compared:

            print(
                f"    [WARNING] {filename}: "
                f"STEP 1 ha {n_compared} jet, "
                f"STEP 2 ne ha {n_step2_jets}"
            )

        # --------------------------------------------------------------
        # DeltaR jet-tau
        #
        # [evento][jet][tau]
        # --------------------------------------------------------------

        jet_eta_c, tau_eta_c = ak.unzip(
            ak.cartesian(
                [
                    jet_eta_v,
                    real_tau_eta_v,
                ],
                nested=True,
            )
        )

        jet_phi_c, tau_phi_c = ak.unzip(
            ak.cartesian(
                [
                    jet_phi_v,
                    real_tau_phi_v,
                ],
                nested=True,
            )
        )

        dr_matrix = delta_r(
            jet_eta_c,
            jet_phi_c,
            tau_eta_c,
            tau_phi_c,
        )

        # Per ogni jet:
        # DeltaR rispetto al tau adronico vero più vicino
        dr_min = ak.min(
            dr_matrix,
            axis=-1,
        )

        # --------------------------------------------------------------
        # Separazione coerenti / incoerenti
        #
        # Anche qui uso ESATTAMENTE il subset dello STEP 1.
        # --------------------------------------------------------------

        dr_min_consistent = (
            dr_min[match_ok]
        )

        dr_min_inconsistent = (
            dr_min[~match_ok]
        )

        if ak.count(
            dr_min_consistent
        ) > 0:

            dr_consistent_parts.append(
                dr_min_consistent
            )

        if ak.count(
            dr_min_inconsistent
        ) > 0:

            dr_inconsistent_parts.append(
                dr_min_inconsistent
            )

        # --------------------------------------------------------------
        # Diagnostica del singolo file
        # --------------------------------------------------------------

        print(
            f"    eventi: "
            f"{n_events}"
        )

        print(
            f"    eventi validi STEP 0: "
            f"{n_evt_valid}"
        )

        print(
            f"    jet label==15: "
            f"{n_jet15_total}"
        )

        print(
            f"    jet label==15 con mask==0: "
            f"{n_jet15_mask0}"
        )

        print(
            f"    jet label==15 con mask!=0: "
            f"{n_jet15_nonzero}"
        )

        # Check interno fondamentale
        if (
            n_jet15_mask0
            + n_jet15_nonzero
            != n_jet15_total
        ):
            print(
                "    [WARNING] "
                "label==15 != "
                "mask==0 + mask!=0"
            )

        print(
            f"    jet confrontati STEP 1: "
            f"{n_compared}"
        )

        print(
            f"    coerenti: "
            f"{n_consistent}"
        )

        print(
            f"    incoerenti: "
            f"{n_inconsistent}"
        )

        print(
            f"    jet usati STEP 2: "
            f"{n_step2_jets}"
        )

    # ==================================================================
    # RISULTATI AGGREGATI
    # ==================================================================

    section(
        "RISULTATI AGGREGATI SU TUTTI I FILE"
    )

    # ------------------------------------------------------------------
    # STEP 0
    # ------------------------------------------------------------------

    section(
        "STEP 0 - I tau adronici veri condividono la stessa mask entro evento?"
    )

    print(
        f"   file analizzati: "
        f"{len(loaded)}"
    )

    print(
        f"   eventi totali: "
        f"{aggregate_events}"
    )

    print(
        f"   eventi con almeno un tau vero mask!=0: "
        f"{aggregate_events_with_ref}"
    )

    frac_inconsistent_tau = (
        aggregate_events_inconsistent_tau
        / aggregate_events_with_ref
        if aggregate_events_with_ref > 0
        else np.nan
    )

    print(
        f"   di questi, con mask NON uniforme "
        f"tra i tau veri: "
        f"{aggregate_events_inconsistent_tau}"
        f"  ({100.0 * frac_inconsistent_tau:.3f}%)"
    )

    print(
        f"   eventi validi per STEP 1/2: "
        f"{aggregate_events_valid}"
        f"  ("
        f"{100.0 * aggregate_events_valid / max(aggregate_events, 1):.3f}%"
        f")"
    )

    print(
        "\n   -> Gli eventi con almeno un tau vero mask!=0 "
        "e mask uniforme costituiscono il campione"
    )

    print(
        "      usato per il confronto jet vs mask di riferimento "
        "dei tau veri."
    )

    # ------------------------------------------------------------------
    # STEP 1
    # ------------------------------------------------------------------

    section(
        "STEP 1 - Coerenza mask jet(label==15, mask!=0) "
        "vs tau veri dello stesso evento"
    )

    print(
        f"   jet label==15 totali: "
        f"{aggregate_jet15_total}"
    )

    frac_mask0 = (
        aggregate_jet15_mask0
        / aggregate_jet15_total
        if aggregate_jet15_total > 0
        else np.nan
    )

    frac_mask_nonzero = (
        aggregate_jet15_nonzero_total
        / aggregate_jet15_total
        if aggregate_jet15_total > 0
        else np.nan
    )

    print(
        f"   di cui con mask==0 "
        f"(baseline esclusa): "
        f"{aggregate_jet15_mask0}"
        f"  ({100.0 * frac_mask0:.3f}%)"
    )

    print(
        f"   jet label==15 con mask!=0: "
        f"{aggregate_jet15_nonzero_total}"
        f"  ({100.0 * frac_mask_nonzero:.3f}%)"
    )

    # --------------------------------------------------------------
    # Check di consistenza:
    # mask==0 + mask!=0 deve ricostruire label==15
    # --------------------------------------------------------------

    print()

    print(
        "   CHECK decomposizione label==15:"
    )

    print(
        f"      mask==0 + mask!=0 = "
        f"{aggregate_jet15_mask0 + aggregate_jet15_nonzero_total}"
    )

    print(
        f"      label==15 totale   = "
        f"{aggregate_jet15_total}"
    )

    if (
        aggregate_jet15_mask0
        + aggregate_jet15_nonzero_total
        == aggregate_jet15_total
    ):
        print(
            "      -> OK"
        )
    else:
        print(
            "      -> ERRORE"
        )

    # --------------------------------------------------------------
    # Jet effettivamente confrontati
    # --------------------------------------------------------------

    frac_nonzero_compared = (
        aggregate_compared
        / aggregate_jet15_nonzero_total
        if aggregate_jet15_nonzero_total > 0
        else np.nan
    )

    print()

    print(
        f"   jet label==15 & mask!=0 "
        f"confrontati in eventi validi: "
        f"{aggregate_compared}"
        f"  ({100.0 * frac_nonzero_compared:.3f}% "
        f"dei label==15 & mask!=0)"
    )

    print()

    print(
        f"   jet confrontati: "
        f"{aggregate_compared}"
    )

    frac_consistent = (
        aggregate_consistent
        / aggregate_compared
        if aggregate_compared > 0
        else np.nan
    )

    frac_inconsistent = (
        aggregate_inconsistent
        / aggregate_compared
        if aggregate_compared > 0
        else np.nan
    )

    print(
        f"   coerenti "
        f"(stessa mask del tau vero): "
        f"{aggregate_consistent}"
        f"  ({100.0 * frac_consistent:.3f}%)"
    )

    print(
        f"   INCOERENTI "
        f"(mask diversa): "
        f"{aggregate_inconsistent}"
        f"  ({100.0 * frac_inconsistent:.3f}%)"
    )

    # ------------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------------

    section(
        "STEP 2 - DeltaR(jet, tau vero piu' vicino) "
        "per coerenti vs incoerenti"
    )

    print(
        "   SUBSET USATO:"
    )

    print(
        "      HadronConeExclTruthLabelID == 15"
    )

    print(
        "      parentHiggsParentsMask != 0"
    )

    print(
        "      stesso identico subset dello STEP 1"
    )

    print()

    # --------------------------------------------------------------
    # DeltaR coerenti
    # --------------------------------------------------------------

    if dr_consistent_parts:

        dr_consistent_all = ak.concatenate(
            dr_consistent_parts
        )

        summarize(
            dr_consistent_all,
            "DeltaR min - jet COERENTI "
            "con la mask del tau",
        )

    else:

        print(
            "   DeltaR min - jet COERENTI: "
            "nessun valore disponibile"
        )

    # --------------------------------------------------------------
    # DeltaR incoerenti
    # --------------------------------------------------------------

    if dr_inconsistent_parts:

        dr_inconsistent_all = ak.concatenate(
            dr_inconsistent_parts
        )

        summarize(
            dr_inconsistent_all,
            "DeltaR min - jet INCOERENTI "
            "con la mask del tau",
        )

    else:

        print(
            "   DeltaR min - jet INCOERENTI: "
            "nessun valore disponibile"
        )

    # ------------------------------------------------------------------
    # Check quantità STEP 2
    # ------------------------------------------------------------------

    n_dr_consistent = 0
    n_dr_inconsistent = 0

    if dr_consistent_parts:

        n_dr_consistent = int(
            ak.sum(
                ak.num(
                    dr_consistent_parts,
                    axis=1,
                )
            )
        )

    if dr_inconsistent_parts:

        n_dr_inconsistent = int(
            ak.sum(
                ak.num(
                    dr_inconsistent_parts,
                    axis=1,
                )
            )
        )

    print()

    print(
        "   CHECK STEP 2:"
    )

    print(
        f"      DeltaR coerenti:   "
        f"{n_dr_consistent}"
    )

    print(
        f"      DeltaR incoerenti: "
        f"{n_dr_inconsistent}"
    )

    print(
        f"      totale STEP 2:     "
        f"{n_dr_consistent + n_dr_inconsistent}"
    )

    print(
        f"      jet confrontati STEP 1: "
        f"{aggregate_compared}"
    )

    if (
        n_dr_consistent
        + n_dr_inconsistent
        == aggregate_compared
    ):
        print(
            "      -> OK: STEP 2 usa esattamente "
            "gli stessi jet dello STEP 1."
        )
    else:
        print(
            "      -> WARNING: il numero di jet "
            "non coincide."
        )

    # ------------------------------------------------------------------
    # Interpretazione
    # ------------------------------------------------------------------

    print()
    print(
        "   Interpretazione:"
    )

    print(
        "   - Se i due gruppi hanno DeltaR simile "
        "(entrambi piccoli),"
    )

    print(
        "     l'incoerenza di mask non sembra spiegabile "
        "semplicemente con una distanza geometrica maggiore."
    )

    print(
        "   - Se gli incoerenti hanno DeltaR sistematicamente "
        "piu' grande,"
    )

    print(
        "     questo suggerisce che siano geometricamente meno "
        "correlati al tau vero piu' vicino."
    )

    print(
        "   - Lo STEP 2 non contiene i jet label==15 con mask==0: "
        "essi sono esclusi gia' dalla definizione del subset."
    )

    print(
        "   - Questa diagnostica non dimostra da sola la causa "
        "dell'incoerenza: distingue la componente geometrica "
        "da quella genealogica."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    loaded = load_files()

    if not loaded:

        print(
            "\nNessun file disponibile. "
            "Controllare ROOT_DIR e i nomi dei file."
        )

        return

    analyze_label15_higgs_consistency(
        loaded
    )

    section(
        "FINE DIAGNOSTICA"
    )


if __name__ == "__main__":
    main()