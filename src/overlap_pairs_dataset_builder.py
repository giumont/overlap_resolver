"""
================================================================================
Costruzione del dataset di coppie (jet reco, tau reco) a partire da file
.root, con:
  - slicing a livello di analisi (reco reali)
  - slicing di overlap geometrico (DeltaR < dr_thr)
  - truth label di coppia indicizzata {FF, FT, TF, TT}
  - conversione in (X, y, event_id) compatibile col preprocessing esistente
  - checkpointing chunked (stesso schema di path di checkpoint_io.py)
  - split train/val/test raggruppato per evento (no data leakage)

Riusa senza modificarle:
  - obj_3_1.load_files / get_analysis_selection / delta_r
  - overlap_kinematics.build_pair_kinematics_and_labels
  - checkpoint_io._strip_npz / _manifest_path / _chunk_dir_path / _chunk_file_path
    (path helper, gia' generici)
  - checkpoint_io._load_chunks_into_arrays (estesa con load_event_id, v. patch)

Non riusa (per incompatibilita' di schema, v. nota nel manifest):
  - checkpoint_io._save_manifest / _read_manifest_raw
  - merge_datasets.check_checkpoint_progress / merge_checkpoint_chunks
  Questi sono specifici del manifest "jet-tagging" (target_indices fisso,
  n_tracks_per_jet, ecc.), concetti che non esistono per le coppie
  jet-tau, dove il totale di coppie non e' noto a priori. Si aggiungono
  quindi un manifest e delle funzioni di merge/progress parallele.
"""

import os

import numpy as np
import awkward as ak

from obj_3_1 import load_files, get_analysis_selection
from overlap_kinematics import build_pair_kinematics_and_labels

from flavour_tag_ml.checkpoint_io import (
    _manifest_path,
    _chunk_dir_path,
    _chunk_file_path,
    _load_chunks_into_arrays,
)


# ======================================================================
# SELEZIONE A LIVELLO DI ANALISI (wrapper generico su get_analysis_selection)
# ======================================================================

def select_analysis_objects(tree, n_entries, jet_analysis_branch, tau_analysis_branch):
    """
    Slicing a livello di analisi per jet e tau reco. Wrapper parametrico
    su get_analysis_selection (obj_3_1); i nomi dei branch booleani
    "isAnalysisXxx" sono parametri di chiamata, coerentemente con
    JET_SELECTION_MODE/TAU_SELECTION_MODE di overlap_kinematics.py.

    Returns
    -------
    jet_sel, tau_sel : awkward.Array (jagged, dtype bool)
    """
    jet_sel = get_analysis_selection(tree, jet_analysis_branch, n_entries)
    tau_sel = get_analysis_selection(tree, tau_analysis_branch, n_entries)
    return jet_sel, tau_sel


# ======================================================================
# TRUTH LABEL DI COPPIA (FF / FT / TF / TT), indicizzata
# ======================================================================

# Ordine (jet, tau): "TF" = jet vero, tau falso; "FT" = jet falso, tau vero
DEFAULT_PAIR_LABEL_INDEX = {"FF": 0, "FT": 1, "TF": 2, "TT": 3}


def build_pair_truth_index(jet_label_ct, tau_label_ct, label_index_map=None):
    """
    Codifica la truth label di coppia (jet, tau) in un intero, a partire
    dalle label booleane per-oggetto gia' propagate a livello di coppia
    (jet_label_ct, tau_label_ct = pair_info["jet_label"]/["tau_label"]
    come restituiti da build_pair_kinematics_and_labels).

    Parameters
    ----------
    jet_label_ct, tau_label_ct : awkward.Array (jagged, dtype bool)
    label_index_map : dict, optional
        {"TT": int, "TF": int, "FT": int, "FF": int}. Default
        DEFAULT_PAIR_LABEL_INDEX = {"FF": 0, "FT": 1, "TF": 2, "TT": 3}.

    Returns
    -------
    label_idx_ct : awkward.Array (jagged, dtype int64), stessa struttura
        pair-level di jet_label_ct/tau_label_ct.
    """
    if label_index_map is None:
        label_index_map = DEFAULT_PAIR_LABEL_INDEX

    jet_lab = ak.values_astype(jet_label_ct, bool)
    tau_lab = ak.values_astype(tau_label_ct, bool)

    label_idx_ct = ak.where(
        jet_lab & tau_lab, label_index_map["TT"],
        ak.where(
            jet_lab & ~tau_lab, label_index_map["TF"],
            ak.where(
                ~jet_lab & tau_lab, label_index_map["FT"],
                label_index_map["FF"],
            ),
        ),
    )
    return ak.values_astype(label_idx_ct, np.int64)


# ======================================================================
# INDICE DI EVENTO PER-COPPIA (per grouping anti-leakage)
# ======================================================================

def build_pair_event_index(pair_info, key="jet_pt", file_offset=0):
    """
    Costruisce un indice di evento univoco per ogni coppia (jet, tau), da
    usare come "group" in split_pairs_by_event per evitare che coppie
    dello stesso evento finiscano in split diversi.

    Parameters
    ----------
    pair_info : dict
        Come restituito da build_pair_kinematics_and_labels(...).
    key : str, default="jet_pt"
        Una qualsiasi chiave pair-level di pair_info, usata solo per la
        sua struttura jagged [n_eventi][n_jet][n_tau] su cui broadcastare
        l'indice di evento.
    file_offset : int, default=0
        Numero cumulativo di eventi gia' processati nei file .root
        precedenti, cosi' l'indice resta univoco quando si concatenano
        piu' file (passare l'offset corrente tra una chiamata e l'altra).

    Returns
    -------
    event_id_ct : awkward.Array (jagged, dtype int64), stessa struttura
        pair-level delle altre variabili di pair_info.
    """
    n_events = len(pair_info[key])
    event_id = ak.Array(np.arange(n_events, dtype=np.int64) + file_offset)
    event_id_ct, _ = ak.broadcast_arrays(event_id, pair_info[key])
    return event_id_ct


# ======================================================================
# CONVERSIONE COPPIE -> (X, y, event_id) PER IL PREPROCESSING
# ======================================================================

def flatten_pairs_for_ml(pair_info, dr_thr, feature_keys, label_idx_ct, event_id_ct):
    """
    Applica lo slicing di overlap geometrico (DeltaR < dr_thr, gia'
    disponibile in pair_info["pair_dr"]) e appiattisce le coppie
    selezionate in (X, y, event_id), nello stesso formato/dtype prodotto
    dalla pipeline esistente (X float32, y float32) piu' un terzo array
    event_id (int64) per il grouping. Generalizza get_overlapping_pairs_kinematics
    di overlap_kinematics.py, che riusa lo stesso pattern flat_overlap.

    Parameters
    ----------
    pair_info : dict
        Come restituito da build_pair_kinematics_and_labels(...).
    dr_thr : float
        Soglia di overlap geometrico DeltaR.
    feature_keys : list of str
        Sottoinsieme ordinato di chiavi di pair_info da usare come colonne
        di X (es. ["pair_dr", "pair_pt_ratio", "jet_pt", "tau_pt", ...]).
    label_idx_ct : awkward.Array
        Come restituito da build_pair_truth_index(...).
    event_id_ct : awkward.Array
        Come restituito da build_pair_event_index(...).

    Returns
    -------
    X : np.ndarray, shape (n_pairs_sel, len(feature_keys)), dtype float32
    y : np.ndarray, shape (n_pairs_sel,), dtype float32
    event_id : np.ndarray, shape (n_pairs_sel,), dtype int64
    """
    overlap_mask = pair_info["pair_dr"] < dr_thr
    flat_mask = ak.to_numpy(ak.flatten(overlap_mask, axis=None)).astype(bool)

    def flat_sel(arr):
        flat_arr = ak.to_numpy(ak.flatten(arr, axis=None))
        return flat_arr[flat_mask]

    cols = [flat_sel(pair_info[k]).astype(np.float32) for k in feature_keys]
    X = np.stack(cols, axis=1) if cols else np.empty((flat_mask.sum(), 0), dtype=np.float32)
    y = flat_sel(label_idx_ct).astype(np.float32)
    event_id = flat_sel(event_id_ct).astype(np.int64)

    return X, y, event_id


# ======================================================================
# CHECKPOINTING (chunked) - schema di path riusato da checkpoint_io.py,
# manifest dedicato alle coppie (v. nota in testa al file)
# ======================================================================

def _save_pair_chunk(chunk_dir, chunk_idx, X, y, event_id):
    """Salva un singolo chunk di coppie: X, y, event_id."""
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_file = _chunk_file_path(chunk_dir, chunk_idx)
    np.savez_compressed(chunk_file, X=X, y=y, event_id=event_id)
    return chunk_file


def _save_pair_manifest(manifest_file, chunk_sizes, feature_names, label_index_map, dr_thr, complete):
    """
    Manifest leggero per il checkpointing delle coppie: mai contiene
    X/y/event_id (riscriverlo ad ogni chunk resta economico), stesso
    principio di _save_manifest in checkpoint_io.py.
    """
    np.savez_compressed(
        manifest_file,
        chunk_sizes=np.array(chunk_sizes, dtype=np.int64),
        feature_names=np.array(feature_names, dtype=object),
        label_index_map=np.array(list(label_index_map.items()), dtype=object),
        dr_thr=dr_thr,
        complete=complete,
    )


def _read_pair_manifest_raw(manifest_file):
    """Legge il manifest di coppie cosi' com'e' salvato, nessuna validazione."""
    with np.load(manifest_file, allow_pickle=True) as loaded:
        return {
            "chunk_sizes": loaded["chunk_sizes"].tolist(),
            "feature_names": loaded["feature_names"].tolist(),
            "label_index_map": dict(loaded["label_index_map"].tolist()),
            "dr_thr": float(loaded["dr_thr"]),
            "complete": bool(loaded["complete"]),
        }


def check_pair_checkpoint_progress(save_path, save_name, verbose=True):
    """
    Equivalente pair-specific di check_checkpoint_progress
    (merge_datasets.py): qui non esiste un n_total noto a priori (il
    numero totale di coppie dipende dai dati), quindi si riporta solo
    quanto e' stato salvato finora.

    Returns
    -------
    dict con n_done, n_chunks, complete. None se non c'e' un manifest.
    """
    manifest_file = _manifest_path(save_path, save_name)
    if not os.path.exists(manifest_file):
        if verbose:
            print(f"NO CHECKPOINT FOUND at: {manifest_file}")
        return None

    manifest = _read_pair_manifest_raw(manifest_file)
    n_done = sum(manifest["chunk_sizes"])

    if verbose:
        status = "COMPLETE" if manifest["complete"] else "PARTIAL"
        print(f"PAIR CHECKPOINT ({status}): {manifest_file}")
        print(f"CHECK - Coppie salvate: {n_done} in {len(manifest['chunk_sizes'])} chunk(s)")

    return {
        "n_done": n_done,
        "n_chunks": len(manifest["chunk_sizes"]),
        "complete": manifest["complete"],
    }


def merge_pair_checkpoint_chunks(save_path, save_name, output_file=None, delete_chunks_after=False, verbose=True):
    """
    Equivalente pair-specific di merge_checkpoint_chunks
    (merge_datasets.py): unisce i chunk (jet, tau) in un unico file .npz
    con X, y, event_id, feature_names, riusando _load_chunks_into_arrays
    (checkpoint_io.py) con load_event_id=True.

    Returns
    -------
    X, y, event_id, feature_names
    """
    manifest_file = _manifest_path(save_path, save_name)
    chunk_dir = _chunk_dir_path(save_path, save_name)

    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f"No manifest found at: {manifest_file}")

    manifest = _read_pair_manifest_raw(manifest_file)
    chunk_sizes = manifest["chunk_sizes"]
    feature_names = manifest["feature_names"]
    n_features = len(feature_names)

    X, y, event_id = _load_chunks_into_arrays(chunk_dir, chunk_sizes, n_features, load_event_id=True)

    if verbose:
        print(f"MERGING PAIR CHECKPOINT: {manifest_file}")
        print("CHECK - X shape:", X.shape)
        print("CHECK - y shape:", y.shape)
        print("CHECK - event_id shape:", event_id.shape)

    write_to_disk = output_file != ""
    if write_to_disk:
        if output_file is None:
            output_file = os.path.join(save_path, f"{save_name}_merged.npz")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        np.savez_compressed(
            output_file,
            X=X, y=y, event_id=event_id,
            feature_names=np.array(feature_names, dtype=object),
            label_index_map=np.array(list(manifest["label_index_map"].items()), dtype=object),
            dr_thr=manifest["dr_thr"],
        )
        if verbose:
            print("MERGED PAIR DATASET SAVED:", output_file)

        if delete_chunks_after:
            for i in range(len(chunk_sizes)):
                chunk_file = _chunk_file_path(chunk_dir, i)
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            if os.path.isdir(chunk_dir) and not os.listdir(chunk_dir):
                os.rmdir(chunk_dir)
            os.remove(manifest_file)
            if verbose:
                print("CHUNK FILES AND MANIFEST REMOVED:", chunk_dir, "/", manifest_file)

    return X, y, event_id, feature_names


# ======================================================================
# ORCHESTRATORE: loop sui file .root -> coppie -> overlap -> label -> checkpoint
# ======================================================================

def build_pair_dataset_from_root(
    save_path, save_name,
    jet_analysis_branch, tau_analysis_branch,
    jet_eta_branch, jet_phi_branch, jet_pt_branch, jet_mass_branch, jet_n_muons_branch,
    tau_eta_branch, tau_phi_branch, tau_pt_branch, tau_nProng_branch, tau_decayMode_branch, tau_charge_branch,
    jet_truth_label_fn, tau_truth_label_fn,
    feature_keys,
    dr_thr=0.4,
    label_index_map=None,
    verbose=True,
):
    """
    Pipeline completa. Per ogni file .root restituito da load_files():
      1) select_analysis_objects()            -> jet_sel, tau_sel
      2) jet_truth_label_fn / tau_truth_label_fn -> jet_label, tau_label
         (generalizzazione parametrica di label_jets_and_taus)
      3) build_pair_kinematics_and_labels()    -> pair_info [RIUSATA]
      4) build_pair_truth_index()              -> label_idx_ct
      5) build_pair_event_index()              -> event_id_ct
      6) flatten_pairs_for_ml()                -> X, y, event_id (overlap gia' applicato)
      7) _save_pair_chunk() + _save_pair_manifest() -> checkpoint incrementale

    Parameters
    ----------
    save_path, save_name : str
        Come in checkpoint_io.py / merge_datasets.py.
    jet_*_branch, tau_*_branch : str
        Nomi dei branch nel file .root, tutti parametri di chiamata.
    jet_truth_label_fn, tau_truth_label_fn : callable
        Firma (awkward.Array eventi_completi, selection_mask) -> awkward.Array
        bool (label vero/falso per gli oggetti selezionati). Analoghe al
        ruolo di label_jets_and_taus in overlap_kinematics.py, ma lasciate
        parametriche per generalita' (dipendono dai branch di verita'
        scelti dal chiamante, es. HadronConeExclTruthLabelID==5 per i jet,
        tau_truth_IsHadronicTau per i tau).
    feature_keys : list of str
        Colonne di X, tra le chiavi restituite da build_pair_kinematics_and_labels.
    dr_thr : float, default=0.4
        Soglia di overlap geometrico.
    label_index_map : dict, optional
        V. build_pair_truth_index. Default DEFAULT_PAIR_LABEL_INDEX.

    Returns
    -------
    dict con n_pairs_saved, n_files_processed, manifest_file.
    """
    if label_index_map is None:
        label_index_map = DEFAULT_PAIR_LABEL_INDEX

    loaded = load_files()
    if not loaded:
        if verbose:
            print("Nessun file .root disponibile.")
        return None

    manifest_file = _manifest_path(save_path, save_name)
    chunk_dir = _chunk_dir_path(save_path, save_name)
    os.makedirs(chunk_dir, exist_ok=True)

    branches = [
        jet_eta_branch, jet_phi_branch, jet_pt_branch, jet_mass_branch, jet_n_muons_branch,
        tau_eta_branch, tau_phi_branch, tau_pt_branch, tau_nProng_branch, tau_decayMode_branch, tau_charge_branch,
        jet_analysis_branch, tau_analysis_branch,
    ]

    chunk_sizes = []
    chunk_idx = 0
    event_offset = 0

    for item in loaded:
        tree, n_entries = item["tree"], item["n_entries"]

        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            if verbose:
                print(f"[SKIP] branch mancanti: {missing}")
            continue

        a = tree.arrays(branches, entry_stop=n_entries, library="ak")

        jet_sel, tau_sel = select_analysis_objects(
            tree, n_entries, jet_analysis_branch, tau_analysis_branch
        )

        jet_label = jet_truth_label_fn(a, jet_sel)
        tau_label = tau_truth_label_fn(a, tau_sel)

        jet_eta = a[jet_eta_branch][jet_sel]
        jet_phi = a[jet_phi_branch][jet_sel]
        jet_pt = a[jet_pt_branch][jet_sel]
        jet_mass = a[jet_mass_branch][jet_sel]
        jet_n_muons = a[jet_n_muons_branch][jet_sel]

        tau_eta = a[tau_eta_branch][tau_sel]
        tau_phi = a[tau_phi_branch][tau_sel]
        tau_pt = a[tau_pt_branch][tau_sel]
        tau_nProng = a[tau_nProng_branch][tau_sel]
        tau_decayMode = a[tau_decayMode_branch][tau_sel]
        tau_charge = a[tau_charge_branch][tau_sel]

        # --- coppie + cinematica: funzione ESISTENTE, non modificata ---
        pair_info = build_pair_kinematics_and_labels(
            jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
            tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
        )

        label_idx_ct = build_pair_truth_index(
            pair_info["jet_label"], pair_info["tau_label"], label_index_map
        )
        event_id_ct = build_pair_event_index(pair_info, key="jet_pt", file_offset=event_offset)

        X, y, event_id = flatten_pairs_for_ml(
            pair_info, dr_thr, feature_keys, label_idx_ct, event_id_ct
        )

        chunk_file = _save_pair_chunk(chunk_dir, chunk_idx, X, y, event_id)
        chunk_sizes.append(X.shape[0])
        chunk_idx += 1
        event_offset += n_entries

        # manifest riscritto ad ogni chunk: economico, mai contiene X/y/event_id
        _save_pair_manifest(manifest_file, chunk_sizes, feature_keys, label_index_map, dr_thr, complete=False)

        if verbose:
            print(f"[OK] chunk {chunk_idx - 1}: {X.shape[0]} coppie salvate in {chunk_file}")

    _save_pair_manifest(manifest_file, chunk_sizes, feature_keys, label_index_map, dr_thr, complete=True)

    return {
        "n_pairs_saved": int(sum(chunk_sizes)),
        "n_files_processed": len(chunk_sizes),
        "manifest_file": manifest_file,
    }


# ======================================================================
# SPLIT TRAIN / VAL / TEST SENZA DATA LEAKAGE TRA EVENTI
# ======================================================================

def split_pairs_by_event(event_id, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Divide gli indici di riga (di X/y/event_id) in train/val/test
    assicurando che tutte le coppie dello stesso evento (stesso
    event_id, v. build_pair_event_index) finiscano nello stesso split,
    per evitare data leakage tra split.

    Parameters
    ----------
    event_id : np.ndarray, shape (n_pairs,)
        Come restituito da flatten_pairs_for_ml / merge_pair_checkpoint_chunks.
    train_frac, val_frac, test_frac : float
        Devono sommare a 1 (entro tolleranza numerica). Le frazioni sono
        calcolate sul numero di EVENTI unici, non sul numero di coppie.
    seed : int

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        Indici di riga (in X/y/event_id) per ciascuno split.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac deve essere 1.0")

    rng = np.random.default_rng(seed)
    unique_events = np.unique(event_id)
    rng.shuffle(unique_events)

    n_events = len(unique_events)
    n_train = int(round(train_frac * n_events))
    n_val = int(round(val_frac * n_events))

    train_events = unique_events[:n_train]
    val_events = unique_events[n_train:n_train + n_val]
    test_events = unique_events[n_train + n_val:]

    train_idx = np.nonzero(np.isin(event_id, train_events))[0]
    val_idx = np.nonzero(np.isin(event_id, val_events))[0]
    test_idx = np.nonzero(np.isin(event_id, test_events))[0]

    return train_idx, val_idx, test_idx