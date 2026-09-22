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
================================================================================
"""

import os
from pathlib import Path

import numpy as np
import awkward as ak

import obj_3_1
from overlap_kinematics import build_pair_kinematics_and_labels

from flavour_tag_ml.checkpoint_io import (
    _manifest_path,
    _chunk_dir_path,
    _chunk_file_path,
    _load_chunks_into_arrays,
)


def select_analysis_objects(tree, n_entries, jet_analysis_branch, tau_analysis_branch):
    jet_sel = get_analysis_selection(tree, jet_analysis_branch, n_entries)
    tau_sel = get_analysis_selection(tree, tau_analysis_branch, n_entries)
    return jet_sel, tau_sel


DEFAULT_PAIR_LABEL_INDEX = {"FF": 0, "FT": 1, "TF": 2, "TT": 3}


def build_pair_truth_index(jet_label_ct, tau_label_ct, label_index_map=None):
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


def build_pair_event_index(pair_info, key="jet_pt", file_offset=0):
    n_events = len(pair_info[key])
    event_id = ak.Array(np.arange(n_events, dtype=np.int64) + file_offset)
    event_id_ct, _ = ak.broadcast_arrays(event_id, pair_info[key])
    return event_id_ct


def flatten_pairs_for_ml(pair_info, dr_thr, feature_keys, label_idx_ct, event_id_ct):
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


def _save_pair_chunk(chunk_dir, chunk_idx, X, y, event_id):
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_file = _chunk_file_path(chunk_dir, chunk_idx)
    np.savez_compressed(chunk_file, X=X, y=y, event_id=event_id)
    return chunk_file


def _save_pair_manifest(manifest_file, chunk_sizes, feature_names, label_index_map, dr_thr, complete):
    np.savez_compressed(
        manifest_file,
        chunk_sizes=np.array(chunk_sizes, dtype=np.int64),
        feature_names=np.array(feature_names, dtype=object),
        label_index_map=np.array(list(label_index_map.items()), dtype=object),
        dr_thr=dr_thr,
        complete=complete,
    )


def _read_pair_manifest_raw(manifest_file):
    with np.load(manifest_file, allow_pickle=True) as loaded:
        return {
            "chunk_sizes": loaded["chunk_sizes"].tolist(),
            "feature_names": loaded["feature_names"].tolist(),
            "label_index_map": dict(loaded["label_index_map"].tolist()),
            "dr_thr": float(loaded["dr_thr"]),
            "complete": bool(loaded["complete"]),
        }


def check_pair_checkpoint_progress(save_path, save_name, verbose=True):
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
def build_pair_dataset_from_root(
    save_path, save_name,
    jet_analysis_branch, tau_analysis_branch,
    jet_eta_branch, jet_phi_branch, jet_pt_branch,
    tau_eta_branch, tau_phi_branch, tau_pt_branch,
    jet_truth_label_fn, tau_truth_label_fn,
    root_dir=None,
    jet_truth_label_branch="recojet_antikt4PFlow_HadronConeExclTruthLabelID",
    tau_truth_label_branch="tau_truth_IsHadronicTau",
    extra_jet_branches=None,
    extra_tau_branches=None,
    met_branch="met_met___NOSYS",
    met_phi_branch="met_phi___NOSYS",
    compute_pt_ratio=True,
    compute_met_proj=True,
    compute_mt=True,
    feature_keys=None,
    dr_thr=0.4,
    label_index_map=None,
    verbose=True,
):
    """
    Pipeline completa per la creazione del dataset di coppie.
    """
    if root_dir is not None:
        obj_3_1.ROOT_DIR = Path(root_dir)

    if label_index_map is None:
        label_index_map = DEFAULT_PAIR_LABEL_INDEX

    if isinstance(extra_jet_branches, (list, tuple)):
        extra_jet_branches = {b: b for b in extra_jet_branches}
    elif extra_jet_branches is None:
        extra_jet_branches = {}

    if isinstance(extra_tau_branches, (list, tuple)):
        extra_tau_branches = {b: b for b in extra_tau_branches}
    elif extra_tau_branches is None:
        extra_tau_branches = {}

    if feature_keys is None:
        feature_keys = [
            "pair_dr", "pair_deta", "pair_dphi",
            "jet_pt", "jet_eta", "jet_phi",
            "tau_pt", "tau_eta", "tau_phi",
        ]
        if compute_pt_ratio:
            feature_keys.append("pair_pt_ratio")
        if compute_met_proj:
            feature_keys.append("tau_met_proj")
        if compute_mt:
            feature_keys.append("tau_mt")
        feature_keys.extend(list(extra_jet_branches.keys()) + list(extra_tau_branches.keys()))
        feature_keys = list(dict.fromkeys(feature_keys))

    loaded = load_files()
    if not loaded:
        if verbose:
            print(f"Nessun file .root disponibile in: {obj_3_1.ROOT_DIR}")
        return None

    manifest_file = _manifest_path(save_path, save_name)
    chunk_dir = _chunk_dir_path(save_path, save_name)
    os.makedirs(chunk_dir, exist_ok=True)

    # Inclusione esplicita dei branch di verità e cinematici principali
    core_branches = [
        jet_analysis_branch, tau_analysis_branch,
        jet_eta_branch, jet_phi_branch, jet_pt_branch,
        tau_eta_branch, tau_phi_branch, tau_pt_branch,
    ]
    if jet_truth_label_branch:
        core_branches.append(jet_truth_label_branch)
    if tau_truth_label_branch:
        core_branches.append(tau_truth_label_branch)

    if (compute_met_proj or compute_mt) and met_branch and met_phi_branch:
        core_branches.extend([met_branch, met_phi_branch])

    extra_branches = [b for b in list(extra_jet_branches.values()) + list(extra_tau_branches.values()) if isinstance(b, str)]
    branches = list(dict.fromkeys(core_branches + extra_branches))

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

        tau_eta = a[tau_eta_branch][tau_sel]
        tau_phi = a[tau_phi_branch][tau_sel]
        tau_pt = a[tau_pt_branch][tau_sel]

        jet_mass = a[extra_jet_branches["jet_mass"]][jet_sel] if "jet_mass" in extra_jet_branches and isinstance(extra_jet_branches["jet_mass"], str) else ak.zeros_like(jet_pt)
        jet_n_muons = a[extra_jet_branches["jet_n_muons"]][jet_sel] if "jet_n_muons" in extra_jet_branches and isinstance(extra_jet_branches["jet_n_muons"], str) else ak.zeros_like(jet_pt)

        tau_nProng = a[extra_tau_branches["tau_nProng"]][tau_sel] if "tau_nProng" in extra_tau_branches and isinstance(extra_tau_branches["tau_nProng"], str) else ak.zeros_like(tau_pt)
        tau_decayMode = a[extra_tau_branches["tau_decayMode"]][tau_sel] if "tau_decayMode" in extra_tau_branches and isinstance(extra_tau_branches["tau_decayMode"], str) else ak.zeros_like(tau_pt)
        tau_charge = a[extra_tau_branches["tau_charge"]][tau_sel] if "tau_charge" in extra_tau_branches and isinstance(extra_tau_branches["tau_charge"], str) else ak.zeros_like(tau_pt)

        met_val = a[met_branch] if ((compute_met_proj or compute_mt) and met_branch in a.fields) else None
        met_phi_val = a[met_phi_branch] if ((compute_met_proj or compute_mt) and met_phi_branch in a.fields) else None

        pair_info = build_pair_kinematics_and_labels(
            jet_pt, jet_eta, jet_phi, jet_mass, jet_n_muons, jet_label,
            tau_pt, tau_eta, tau_phi, tau_nProng, tau_decayMode, tau_charge, tau_label,
            met=met_val, met_phi=met_phi_val,
            compute_pt_ratio=compute_pt_ratio,
            compute_met_proj=compute_met_proj,
            compute_mt=compute_mt,
        )

        for key, branch_name in extra_jet_branches.items():
            if key in pair_info:
                continue
            val = a[branch_name][jet_sel]
            val_ct, _ = ak.broadcast_arrays(val[:, :, None], tau_pt[:, None, :])
            pair_info[key] = val_ct

        for key, branch_name in extra_tau_branches.items():
            if key in pair_info:
                continue
            val = a[branch_name][tau_sel]
            _, val_ct = ak.broadcast_arrays(jet_pt[:, :, None], val[:, None, :])
            pair_info[key] = val_ct

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

        _save_pair_manifest(manifest_file, chunk_sizes, feature_keys, label_index_map, dr_thr, complete=False)

        if verbose:
            print(f"[OK] chunk {chunk_idx - 1}: {X.shape[0]} coppie salvate in {chunk_file}")

    _save_pair_manifest(manifest_file, chunk_sizes, feature_keys, label_index_map, dr_thr, complete=True)

    return {
        "n_pairs_saved": int(sum(chunk_sizes)),
        "n_files_processed": len(chunk_sizes),
        "manifest_file": manifest_file,
    }

def split_pairs_by_event(event_id, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
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