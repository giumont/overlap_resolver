"""
Build the (reco jet, reco tau) pair dataset for the HH -> bb tau tau study.

Role in the project: starting from the ntuples, this module produces the
pair-level table on which the b-jet / tau-jet identification is studied
(objective 3.1, overlap between the two kinds of reconstructed objects)
and on which the classification quality and the manual combination of
discriminants are evaluated (objective 3.2). The steps are:

1. select analysis-level jets and taus;
2. form all (jet, tau) pairs per event and keep the geometrically
   overlapping ones (DeltaR < ``dr_thr``);
3. assign each pair a truth index in {FF, FT, TF, TT} (first letter:
   b-jet truth, second letter: hadronic-tau truth);
4. flatten to ``(X, y, event_id)``, compatible with the existing
   preprocessing, and save it in chunks (one per input file) with the same
   checkpoint path scheme as ``flavour_tag_ml.checkpoint_io``;
5. split train/val/test grouped by event, to avoid leakage between pairs
   of the same event.

Identification discriminants (e.g. ``params.JET_SCORE_BRANCH``,
``params.TAU_ID_SCORE_BRANCH``) can be added to the features through
``extra_jet_branches`` / ``extra_tau_branches``.

Defaults (branch names, DeltaR threshold, label index, split fractions) are
defined in ``params.py``.
"""

import os
from pathlib import Path

import awkward as ak
import numpy as np

import obj_3_1
import params
from overlap_kinematics import build_pair_kinematics_and_labels

from flavour_tag_ml.checkpoint_io import (
    _manifest_path,
    _chunk_dir_path,
    _chunk_file_path,
    _load_chunks_into_arrays,
)


def select_analysis_objects(tree, n_entries, jet_analysis_branch, tau_analysis_branch):
    """
    Build the analysis-level selection masks for jets and taus.

    Parameters
    ----------
    tree : uproot.TTree
        Tree to read the selection branches from.

    n_entries : int
        Number of events to read.

    jet_analysis_branch, tau_analysis_branch : str
        Branches holding the per-object analysis flag for jets and taus
        (e.g. `params.JET_IS_ANALYSIS_BRANCH`, `params.TAU_IS_ANALYSIS_BRANCH`).

    Returns
    -------
    jet_sel, tau_sel : ak.Array
        Jagged selection masks, as returned by
        `obj_3_1.get_analysis_selection`.
    """
    jet_sel = obj_3_1.get_analysis_selection(tree, jet_analysis_branch, n_entries)
    tau_sel = obj_3_1.get_analysis_selection(tree, tau_analysis_branch, n_entries)
    return jet_sel, tau_sel


def build_pair_truth_index(jet_label_ct, tau_label_ct, label_index_map=None):
    """
    Map per-pair jet and tau truth flags to the integer pair label.

    Parameters
    ----------
    jet_label_ct, tau_label_ct : ak.Array
        Truth flags (cast to bool) broadcast to the (event, jet, tau)
        pair structure. True means a true b-jet / true hadronic tau.

    label_index_map : dict or None, default=None
        Maps ``"FF"``, ``"FT"``, ``"TF"``, ``"TT"`` to integer labels
        (first letter: jet, second: tau). None uses
        `params.PAIR_LABEL_INDEX`.

    Returns
    -------
    ak.Array
        Same structure as the inputs, with int64 labels.
    """
    if label_index_map is None:
        label_index_map = params.PAIR_LABEL_INDEX

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
    """
    Assign to every pair the index of the event it belongs to.

    Parameters
    ----------
    pair_info : dict of ak.Array
        Pair-level quantities with (event, jet, tau) structure.

    key : str, default="jet_pt"
        Entry of `pair_info` used to get the number of events and the
        structure to broadcast to.

    file_offset : int, default=0
        Added to the event index so that indices are unique across input
        files (the builder passes the cumulative number of events already
        processed).

    Returns
    -------
    ak.Array
        int64 event index with the same structure as ``pair_info[key]``.
    """
    n_events = len(pair_info[key])
    event_id = ak.Array(np.arange(n_events, dtype=np.int64) + file_offset)
    event_id_ct, _ = ak.broadcast_arrays(event_id, pair_info[key])
    return event_id_ct


def flatten_pairs_for_ml(pair_info, dr_thr, feature_keys, label_idx_ct, event_id_ct):
    """
    Keep the overlapping pairs and flatten them to ML arrays.

    Parameters
    ----------
    pair_info : dict of ak.Array
        Pair-level quantities; must contain ``"pair_dr"`` and every key in
        `feature_keys`.

    dr_thr : float
        Pairs with ``pair_dr < dr_thr`` are kept.

    feature_keys : list of str
        Entries of `pair_info` used as columns of ``X``, in this order.

    label_idx_ct : ak.Array
        Pair truth index (see `build_pair_truth_index`).

    event_id_ct : ak.Array
        Pair event index (see `build_pair_event_index`).

    Returns
    -------
    X : numpy.ndarray, shape (n_pairs, len(feature_keys)), float32
    y : numpy.ndarray, shape (n_pairs,), float32
        Integer pair labels stored as float32.
    event_id : numpy.ndarray, shape (n_pairs,), int64
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


def _save_pair_chunk(chunk_dir, chunk_idx, X, y, event_id):
    """
    Save one chunk of the pair dataset as a compressed ``.npz``.

    Parameters
    ----------
    chunk_dir : str or pathlib.Path
        Chunk directory (created if missing).

    chunk_idx : int
        Chunk index, used to build the file name via
        ``flavour_tag_ml.checkpoint_io._chunk_file_path``.

    X, y, event_id : numpy.ndarray
        Arrays returned by `flatten_pairs_for_ml`.

    Returns
    -------
    str or pathlib.Path
        Path of the written chunk file.
    """
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_file = _chunk_file_path(chunk_dir, chunk_idx)
    np.savez_compressed(chunk_file, X=X, y=y, event_id=event_id)
    return chunk_file


def _save_pair_manifest(manifest_file, chunk_sizes, feature_names, label_index_map, dr_thr, complete):
    """
    Write the manifest describing the chunked pair dataset.

    Parameters
    ----------
    manifest_file : str or pathlib.Path
        Manifest path.

    chunk_sizes : list of int
        Number of pairs in each saved chunk.

    feature_names : list of str
        Column names of ``X``.

    label_index_map : dict
        Mapping ``{"FF", "FT", "TF", "TT"} -> int`` used for ``y``.

    dr_thr : float
        DeltaR threshold used to select the pairs.

    complete : bool
        False while the build is running (allows resuming/inspection),
        True once every input file has been processed.

    Returns
    -------
    None
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
    """
    Read a manifest written by `_save_pair_manifest`.

    Parameters
    ----------
    manifest_file : str or pathlib.Path
        Manifest path.

    Returns
    -------
    dict
        Keys ``chunk_sizes`` (list of int), ``feature_names`` (list of
        str), ``label_index_map`` (dict), ``dr_thr`` (float) and
        ``complete`` (bool).
    """
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
    Report how many pairs are stored in a pair-dataset checkpoint.

    Parameters
    ----------
    save_path : str or pathlib.Path
        Directory of the checkpoint.

    save_name : str
        Checkpoint name, used by ``checkpoint_io`` to locate manifest and
        chunk directory.

    verbose : bool, default=True
        If True, print the status (COMPLETE/PARTIAL) and the counts.

    Returns
    -------
    dict or None
        None if no manifest exists. Otherwise ``n_done`` (total pairs
        saved), ``n_chunks`` and ``complete``.
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
    Concatenate the chunks of a checkpoint into single arrays.

    Parameters
    ----------
    save_path, save_name : str
        Locate the checkpoint (see `check_pair_checkpoint_progress`).

    output_file : str or None, default=None
        Where to write the merged ``.npz`` (containing X, y, event_id,
        feature_names, label_index_map, dr_thr):

        - None: ``<save_path>/<save_name>_merged.npz``.
        - ``""``: nothing is written, arrays are only returned.

    delete_chunks_after : bool, default=False
        If True and the merged file was written, delete chunk files, the
        chunk directory (if empty) and the manifest. Ignored when
        ``output_file=""``.

    verbose : bool, default=True
        Print shapes and file operations.

    Returns
    -------
    X : numpy.ndarray, shape (n_pairs, n_features)
    y : numpy.ndarray, shape (n_pairs,)
    event_id : numpy.ndarray, shape (n_pairs,)
    feature_names : list of str

    Raises
    ------
    FileNotFoundError
        If the manifest does not exist.
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


def build_pair_dataset_from_root(
    save_path, save_name,
    jet_analysis_branch, tau_analysis_branch,
    jet_eta_branch, jet_phi_branch, jet_pt_branch,
    tau_eta_branch, tau_phi_branch, tau_pt_branch,
    jet_truth_label_fn, tau_truth_label_fn,
    root_dir=None,
    jet_truth_label_branch=params.JET_TRUTH_LABEL_BRANCH,
    tau_truth_label_branch=params.TAU_TRUTH_MATCH_BRANCH,
    extra_jet_branches=None,
    extra_tau_branches=None,
    met_branch=params.MET_BRANCH,
    met_phi_branch=params.MET_PHI_BRANCH,
    compute_pt_ratio=params.COMPUTE_PT_RATIO,
    compute_met_proj=params.COMPUTE_MET_PROJ,
    compute_mt=params.COMPUTE_MT,
    feature_keys=None,
    dr_thr=params.PAIR_DATASET_DR_THRESHOLD,
    label_index_map=None,
    verbose=True,
):
    """
    Build the chunked (jet, tau) pair dataset from the ``.root`` files.

    For each input file (as listed by `obj_3_1.load_files`): read the
    needed branches, select analysis-level jets and taus, compute pair
    kinematics and truth labels with
    `overlap_kinematics.build_pair_kinematics_and_labels`, keep pairs with
    ``pair_dr < dr_thr`` and save them as one chunk. The manifest is
    updated after every chunk (``complete=False``) and marked complete at
    the end. Files lacking any required branch are skipped.

    Parameters
    ----------
    save_path, save_name : str
        Checkpoint location, resolved through ``checkpoint_io``.

    jet_analysis_branch, tau_analysis_branch : str
        Per-object analysis flags used for the object selection.

    jet_eta_branch, jet_phi_branch, jet_pt_branch : str
        Jet kinematic branches.

    tau_eta_branch, tau_phi_branch, tau_pt_branch : str
        Tau kinematic branches.

    jet_truth_label_fn, tau_truth_label_fn : callable
        ``fn(a, sel)`` where ``a`` is the ``ak.Array`` of all read
        branches and ``sel`` the jet (tau) selection mask. Must return the
        truth flag (True = true b-jet / true hadronic tau) of the selected
        objects.

    root_dir : str or pathlib.Path or None, default=None
        If given, overrides ``obj_3_1.ROOT_DIR`` (global side effect on
        `obj_3_1`); None keeps the value from `params.ROOT_DIR`.

    jet_truth_label_branch : str, default=`params.JET_TRUTH_LABEL_BRANCH`
        Branch read for the jet truth label (skipped if empty).

    tau_truth_label_branch : str, default=`params.TAU_TRUTH_MATCH_BRANCH`
        Branch read for the tau truth label (skipped if empty).

    extra_jet_branches, extra_tau_branches : list of str or dict or None, default=None
        Additional per-object branches (e.g. identification scores) added
        as pair features. A list is converted to ``{branch: branch}``; in
        a dict, keys are feature names and values branch names. The keys
        ``jet_mass`` and ``jet_n_muons`` (jets) and ``tau_nProng``,
        ``tau_decayMode`` and ``tau_charge`` (taus) are also passed to the
        pair-kinematics builder; if absent, these quantities are filled
        with zeros.

    met_branch, met_phi_branch : str, default=`params.MET_BRANCH`, `params.MET_PHI_BRANCH`
        MET magnitude and azimuth, read only if `compute_met_proj` or
        `compute_mt` is True.

    compute_pt_ratio : bool, default=`params.COMPUTE_PT_RATIO`
        Compute ``pair_pt_ratio`` (jet pT / tau pT) and add it to the
        default features.

    compute_met_proj : bool, default=`params.COMPUTE_MET_PROJ`
        Compute ``tau_met_proj`` (MET projected on the tau direction) and
        add it to the default features.

    compute_mt : bool, default=`params.COMPUTE_MT`
        Compute ``tau_mt`` (tau-MET transverse mass) and add it to the
        default features.

    feature_keys : list of str or None, default=None
        Columns of ``X``. None uses `params.PAIR_BASE_FEATURE_KEYS`, plus
        the enabled optional variables and the keys of the extra
        branches, without duplicates.

    dr_thr : float, default=`params.PAIR_DATASET_DR_THRESHOLD`
        Maximum DeltaR of a stored pair.

    label_index_map : dict or None, default=None
        ``{"FF", "FT", "TF", "TT"} -> int``; None uses
        `params.PAIR_LABEL_INDEX`.

    verbose : bool, default=True
        Print per-chunk progress and skipped files.

    Returns
    -------
    dict or None
        None if no ``.root`` file is available. Otherwise
        ``n_pairs_saved``, ``n_files_processed`` (files actually saved)
        and ``manifest_file``.
    """
    if root_dir is not None:
        obj_3_1.ROOT_DIR = Path(root_dir)

    if label_index_map is None:
        label_index_map = params.PAIR_LABEL_INDEX

    if isinstance(extra_jet_branches, (list, tuple)):
        extra_jet_branches = {b: b for b in extra_jet_branches}
    elif extra_jet_branches is None:
        extra_jet_branches = {}

    if isinstance(extra_tau_branches, (list, tuple)):
        extra_tau_branches = {b: b for b in extra_tau_branches}
    elif extra_tau_branches is None:
        extra_tau_branches = {}

    if feature_keys is None:
        feature_keys = list(params.PAIR_BASE_FEATURE_KEYS)
        if compute_pt_ratio:
            feature_keys.append("pair_pt_ratio")
        if compute_met_proj:
            feature_keys.append("tau_met_proj")
        if compute_mt:
            feature_keys.append("tau_mt")
        feature_keys.extend(list(extra_jet_branches.keys()) + list(extra_tau_branches.keys()))
        feature_keys = list(dict.fromkeys(feature_keys))

    loaded = obj_3_1.load_files()
    if not loaded:
        if verbose:
            print(f"Nessun file .root disponibile in: {obj_3_1.ROOT_DIR}")
        return None

    manifest_file = _manifest_path(save_path, save_name)
    chunk_dir = _chunk_dir_path(save_path, save_name)
    os.makedirs(chunk_dir, exist_ok=True)

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

    extra_branches = [
        b for b in list(extra_jet_branches.values()) + list(extra_tau_branches.values())
        if isinstance(b, str)
    ]
    branches = list(dict.fromkeys(core_branches + extra_branches))

    def optional_branch(a, extra, key, sel, template):
        if key in extra and isinstance(extra[key], str):
            return a[extra[key]][sel]
        return ak.zeros_like(template)

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

        jet_mass = optional_branch(a, extra_jet_branches, "jet_mass", jet_sel, jet_pt)
        jet_n_muons = optional_branch(a, extra_jet_branches, "jet_n_muons", jet_sel, jet_pt)

        tau_nProng = optional_branch(a, extra_tau_branches, "tau_nProng", tau_sel, tau_pt)
        tau_decayMode = optional_branch(a, extra_tau_branches, "tau_decayMode", tau_sel, tau_pt)
        tau_charge = optional_branch(a, extra_tau_branches, "tau_charge", tau_sel, tau_pt)

        use_met = compute_met_proj or compute_mt
        met_val = a[met_branch] if (use_met and met_branch in a.fields) else None
        met_phi_val = a[met_phi_branch] if (use_met and met_phi_branch in a.fields) else None

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


def split_pairs_by_event(
    event_id,
    train_frac=params.SPLIT_TRAIN_FRAC,
    val_frac=params.SPLIT_VAL_FRAC,
    test_frac=params.SPLIT_TEST_FRAC,
    seed=params.SPLIT_SEED,
):
    """
    Split pairs into train/val/test sets grouped by event.

    Unique events are shuffled and assigned to the three sets, so all
    pairs of one event end up in the same set (no leakage between pairs
    that share an event). The fractions therefore refer to events, not to
    pairs: the resulting pair fractions are only approximately equal to
    them.

    Parameters
    ----------
    event_id : numpy.ndarray
        Event index of each pair (see `build_pair_event_index`).

    train_frac, val_frac, test_frac : float
        Fractions of events in each set; must sum to 1. Defaults are
        `params.SPLIT_TRAIN_FRAC`, `params.SPLIT_VAL_FRAC`,
        `params.SPLIT_TEST_FRAC`. The test set takes the remaining events
        after rounding.

    seed : int, default=`params.SPLIT_SEED`
        Seed of the random generator.

    Returns
    -------
    train_idx, val_idx, test_idx : numpy.ndarray
        Indices of the pairs belonging to each set.

    Raises
    ------
    ValueError
        If the three fractions do not sum to 1.
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