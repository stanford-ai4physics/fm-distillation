import torch
import torch.distributed as dist
import h5py
import hashlib
import shutil
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import requests
import re
import os
from urllib.parse import urljoin
import numpy as np
from pathlib import Path


class _LRUFileCache:
    """Bounded LRU cache for h5py.File handles.

    Without a cap, persistent DataLoader workers accumulate an open handle
    for every file they visit. With random pretrain shuffling across thousands
    of companion files this exhausts node RAM. maxsize=128 keeps at most 128
    handles per worker (256 total with source+teacher) while absorbing most
    locality in the random-access pattern.
    """

    def __init__(self, maxsize=128):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, key, opener):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        fh = opener(key)
        if len(self._cache) >= self._maxsize:
            _, old_fh = self._cache.popitem(last=False)
            try:
                old_fh.close()
            except Exception:
                pass
        self._cache[key] = fh
        return fh

    def close_all(self):
        for fh in self._cache.values():
            try:
                fh.close()
            except Exception:
                pass
        self._cache.clear()


def collate_point_cloud(batch, max_part=5000):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]

    # Stack once to avoid repeated slicing
    point_clouds = torch.stack(batch_X)  # (B, N, F)
    labels = torch.stack(batch_y)  # (B, num_classes)

    # Use validity mask based on feature index 2
    valid_mask = point_clouds[:, :, 2] != 0
    max_particles = min(valid_mask.sum(dim=1).max().item(), max_part)
    max_particles = point_clouds.shape[1]

    # Truncate point clouds
    truncated_X = point_clouds[:, :max_particles, :].contiguous()  # (B, M, F)
    result = {"X": truncated_X, "y": labels}

    # sample_key is always present: (B, 2) tensor of (file_idx, sample_idx)
    result["sample_key"] = torch.stack([item["sample_key"] for item in batch])

    # teacher_logits is present only when a teacher labels file was loaded
    if all(item.get("teacher_logits") is not None for item in batch):
        result["teacher_logits"] = torch.stack(
            [item["teacher_logits"] for item in batch]
        )
    else:
        result["teacher_logits"] = None

    # teacher_cls_embed is present only when a teacher labels file with a
    # `teacher_cls_embed` dataset was loaded (CLS-MSE feature distillation).
    if all(item.get("teacher_cls_embed") is not None for item in batch):
        result["teacher_cls_embed"] = torch.stack(
            [item["teacher_cls_embed"] for item in batch]
        )
    else:
        result["teacher_cls_embed"] = None

    # Handle optional fields in a loop to reduce code duplication
    optional_fields = ["cond", "pid", "add_info", "data_pid", "vertex_pid", "weight"]
    for field in optional_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            # Truncate if it's sequence-like (i.e., has 2 or more dims)
            if stacked.dim() >= 2 and stacked.shape[1] >= max_particles:
                stacked = stacked[:, :max_particles].contiguous()
            result[field] = stacked
        else:
            result[field] = None

    return result


def get_url(
    dataset_name,
    dataset_type,
    base_url="https://portal.nersc.gov/cfs/dasrepo/omnilearned/",
):
    url = f"{base_url}/{dataset_name}/{dataset_type}/"
    try:
        requests.head(url, allow_redirects=True, timeout=5)
        return url
    except requests.RequestException:
        print(
            "ERROR: Request timed out, visit https://www.nersc.gov/users/status for status on  portal.nersc.gov"
        )
        return None


def download_h5_files(base_url, destination_folder):
    """
    Downloads all .h5 files from the specified directory URL.

    Args:
        base_url (str): The base URL of the directory containing the .h5 files.
        destination_folder (str): The local folder to save the downloaded files.
    """

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}")
        return

    file_links = re.findall(r'href="([^"]+\.h5)"', response.text)

    for file_name in file_links:
        file_url = urljoin(base_url, file_name)
        file_path = os.path.join(destination_folder, file_name)

        print(f"Downloading {file_url} to {file_path}")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_name}")


class HEPDataset(Dataset):
    def __init__(
        self,
        file_paths,
        file_indices=None,
        use_cond=False,
        use_pid=False,
        pid_idx=4,
        use_add=False,
        num_add=4,
        label_shift=0,
        clip_inputs=False,
        mode="",
        nevts=-1,
        teacher_file_paths=None,
    ):
        """
        Args:
            file_paths (list): List of file paths.
            use_pid (bool): Flag to select if PID information is used during training
            use_add (bool): Flags to select if additional information besides kinematics are used
            teacher_file_paths (list): List parallel to file_paths; each entry is
                the path to a companion teacher-logits .h5 (with a `teacher_logits`
                dataset indexed by sample_idx) for the matching source file, or
                None if the source file already has `teacher_logits` merged into
                it directly (see merge_teacher_logits.py) or no teacher logits are
                available for it. A source file's own `teacher_logits` dataset,
                when present, always takes priority over the companion file --
                merged files need no second handle open per sample.
        """
        self.use_cond = use_cond
        self.use_pid = use_pid
        self.use_add = use_add
        self.pid_idx = pid_idx
        self.num_add = num_add
        self.label_shift = label_shift

        self.file_paths = file_paths
        # Bounded LRU (see _LRUFileCache docstring): with thousands of files in
        # the full pretrain mixture, sizing this to len(file_paths) means it
        # never evicts, so all num_workers persistent workers accumulate an
        # open h5py.File handle for every unique file they touch over the
        # course of training -> unbounded host RAM growth that can eventually
        # stall/OOM-kill a worker. Keep the cap at 128 regardless of dataset
        # size, matching what the class was actually designed for.
        self._file_cache = _LRUFileCache(maxsize=128)
        self.file_indices = file_indices
        self.clip_inputs = clip_inputs
        self.mode = mode
        self.nevts = int(nevts)
        if self.nevts < 0:
            self.nevts = len(self.file_indices)

        self.teacher_file_paths = teacher_file_paths
        self._teacher_cache = _LRUFileCache(maxsize=128)

        # random.shuffle(self.file_indices)  # Shuffle data entries globally

    def __len__(self):
        return min(self.nevts, len(self.file_indices))

    def _get_file(self, file_idx):
        # rdcc_nbytes=0: single random-row reads get no benefit from h5py’s
        # default 1 MB per-dataset chunk cache; disabling it saves ~1 MB per
        # open handle, which matters when the LRU holds up to 128 handles.
        return self._file_cache.get(
            file_idx,
            lambda idx: h5py.File(self.file_paths[idx], "r", rdcc_nbytes=0),
        )

    def _get_teacher_file(self, file_idx):
        return self._teacher_cache.get(
            file_idx,
            lambda idx: h5py.File(self.teacher_file_paths[idx], "r", rdcc_nbytes=0),
        )

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_indices[idx]
        f = self._get_file(file_idx)

        sample = {}
        sample["sample_key"] = torch.tensor(
            [file_idx, sample_idx], dtype=torch.int64
        )

        sample["X"] = torch.tensor(f["data"][sample_idx], dtype=torch.float32)
        if self.clip_inputs:
            # Enforce particles to be inside R=0.8 and pT > 0.5 MeV
            mask_part = (torch.hypot(sample["X"][:, 0], sample["X"][:, 1]) < 0.8) & (
                sample["X"][:, 2] > 0.0
            )
            sample["X"][:, 3] = np.clip(
                sample["X"][:, 3], a_min=sample["X"][:, 2], a_max=None
            )
            sample["X"] = sample["X"] * mask_part.unsqueeze(-1).float()

        label = f["pid"][sample_idx]

        if self.mode == "regression":
            pid_dtype = torch.float32
        else:
            pid_dtype = torch.int64

        sample["y"] = torch.tensor(label - self.label_shift, dtype=pid_dtype)
        if "global" in f and self.use_cond:
            sample["cond"] = torch.tensor(f["global"][sample_idx], dtype=torch.float32)

        if self.use_pid:
            sample["pid"] = sample["X"][:, self.pid_idx].int()
            sample["X"] = torch.cat(
                (sample["X"][:, : self.pid_idx], sample["X"][:, self.pid_idx + 1 :]),
                dim=1,
            )
        if self.use_add:
            # Assume any additional info appears last
            sample["add_info"] = sample["X"][:, -self.num_add :]
            sample["X"] = sample["X"][:, : -self.num_add]

        if self.mode in ["segmentation", "ftag"]:
            if self.mode == "segmentation":
                data_dtype = torch.float32
            elif self.mode == "ftag":
                data_dtype = torch.int64

            sample["data_pid"] = torch.tensor(
                f["data_pid"][sample_idx], dtype=data_dtype
            )

        if "teacher_logits" in f:
            # Merged file (merge_teacher_logits.py): logits live alongside
            # `data`/`pid` in the same handle, no second file open needed.
            sample["teacher_logits"] = torch.from_numpy(
                f["teacher_logits"][sample_idx].astype(np.float32)
            )
        elif (
            self.teacher_file_paths is not None
            and self.teacher_file_paths[file_idx] is not None
        ):
            tf = self._get_teacher_file(file_idx)
            sample["teacher_logits"] = torch.from_numpy(
                tf["teacher_logits"][sample_idx].astype(np.float32)
            )
        else:
            sample["teacher_logits"] = None

        if "teacher_cls_embed" in f:
            # Merged file: cls_embed lives alongside `data`/`pid`, no second
            # file open needed.
            sample["teacher_cls_embed"] = torch.from_numpy(
                f["teacher_cls_embed"][sample_idx].astype(np.float32)
            )
        elif (
            self.teacher_file_paths is not None
            and self.teacher_file_paths[file_idx] is not None
        ):
            tf = self._get_teacher_file(file_idx)
            # Opt-in field (tools/preprocess/build_teacher_h5.py --include-cls-embed): older
            # companion files built without it simply won't have this key.
            if "teacher_cls_embed" in tf:
                sample["teacher_cls_embed"] = torch.from_numpy(
                    tf["teacher_cls_embed"][sample_idx].astype(np.float32)
                )
            else:
                sample["teacher_cls_embed"] = None
        else:
            sample["teacher_cls_embed"] = None

        return sample

    def __del__(self):
        self._file_cache.close_all()
        self._teacher_cache.close_all()


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    distributed=True,
    use_cond=False,
    use_pid=False,
    pid_idx=4,
    use_add=False,
    num_add=4,
    num_workers=16,
    rank=0,
    size=1,
    clip_inputs=False,
    mode="",
    shuffle=True,
    nevts=-1,
    teacher_labels_dir=None,
    teacher_tag=None,
    num_chunks=1,
    chunk_idx=0,
):
    supported_datasets = [
        "top",
        "qg",
        "pretrain",
        "atlas",
        "aspen",
        "jetclass",
        "jetclass2",
        "h1",
        "toy",
        "cms_qcd",
        "cms_bsm",
        "cms_top",
        "aspen_bsm",
        "aspen_bsm_ad_sb",
        "aspen_bsm_ad_sr",
        "aspen_top_ad_sb",
        "aspen_top_ad_sr",
        "aspen_top_ad_sr_hl",
        "qcd_dijet",
        "jetnet150",
        "jetnet30",
        "dctr",
        "atlas_flav",
        "custom",
        "camels",
        "quijote",
        "microboone",
        "aspen_bsm_ad_sb",
        "aspen_bsm_ad_sr",
        "aspen_bsm_ad_sr_hl",
        "lhco_ad_data",
        "lhco_ad_bkg",
        "lhco_ad",
        "collide",
    ]
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}."
        )

    if dataset_name == "pretrain":
        names = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]
        types = [dataset_type]
    else:
        names = [dataset_name]
        types = [dataset_type]

    dataset_paths = [os.path.join(path, name, type) for name in names for type in types]

    file_list = []
    file_index_parts = []
    index_shift = 0
    teacher_on = teacher_labels_dir is not None and teacher_tag is not None
    teacher_file_list = [] if teacher_on else None
    for iname, dataset_path in enumerate(dataset_paths):
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        if not any(dataset_path.iterdir()):
            print(f"Fetching download url for dataset {names[iname]}")
            url = get_url(names[iname], dataset_type)
            if url is None:
                raise ValueError(f"No download URL found for dataset '{dataset_name}'.")
            download_h5_files(url, dataset_path)

        # Sort for deterministic ordering across runs/filesystems; a saved
        # file_index.npy is only valid against the same order.
        h5_files = sorted(
            list(dataset_path.glob("*.h5")) + list(dataset_path.glob("*.hdf5"))
        )
        file_list.extend(map(str, h5_files))  # Convert to string paths

        # Companion teacher-logits .h5, one per source file, in the SAME order.
        # Path mirrors tools/preprocess/build_teacher_h5.py:
        #   <teacher_labels_dir>/<dataset>/<split>/<source_stem>.h5
        # A source file already carrying its own `teacher_logits` dataset (see
        # merge_teacher_logits.py) needs no companion -- HEPDataset reads it
        # straight from the source handle, so we append None here.
        if teacher_on:
            for h5 in h5_files:
                try:
                    with h5py.File(h5, "r") as hf:
                        merged_in = "teacher_logits" in hf
                except OSError:
                    merged_in = False
                if merged_in:
                    teacher_file_list.append(None)
                    continue
                companion = (
                    Path(teacher_labels_dir)
                    / names[iname]
                    / dataset_type
                    / (h5.stem + ".h5")
                )
                if not companion.is_file():
                    raise FileNotFoundError(
                        f"Missing teacher companion for {names[iname]}/{dataset_type}: "
                        f"{companion} (run tools/preprocess/build_teacher_h5.py for this dataset/split, "
                        "or merge_teacher_logits.py to fold it into the source file)"
                    )
                teacher_file_list.append(str(companion))

        def _validate_index(arr, files):
            if arr.size == 0:
                return False
            fids = np.unique(arr[:, 0])
            if int(fids.max()) >= len(files) or int(fids.min()) < 0:
                return False
            for fid in fids:
                max_si = int(arr[arr[:, 0] == fid, 1].max())
                try:
                    with h5py.File(files[int(fid)], "r") as f:
                        if max_si >= len(f["data"]):
                            return False
                except Exception:
                    return False
            return True

        def _file_fingerprint(files):
            # Cheap (stat, not h5py open) fingerprint of the file list so a
            # later run can skip _validate_index's per-file HDF5 opens
            # entirely once nothing on disk has changed since it last passed.
            parts = [f"{f.name}:{(st := f.stat()).st_size}:{int(st.st_mtime)}" for f in files]
            return hashlib.sha1("\n".join(parts).encode()).hexdigest()

        def _validated(candidate, arr, files):
            # _validate_index opens every unique file referenced in the index
            # (up to ~thousands for the pretrain mixture) via h5py -- doing
            # that independently on all DDP ranks is a Nx redundant Lustre
            # burst. Only rank 0 pays the cost (using a fingerprint sentinel
            # to skip it entirely on repeat runs against unchanged data);
            # every other rank just trusts the broadcast result.
            distributed_on = dist.is_available() and dist.is_initialized()
            if rank == 0 or not distributed_on:
                fp_path = candidate.with_suffix(candidate.suffix + ".fp")
                current_fp = _file_fingerprint(files)
                if fp_path.is_file() and fp_path.read_text().strip() == current_fp:
                    is_valid = True
                else:
                    is_valid = _validate_index(arr, files)
                    if is_valid:
                        try:
                            fp_path.write_text(current_fp)
                        except OSError:
                            pass
            else:
                is_valid = None
            if distributed_on:
                result = [is_valid]
                dist.broadcast_object_list(result, src=0)
                is_valid = result[0]
            return bool(is_valid)

        scratch_root = os.environ.get("SCRATCH")
        scratch_index = (
            Path(scratch_root)
            / "omnilearned_cache"
            / names[iname]
            / dataset_type
            / "file_index.npy"
            if scratch_root
            else None
        )

        index_file = dataset_path / "file_index.npy"

        # Stage CFS index → $SCRATCH once via rank 0 before any rank reads.
        # Concurrent np.load() from many DDP ranks against the same CFS file
        # has been observed to fail with "could only read 0 elements".
        # $SCRATCH (also Lustre, but tuned for parallel I/O) handles the
        # concurrent read pattern reliably once the file is freshly written.
        if scratch_index is not None and index_file.is_file():
            if rank == 0 and not scratch_index.is_file():
                scratch_index.parent.mkdir(parents=True, exist_ok=True)
                tmp = scratch_index.with_suffix(scratch_index.suffix + ".tmp")
                shutil.copyfile(str(index_file), str(tmp))
                tmp.rename(scratch_index)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        indices_arr = None
        # Prefer scratch_index over the CFS copy.
        for candidate in (scratch_index, index_file):
            if candidate is None or not candidate.is_file():
                continue
            # Load fully into RAM: mmap on Lustre/CFS is unreliable with
            # concurrent DDP ranks and can fail in mmap.mmap with large indices.
            arr = np.load(candidate)
            if _validated(candidate, arr, h5_files):
                indices_arr = arr
                break
            if rank == 0:
                print(
                    f"WARNING: cached index {candidate} is inconsistent with current "
                    f"file ordering for dataset {names[iname]}; will regenerate"
                )

        if indices_arr is None:
            print(f"Creating index list for dataset {names[iname]}")
            new_indices = []
            for file_idx, path in enumerate(h5_files):
                try:
                    with h5py.File(path, "r") as f:
                        num_samples = len(f["data"])
                        new_indices.extend([(file_idx, i) for i in range(num_samples)])
                except Exception as e:
                    print(f"ERROR: File {path} is likely corrupted: {e}")
            indices_arr = np.array(new_indices, dtype=np.int32)

            save_target = None
            if os.access(dataset_path, os.W_OK):
                save_target = index_file
            elif scratch_index is not None:
                if rank == 0:
                    scratch_index.parent.mkdir(parents=True, exist_ok=True)
                save_target = scratch_index
            if save_target is not None and rank == 0:
                np.save(save_target, indices_arr)
                print(f"Saved index to {save_target}; {len(indices_arr)} events")
            elif save_target is None:
                print(
                    f"WARNING: no writable cache location for index; "
                    f"recomputed in-memory ({len(indices_arr)} events)"
                )

        if num_chunks > 1:
            if not (0 <= chunk_idx < num_chunks):
                raise ValueError(
                    f"chunk_idx={chunk_idx} out of range for num_chunks={num_chunks}"
                )
            total = len(indices_arr)
            c0 = total * chunk_idx // num_chunks
            c1 = total * (chunk_idx + 1) // num_chunks
            indices_arr = indices_arr[c0:c1]
            if rank == 0:
                print(
                    f"Chunk {chunk_idx}/{num_chunks} on {names[iname]}: "
                    f"events {c0}..{c1} of {total}"
                )

        if shuffle:
            indices = indices_arr[rank::size]
        else:
            total = len(indices_arr)
            indices = indices_arr[total * rank // size : total * (rank + 1) // size]

        # Keep the index as a compact numpy array, NOT a Python list of tuples.
        # The full pretrain index is ~1e9 rows: as int32 that's ~8 GB, but as a
        # list of (int, int) tuples it balloons to ~75 GB and, once forked into
        # num_workers dataloader processes, COW-copies per object and OOMs the
        # node. Shift the file-id column in place; concatenate once after the loop.
        shifted = np.asarray(indices, dtype=np.int32).copy()
        shifted[:, 0] += index_shift
        file_index_parts.append(shifted)
        index_shift += len(h5_files)

    file_indices = (
        np.concatenate(file_index_parts)
        if file_index_parts
        else np.empty((0, 2), dtype=np.int32)
    )

    # Shift labels if they are not used for pretrain
    label_shift = {
        "jetclass": 2,
        "jetclass2": 12,
        "aspen": 200,
        "cms_qcd": 201,
        "cms_bsm": 202,
    }

    data = HEPDataset(
        file_list,
        file_indices,
        use_cond=use_cond,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        label_shift=label_shift.get(dataset_name, 0),
        clip_inputs=clip_inputs,
        mode=mode,
        nevts=nevts,
        teacher_file_paths=teacher_file_list,
    )

    loader = DataLoader(
        data,
        batch_size=batch,
        pin_memory=torch.cuda.is_available(),
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=collate_point_cloud,
    )
    return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="top",
        help="Dataset name to download",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="./",
        help="Folder to save the dataset",
    )
    args = parser.parse_args()

    for tag in ["train", "test", "val"]:
        load_data(args.dataset, args.folder, dataset_type=tag, distributed=False)
