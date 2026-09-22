"""Concatenate per-chunk/rank logit files into one .npz per split.

Input filenames produced by evaluate.py follow either:
    outputs_<tag>_<dataset>_<split>_chunk<K>of<N>_rank<r>.npz   (num_chunks > 1)
    outputs_<tag>_<dataset>_<split>_rank<r>.npz                 (num_chunks == 1)

This script groups files by <split> and writes:
    <tag>_<dataset>_<split>.npz
into the same directory (or --outdir if provided). The split name is part of
the output filename, so you can always tell train/val/test apart.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

CHUNK_RE = re.compile(
    r"^outputs_(?P<tag>.+)_(?P<dataset>[^_]+)_(?P<split>train|val|test)"
    r"(?:_chunk\d+of\d+)?_rank\d+\.npz$"
)


def group_chunks(indir, tag, dataset):
    """Return {split: [sorted chunk paths]} for files matching tag+dataset."""
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(indir, "outputs_*.npz")):
        m = CHUNK_RE.match(os.path.basename(path))
        if not m:
            continue
        if tag and m["tag"] != tag:
            continue
        if dataset and m["dataset"] != dataset:
            continue
        groups[m["split"]].append(path)
    for split in groups:
        groups[split].sort()
    return groups


def concat_split(paths):
    """Concatenate all chunks for one split. Returns dict of arrays."""
    keys = None
    buffers = defaultdict(list)
    for p in paths:
        with np.load(p) as z:
            if keys is None:
                keys = list(z.keys())
            for k in keys:
                buffers[k].append(z[k])
    out = {}
    for k, arrs in buffers.items():
        # `cond` is written as np.array([]) when absent — skip concat in that case.
        if all(a.size == 0 for a in arrs):
            out[k] = np.array([])
        else:
            out[k] = np.concatenate(arrs, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory with chunk .npz files")
    ap.add_argument("--outdir", default=None, help="Where to write merged files (default: --indir)")
    ap.add_argument("--tag", default="", help="Filter on save tag (e.g. fine_tune_top_l)")
    ap.add_argument("--dataset", default="", help="Filter on dataset name (e.g. top)")
    args = ap.parse_args()

    outdir = args.outdir or args.indir
    os.makedirs(outdir, exist_ok=True)

    groups = group_chunks(args.indir, args.tag, args.dataset)
    if not groups:
        print(f"No matching chunk files found in {args.indir}")
        return

    for split, paths in sorted(groups.items()):
        print(f"[{split}] merging {len(paths)} chunks")
        merged = concat_split(paths)
        name_tag = args.tag or "merged"
        name_ds = args.dataset or "data"
        out_path = os.path.join(outdir, f"{name_tag}_{name_ds}_{split}.npz")
        np.savez(out_path, **merged)
        n = next(iter(merged.values())).shape[0] if merged else 0
        print(f"  -> {out_path}  ({n} events)")


if __name__ == "__main__":
    main()
