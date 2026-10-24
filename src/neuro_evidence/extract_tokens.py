"""Extract NeuroVFM tokens for every scan in the OASIS canonical manifest, from the raw (unstripped) NIfTIs.

Each scan is encoded once per variant and written in the cached-token format read by
models.encoders.CachedTokenEncoder (`<PTID>__<VISCODE>__<EXAMDATE>.pt`), so a variant dir can be
used directly as `token_dir`:
  global: original NeuroVFM encoder, whole scan in one attention sequence (local_window: null)
  local:  attention restricted to non-overlapping windows of --local-window patches

Token selection follows the `neurovfm:` config section (strict pipeline: NeuroVFM's own background
rule, no masks). Existing outputs are skipped, so the job can be restarted. Shard with
--shard i --num-shards n to run several processes in parallel.

  cd src && python -m neuro_evidence.extract_tokens --local-window 2 4 4 --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from env.data import remap_data_path, token_name

from .data import load_config
from .encoder import NeuroVFMEncoder, TokenBatch

DATA_ROOT = "/mnt/shared/datasets/NeuRL-dataset"
MANIFEST = f"{DATA_ROOT}/oasis/processed_mri/oasis_neurovfm_canonical_manifest.csv"
OUT_ROOT = f"{DATA_ROOT}/oasis/processed"


def to_cache(batch: TokenBatch) -> dict:
    """Same keys/dtypes as the existing OASIS token cache, plus provenance."""
    return {
        "embeddings": torch.from_numpy(batch.embeddings).half(),
        "coords": torch.from_numpy(batch.coords).short(),
        "token_count": int(batch.embeddings.shape[0]),
        "embedding_dim": int(batch.embeddings.shape[1]),
        "grid_shape": torch.tensor(batch.dense_grid_shape, dtype=torch.int32),
        "neurovfm_foreground": torch.from_numpy(batch.neurovfm_foreground),
        "scan_path": batch.scan_path,
        "volume_size_dhw": list(batch.volume_size_dhw),
        "metadata": json.dumps(batch.metadata),
    }


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--local-window", type=int, nargs=3, default=[2, 4, 4], metavar=("D", "H", "W"))
    parser.add_argument("--variants", nargs="+", choices=["global", "local"], default=["global", "local"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Only the first N scans of this shard (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    w = args.local_window
    windows = {"global": None, "local": w}
    out_dirs = {
        "global": Path(args.out_root) / "neurovfm_tokens_raw_global",
        "local": Path(args.out_root) / f"neurovfm_tokens_raw_local_w{w[0]}x{w[1]}x{w[2]}",
    }
    for v in args.variants:
        out_dirs[v].mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.manifest)
    meta = meta.iloc[args.shard :: args.num_shards]
    if args.limit:
        meta = meta.iloc[: args.limit]
    enc = NeuroVFMEncoder(cfg)
    log = Path(args.out_root) / f"extract_tokens_shard{args.shard}of{args.num_shards}.log"
    done = failed = 0
    t0 = time.time()
    with open(log, "a") as fh:
        for i, row in enumerate(meta.itertuples(index=False)):
            name = token_name(row.PTID, row.VISCODE, row.EXAMDATE)
            todo = [v for v in args.variants if not (out_dirs[v] / name).exists()]
            if not todo:
                continue
            scan = remap_data_path(row.source_image_path, args.data_root)
            try:
                for v in todo:
                    cfg["neurovfm"]["local_window"] = windows[v]
                    save_atomic(to_cache(enc.extract(scan)), out_dirs[v] / name)
                done += 1
            except Exception:
                failed += 1
                fh.write(f"FAIL {name} {scan}\n{traceback.format_exc()}\n")
            if (i + 1) % 25 == 0:
                msg = f"[{i + 1}/{len(meta)}] extracted={done} failed={failed} {time.time() - t0:.0f}s"
                print(msg, flush=True)
                fh.write(msg + "\n")
                fh.flush()
        msg = f"DONE shard {args.shard}/{args.num_shards}: extracted={done} failed={failed} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        fh.write(msg + "\n")


if __name__ == "__main__":
    main()
