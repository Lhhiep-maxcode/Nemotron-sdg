#!/usr/bin/env python3
"""Edit config/knowledge.yaml, then run this file to synthesize a JSONL tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_sdg.streaming import run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/knowledge.yaml")
    parser.add_argument("--input-path", type=Path, help="a folder (recursive *.jsonl) or one JSONL file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-process", type=int, help="number of files processed simultaneously")
    parser.add_argument("--concurrency-per-process", type=int, help="concurrent samples inside each file worker")
    parser.add_argument("--limit", type=int, help="first N nonblank input rows per file, including resumed rows")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    config_path = args.config.resolve()
    overrides = OmegaConf.load(config_path)
    pipeline_path = config_path.parent / overrides.pop("pipeline_config", "pipeline.yaml")
    cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(pipeline_path), overrides), resolve=True)
    batch = cfg["batch"]
    for key in ("input_path", "output_dir", "num_process", "concurrency_per_process", "limit"):
        value = getattr(args, key)
        if value is not None:
            batch[key] = str(value) if isinstance(value, Path) else value
    for key in ("input_path", "output_dir"):
        path = Path(batch[key]).expanduser()
        batch[key] = str(path if path.is_absolute() else (config_path.parent / path).resolve())
    try:
        failures = run_directory(cfg)
    except KeyboardInterrupt:
        print("Stopped. Saved rows are retained; run the same command to resume.", flush=True)
        raise SystemExit(130)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
