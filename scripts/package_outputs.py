#!/usr/bin/env python
"""Package the (actual) output directory into a ZIP file."""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run without installing

from tcav_imagenette.config import add_common_args, config_from_args


def package_directory(out_dir: Path, zip_path: Path, exclude_checkpoints: bool = False) -> int:
    """Zip every file of ``out_dir`` under a top-level folder named like the directory.
    Returns the number of files written; raises if the archive fails its integrity test."""
    out_dir, zip_path = Path(out_dir), Path(zip_path)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"output directory not found: {out_dir}")
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.resolve() != zip_path.resolve())
    if exclude_checkpoints:
        files = [p for p in files if p.suffix != ".pt"]
    if not files:
        raise RuntimeError(f"nothing to package in {out_dir}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, Path(out_dir.name) / p.relative_to(out_dir))
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP integrity check failed at {bad}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--zip-path", default=None, help="output ZIP (default: <out-dir>.zip next to the directory)")
    parser.add_argument("--exclude-checkpoints", action="store_true", help="leave out *.pt files (model, features)")
    args = parser.parse_args()

    cfg = config_from_args(args)
    zip_path = Path(args.zip_path) if args.zip_path else cfg.out_path.resolve().with_suffix(".zip")
    n = package_directory(cfg.out_path, zip_path, args.exclude_checkpoints)
    print(f"wrote {n} files to {zip_path}")


if __name__ == "__main__":
    main()
