#!/usr/bin/env python3
"""
Minimal strict downloader: given a required data_source, data_path and climatology selection
(download grid, area and the chosen climatology). Retries on failure. Exits non‑zero if any
required file cannot be downloaded or is missing in the configuration.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
from typing import Optional, Tuple
import ast

import requests

from conf.config import CMEMS_datasets_options


def download_with_retries(url: str, outpath: str, retries: int = 3, delay: int = 5) -> bool:
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"[{attempt}/{retries}] Downloading {url} -> {outpath}")
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(outpath, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
            print(f"Saved: {outpath}")
            return True
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries:
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    print(f"Failed to download after {retries} attempts: {url}")
    return False


def resolve_climatology(meta: dict, clim_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given dataset meta and a required clim_key, return (filename, url).
    If config.clim_file is a dict, clim_key must be one of its keys.
    If config.clim_file is a string/list, clim_key must equal the filename and config must provide top-level clim_url.
    """
    entry = meta.get("clim_file")
    if entry is None:
        return None, None

    # dict: require clim_key present
    if isinstance(entry, dict):
        if clim_key not in entry:
            return None, None
        info = entry[clim_key]
        if not isinstance(info, dict):
            return None, None
        fname = info.get("file")
        url = info.get("url") or info.get("ur")
        return fname, url

    # list/tuple or string: treat clim_key as filename and use top-level clim_url
    if isinstance(entry, (list, tuple)):
        # check provided key matches one of entries (or matches first item)
        items = [str(i) for i in entry]
        if clim_key not in items and clim_key != items[0]:
            # allow using first item if user provided that name but not in list
            return None, None
        fname = clim_key
    elif isinstance(entry, str):
        if clim_key != entry:
            return None, None
        fname = clim_key
    else:
        return None, None

    url = meta.get("clim_url")
    return fname, url


def main():
    p = argparse.ArgumentParser(description="Download grid, area and climatology for dataset (strict).")
    p.add_argument("data_source", help="Dataset id (key in CMEMS_datasets_options)")
    p.add_argument("data_path", help="Directory to save files")
    p.add_argument("climatology", help="Climatology selection key (required)")
    p.add_argument("--retries", type=int, default=3, help="Download retries")
    p.add_argument("--delay", type=int, default=5, help="Delay between retries (s)")
    args = p.parse_args()

    # ds = ast.literal_eval(args.data_source)[0]
    ds = args.data_source
    print(type(ds), ds)
    return

    data_path = args.data_path
    clim_key = args.climatology

    if ds not in CMEMS_datasets_options:
        print(f"ERROR: dataset '{ds}' not found in config", file=sys.stderr)
        sys.exit(2)

    meta = CMEMS_datasets_options[ds]

    # prepare required tasks (grid, area, climatology)
    tasks = []

    grid_url = meta.get("grid_url")
    grid_file = meta.get("grid_file")
    if not grid_url or not grid_file:
        print("ERROR: grid_url or grid_file missing in config for dataset", file=sys.stderr)
        sys.exit(3)
    tasks.append((grid_url, os.path.join(data_path, str(grid_file))))

    area_url = meta.get("area_url")
    area_file = meta.get("area_file")
    if not area_url or not area_file:
        print("ERROR: area_url or area_file missing in config for dataset", file=sys.stderr)
        sys.exit(4)
    tasks.append((area_url, os.path.join(data_path, str(area_file))))

    clim_fname, clim_url = resolve_climatology(meta, clim_key)
    if not clim_fname or not clim_url:
        print(f"ERROR: climatology '{clim_key}' not found or missing url/file in config for dataset", file=sys.stderr)
        print("Available clim entries:", list(meta.get("clim_file").keys()) if isinstance(meta.get("clim_file"), dict) else meta.get("clim_file"))
        sys.exit(5)
    tasks.append((clim_url, os.path.join(data_path, str(clim_fname))))

    # download missing files; require success for all
    any_failed = False
    for url, outpath in tasks:
        if os.path.exists(outpath):
            print(f"Exists: {outpath} (skipping)")
            continue
        ok = download_with_retries(url, outpath, retries=args.retries, delay=args.delay)
        if not ok:
            any_failed = True

    if any_failed:
        print("One or more downloads failed.", file=sys.stderr)
        sys.exit(6)

    print("All required files saved or already present.")
    sys.exit(0)


if __name__ == "__main__":
    main()