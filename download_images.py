#!/usr/bin/env python3
"""
Download images from the image_url column of multimodal_test_public.tsv
Usage: python3 download_images.py [--tsv PATH] [--out DIR] [--workers N]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# ── Config defaults ────────────────────────────────────────────────────────────
DEFAULT_TSV     = "multimodal_test_public.tsv"
DEFAULT_OUT_DIR = "downloaded_images"
DEFAULT_WORKERS = 8
TIMEOUT         = 15          # seconds per request
MAX_RETRIES     = 3
RETRY_BACKOFF   = 2           # seconds between retries
# ──────────────────────────────────────────────────────────────────────────────


def get_extension(url: str) -> str:
    """Extract file extension from URL, defaulting to .jpg."""
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.split("?")[0].lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"} else ".jpg"


def sanitize_filename(url: str, row_id: str) -> str:
    """Return filename as <row_id><ext>."""
    ext = get_extension(url)
    return f"{row_id}{ext}"


def download_one(url: str, dest: Path) -> tuple[str, bool, str]:
    """Download a single URL to dest. Returns (url, success, message)."""
    if dest.exists():
        return url, True, f"skipped (already exists): {dest.name}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return url, True, f"saved: {dest.name}"
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                return url, False, f"failed after {MAX_RETRIES} attempts: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Download images from TSV")
    parser.add_argument("--tsv",     default=DEFAULT_TSV,     help="Path to the .tsv file")
    parser.add_argument("--out",     default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers")
    args, _ = parser.parse_known_args()

    # ── Load TSV ───────────────────────────────────────────────────────────────
    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        sys.exit(f"[ERROR] TSV not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    if "image_url" not in df.columns:
        sys.exit(f"[ERROR] No 'image_url' column found. Columns: {df.columns.tolist()}")

    # Keep only rows that have an image URL
    df = df[df["image_url"].notna() & (df["image_url"].str.strip() != "")]
    total = len(df)
    print(f"Found {total} rows with image URLs.")

    # ── Prepare output dir ─────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build work list ────────────────────────────────────────────────────────
    id_col = "id" if "id" in df.columns else df.columns[0]
    tasks = []
    for _, row in df.iterrows():
        url    = row["image_url"].strip()
        row_id = str(row[id_col]).strip()
        fname  = sanitize_filename(url, row_id)
        dest   = out_dir / fname
        tasks.append((url, dest))

    # ── Download ───────────────────────────────────────────────────────────────
    success_count = 0
    fail_count    = 0
    failed_urls   = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, url, dest): url for url, dest in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            url, ok, msg = future.result()
            status = "✓" if ok else "✗"
            print(f"[{i:>{len(str(total))}}/{total}] {status} {msg}")
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_urls.append(url)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done. Success: {success_count}  |  Failed: {fail_count}")
    print(f"Images saved to: {out_dir.resolve()}")

    if failed_urls:
        failed_log = out_dir / "failed_urls.txt"
        failed_log.write_text("\n".join(failed_urls))
        print(f"Failed URLs logged to: {failed_log}")


if __name__ == "__main__":
    main()
