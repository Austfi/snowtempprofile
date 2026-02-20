#!/usr/bin/env python3
"""
Download and cache a USGS window locally for fast/reproducible reruns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache USGS dataset locally.")
    parser.add_argument(
        "--station",
        default="senator_beck",
        choices=["senator_beck", "independence_pass", "berthoud_pass"],
        help="Station key from usgs_collector.STATIONS",
    )
    parser.add_argument("--start", default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force API fetch even if cache exists.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/usgs_cache",
        help="Cache directory for raw cached files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    notebooks_dir = repo_root / "notebooks"
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))

    from usgs_collector import STATIONS, fetch_usgs_iv_cached, simplify_columns  # noqa: E402

    site = STATIONS[args.station]
    cache_dir = (repo_root / args.cache_dir).resolve()

    df_raw = fetch_usgs_iv_cached(
        site=site,
        start_date=args.start,
        end_date=args.end,
        cache_dir=cache_dir,
        refresh=args.refresh,
    )
    if df_raw.empty:
        print("No data returned.")
        return 1

    # Save one stable simplified file for notebook use
    df_simple = simplify_columns(df_raw)
    out_simple = cache_dir / f"{args.station}_{args.start}_{args.end}_simplified.csv"
    df_simple.to_csv(out_simple)

    print("\nCache complete")
    print("=" * 70)
    print(f"station key : {args.station}")
    print(f"site id     : {site}")
    print(f"rows raw    : {len(df_raw)}")
    print(f"rows simple : {len(df_simple)}")
    print(f"time min    : {df_simple.index.min()}")
    print(f"time max    : {df_simple.index.max()}")
    print(f"simplified  : {out_simple}")
    print(f"columns     : {df_simple.columns.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
