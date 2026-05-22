"""Skeleton importer for surviving / successful companies.

Not yet implemented. Recommended sources (pick one and stick with it for
reproducibility):

- Crunchbase API (paid). Filter to companies still listed as `operating` with
  founded_date older than `today - PREDICTION_HORIZON_YEARS` so they qualify
  as label=0.
- PitchBook / Tracxn (paid).
- A public snapshot CSV (e.g. ProductHunt scrapers, GitHub awesome lists) for
  bootstrapping; lower quality but free.

Whichever source: produce rows that match `data/companies_raw.csv` schema with
`outcome = operating` and `last_observed_date = today`. Funding history goes
into `data/funding_events.csv`.

The class imbalance matters: if your survivor pool is much larger than the
failed pool, downsample at training time (or use class_weight, which we
already do in `train.py`).
"""

from __future__ import annotations

import argparse


def fetch_survivors() -> list[dict]:
    raise NotImplementedError(
        "Surviving-company importer not implemented. Pick a source (Crunchbase "
        "API recommended) and emit rows matching the companies_raw.csv schema."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import surviving companies.")
    parser.add_argument("--companies-out", default="data/survivor_companies.csv")
    parser.add_argument("--events-out", default="data/survivor_events.csv")
    args = parser.parse_args()
    rows = fetch_survivors()
    print(f"would write {len(rows)} survivor rows to {args.companies_out} and events to {args.events_out}")


if __name__ == "__main__":
    main()
