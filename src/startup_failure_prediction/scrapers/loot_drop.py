"""Skeleton scraper for loot-drop.io failed-company database.

Not yet implemented. The site loads its database via JavaScript, so HTML
scraping is empty. Two viable paths:

1. Reverse-engineer the XHR/JSON request that the database view fires when
   the page loads (look in DevTools Network tab). Replay it with `requests`,
   paginate, and persist rows.
2. Drive a headless browser (Playwright recommended) and intercept the
   network request OR scrape the rendered DOM.

The output schema must match `data/companies_raw.csv` and `data/funding_events.csv`
so the snapshot pipeline can ingest it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path


LOOT_DROP_DATABASE_URL = "https://www.loot-drop.io/database-view"
LOOT_DROP_REBUILDS_URL = "https://www.loot-drop.io/rebuilds"


@dataclass
class LootDropRecord:
    company_id: str
    company_name: str
    industry: str
    product_type: str
    country: str
    founded_date: str
    outcome: str
    outcome_date: str
    last_observed_date: str
    market_score: int
    scalability_score: int
    company_description: str
    founder_statement: str
    failure_reason: str


def fetch_records() -> list[LootDropRecord]:
    """Fetch all failed-company records from loot-drop.

    TODO: implement one of the two strategies described in the module docstring.
    """
    raise NotImplementedError(
        "loot-drop scraper not implemented. The page is JS-rendered; either "
        "intercept the XHR JSON endpoint, or drive Playwright. See module "
        "docstring."
    )


def write_records(
    records: list[LootDropRecord],
    output_companies_path: str | Path,
    output_events_path: str | Path,
) -> None:
    output_companies_path = Path(output_companies_path)
    output_companies_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with output_companies_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    # Funding events are not provided directly by loot-drop, but the schema
    # column still needs to exist for the snapshot loader.
    output_events_path = Path(output_events_path)
    with output_events_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["company_id", "round_date", "round_name", "amount_usd"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape loot-drop.io failed companies.")
    parser.add_argument("--companies-out", default="data/loot_drop_companies.csv")
    parser.add_argument("--events-out", default="data/loot_drop_events.csv")
    args = parser.parse_args()
    records = fetch_records()
    write_records(records, args.companies_out, args.events_out)
    print(f"wrote {len(records)} records to {args.companies_out}")


if __name__ == "__main__":
    main()
