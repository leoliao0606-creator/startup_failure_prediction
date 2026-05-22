from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import (
    PREDICTION_HORIZON_YEARS,
    REFERENCE_TODAY,
    SNAPSHOT_DAYS_AFTER_FUNDING,
    SNAPSHOT_MIN_AGE_DAYS,
    SNAPSHOT_YEARLY_OFFSETS,
)


@dataclass(frozen=True)
class FundingEvent:
    round_date: date
    round_name: str
    amount_usd: float


@dataclass(frozen=True)
class Company:
    company_id: str
    company_name: str
    industry: str
    product_type: str
    country: str
    founded_date: date
    outcome: str
    outcome_date: date | None
    last_observed_date: date
    market_score: float
    scalability_score: float
    company_description: str
    founder_statement: str
    funding_events: tuple[FundingEvent, ...]


def _parse_date(value: object) -> date | None:
    if value is None or value == "" or pd.isna(value):
        return None
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def load_companies(
    companies_path: str | Path,
    funding_events_path: str | Path,
) -> list[Company]:
    companies_df = pd.read_csv(companies_path)
    events_df = pd.read_csv(funding_events_path)

    events_by_company: dict[str, list[FundingEvent]] = {}
    for record in events_df.to_dict(orient="records"):
        cid = str(record["company_id"])
        events_by_company.setdefault(cid, []).append(
            FundingEvent(
                round_date=_parse_date(record["round_date"]),
                round_name=str(record["round_name"]),
                amount_usd=float(record["amount_usd"]),
            )
        )
    for events in events_by_company.values():
        events.sort(key=lambda event: event.round_date)

    companies: list[Company] = []
    for record in companies_df.to_dict(orient="records"):
        cid = str(record["company_id"])
        companies.append(
            Company(
                company_id=cid,
                company_name=str(record["company_name"]),
                industry=str(record["industry"]),
                product_type=str(record["product_type"]),
                country=str(record["country"]),
                founded_date=_parse_date(record["founded_date"]),
                outcome=str(record["outcome"]),
                outcome_date=_parse_date(record.get("outcome_date")),
                last_observed_date=_parse_date(record["last_observed_date"]),
                market_score=float(record["market_score"]),
                scalability_score=float(record["scalability_score"]),
                company_description=str(record.get("company_description") or ""),
                founder_statement=str(record.get("founder_statement") or ""),
                funding_events=tuple(events_by_company.get(cid, [])),
            )
        )
    return companies


def _horizon_date(snapshot_date: date, horizon_years: int) -> date:
    return snapshot_date + timedelta(days=int(round(horizon_years * 365.25)))


def label_for_snapshot(
    company: Company,
    snapshot_date: date,
    horizon_years: int,
) -> tuple[int | None, bool, str]:
    """Return (label, censored, reason).

    label = 1 if failed within `horizon_years` after snapshot_date,
    label = 0 if confirmed alive at snapshot_date + horizon_years,
    label = None and censored=True if we cannot tell yet.
    """
    horizon = _horizon_date(snapshot_date, horizon_years)
    if company.outcome == "failed":
        if company.outcome_date is None:
            return None, True, "failed_without_date"
        if company.outcome_date <= horizon:
            return 1, False, "failed_within_horizon"
        return 0, False, "failed_after_horizon"

    if company.outcome == "operating":
        if company.last_observed_date >= horizon:
            return 0, False, "survived_through_horizon"
        return None, True, "operating_but_horizon_not_reached"

    return None, True, "unknown_outcome"


def candidate_snapshot_dates(company: Company) -> list[date]:
    candidates: set[date] = set()
    for offset_years in SNAPSHOT_YEARLY_OFFSETS:
        candidate = company.founded_date + timedelta(days=int(round(offset_years * 365.25)))
        if candidate >= company.founded_date + timedelta(days=SNAPSHOT_MIN_AGE_DAYS):
            candidates.add(candidate)

    for event in company.funding_events:
        candidate = event.round_date + timedelta(days=SNAPSHOT_DAYS_AFTER_FUNDING)
        if candidate >= company.founded_date + timedelta(days=SNAPSHOT_MIN_AGE_DAYS):
            candidates.add(candidate)

    end_date = company.outcome_date or company.last_observed_date
    if end_date is not None:
        candidates = {d for d in candidates if d < end_date}

    candidates = {d for d in candidates if d <= REFERENCE_TODAY}
    return sorted(candidates)


def features_at(company: Company, snapshot_date: date) -> dict[str, object]:
    age_days = (snapshot_date - company.founded_date).days
    age_years = age_days / 365.25
    funding_total = 0.0
    funding_rounds = 0
    last_round_days_ago: float | None = None
    for event in company.funding_events:
        if event.round_date <= snapshot_date:
            funding_total += event.amount_usd
            funding_rounds += 1
            last_round_days_ago = (snapshot_date - event.round_date).days

    return {
        "company_id": company.company_id,
        "snapshot_date": snapshot_date.isoformat(),
        "snapshot_year": snapshot_date.year,
        "age_years_at_snapshot": round(age_years, 3),
        "funding_total_usd_at_snapshot": funding_total,
        "funding_rounds_at_snapshot": funding_rounds,
        "days_since_last_round": (
            int(last_round_days_ago) if last_round_days_ago is not None else -1
        ),
        "industry": company.industry,
        "product_type": company.product_type,
        "country": company.country,
        "market_score": company.market_score,
        "scalability_score": company.scalability_score,
        "company_description": company.company_description,
        "founder_statement": company.founder_statement,
    }


def build_snapshots(
    companies: Iterable[Company],
    horizon_years: int = PREDICTION_HORIZON_YEARS,
    include_censored: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for company in companies:
        for snapshot_date in candidate_snapshot_dates(company):
            label, censored, reason = label_for_snapshot(
                company,
                snapshot_date,
                horizon_years,
            )
            if censored and not include_censored:
                continue
            row = features_at(company, snapshot_date)
            row["label"] = label if label is not None else -1
            row["censored"] = censored
            row["censor_reason"] = reason
            row["horizon_years"] = horizon_years
            row["company_name"] = company.company_name
            row["outcome"] = company.outcome
            row["outcome_date"] = (
                company.outcome_date.isoformat() if company.outcome_date else ""
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame.sort_values(["snapshot_date", "company_id"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def iter_snapshots(
    companies_path: str | Path,
    funding_events_path: str | Path,
    horizon_years: int = PREDICTION_HORIZON_YEARS,
    include_censored: bool = False,
) -> Iterator[dict[str, object]]:
    companies = load_companies(companies_path, funding_events_path)
    frame = build_snapshots(
        companies,
        horizon_years=horizon_years,
        include_censored=include_censored,
    )
    yield from frame.to_dict(orient="records")
