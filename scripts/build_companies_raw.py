"""Generate synthetic company-level seed data with proper time fields.

This is a stop-gap dataset that demonstrates the snapshot pipeline. Real
training requires scraped failed companies (loot-drop.io) plus a balanced
set of surviving companies (Crunchbase, etc.) — see
`src/startup_failure_prediction/scrapers/`.

Each company gets:
- founded_date (with month for sub-year precision)
- outcome (failed | operating) + outcome_date for failed, last_observed_date for survivors
- a plausible funding history with dated rounds

Funding history is generated deterministically from total funding and round
count so the CSV is reproducible from this script.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

REFERENCE_TODAY = date(2026, 5, 21)
OUTPUT_COMPANIES = Path("data/companies_raw.csv")
OUTPUT_EVENTS = Path("data/funding_events.csv")


@dataclass(frozen=True)
class CompanySeed:
    company_id: str
    name: str
    industry: str
    product_type: str
    country: str
    founded_year: int
    founded_month: int
    market_score: int
    scalability_score: int
    company_description: str
    founder_statement: str
    funding_total_usd: int
    funding_rounds: int
    outcome: str
    operating_years: int | None
    failure_reason: str = ""


SEEDS: list[CompanySeed] = [
    # ---- failed companies (existing 15) ----
    CompanySeed("co_flashcart", "FlashCart Market", "Ecommerce", "Marketplace", "USA", 2019, 3, 42, 55,
                "Same day grocery marketplace with heavy subsidies and weak repeat demand",
                "Growth depended on discounts before retention was proven",
                42_000_000, 4, "failed", 4, "High burn and weak unit economics"),
    CompanySeed("co_tokennest", "TokenNest", "Crypto", "Infrastructure", "Singapore", 2020, 5, 38, 62,
                "Crypto wallet infrastructure for speculative apps with volatile demand",
                "The team chased protocol partnerships before revenue was stable",
                18_000_000, 3, "failed", 3, "Regulatory pressure and shrinking developer demand"),
    CompanySeed("co_medroute", "MedRoute AI", "Healthcare", "SaaS", "USA", 2018, 2, 45, 58,
                "AI care routing platform with long hospital sales cycles and unclear reimbursement",
                "Hospitals liked pilots but procurement rarely converted",
                9_000_000, 2, "failed", 5, "Sales cycles exceeded runway"),
    CompanySeed("co_droneshelf", "DroneShelf", "Robotics", "Hardware", "Germany", 2017, 4, 40, 48,
                "Autonomous retail inventory drones requiring custom hardware and store retrofits",
                "Every deployment needed custom work from the engineering team",
                26_000_000, 3, "failed", 5, "Hardware cost and integration risk were too high"),
    CompanySeed("co_eduspark", "EduSpark Live", "Edtech", "Consumer App", "India", 2021, 1, 44, 60,
                "Live tutoring marketplace in a crowded test prep market with low switching cost",
                "Discounting brought users but paid retention stayed weak",
                5_000_000, 2, "failed", 3, "Customer acquisition cost exceeded lifetime value"),
    CompanySeed("co_metatown", "MetaTown Work", "Collaboration", "Virtual Platform", "USA", 2020, 7, 35, 70,
                "Virtual office platform tied to novelty behavior and low daily utility",
                "Usage was interesting during demos but not part of daily work",
                30_000_000, 3, "failed", 3, "Demand faded after the initial launch wave"),
    CompanySeed("co_greenscoot", "GreenScoot", "Mobility", "Hardware", "France", 2018, 6, 41, 50,
                "Shared scooter fleet with high repair cost and city permit exposure",
                "The model required dense utilization before city rules changed",
                22_000_000, 4, "failed", 4, "Operating cost and regulation compressed margins"),
    CompanySeed("co_adsignal", "AdSignal Loop", "Martech", "SaaS", "UK", 2019, 4, 47, 64,
                "Attribution platform exposed to privacy changes and crowded ad tooling",
                "Customers questioned accuracy as tracking data disappeared",
                7_500_000, 2, "failed", 4, "Privacy changes reduced signal quality"),
    CompanySeed("co_mealkit", "MealKit Cloud", "Foodtech", "Subscription", "USA", 2017, 5, 39, 52,
                "Subscription meal kits with complex logistics high spoilage and frequent churn",
                "The brand grew quickly but fulfillment quality varied by region",
                14_000_000, 3, "failed", 5, "Logistics cost and churn prevented scale"),
    CompanySeed("co_indiebank", "IndieBank", "Fintech", "Consumer App", "Brazil", 2019, 2, 43, 66,
                "Neobank for freelancers with high customer acquisition cost and thin interchange margins",
                "The product was useful but distribution was more expensive than expected",
                12_000_000, 2, "failed", 4, "Revenue per customer was too low"),
    CompanySeed("co_fitmirror", "FitMirror Home", "Fitness", "Hardware", "USA", 2018, 1, 46, 54,
                "Connected fitness mirror with expensive hardware and demand pulled forward by lockdowns",
                "Hardware margin left little room for content investment",
                55_000_000, 4, "failed", 5, "Inventory and support costs outgrew revenue"),
    CompanySeed("co_legalbot", "LegalBot Pro", "Legaltech", "SaaS", "Canada", 2019, 8, 49, 57,
                "SMB legal automation tool in regulated workflows where trust was hard to earn",
                "Customers wanted attorney review for the highest value tasks",
                6_000_000, 2, "failed", 4, "Low trust slowed adoption"),
    CompanySeed("co_quickhire", "QuickHire VR", "HR Tech", "SaaS", "USA", 2020, 3, 43, 61,
                "VR interviewing product launched before customers had budget for new hiring hardware",
                "The workflow created extra setup for recruiters",
                8_500_000, 2, "failed", 3, "Market timing and adoption friction were poor"),
    CompanySeed("co_nanofarm", "NanoFarm Boxes", "Agtech", "Hardware", "Netherlands", 2018, 4, 37, 49,
                "Indoor micro farm appliance with high capex uncertain yields and maintenance needs",
                "The system worked technically but payback periods were too long",
                17_000_000, 3, "failed", 4, "Unit economics never reached target"),
    CompanySeed("co_socialpulse", "SocialPulse", "Social", "Consumer App", "USA", 2021, 6, 34, 68,
                "Consumer social app with celebrity launch weak retention and high moderation cost",
                "The audience arrived in bursts but did not build a durable habit",
                11_000_000, 3, "failed", 2, "Retention fell after launch attention faded"),

    # ---- additional failed companies for diversity ----
    CompanySeed("co_brickbuy", "BrickBuy", "Proptech", "Marketplace", "USA", 2015, 5, 36, 51,
                "Instant home offer marketplace with capital intensive inventory and exposure to housing cycles",
                "We bought homes faster than we could resell when rates moved",
                90_000_000, 5, "failed", 7, "Inventory exposure during a rate shock"),
    CompanySeed("co_bytebrew", "ByteBrew Studio", "Gaming", "Consumer App", "Japan", 2016, 9, 38, 58,
                "Mobile game studio dependent on a single hit title with weak follow-up monetization",
                "Our second title never matched the first launch",
                14_000_000, 3, "failed", 6, "Single title dependency and rising user acquisition costs"),
    CompanySeed("co_clinicpod", "ClinicPod", "Healthcare", "Hardware", "USA", 2014, 2, 33, 47,
                "Telehealth kiosk hardware deployed in pharmacies with low utilization and high service costs",
                "Each kiosk needed a technician within two days when something broke",
                70_000_000, 5, "failed", 7, "Utilization stayed below the breakeven line"),
    CompanySeed("co_groovenote", "GrooveNote", "Music", "Consumer App", "Sweden", 2017, 11, 42, 64,
                "Social music creation app reliant on viral loops with weak creator monetization",
                "Creators came for the trend but left for established tools",
                9_500_000, 2, "failed", 5, "Creator retention never reached durable levels"),
    CompanySeed("co_routemind", "RouteMind", "Logistics", "SaaS", "USA", 2013, 4, 44, 56,
                "Last mile routing software for small carriers competing with free tools from large platforms",
                "Customers told us the free version was good enough",
                12_000_000, 3, "failed", 8, "Crowded out by free tooling from large platforms"),
    CompanySeed("co_pixelpath", "PixelPath AR", "AR/VR", "Consumer App", "USA", 2019, 9, 37, 67,
                "AR shopping overlay reliant on developer adoption and expensive 3D content creation",
                "Brands liked demos but every deployment needed a custom 3D team",
                21_000_000, 3, "failed", 4, "3D content production cost killed margins"),
    CompanySeed("co_yieldyard", "YieldYard", "Climate", "Hardware", "USA", 2016, 5, 39, 49,
                "Distributed solar microgrid installer with regulatory friction and long permitting timelines",
                "Permits varied so much between counties that growth was uneven",
                33_000_000, 4, "failed", 6, "Regulatory variation slowed expansion"),
    CompanySeed("co_swiftbrief", "SwiftBrief", "Legaltech", "Consumer App", "USA", 2020, 10, 41, 60,
                "Consumer legal Q&A app pivoting between subscription and pay-per-use without retention",
                "We changed the pricing four times in two years",
                4_500_000, 2, "failed", 3, "Pricing model never settled"),
    CompanySeed("co_caretrack", "CareTrack Wear", "Healthcare", "Hardware", "USA", 2014, 6, 40, 52,
                "Wearable for chronic disease monitoring with reimbursement uncertainty and clinical evidence gaps",
                "Payers wanted larger studies than our runway allowed",
                48_000_000, 4, "failed", 8, "Reimbursement evidence cycle outran capital"),
    CompanySeed("co_finlearn", "FinLearn", "Edtech", "Consumer App", "USA", 2020, 4, 36, 65,
                "Personal finance learning app with cyclic engagement and weak monetization beyond freemium",
                "Engagement spiked during tax season then disappeared",
                7_000_000, 2, "failed", 3, "Engagement was seasonal and monetization was weak"),

    # ---- operating companies (existing 15) ----
    CompanySeed("co_ledgerflow", "LedgerFlow", "Fintech", "SaaS", "USA", 2018, 3, 82, 86,
                "Revenue automation platform with clear buyer ROI recurring usage and strong retention",
                "Customers expanded usage after the first finance workflow",
                15_000_000, 3, "operating", None),
    CompanySeed("co_clinicbridge", "ClinicBridge", "Healthcare", "SaaS", "Canada", 2017, 5, 78, 82,
                "Patient coordination software for clinics with measurable admin savings and low churn",
                "The product entered through one department and expanded across sites",
                10_000_000, 2, "operating", None),
    CompanySeed("co_shopstack", "ShopStack", "Ecommerce", "SaaS", "USA", 2016, 4, 79, 88,
                "Composable checkout tools for merchants with recurring revenue and partner channels",
                "Merchant integrations made the product sticky after onboarding",
                30_000_000, 4, "operating", None),
    CompanySeed("co_freightgrid", "FreightGrid", "Logistics", "SaaS", "Germany", 2017, 2, 81, 84,
                "Freight planning software that reduces empty miles and integrates with existing systems",
                "Operators adopted it because savings showed up in weekly routing",
                24_000_000, 3, "operating", None),
    CompanySeed("co_codeharbor", "CodeHarbor", "Devtools", "SaaS", "USA", 2019, 6, 76, 90,
                "Developer workflow platform with bottom up adoption and strong team expansion",
                "Individual developers started free and teams converted for governance",
                12_000_000, 2, "operating", None),
    CompanySeed("co_datacove", "DataCove", "Security", "SaaS", "UK", 2018, 9, 84, 87,
                "Cloud data security platform with urgent compliance demand and enterprise renewals",
                "Security teams had budget and a clear audit deadline",
                20_000_000, 3, "operating", None),
    CompanySeed("co_payatlas", "PayAtlas", "Fintech", "Infrastructure", "Singapore", 2017, 1, 80, 89,
                "Payment infrastructure API with high transaction volume and durable platform integrations",
                "Once merchants integrated the API churn was very low",
                28_000_000, 4, "operating", None),
    CompanySeed("co_farmops", "FarmOps AI", "Agtech", "SaaS", "USA", 2020, 3, 74, 80,
                "Farm operations analytics sold to cooperatives with seasonal ROI and software delivery",
                "The team focused on crop planning workflows with repeat annual budgets",
                9_000_000, 2, "operating", None),
    CompanySeed("co_learnloop", "LearnLoop", "Edtech", "SaaS", "India", 2019, 7, 73, 82,
                "Learning management software for schools with administrative reporting and renewal contracts",
                "Schools renewed because reporting reduced manual work for teachers",
                7_000_000, 2, "operating", None),
    CompanySeed("co_carbonledger", "CarbonLedger", "Climate", "SaaS", "Denmark", 2018, 11, 83, 85,
                "Carbon accounting software driven by regulation procurement requirements and audit trails",
                "Compliance pressure created budget and a clear renewal event",
                16_000_000, 3, "operating", None),
    CompanySeed("co_talentmap", "TalentMap", "HR Tech", "SaaS", "Netherlands", 2019, 4, 75, 83,
                "Workforce planning software for mid market companies with repeat budget cycles",
                "Customers used the planning model every quarter",
                8_000_000, 2, "operating", None),
    CompanySeed("co_homecare", "HomeCare Hub", "Healthcare", "Marketplace", "USA", 2018, 5, 77, 78,
                "Home care coordination marketplace with vetted supply insurance workflows and repeat demand",
                "The team limited geography until supply quality was reliable",
                13_000_000, 3, "operating", None),
    CompanySeed("co_warehouseiq", "WarehouseIQ", "Robotics", "SaaS", "Germany", 2017, 8, 78, 81,
                "Warehouse optimization software for robots and workers with limited hardware exposure",
                "The product improved throughput without forcing customers into new equipment",
                19_000_000, 3, "operating", None),
    CompanySeed("co_privacykit", "PrivacyKit", "Martech", "SaaS", "Canada", 2020, 2, 76, 84,
                "Privacy first analytics platform aligned with consent rules and first party data strategies",
                "Customers bought it because privacy changes made old tooling risky",
                9_000_000, 2, "operating", None),
    CompanySeed("co_teamcanvas", "TeamCanvas", "Collaboration", "SaaS", "Australia", 2020, 6, 72, 81,
                "Team planning workspace with recurring rituals templates and strong weekly active usage",
                "Usage attached to planning meetings that already happened every week",
                6_000_000, 2, "operating", None),

    # ---- additional operating companies for diversity ----
    CompanySeed("co_routesage", "RouteSage", "Logistics", "SaaS", "USA", 2013, 6, 84, 86,
                "Long haul carrier dispatch software with measurable fuel savings and multi year contracts",
                "Once dispatchers adopted the screens contracts renewed for years",
                40_000_000, 5, "operating", None),
    CompanySeed("co_clinicrouter", "ClinicRouter", "Healthcare", "Infrastructure", "USA", 2014, 4, 79, 83,
                "Healthcare data routing infrastructure with HIPAA controls and clear integration points",
                "Hospitals chose it because it slotted into existing EHR plumbing",
                32_000_000, 4, "operating", None),
    CompanySeed("co_payanchor", "PayAnchor", "Fintech", "Infrastructure", "USA", 2015, 9, 82, 88,
                "Reconciliation infrastructure for marketplaces with high transaction volume and strong defaults",
                "Once a marketplace integrated reconciliation they did not rebuild it",
                55_000_000, 5, "operating", None),
    CompanySeed("co_orbitlearn", "OrbitLearn", "Edtech", "SaaS", "UK", 2016, 3, 75, 84,
                "Workforce training platform sold to enterprise L&D teams with annual procurement cycles",
                "L&D teams renewed because compliance modules mapped to their audits",
                25_000_000, 4, "operating", None),
    CompanySeed("co_sentrymesh", "SentryMesh", "Security", "Infrastructure", "Israel", 2015, 10, 86, 88,
                "Cloud workload security platform aligned with compliance frameworks and SOC operations",
                "Customers ran us inside their SOC runbooks within the first quarter",
                45_000_000, 5, "operating", None),
    CompanySeed("co_devsignal", "DevSignal", "Devtools", "SaaS", "USA", 2017, 11, 78, 89,
                "Engineering productivity analytics with bottom up adoption inside platform teams",
                "Platform teams used the dashboards in their weekly reviews",
                18_000_000, 3, "operating", None),
    CompanySeed("co_smbloan", "SMBLoan Direct", "Fintech", "Consumer App", "USA", 2014, 7, 73, 79,
                "SMB working capital lender with proprietary underwriting and stable repeat borrowers",
                "Repeat borrowers were the majority of originations by year three",
                60_000_000, 6, "operating", None),
    CompanySeed("co_seedbed", "Seedbed Agtech", "Agtech", "SaaS", "Brazil", 2018, 8, 74, 80,
                "Crop planning software for large grain farms with measurable yield improvement and renewal cycles",
                "Farms renewed once we showed yield uplift over a full season",
                11_000_000, 3, "operating", None),
    CompanySeed("co_civicgrid", "CivicGrid", "Climate", "SaaS", "Germany", 2019, 5, 77, 82,
                "Grid optimization software for municipal utilities with regulatory tailwinds and procurement cycles",
                "Utilities adopted us because regulators required the reporting we already built",
                14_000_000, 3, "operating", None),
    CompanySeed("co_routinely", "Routinely", "HR Tech", "SaaS", "Canada", 2019, 9, 72, 82,
                "Manager workflow software with weekly active usage and durable expansion across teams",
                "Managers used it during one on ones and the habit stuck",
                7_500_000, 2, "operating", None),
]


def funding_share(round_index: int, total_rounds: int) -> float:
    if total_rounds <= 0:
        return 0.0
    weights = [(i + 1) ** 1.6 for i in range(total_rounds)]
    return weights[round_index] / sum(weights)


def round_name(round_index: int, total_rounds: int) -> str:
    standard = ["Seed", "Series A", "Series B", "Series C", "Series D", "Series E", "Series F"]
    if round_index < len(standard):
        return standard[round_index]
    return f"Series {chr(ord('A') + round_index - 1)}"


def funding_events_for(seed: CompanySeed, founded_date: date, end_date: date) -> list[dict]:
    events: list[dict] = []
    if seed.funding_rounds <= 0 or seed.funding_total_usd <= 0:
        return events

    horizon_days = max((end_date - founded_date).days - 60, 365)
    spacing_days = max(horizon_days // max(seed.funding_rounds, 1), 90)

    for i in range(seed.funding_rounds):
        offset_days = 150 + i * spacing_days
        round_date = founded_date + timedelta(days=offset_days)
        if round_date > end_date - timedelta(days=30):
            round_date = end_date - timedelta(days=30 + (seed.funding_rounds - i - 1) * 60)
        amount = int(round(seed.funding_total_usd * funding_share(i, seed.funding_rounds)))
        events.append(
            {
                "company_id": seed.company_id,
                "round_date": round_date.isoformat(),
                "round_name": round_name(i, seed.funding_rounds),
                "amount_usd": amount,
            }
        )
    return events


def main() -> None:
    OUTPUT_COMPANIES.parent.mkdir(parents=True, exist_ok=True)

    company_rows: list[dict] = []
    event_rows: list[dict] = []

    for seed in SEEDS:
        founded_date = date(seed.founded_year, seed.founded_month, 15)
        if seed.outcome == "failed":
            assert seed.operating_years is not None
            outcome_date = founded_date + timedelta(days=seed.operating_years * 365 + 60)
            last_observed_date = outcome_date
        else:
            outcome_date = None
            last_observed_date = REFERENCE_TODAY

        end_date_for_funding = outcome_date if outcome_date else REFERENCE_TODAY
        events = funding_events_for(seed, founded_date, end_date_for_funding)
        event_rows.extend(events)

        company_rows.append(
            {
                "company_id": seed.company_id,
                "company_name": seed.name,
                "industry": seed.industry,
                "product_type": seed.product_type,
                "country": seed.country,
                "founded_date": founded_date.isoformat(),
                "outcome": seed.outcome,
                "outcome_date": outcome_date.isoformat() if outcome_date else "",
                "last_observed_date": last_observed_date.isoformat(),
                "market_score": seed.market_score,
                "scalability_score": seed.scalability_score,
                "company_description": seed.company_description,
                "founder_statement": seed.founder_statement,
                "failure_reason": seed.failure_reason,
            }
        )

    company_fields = list(company_rows[0].keys())
    with OUTPUT_COMPANIES.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=company_fields)
        writer.writeheader()
        writer.writerows(company_rows)

    event_fields = ["company_id", "round_date", "round_name", "amount_usd"]
    with OUTPUT_EVENTS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(event_rows)

    print(f"wrote {len(company_rows)} companies to {OUTPUT_COMPANIES}")
    print(f"wrote {len(event_rows)} funding events to {OUTPUT_EVENTS}")
    print(json.dumps({"failed": sum(1 for c in company_rows if c["outcome"] == "failed"),
                      "operating": sum(1 for c in company_rows if c["outcome"] == "operating")}, indent=2))


if __name__ == "__main__":
    main()
