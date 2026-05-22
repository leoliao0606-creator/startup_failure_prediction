"""Scraper skeletons for real failed and surviving company data.

These are stubs. Real implementations are blocked on:
- loot-drop.io serving content via JS-loaded data; needs a headless browser or
  reverse-engineered XHR/JSON endpoint.
- A successful-company source (Crunchbase, PitchBook, Tracxn) needs an API key.

Each scraper writes into the same schema consumed by `snapshots.load_companies`:
- data/companies_raw.csv
- data/funding_events.csv
"""
