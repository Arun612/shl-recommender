"""
SHL Catalog Scraper
-------------------
Run this locally (not in sandbox) to generate data/catalog.json.
Usage: python scrape_catalog.py

Scrapes Individual Test Solutions (type=1) from:
https://www.shl.com/solutions/products/product-catalog/
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://www.shl.com"
CATALOG_URL = (
    "https://www.shl.com/solutions/products/product-catalog/"
    "?start={start}&type=1&action_doFilteringForm=Search"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PAGE_SIZE = 12  # SHL returns 12 items per page


def get_page(start: int, session: requests.Session) -> BeautifulSoup:
    url = CATALOG_URL.format(start=start)
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    """Extract assessment stubs from a catalog listing page."""
    items = []
    # SHL catalog rows: each assessment is in a <tr> or a card element
    # Adjust selector after inspecting actual HTML
    rows = soup.select("tr.product-catalogue__row")
    if not rows:
        # fallback: any link inside the product table
        rows = soup.select("table.custom-table tbody tr")

    for row in rows:
        # Name + URL
        link_tag = row.select_one("a")
        if not link_tag:
            continue
        name = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        url = BASE_URL + href if href.startswith("/") else href

        # Test type badges (K=Knowledge, P=Personality, A=Ability, B=Biodata, S=Simulation)
        badges = row.select("span.product-catalogue__key")
        test_types = [b.get_text(strip=True) for b in badges if b.get_text(strip=True)]

        # Remote / adaptive flags
        tds = row.find_all("td")
        remote_testing = False
        adaptive = False
        if len(tds) >= 4:
            # columns vary; scan for checkmark symbols
            for td in tds:
                txt = td.get_text(strip=True)
                if "●" in txt or "✓" in txt or td.find("span", class_="catalogue__circle"):
                    # heuristic: remote and adaptive columns
                    pass
            # More reliable: check data attributes or specific column positions
            # SHL typically has cols: Name | Remote | Adaptive | Test Types | Duration
            if len(tds) > 2:
                remote_testing = bool(tds[1].find("span", class_="catalogue__circle--green") or
                                      "yes" in tds[1].get_text(strip=True).lower())
                adaptive = bool(tds[2].find("span", class_="catalogue__circle--green") or
                                "yes" in tds[2].get_text(strip=True).lower())

        items.append({
            "name": name,
            "url": url,
            "test_types": test_types,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive,
            "description": "",   # filled in by detail scrape
            "duration": "",
            "languages": [],
        })

    return items


def scrape_detail(url: str, session: requests.Session) -> dict:
    """Scrape individual assessment detail page for description, duration, languages."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Description: first substantial paragraph
        desc = ""
        for tag in soup.select("div.product-hero__description p, .product__description p, p"):
            text = tag.get_text(strip=True)
            if len(text) > 60:
                desc = text
                break

        # Duration
        duration = ""
        dur_el = soup.find(string=re.compile(r"\d+\s*min", re.I))
        if dur_el:
            duration = dur_el.strip()

        # Languages — look for a languages list
        lang_section = soup.find(string=re.compile(r"language", re.I))
        languages = []
        if lang_section and lang_section.parent:
            lang_text = lang_section.parent.get_text(separator=", ", strip=True)
            # crude extraction
            languages = [l.strip() for l in lang_text.split(",") if len(l.strip()) > 1][:10]

        return {"description": desc, "duration": duration, "languages": languages}
    except Exception as e:
        print(f"  [warn] Could not fetch detail {url}: {e}")
        return {"description": "", "duration": "", "languages": []}


def total_items(soup: BeautifulSoup) -> int:
    """Try to extract total count from page."""
    el = soup.find(string=re.compile(r"\d+\s+result", re.I))
    if el:
        m = re.search(r"(\d+)", el)
        if m:
            return int(m.group(1))
    return 300  # safe upper bound


def main():
    out_path = Path("data/catalog.json")
    out_path.parent.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    print("Fetching page 0 to get total count...")
    soup0 = get_page(0, session)
    total = total_items(soup0)
    print(f"Total assessments reported: {total}")

    all_items: list[dict] = []
    start = 0
    while start < total:
        print(f"  Listing page start={start} ...")
        soup = get_page(start, session) if start > 0 else soup0
        items = parse_listing_page(soup)
        if not items:
            print("  No items found on this page — stopping.")
            break
        all_items.extend(items)
        start += PAGE_SIZE
        time.sleep(1.0)   # polite delay

    print(f"\nScrapped {len(all_items)} items. Now fetching detail pages...")

    for i, item in enumerate(all_items):
        print(f"  [{i+1}/{len(all_items)}] {item['name']}")
        detail = scrape_detail(item["url"], session)
        item.update(detail)
        time.sleep(0.8)

    out_path.write_text(json.dumps(all_items, indent=2, ensure_ascii=False))
    print(f"\nDone. Saved {len(all_items)} assessments to {out_path}")


if __name__ == "__main__":
    main()
