#!/usr/bin/env python3
"""
Scrape course pages from kurser.ku.dk.

Downloads HTML pages for physics/CS/math courses from KU's course catalog.
Uses the sitemap for discovery and filters by course code prefix.

Usage:
    python3 scrape_kurser.py                    # physics/CS/math only (~150 courses)
    python3 scrape_kurser.py --all              # all KU courses (~3400)
    python3 scrape_kurser.py --prefixes NFYK    # only physics
"""

import argparse
import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PAGES_DIR = DATA_DIR / "pages"

# Course code prefixes for SCIENCE faculty departments:
# 4th letter: A=general, B=bachelor, K=kandidat(master)
#   NFY = Physics (Niels Bohr Institute)
#   NDA = Computer Science (DIKU)
#   NMA = Mathematics
#   NFA = Applied Math & CS (cross-department)
DEFAULT_PREFIXES = (
    "NFYA", "NFYB", "NFYK",  # Physics: general, bachelor, master
    "NDAA", "NDAB", "NDAK",  # CS: general, bachelor, master
    "NMAA", "NMAB", "NMAK",  # Math: general, bachelor, master
    "NFAB",                   # Applied Math & CS
)

HEADERS = {
    "User-Agent": "KU-Course-Scraper/1.0 (educational-project)"
}


def get_all_course_urls():
    """Fetch all course URLs from KU's sitemap."""
    print("Fetching sitemap from kurser.ku.dk...")
    resp = requests.get("https://kurser.ku.dk/sitemap.xml", headers=HEADERS, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []
    for loc in root.findall(".//s:loc", ns):
        url = loc.text.strip()
        if "/course/" in url:
            # Normalize to https
            url = url.replace("http://", "https://")
            urls.append(url)

    print(f"Found {len(urls)} total course URLs in sitemap.")
    return urls


def filter_urls_by_prefix(urls, prefixes):
    """Filter course URLs to only matching code prefixes.

    Also deduplicates by course code, keeping the URL that sorts last
    (typically the latest academic year).
    """
    best = {}
    for url in urls:
        code = url.split("/course/")[-1].split("/")[0]
        if any(code.upper().startswith(p) for p in prefixes):
            if code not in best or url > best[code]:
                best[code] = url

    result = list(best.values())
    print(f"Filtered to {len(result)} courses matching prefixes: {prefixes}")
    return result


def download_page(url, pages_dir):
    """Download a single course page. Returns 'ok', 'skip', or 'fail'."""
    code = url.split("/course/")[-1].split("/")[0]
    filepath = pages_dir / f"{code}.html"

    if filepath.exists():
        return "skip"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        # Check for error pages (title contains "Error")
        if "<title>\n    Error\n</title>" in resp.text:
            print(f"  WARN {code}: error page returned, skipping")
            return "fail"

        filepath.write_text(resp.text, encoding="utf-8")
        return "ok"

    except Exception as e:
        print(f"  FAIL {code}: {e}")
        return "fail"


def main():
    parser = argparse.ArgumentParser(description="Scrape course pages from kurser.ku.dk")
    parser.add_argument("--all", action="store_true", help="Download ALL courses (not just physics/CS/math)")
    parser.add_argument("--prefixes", nargs="+", default=list(DEFAULT_PREFIXES),
                        help=f"Course code prefixes to include (default: {DEFAULT_PREFIXES})")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    args = parser.parse_args()

    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    urls = get_all_course_urls()

    if not args.all:
        urls = filter_urls_by_prefix(urls, tuple(p.upper() for p in args.prefixes))

    ok = skip = fail = 0
    total = len(urls)
    print(f"\nDownloading {total} pages to {PAGES_DIR}/ ...")

    for i, url in enumerate(urls, 1):
        code = url.split("/course/")[-1].split("/")[0]
        result = download_page(url, PAGES_DIR)

        if result == "ok":
            ok += 1
            print(f"  [{i}/{total}] {code} OK")
        elif result == "skip":
            skip += 1
        else:
            fail += 1

        if result != "skip":
            time.sleep(args.delay)

    existing = len(list(PAGES_DIR.glob("*.html")))
    print(f"\nDone! new={ok} skipped={skip} failed={fail}")
    print(f"Total HTML files on disk: {existing}")
    print(f"\nNext step: python3 {SCRIPT_DIR / 'parse_kurser.py'}")


if __name__ == "__main__":
    main()
