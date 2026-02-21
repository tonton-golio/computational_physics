#!/usr/bin/env python3
"""
Parse downloaded KU course HTML files into structured JSON.

Reads HTML files from data/pages/ and extracts course metadata using
BeautifulSoup with the actual kurser.ku.dk DOM structure.

Usage:
    python3 parse_kurser.py
    python3 parse_kurser.py --pages-dir ./data/pages --output ./data/courses_raw.json
"""

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_PAGES_DIR = DATA_DIR / "pages"
DEFAULT_OUTPUT = DATA_DIR / "courses_raw.json"


def get_section_text(soup, css_class):
    """Extract text from a course-item section by its CSS class."""
    section = soup.find("div", class_=css_class)
    if not section:
        return ""
    # The content is inside the collapse div (child of the section)
    collapse = section.find("div", class_="collapse")
    if collapse:
        return collapse.get_text(" ", strip=True)
    return section.get_text(" ", strip=True)


def get_section_html(soup, css_class):
    """Extract the inner collapse div from a course-item section."""
    section = soup.find("div", class_=css_class)
    if not section:
        return None
    return section.find("div", class_="collapse")


def parse_sidebar_dl(panel):
    """Parse <dl> key-value pairs from the sidebar panel."""
    info = {}
    if not panel:
        return info
    dl = panel.find("dl", class_="dl-horizontal")
    if not dl:
        return info
    dts = dl.find_all("dt")
    dds = dl.find_all("dd")
    for dt, dd in zip(dts, dds):
        key = dt.get_text(strip=True).lower().replace(" ", "_")
        val = dd.get_text(" ", strip=True)
        info[key] = val
    return info


def parse_sidebar_sections(panel):
    """Parse h5-titled list sections from sidebar (department, faculty, etc.)."""
    info = {}
    if not panel:
        return info
    for h5 in panel.find_all("h5", class_="panel-title"):
        key = h5.get_text(strip=True).lower().replace(" ", "_")
        ul = h5.find_next_sibling("ul")
        if ul:
            items = [li.get_text(strip=True) for li in ul.find_all("li")]
            info[key] = items[0] if len(items) == 1 else items
    return info


def parse_learning_outcomes(soup):
    """Parse Knowledge / Skills / Competences from the learning outcome section."""
    outcomes = {"knowledge": [], "skills": [], "competences": []}
    section_html = get_section_html(soup, "course-description")
    if not section_html:
        return outcomes

    # Find strong/b tags that indicate section headers
    current = None
    for elem in section_html.children:
        text = ""
        if hasattr(elem, "get_text"):
            text = elem.get_text(strip=True).lower()
        elif isinstance(elem, str):
            text = elem.strip().lower()

        # Detect section headers
        if "knowledge" in text:
            current = "knowledge"
            continue
        elif "skills" in text or "skill" in text:
            current = "skills"
            continue
        elif "competence" in text:
            current = "competences"
            continue

        # Collect list items under current section
        if current and hasattr(elem, "find_all"):
            for li in elem.find_all("li"):
                item = li.get_text(strip=True)
                if item:
                    outcomes[current].append(item)

    return outcomes


def parse_workload(soup):
    """Parse workload table from the course-load section."""
    workload = {}
    section_html = get_section_html(soup, "course-load")
    if not section_html:
        return workload

    items = section_html.find_all("li")
    # Items come in pairs: [Category, Hours, Category, Hours, ...]
    # First two are headers ("Category", "Hours"), skip them
    i = 0
    while i < len(items):
        text = items[i].get_text(strip=True)
        if text.lower() in ("category", "hours", "kategori", "timer"):
            i += 1
            continue
        # This should be a category name, next should be hours
        if i + 1 < len(items):
            hours_text = items[i + 1].get_text(strip=True).replace(",", ".")
            try:
                hours = float(hours_text) if "." in hours_text else int(hours_text)
                key = text.lower().replace(" ", "_")
                workload[key] = hours
                i += 2
                continue
            except ValueError:
                pass
        i += 1

    return workload


def parse_exam(soup):
    """Parse exam info from the course-exams section."""
    exam = {}
    # There can be multiple exam sections (course-exams1, course-exams2, etc.)
    for section in soup.find_all("div", class_="course-item"):
        classes = section.get("class", [])
        if not any(c.startswith("course-exams") for c in classes):
            continue
        collapse = section.find("div", class_="collapse")
        if not collapse:
            continue
        dl = collapse.find("dl")
        if dl:
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True).lower().replace(" ", "_")
                val = dd.get_text(" ", strip=True)
                exam[key] = val
    return exam


def parse_course(filepath):
    """Parse a single course HTML file into a structured dict."""
    try:
        html = filepath.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        # Check for error pages
        title_tag = soup.find("title")
        if title_tag and "error" in title_tag.get_text().lower():
            return None

        code = filepath.stem
        course = {
            "code": code,
            "url": f"https://kurser.ku.dk/course/{code}",
        }

        # Title
        h1 = soup.find("h1", class_="courseTitle")
        if h1:
            raw_title = h1.get_text(strip=True)
            # Remove course code prefix from title
            course["title"] = re.sub(rf"^{re.escape(code)}\s*", "", raw_title).strip()
        else:
            course["title"] = code

        # Education / program
        course["education"] = get_section_text(soup, "course-name")

        # Description (content section)
        course["description"] = get_section_text(soup, "course-content")

        # Learning outcomes
        course["learning_outcomes"] = parse_learning_outcomes(soup)

        # Full learning outcome text (for embedding)
        course["learning_outcome_text"] = get_section_text(soup, "course-description")

        # Sidebar panel info - find the "Course information" panel specifically
        # (the first panel-default is often the schedule panel)
        panel = None
        for p in soup.find_all("div", class_="panel-default"):
            heading = p.find(["h3", "div"], class_="panel-title")
            if heading and "course information" in heading.get_text().lower():
                panel = p
                break
            # Danish: "Kursusopl" or "Kursusfakta"
            if heading and any(kw in heading.get_text().lower() for kw in ("kursus", "kursusopl")):
                panel = p
                break
        sidebar_dl = parse_sidebar_dl(panel)
        sidebar_sections = parse_sidebar_sections(panel)

        # Extract key fields from sidebar (English or Danish labels)
        course["language"] = (sidebar_dl.get("language", "")
                              or sidebar_dl.get("sprog", ""))
        ects_raw = (sidebar_dl.get("credit", "")
                    or sidebar_dl.get("point", ""))
        ects_match = re.search(r"([\d.,]+)", ects_raw)
        course["ects"] = float(ects_match.group(1).replace(",", ".")) if ects_match else None
        course["level"] = (sidebar_dl.get("level", "")
                           or sidebar_dl.get("niveau", ""))
        course["duration"] = (sidebar_dl.get("duration", "")
                              or sidebar_dl.get("varighed", ""))
        course["placement"] = (sidebar_dl.get("placement", "")
                               or sidebar_dl.get("placering", ""))

        course["department"] = (sidebar_sections.get("contracting_department", "")
                                or sidebar_sections.get("udbydende_institut", ""))
        course["faculty"] = (sidebar_sections.get("contracting_faculty", "")
                             or sidebar_sections.get("udbydende_fakultet", ""))
        course["study_board"] = (sidebar_sections.get("study_board", "")
                                 or sidebar_sections.get("studienævn", ""))

        # Exam
        course["exam"] = parse_exam(soup)

        # Workload
        course["workload"] = parse_workload(soup)

        # Prerequisites
        course["prerequisites"] = get_section_text(soup, "course-prerequisites")
        course["recommended_qualifications"] = get_section_text(soup, "course-skills")

        # Teaching method
        course["teaching_method"] = get_section_text(soup, "course-form")

        return course

    except Exception as e:
        print(f"  ERROR parsing {filepath.name}: {e}")
        return {"code": filepath.stem, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Parse KU course HTML files into JSON")
    parser.add_argument("--pages-dir", type=Path, default=DEFAULT_PAGES_DIR,
                        help=f"Directory with HTML files (default: {DEFAULT_PAGES_DIR})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if not args.pages_dir.exists() or not any(args.pages_dir.glob("*.html")):
        print(f"No HTML files found in {args.pages_dir}/")
        print(f"Run scrape_kurser.py first.")
        exit(1)

    html_files = sorted(args.pages_dir.glob("*.html"))
    print(f"Parsing {len(html_files)} HTML files...")

    courses = []
    errors = 0
    skipped = 0
    for i, filepath in enumerate(html_files, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(html_files)}")
        result = parse_course(filepath)
        if result is None:
            skipped += 1
        elif "error" in result:
            errors += 1
            courses.append(result)
        else:
            courses.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Parsed {len(courses)} courses ({errors} errors, {skipped} error pages skipped)")
    print(f"Output: {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")

    # Show sample
    valid = [c for c in courses if "error" not in c]
    if valid:
        sample = valid[0]
        print(f"\nSample: {sample['code']} - {sample.get('title', '?')}")
        print(f"  Department: {sample.get('department', '?')}")
        print(f"  ECTS: {sample.get('ects', '?')}")
        print(f"  Language: {sample.get('language', '?')}")

    print(f"\nNext step: python3 {SCRIPT_DIR / 'build_course_list.py'}")


if __name__ == "__main__":
    main()
