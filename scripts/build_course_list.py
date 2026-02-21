#!/usr/bin/env python3
"""
Build the ranked course list for the website.

Pipeline:
1. Load parsed courses from courses_raw.json
2. Filter to physics/CS departments
3. Load existing website topics (from topic-config.ts + content/)
4. Embed course descriptions and website topic descriptions (OpenAI)
5. Compute cosine similarity to find already-covered and most-relevant courses
6. Sort: already-on-website first, then by relevance to existing content
7. Output final course_list.json

Usage:
    python3 build_course_list.py
    python3 build_course_list.py --similarity-threshold 0.75
    python3 build_course_list.py --no-embed   # skip embedding, just filter
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from openai import OpenAI

# Load .env from project root if OPENAI_API_KEY not already set
_env_file = Path(__file__).resolve().parents[1] / ".env"
if not os.getenv("OPENAI_API_KEY") and _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
            break

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"

DEFAULT_INPUT = DATA_DIR / "courses_raw.json"
DEFAULT_OUTPUT = DATA_DIR / "course_list.json"
EMBEDDINGS_CACHE = DATA_DIR / "embeddings_cache.json"

TOPIC_CONFIG_PATH = PROJECT_ROOT / "src" / "lib" / "topic-config.ts"
CONTENT_ROOT = PROJECT_ROOT / "content" / "topics"

EMBED_MODEL = "text-embedding-3-small"

# Manual overrides: course code -> website topic ID
# These are KU courses that directly correspond to existing website topics
# but may fall below the automatic similarity threshold.
MANUAL_ON_WEBSITE = {
    "NFYK13006U": "quantum-optics",
    "NFYK18005U": "complex-physics",
    "NFYK13011U": "applied-statistics",
    "NFYK10005U": "continuum-mechanics",
    "NDAK24002U": "advanced-deep-learning",
    "NFYK20002U": "applied-machine-learning",
    "NDAK21003U": "online-reinforcement-learning",
}

# Department keywords for filtering to physics/CS/math
DEPARTMENT_KEYWORDS = [
    "niels bohr",
    "computer science",
    "datalogi",
    "mathematical sciences",
    "matematik",
    "physics",
    "fysik",
]

# Course code prefixes that are always included regardless of department text
INCLUDE_CODE_PREFIXES = ("NFYK", "NDAK", "NMAK", "NFAB")


def load_parsed_courses(path):
    """Load courses from courses_raw.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_physics_cs(courses):
    """Filter courses to physics, CS, and math departments."""
    filtered = []
    for c in courses:
        if "error" in c:
            continue

        code = c.get("code", "").upper()
        dept = (c.get("department", "") or "").lower()
        faculty = (c.get("faculty", "") or "").lower()

        # Include by code prefix
        if any(code.startswith(p) for p in INCLUDE_CODE_PREFIXES):
            filtered.append(c)
            continue

        # Include by department keyword
        combined = f"{dept} {faculty}"
        if any(kw in combined for kw in DEPARTMENT_KEYWORDS):
            filtered.append(c)
            continue

    return filtered


def load_website_topics():
    """Load existing website topics from topic-config.ts and content files."""
    topics = []

    # Parse topic-config.ts
    config_text = TOPIC_CONFIG_PATH.read_text(encoding="utf-8")

    # Extract topic entries with regex (simpler than a full TS parser)
    pattern = r'"([^"]+)":\s*\{[^}]*title:\s*"([^"]+)"[^}]*description:\s*"([^"]+)"'
    for match in re.finditer(pattern, config_text, re.DOTALL):
        topic_id = match.group(1)
        title = match.group(2)
        description = match.group(3)

        # Also gather lesson titles from content dir
        lesson_texts = []
        topic_dir = CONTENT_ROOT / topic_id
        if topic_dir.exists():
            for md_file in sorted(topic_dir.glob("*.md")):
                if md_file.stem in ("home", "intro", "introduction"):
                    # Read home page content for richer representation
                    content = md_file.read_text(encoding="utf-8")
                    lesson_texts.append(content[:2000])
                else:
                    # Just use lesson filename as a hint
                    lesson_name = md_file.stem.replace("-", " ").replace("_", " ")
                    lesson_texts.append(lesson_name)

        topics.append({
            "id": topic_id,
            "title": title,
            "description": description,
            "lesson_hints": lesson_texts,
        })

    return topics


def build_embedding_text_for_course(course):
    """Build a text snippet for embedding a KU course."""
    parts = [course.get("title", "")]
    if course.get("description"):
        parts.append(course["description"])
    if course.get("learning_outcome_text"):
        parts.append(course["learning_outcome_text"][:1000])
    return "\n\n".join(parts)[:4000]


def build_embedding_text_for_topic(topic):
    """Build a text snippet for embedding a website topic."""
    parts = [topic["title"], topic["description"]]
    for hint in topic.get("lesson_hints", []):
        parts.append(hint)
    return "\n\n".join(parts)[:4000]


def embed_texts(client, texts, batch_size=64):
    """Embed a list of texts using OpenAI API, returning numpy array."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors.extend([item.embedding for item in ordered])
        if i + batch_size < len(texts):
            print(f"  Embedded {i + batch_size}/{len(texts)}...")
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def load_embedding_cache():
    """Load cached embeddings to avoid re-embedding."""
    if EMBEDDINGS_CACHE.exists():
        with open(EMBEDDINGS_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_embedding_cache(cache):
    """Save embedding cache."""
    with open(EMBEDDINGS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Build ranked course list for the website")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Input courses JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output course list JSON (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--similarity-threshold", type=float, default=0.72,
                        help="Cosine similarity threshold for 'already on website' (default: 0.72)")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip embedding step (just filter and output)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding API batch size (default: 64)")
    args = parser.parse_args()

    # 1. Load parsed courses
    print(f"Loading courses from {args.input}...")
    all_courses = load_parsed_courses(args.input)
    print(f"  Loaded {len(all_courses)} total courses")

    # 2. Filter to physics/CS/math
    courses = filter_physics_cs(all_courses)
    print(f"  Filtered to {len(courses)} physics/CS/math courses")

    # 3. Load website topics
    print(f"\nLoading website topics from {TOPIC_CONFIG_PATH}...")
    topics = load_website_topics()
    print(f"  Found {len(topics)} existing topics: {[t['title'] for t in topics]}")

    # 4. Build output records
    records = []
    for c in courses:
        records.append({
            "code": c["code"],
            "title": c.get("title", ""),
            "department": c.get("department", ""),
            "description": c.get("description", ""),
            "url": c.get("url", ""),
            "ects": c.get("ects"),
            "language": c.get("language", ""),
            "level": c.get("level", ""),
            "learning_outcomes": c.get("learning_outcomes", {}),
            "on_website": False,
            "matched_topic": None,
            "similarity_score": 0.0,
        })

    # Apply manual overrides (works with or without embedding)
    for record in records:
        if record["code"] in MANUAL_ON_WEBSITE:
            record["on_website"] = True
            record["matched_topic"] = MANUAL_ON_WEBSITE[record["code"]]

    if args.no_embed:
        # Sort: on_website first, then alphabetically
        records.sort(key=lambda r: (-int(r["on_website"]), r["title"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        on_site = sum(1 for r in records if r["on_website"])
        print(f"\nWrote {len(records)} courses to {args.output} (no embedding)")
        print(f"  {on_site} marked as on website (manual overrides only)")
        return

    # 5. Embed everything
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nOPENAI_API_KEY not set. Run with --no-embed to skip embedding,")
        print("or set the key: export OPENAI_API_KEY=sk-...")
        exit(1)

    client = OpenAI(api_key=api_key)
    cache = load_embedding_cache()

    # Build texts for embedding
    course_texts = [build_embedding_text_for_course(c) for c in courses]
    topic_texts = [build_embedding_text_for_topic(t) for t in topics]

    # Check cache for courses
    texts_to_embed = []
    text_indices = []  # (type, index) tuples
    for i, text in enumerate(course_texts):
        cache_key = f"course:{courses[i]['code']}"
        if cache_key not in cache:
            texts_to_embed.append(text)
            text_indices.append(("course", i))

    for i, text in enumerate(topic_texts):
        cache_key = f"topic:{topics[i]['id']}"
        if cache_key not in cache:
            texts_to_embed.append(text)
            text_indices.append(("topic", i))

    if texts_to_embed:
        print(f"\nEmbedding {len(texts_to_embed)} texts ({len(cache)} cached)...")
        new_vectors = embed_texts(client, texts_to_embed, batch_size=args.batch_size)

        for idx, (typ, orig_idx) in enumerate(text_indices):
            if typ == "course":
                cache_key = f"course:{courses[orig_idx]['code']}"
            else:
                cache_key = f"topic:{topics[orig_idx]['id']}"
            cache[cache_key] = new_vectors[idx].tolist()

        save_embedding_cache(cache)
        print(f"  Cached {len(texts_to_embed)} new embeddings")
    else:
        print(f"\nAll {len(course_texts) + len(topic_texts)} embeddings found in cache")

    # 6. Compute similarity scores
    print("\nComputing similarities...")
    topic_vectors = []
    for t in topics:
        vec = cache[f"topic:{t['id']}"]
        topic_vectors.append(np.array(vec, dtype=np.float32))

    # Compute mean of all website topic vectors (for ranking by overall relevance)
    mean_topic_vec = np.mean(topic_vectors, axis=0)
    mean_topic_vec = mean_topic_vec / np.linalg.norm(mean_topic_vec)  # normalize

    for i, record in enumerate(records):
        course_vec = np.array(cache[f"course:{courses[i]['code']}"], dtype=np.float32)

        # Similarity to closest individual topic (for matching)
        best_sim = 0.0
        best_topic = None
        for j, topic_vec in enumerate(topic_vectors):
            sim = cosine_similarity(course_vec, topic_vec)
            if sim > best_sim:
                best_sim = sim
                best_topic = topics[j]["id"]

        # Similarity to the mean of all website topics (for ranking)
        mean_sim = cosine_similarity(course_vec, mean_topic_vec)

        record["similarity_score"] = round(best_sim, 4)
        record["mean_similarity"] = round(mean_sim, 4)
        record["matched_topic"] = best_topic

        # Apply manual override first, then threshold fallback
        code = courses[i]["code"]
        if code in MANUAL_ON_WEBSITE:
            record["on_website"] = True
            record["matched_topic"] = MANUAL_ON_WEBSITE[code]
        elif best_sim >= args.similarity_threshold:
            record["on_website"] = True

    # 7. Sort: on_website first, then rest by mean similarity to website content
    records.sort(key=lambda r: (-int(r["on_website"]), -r["mean_similarity"]))

    # 8. Output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    on_site = sum(1 for r in records if r["on_website"])
    print(f"\nDone! Wrote {len(records)} courses to {args.output}")
    print(f"  {on_site} marked as already on website (threshold={args.similarity_threshold})")
    print(f"  {len(records) - on_site} new courses to consider adding")

    # Show top matches
    print(f"\nAlready on website:")
    for r in records[:on_site]:
        print(f"  [ON SITE] best={r['similarity_score']:.3f} mean={r['mean_similarity']:.3f} {r['code']} {r['title']}")
        print(f"            -> {r['matched_topic']}")

    print(f"\nHighest-relevance new courses (sorted by mean similarity to website):")
    new_courses = [r for r in records if not r["on_website"]]
    for r in new_courses[:20]:
        print(f"  mean={r['mean_similarity']:.3f} best={r['similarity_score']:.3f} {r['code']} {r['title']}")
        print(f"          closest to: {r['matched_topic']}")


if __name__ == "__main__":
    main()
