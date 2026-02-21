#!/usr/bin/env python3
"""
Build 2D points data for topic sub-topics.

Pipeline:
1) Read all sub-topic markdown/text content under content/topics/*.
2) Embed each sub-topic with OpenAI text-embedding-3-small.
3) Project embedding vectors to 2D via t-SNE.
4) Write JSON artifact consumed by the Topics points UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = ROOT / "content" / "topics"
DEFAULT_OUTPUT = ROOT / "public" / "data" / "topic-points.json"
CACHE_DIR = ROOT / ".cache" / "topic_points"
EMBED_MODEL = "text-embedding-3-small"
LANDING_PAGE_SLUGS = {"home", "intro", "introduction", "landingPage"}


@dataclass
class TopicDocument:
  topic_id: str
  route_slug: str
  lesson_slug: str
  title: str
  content: str


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
  match = re.match(r"^---\n([\s\S]*?)\n---\n", raw)
  if not match:
    return {}, raw
  meta: dict[str, str] = {}
  for line in match.group(1).split("\n"):
    key, sep, value = line.partition(":")
    if not sep:
      continue
    meta[key.strip()] = value.strip()
  body = raw[len(match.group(0)) :]
  return meta, body


def title_from_markdown(body: str, fallback: str) -> str:
  match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
  return match.group(1).strip() if match else fallback


def topic_route_slug(topic_id: str) -> str:
  return topic_id


def topic_title(topic_id: str) -> str:
  return topic_id.replace("-", " ").title()


def discover_docs(content_root: Path) -> list[TopicDocument]:
  docs: list[TopicDocument] = []
  for topic_dir in sorted([p for p in content_root.iterdir() if p.is_dir()]):
    topic_id = topic_dir.name
    route_slug = topic_route_slug(topic_id)
    files = sorted(
      [p for p in topic_dir.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"}],
      key=lambda p: p.name.lower(),
    )
    for path in files:
      lesson_slug = path.stem
      if lesson_slug in LANDING_PAGE_SLUGS:
        continue
      raw = path.read_text(encoding="utf-8")
      meta, body = parse_frontmatter(raw)
      title = meta.get("title") or title_from_markdown(body, lesson_slug)
      docs.append(
        TopicDocument(
          topic_id=topic_id,
          route_slug=route_slug,
          lesson_slug=lesson_slug,
          title=title,
          content=body,
        )
      )
  return docs


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
  for i in range(0, len(items), batch_size):
    yield items[i : i + batch_size]


def embed_texts(client: OpenAI, texts: list[str], batch_size: int) -> np.ndarray:
  vectors: list[list[float]] = []
  for batch in batched(texts, batch_size):
    response = client.embeddings.create(model=EMBED_MODEL, input=batch)
    ordered = sorted(response.data, key=lambda d: d.index)
    vectors.extend([item.embedding for item in ordered])
  return np.array(vectors, dtype=np.float32)


def normalize_xy(points_2d: np.ndarray) -> np.ndarray:
  mins = points_2d.min(axis=0)
  maxs = points_2d.max(axis=0)
  spans = np.where((maxs - mins) == 0, 1.0, maxs - mins)
  return (points_2d - mins) / spans


def load_cache() -> dict:
  meta_path = CACHE_DIR / "meta.json"
  if not meta_path.exists():
    return {}
  with open(meta_path, "r") as f:
    meta = json.load(f)
  embeddings_path = CACHE_DIR / "embeddings.npy"
  points_path = CACHE_DIR / "points_2d.npy"
  if embeddings_path.exists():
    meta["embeddings"] = np.load(embeddings_path)
  if points_path.exists():
    meta["points_2d"] = np.load(points_path)
  return meta


def save_cache(
  doc_keys: list[str],
  content_hashes: dict[str, str],
  topics: list[str],
  embeddings: np.ndarray,
  points_2d: np.ndarray,
) -> None:
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  meta = {"doc_keys": doc_keys, "content_hashes": content_hashes, "topics": topics}
  with open(CACHE_DIR / "meta.json", "w") as f:
    json.dump(meta, f)
  np.save(CACHE_DIR / "embeddings.npy", embeddings)
  np.save(CACHE_DIR / "points_2d.npy", points_2d)


def build_points(output_path: Path, batch_size: int, max_chars: int) -> None:
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY is required.")

  docs = discover_docs(CONTENT_ROOT)
  if not docs:
    raise RuntimeError(f"No sub-topic content found in {CONTENT_ROOT}")

  # Compute content hashes for change detection
  doc_keys = [f"{doc.topic_id}/{doc.lesson_slug}" for doc in docs]
  embedding_inputs = [f"{doc.title}\n\n{doc.content}"[:max_chars] for doc in docs]
  content_hashes = {
    key: hashlib.sha256(text.encode()).hexdigest()
    for key, text in zip(doc_keys, embedding_inputs)
  }

  # Load cache
  cache = load_cache()
  cached_hashes = cache.get("content_hashes", {})
  cached_doc_keys = cache.get("doc_keys", [])
  cached_topics = cache.get("topics", [])
  cached_embeddings = cache.get("embeddings")
  cached_points = cache.get("points_2d")

  # Index cached embeddings by doc key
  cached_embedding_map: dict[str, np.ndarray] = {}
  if cached_embeddings is not None:
    for j, ckey in enumerate(cached_doc_keys):
      if j < len(cached_embeddings):
        cached_embedding_map[ckey] = cached_embeddings[j]

  # Determine which docs need re-embedding
  to_embed_texts: list[str] = []
  to_embed_indices: list[int] = []
  for i, key in enumerate(doc_keys):
    if cached_hashes.get(key) == content_hashes[key] and key in cached_embedding_map:
      continue
    to_embed_texts.append(embedding_inputs[i])
    to_embed_indices.append(i)

  reused = len(docs) - len(to_embed_texts)
  print(f"Documents: {len(docs)} total, {reused} cached, {len(to_embed_texts)} to embed")

  # Build embedding matrix: start from cached, fill in new/changed
  embed_dim = cached_embeddings.shape[1] if cached_embeddings is not None else 1536
  vectors = np.zeros((len(docs), embed_dim), dtype=np.float32)

  to_embed_set = set(to_embed_indices)
  for i, key in enumerate(doc_keys):
    if i not in to_embed_set:
      vectors[i] = cached_embedding_map[key]

  if to_embed_texts:
    client = OpenAI(api_key=api_key)
    new_vectors = embed_texts(client, to_embed_texts, batch_size=batch_size)
    for j, i in enumerate(to_embed_indices):
      vectors[i] = new_vectors[j]

  # Decide whether t-SNE needs to refit
  current_topics = sorted({doc.topic_id for doc in docs})
  need_tsne_refit = (
    cached_points is None
    or set(current_topics) != set(cached_topics)
    or doc_keys != cached_doc_keys
  )

  if need_tsne_refit:
    reason = "no cache" if cached_points is None else (
      "topics changed" if set(current_topics) != set(cached_topics) else "documents changed"
    )
    print(f"Running t-SNE fit ({reason})")
    perplexity = min(30, max(5, len(docs) // 6))
    if perplexity >= len(docs):
      perplexity = max(1, len(docs) - 1)
    tsne = TSNE(
      n_components=2,
      random_state=42,
      perplexity=perplexity,
      learning_rate="auto",
      init="pca",
    )
    points_2d = tsne.fit_transform(vectors)
  else:
    print("Reusing cached t-SNE positions (no structural changes)")
    points_2d = cached_points
    perplexity = min(30, max(5, len(docs) // 6))
    if perplexity >= len(docs):
      perplexity = max(1, len(docs) - 1)

  points_norm = normalize_xy(points_2d)

  # Save cache
  save_cache(doc_keys, content_hashes, current_topics, vectors, points_2d)

  topics = sorted({doc.topic_id for doc in docs})
  payload = {
    "meta": {
      "generatedAt": datetime.now(timezone.utc).isoformat(),
      "embeddingModel": EMBED_MODEL,
      "projection": "tsne-2d",
      "inputCount": len(docs),
      "perplexity": perplexity,
    },
    "topics": [{"topicId": topic_id, "topicTitle": topic_title(topic_id)} for topic_id in topics],
    "points": [
      {
        "topicId": doc.topic_id,
        "topicTitle": topic_title(doc.topic_id),
        "routeSlug": doc.route_slug,
        "lessonSlug": doc.lesson_slug,
        "lessonTitle": doc.title,
        "x": float(points_norm[i][0]),
        "y": float(points_norm[i][1]),
        "searchText": f"{doc.title}\n{doc.content}".lower()[:max_chars],
      }
      for i, doc in enumerate(docs)
    ],
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
  print(f"Wrote points data to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate topic points JSON.")
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
  parser.add_argument("--batch-size", type=int, default=64, help="Embedding API batch size.")
  parser.add_argument(
    "--max-chars",
    type=int,
    default=6000,
    help="Max chars per lesson included in embedding/search text.",
  )
  args = parser.parse_args()
  build_points(output_path=args.output, batch_size=args.batch_size, max_chars=args.max_chars)


if __name__ == "__main__":
  main()
