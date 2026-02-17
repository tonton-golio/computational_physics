#!/usr/bin/env python3
"""
Build 2D point-cloud data for topic sub-topics.

Pipeline:
1) Read all sub-topic markdown/text content under content/topics/*.
2) Embed each sub-topic with OpenAI text-embedding-3-small.
3) Project embedding vectors to 2D via t-SNE.
4) Write JSON artifact consumed by the Topics point-cloud UI.
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = ROOT / "public" / "data" / "topic-point-cloud.json"
EMBED_MODEL = "text-embedding-3-small"


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
  if topic_id == "online-reinforcement-learning":
    return "online-reinforcement"
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


def build_point_cloud(output_path: Path, batch_size: int, max_chars: int) -> None:
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY is required.")

  docs = discover_docs(CONTENT_ROOT)
  if not docs:
    raise RuntimeError(f"No sub-topic content found in {CONTENT_ROOT}")

  client = OpenAI(api_key=api_key)
  embedding_inputs = [f"{doc.title}\n\n{doc.content}"[:max_chars] for doc in docs]
  vectors = embed_texts(client, embedding_inputs, batch_size=batch_size)

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
  points_norm = normalize_xy(points_2d)

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
  print(f"Wrote point-cloud data to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate topic point-cloud JSON.")
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
  parser.add_argument("--batch-size", type=int, default=64, help="Embedding API batch size.")
  parser.add_argument(
    "--max-chars",
    type=int,
    default=6000,
    help="Max chars per lesson included in embedding/search text.",
  )
  args = parser.parse_args()
  build_point_cloud(output_path=args.output, batch_size=args.batch_size, max_chars=args.max_chars)


if __name__ == "__main__":
  main()
