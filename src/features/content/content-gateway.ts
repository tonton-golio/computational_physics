import { unstable_cache } from "next/cache";
import { TOPICS } from "@/lib/topic-config";
import { AppError } from "@/shared/errors/app-error";
import type { ContentDocument, TopicDefinition, TopicIndex } from "@/shared/types/content";
import { readLessonDocument } from "@/infra/content/file-content-repository";
import { getOrderedLessonSlugs } from "@/features/content/topic-lessons";

function asTopicDefinition(topicId: string): TopicDefinition {
  const topic = TOPICS[topicId as keyof typeof TOPICS];
  if (!topic) {
    throw new AppError("NOT_FOUND", "Topic not found.", 404, { topicId });
  }
  return {
    id: topicId,
    title: topic.title,
    description: topic.description,
    difficulty: topic.difficulty,
    color: topic.color,
    relatedTopics: topic.relatedTopics ?? [],
  };
}

async function loadTopicIndexes(): Promise<TopicIndex[]> {
  return Object.keys(TOPICS).map((topicId) => ({
    topic: asTopicDefinition(topicId),
    lessons: [...getOrderedLessonSlugs(topicId)],
  }));
}

const getTopicIndexesCached = unstable_cache(loadTopicIndexes, ["topics:index"], {
  revalidate: 3600,
  tags: ["topics:index"],
});

export async function getTopicIndexes(): Promise<TopicIndex[]> {
  try {
    return await getTopicIndexesCached();
  } catch {
    // Test runners and non-Next runtimes may not provide incremental cache.
    return loadTopicIndexes();
  }
}

export async function getTopicLessons(topicId: string): Promise<{ topic: TopicDefinition; lessons: ContentDocument[] }> {
  const topic = asTopicDefinition(topicId);
  const lessons = getOrderedLessonSlugs(topicId)
    .map((slug) => readLessonDocument(topicId, slug))
    .filter((doc): doc is ContentDocument => doc !== null);

  return { topic, lessons };
}

export async function getTopicLesson(topicId: string, slug: string): Promise<ContentDocument | null> {
  asTopicDefinition(topicId);
  return readLessonDocument(topicId, slug);
}
