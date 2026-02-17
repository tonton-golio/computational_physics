import type { ContentDocument, TopicDefinition, TopicIndex } from "@/shared/types/content";

export function sortLessonsBySlug(lessons: ContentDocument[]): ContentDocument[] {
  return [...lessons].sort((a, b) => a.slug.localeCompare(b.slug));
}

export function topicIndexFrom(topic: TopicDefinition, lessons: string[]): TopicIndex {
  return {
    topic,
    lessons: [...lessons].sort((a, b) => a.localeCompare(b)),
  };
}
