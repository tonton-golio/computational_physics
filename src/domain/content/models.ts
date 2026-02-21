import type { TopicDefinition, TopicIndex } from "@/shared/types/content";

export function topicIndexFrom(topic: TopicDefinition, lessons: string[]): TopicIndex {
  return {
    topic,
    lessons: [...lessons],
  };
}
