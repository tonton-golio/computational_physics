import type { TopicConfig } from "@/lib/topic-config";

export interface TopicDefinition extends TopicConfig {
  id: string;
  relatedTopics: string[];
}

export interface TopicIndex {
  topic: TopicDefinition;
  lessons: string[];
}

export interface ContentDocument {
  topicId: string;
  slug: string;
  title: string;
  content: string;
  meta: Record<string, string>;
}
