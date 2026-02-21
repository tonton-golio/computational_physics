export interface TopicDefinition {
  id: string;
  title: string;
  description: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced" | "Expert";
  color: string;
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
