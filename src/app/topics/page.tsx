import { TOPICS } from "@/lib/content";
import { TOPIC_ROUTES } from "@/lib/topic-navigation";
import { getSearchableLessonsForTopic } from "@/lib/topic-navigation.server";
import { TopicsSearchGrid } from "@/components/topics/TopicsSearchGrid";

export default function TopicsPage() {
  const entries = TOPIC_ROUTES.map((topic) => ({
    routeSlug: topic.routeSlug,
    meta: TOPICS[topic.contentId],
    lessons: getSearchableLessonsForTopic(topic.contentId),
  }));

  return (
    <div className="h-[calc(100vh-4rem)] w-full overflow-hidden bg-[var(--background)]">
      <div className="h-full w-full">
        <TopicsSearchGrid entries={entries} />
      </div>
    </div>
  );
}
