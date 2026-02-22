import { Suspense } from "react";
import { TOPICS } from "@/lib/topic-config";
import { TOPIC_ROUTES } from "@/lib/topic-navigation";
import { getSearchableLessonsForTopic } from "@/features/content/topic-lessons";
import { TopicsSearchGrid } from "@/components/topics/TopicsSearchGrid";

export default function TopicsPage() {
  const entries = TOPIC_ROUTES.map((topic) => ({
    routeSlug: topic.routeSlug,
    topicId: topic.topicId,
    meta: TOPICS[topic.topicId],
    lessons: getSearchableLessonsForTopic(topic.topicId),
  })).sort((a, b) => a.meta.title.localeCompare(b.meta.title));

  return (
    <div className="h-[calc(100vh-4rem)] w-full overflow-hidden">
      <div className="h-full w-full">
        <Suspense
          fallback={
            <div className="mt-8 rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-6 text-center text-sm text-[var(--text-muted)]">
              Loading topic search...
            </div>
          }
        >
          <TopicsSearchGrid entries={entries} />
        </Suspense>
      </div>
    </div>
  );
}
