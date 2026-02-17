import { Suspense } from "react";
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
