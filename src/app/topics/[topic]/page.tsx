import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { MarkdownContent } from "@/components/content/MarkdownContent";
import { CollapsibleTopicLayout } from "@/components/layout/CollapsibleTopicLayout";
import { TopicSidebar } from "@/components/layout/TopicSidebar";
import { TOPICS } from "@/lib/topic-config";
import { getTopicLesson } from "@/features/content/content-gateway";
import { resolveTopicRoute, topicHref, TOPIC_ROUTES } from "@/lib/topic-navigation";
import {
  isLandingPage,
  getLandingPageSlug,
  getLessonsForTopic,
  getOrderedLessonSlugs,
} from "@/features/content/topic-lessons";

interface TopicPageProps {
  params: Promise<{ topic: string }>;
}

export function generateStaticParams() {
  return TOPIC_ROUTES.map((topic) => ({ topic: topic.routeSlug }));
}

export default async function TopicPage({ params }: TopicPageProps) {
  const { topic } = await params;
  const route = resolveTopicRoute(topic);
  if (!route) notFound();

  const landingSlug = getLandingPageSlug(route.topicId);

  if (!landingSlug) {
    const slugs = getOrderedLessonSlugs(route.topicId);
    if (!slugs.length) redirect("/topics");
    redirect(topicHref(route.routeSlug, slugs[0]));
  }

  const topicMeta = TOPICS[route.topicId];
  const lessons = getLessonsForTopic(route.topicId).filter((l) => !isLandingPage(l.slug));
  const lesson = await getTopicLesson(route.topicId, landingSlug);
  if (!lesson) notFound();

  return (
    <div className="min-h-screen px-4 py-6 md:px-6">
      <CollapsibleTopicLayout
        sidebar={
          <TopicSidebar
            routeSlug={route.routeSlug}
            topicMeta={topicMeta}
            topicId={route.topicId}
            lessons={lessons}
          />
        }
      >
        <article className="rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-5 md:p-7">
          <div className="mb-6 flex flex-wrap items-center gap-3 border-b border-[var(--border-strong)] pb-4">
            <Link href="/topics" className="text-xs text-[var(--accent)] hover:text-[var(--accent-strong)]">
              /topics
            </Link>
            <span className="text-xs text-[var(--text-soft)]">/</span>
            <span className="text-xs text-[var(--text-strong)]">{topicMeta.title}</span>
          </div>
          <MarkdownContent content={lesson.content} />
        </article>
      </CollapsibleTopicLayout>
    </div>
  );
}
