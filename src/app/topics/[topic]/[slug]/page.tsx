import Link from "next/link";
import { notFound } from "next/navigation";
import { MarkdownContent } from "@/components/content/MarkdownContent";
import { CollapsibleTopicLayout } from "@/components/layout/CollapsibleTopicLayout";
import { TopicSidebar } from "@/components/layout/TopicSidebar";
import { TOPICS } from "@/lib/topic-config";
import { getTopicLesson } from "@/features/content/content-gateway";
import { resolveTopicRoute, topicHref, TOPIC_ROUTES } from "@/lib/topic-navigation";
import { isLandingPage, getLessonsForTopic } from "@/features/content/topic-lessons";

interface LessonPageProps {
  params: Promise<{ topic: string; slug: string }>;
}

export function generateStaticParams() {
  return TOPIC_ROUTES.flatMap((topic) =>
    getLessonsForTopic(topic.topicId).map((lesson) => ({
      topic: topic.routeSlug,
      slug: lesson.slug,
    }))
  );
}

export default async function LessonPage({ params }: LessonPageProps) {
  const { topic, slug } = await params;
  const route = resolveTopicRoute(topic);
  if (!route) notFound();

  const topicMeta = TOPICS[route.topicId];
  const lessons = getLessonsForTopic(route.topicId).filter((l) => !isLandingPage(l.slug));
  const lesson = await getTopicLesson(route.topicId, slug);
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
            activeSlug={lesson.slug}
          />
        }
      >
        <article className="rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-5 md:p-7">
          <div className="mb-6 flex flex-wrap items-center gap-3 border-b border-[var(--border-strong)] pb-4">
            <Link href="/topics" className="text-xs text-[var(--accent)] hover:text-[var(--accent-strong)]">
              /topics
            </Link>
            <span className="text-xs text-[var(--text-soft)]">/</span>
            <Link href={topicHref(route.routeSlug)} className="text-xs text-[var(--accent)] hover:text-[var(--accent-strong)]">
              {topicMeta.title}
            </Link>
            <span className="text-xs text-[var(--text-soft)]">/</span>
            <span className="text-xs text-[var(--text-strong)]">{lesson.title}</span>
          </div>
          <MarkdownContent content={lesson.content} />
        </article>
      </CollapsibleTopicLayout>
    </div>
  );
}
