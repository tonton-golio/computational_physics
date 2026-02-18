import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { MarkdownContent } from "@/components/content/MarkdownContent";
import { ExportPdfButton } from "@/components/content/ExportPdfButton";
import { TOPICS } from "@/lib/content";
import { readLessonDocument } from "@/infra/content/file-content-repository";
import { resolveTopicRoute, topicHref, TOPIC_ROUTES } from "@/lib/topic-navigation";
import {
  getLandingPageSlug,
  getLessonsForTopic,
  getOrderedLessonSlugs,
  isLandingPage,
} from "@/lib/topic-navigation.server";

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

  const landingSlug = getLandingPageSlug(route.contentId);

  if (!landingSlug) {
    const slugs = getOrderedLessonSlugs(route.contentId);
    if (!slugs.length) redirect("/topics");
    redirect(topicHref(route.routeSlug, slugs[0]));
  }

  const topicMeta = TOPICS[route.contentId];
  const lessons = getLessonsForTopic(route.contentId).filter((l) => !isLandingPage(l.slug));
  const lesson = readLessonDocument(route.contentId, landingSlug);
  if (!lesson) notFound();

  return (
    <div className="min-h-screen bg-[var(--background)] px-4 py-6 md:px-6">
      <div className="grid w-full gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
        <aside className="h-fit rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-4">
          <div className="border-b border-[var(--border-strong)] pb-3">
            <div className="text-[11px] uppercase tracking-[0.18em] text-[var(--accent)]">Topic</div>
            <div className="mt-1 text-lg font-semibold text-[var(--text-strong)]">{topicMeta.title}</div>
            <p className="mt-2 text-xs text-[var(--text-muted)]">{topicMeta.description}</p>
            <ExportPdfButton topicContentId={route.contentId} topicTitle={topicMeta.title} />
          </div>

          <div className="mt-4 space-y-1 font-mono text-sm">
            {lessons.map((entry) => (
              <Link
                key={entry.slug}
                href={topicHref(route.routeSlug, entry.slug)}
                title={entry.summary}
                className="block rounded-md border border-transparent px-3 py-2 transition text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:bg-[var(--surface-2)] hover:text-[var(--text-strong)]"
              >
                <span className="text-[var(--accent)]">$</span> {entry.title}
                {entry.summary && (
                  <span className="mt-0.5 block font-sans text-[10px] leading-tight text-[var(--text-soft)]">
                    {entry.summary}
                  </span>
                )}
              </Link>
            ))}
          </div>
        </aside>

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
      </div>
    </div>
  );
}
